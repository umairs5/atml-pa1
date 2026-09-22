"""PROSER fine-tuning with classifier and layer-2 data placeholders."""
from __future__ import annotations
import argparse, csv, random
from pathlib import Path
import numpy as np, torch, yaml
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from task4.data import make_split, transforms_for
from task4.model import CifarResNet18
KNOWN=10
def cfg(path):
    c=yaml.safe_load(Path(path).read_text()); b=yaml.safe_load(Path(c.pop('base')).read_text()); b.update(c); return b
def seed(s): random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
def dummy_loss(logits, labels=None):
    if labels is not None:
        known=logits[:,:KNOWN].clone(); known.scatter_(1,labels[:,None],float('-inf')); logits=torch.cat((known,logits[:,KNOWN:]),1)
    return (torch.logsumexp(logits,1)-torch.logsumexp(logits[:,KNOWN:],1)).mean()
def pairs(labels):
    for _ in range(32):
        p=torch.randperm(len(labels),device=labels.device)
        if not labels.eq(labels[p]).any(): return p
    raise RuntimeError('Unable to form different-class mixup pairs.')
@torch.no_grad()
def valid_accuracy(model,loader,device):
    model.eval(); right=total=0
    for x,y in loader: right+=(model(x.to(device))[:,:KNOWN].argmax(1).cpu()==y).sum().item(); total+=len(y)
    return right/total
def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',default='task4/configs/proser.yaml'); p.add_argument('--vanilla-checkpoint',required=True); p.add_argument('--device',default=None); a=p.parse_args(); c=cfg(a.config); seed(c['seed']); device=torch.device(a.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    split=make_split(c['data']['root'],c['data']['split_path'],c['seed'],c['data']['validation_fraction']); train=datasets.CIFAR10(c['data']['root'],train=True,download=True,transform=transforms_for('vanilla',True)); valid=datasets.CIFAR10(c['data']['root'],train=True,download=True,transform=transforms_for('vanilla',False)); loader=DataLoader(Subset(train,split['train_indices']),128,shuffle=True,num_workers=c['training']['num_workers']); vl=DataLoader(Subset(valid,split['validation_indices']),128,num_workers=c['training']['num_workers'])
    model=CifarResNet18(15).to(device); saved=torch.load(a.vanilla_checkpoint,map_location='cpu',weights_only=False)['model']; own=model.state_dict(); own.update({k:v for k,v in saved.items() if k in own and own[k].shape==v.shape}); own['net.fc.weight'][:10]=saved['net.fc.weight']; own['net.fc.bias'][:10]=saved['net.fc.bias']; model.load_state_dict(own)
    opt=torch.optim.SGD(model.parameters(),lr=.001,momentum=.9,weight_decay=5e-4); sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,50); ce=nn.CrossEntropyLoss(); out=Path(c['output']['root'])/'proser'; out.mkdir(parents=True,exist_ok=True); history=[]; best=-1
    for epoch in range(1,51):
        model.train(); total=0
        for x,y in loader:
            x,y=x.to(device),y.to(device); x1,x2=x.chunk(2); y1,y2=y.chunk(2); opt.zero_grad(); logits=model(x1); known=ce(logits[:,:10],y1); classifier=dummy_loss(logits,y1); hidden=model.forward_to_layer2(x2); lam=torch.distributions.Beta(2,2).sample((len(y2),)).to(device).view(-1,1,1,1); mixed=lam*hidden+(1-lam)*hidden[pairs(y2)]; mix_logits=model.net.fc(model.net.avgpool(model.forward_from_layer2(mixed)).flatten(1)); data=dummy_loss(mix_logits); loss=known+classifier+.1*data; loss.backward(); opt.step(); total+=loss.item()
        score=valid_accuracy(model,vl,device); sch.step(); history.append({'epoch':epoch,'total_loss':total/len(loader),'validation_accuracy':score});
        if score>best: best=score; torch.save({'model':model.state_dict(),'config':c,'epoch':epoch,'validation_accuracy':score},out/'best.pt')
        print(f'Epoch {epoch:03d}: validation accuracy={score:.4f}')
    with (out/'history.csv').open('w',newline='',encoding='utf-8') as f: w=csv.DictWriter(f,fieldnames=history[0]); w.writeheader(); w.writerows(history)
if __name__=='__main__': main()
