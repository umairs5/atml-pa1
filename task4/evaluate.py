"""Evaluation-only OSR protocol for frozen Vanilla/GCSC checkpoints."""
from __future__ import annotations
import argparse, csv
from pathlib import Path
import numpy as np, torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from task4.data import make_split, transforms_for, unknown_indices
from task4.model import CifarResNet18
from task4.scores import energy, mahalanobis, mls, msp

@torch.no_grad()
def outputs(model, loader, device):
    model.eval(); features=[]; logits=[]; labels=[]
    for images, target in loader:
        images=images.to(device); features.append(model.forward_features(images).cpu()); logits.append(model(images).cpu()); labels.append(target)
    return tuple(x.numpy() for x in (torch.cat(features), torch.cat(logits), torch.cat(labels)))

def checkpoint(path, device):
    state=torch.load(path, map_location=device, weights_only=False)
    classes=state['model']['net.fc.weight'].shape[0]
    model=CifarResNet18(classes).to(device); model.load_state_dict(state['model']); return model

def score_values(name, logits, features, means=None, variance=None):
    logits, features = torch.tensor(logits), torch.tensor(features)
    if name == "msp": return msp(logits).numpy()
    if name == "mls": return mls(logits).numpy()
    if name == "energy": return energy(logits).numpy()
    if name == "mahalanobis": return mahalanobis(features, means, variance).numpy()
    raise ValueError(f"Unknown score: {name}")

def placeholder_score(logits):
    """PROSER unknownness: strongest dummy logit minus strongest known logit."""
    return (torch.tensor(logits)[:, 10:].amax(1) - torch.tensor(logits)[:, :10].amax(1)).numpy()

def row(method, score, known_validation, known_test, unknown, group, csa):
    threshold=float(np.quantile(known_validation, .95)); labels=np.r_[np.zeros(len(known_test)), np.ones(len(unknown))]; values=np.r_[known_test, unknown]
    return {"method":method,"score":score,"closed_set_accuracy":csa,"unknown_group":group,"auroc":roc_auc_score(labels,values),"threshold":threshold,"known_test_acceptance_rate":float((known_test<=threshold).mean()),"unknown_rejection_rate":float((unknown>threshold).mean()),"fpr_at_95_tpr":float((unknown<=threshold).mean())}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--checkpoint',action='append',required=True,help='method=path'); p.add_argument('--data-root',default='data/cifar'); p.add_argument('--split-path',default='task4/data/splits/cifar10_seed6304.json'); p.add_argument('--results-dir',default='task4/results/final_evaluation'); p.add_argument('--device',default=None); a=p.parse_args()
    device=torch.device(a.device or ('cuda' if torch.cuda.is_available() else 'cpu')); split=make_split(a.data_root,a.split_path,6304,.10); out=Path(a.results_dir); out.mkdir(parents=True,exist_ok=True)
    unknown, groups=unknown_indices(a.data_root); rows=[]
    for specification in a.checkpoint:
        method,path=specification.split('=',1); eval_tf=transforms_for(method,False)
        train=datasets.CIFAR10(a.data_root,train=True,download=True,transform=eval_tf); test=datasets.CIFAR10(a.data_root,train=False,download=True,transform=eval_tf); unknown.transform=eval_tf
        loader=lambda data: DataLoader(data,128,num_workers=2,pin_memory=True)
        model=checkpoint(path,device); train_f,train_z,train_y=outputs(model,loader(Subset(train,split['train_indices'])),device); valid_f,valid_z,_=outputs(model,loader(Subset(train,split['validation_indices'])),device); test_f,test_z,test_y=outputs(model,loader(test),device)
        means=torch.tensor(np.stack([train_f[train_y==c].mean(0) for c in range(10)])); variance=torch.tensor(train_f-np.stack([means.numpy()[y] for y in train_y])).pow(2).mean(0).add(1e-6)
        cache=Path('task4/cache'); cache.mkdir(parents=True,exist_ok=True); np.savez_compressed(cache/f'{method}_known_outputs.npz',train_features=train_f,train_logits=train_z,validation_features=valid_f,validation_logits=valid_z,test_features=test_f,test_logits=test_z,test_labels=test_y)
        known_valid,known_test=valid_z[:,:10],test_z[:,:10]
        csa=float((known_test.argmax(1)==test_y).mean())
        scores=('mls',) if method=='gcsc' else ('mls','proser_placeholder') if method=='proser' else ('msp','mls','energy','mahalanobis')
        def values(name, logits, features):
            return placeholder_score(logits) if name == 'proser_placeholder' else score_values(name, logits[:, :10], features, means, variance)
        for group,indices in groups.items():
            uf,uz,uy=outputs(model,loader(Subset(unknown,indices)),device); np.savez_compressed(cache/f'{method}_{group}_outputs.npz',features=uf,logits=uz,labels=uy)
            for name in scores:
                rows.append(row(method,name,values(name,valid_z,valid_f),values(name,test_z,test_f),values(name,uz,uf),group,csa))
        for name in scores:
            near=np.load(cache/f'{method}_near_outputs.npz'); far=np.load(cache/f'{method}_far_outputs.npz')
            rows.append(row(method,name,values(name,valid_z,valid_f),values(name,test_z,test_f),np.r_[values(name,near['logits'],near['features']),values(name,far['logits'],far['features'])],'all',csa))
    with (out/'osr_metrics.csv').open('w',newline='',encoding='utf-8') as f: w=csv.DictWriter(f,fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
if __name__=='__main__': main()
