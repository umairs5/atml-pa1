"""Train the required Vanilla or GCSC closed-set model; CIFAR-100 is never loaded."""
from __future__ import annotations
import argparse, csv, random
from pathlib import Path
import numpy as np, torch, yaml
from torch import nn
from torch.utils.data import DataLoader, Subset
from task4.data import make_split, transforms_for
from task4.model import CifarResNet18
from torchvision import datasets

def config(path):
    child = yaml.safe_load(Path(path).read_text()); base = yaml.safe_load(Path(child.pop("base")).read_text()); base.update(child); return base
def seed_everything(seed): random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
@torch.no_grad()
def accuracy(model, loader, device):
    model.eval(); correct = total = 0
    for images, labels in loader:
        correct += (model(images.to(device)).argmax(1).cpu() == labels).sum().item(); total += len(labels)
    return correct / total
def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True); parser.add_argument("--device", default=None); args = parser.parse_args()
    cfg = config(args.config); seed_everything(cfg["seed"]); device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    split = make_split(cfg["data"]["root"], cfg["data"]["split_path"], cfg["seed"], cfg["data"]["validation_fraction"])
    train = datasets.CIFAR10(cfg["data"]["root"], train=True, download=True, transform=transforms_for(cfg["method"], True))
    valid = datasets.CIFAR10(cfg["data"]["root"], train=True, download=True, transform=transforms_for(cfg["method"], False))
    loader = DataLoader(Subset(train, split["train_indices"]), cfg["training"]["batch_size"], shuffle=True, num_workers=cfg["training"]["num_workers"], pin_memory=True)
    valid_loader = DataLoader(Subset(valid, split["validation_indices"]), cfg["training"]["batch_size"], num_workers=cfg["training"]["num_workers"], pin_memory=True)
    model = CifarResNet18().to(device); optimizer = torch.optim.SGD(model.parameters(), lr=cfg["training"]["learning_rate"], momentum=cfg["training"]["momentum"], weight_decay=cfg["training"]["weight_decay"]); scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg["training"]["epochs"]); criterion = nn.CrossEntropyLoss()
    output = Path(cfg["output"]["root"]) / cfg["method"]; output.mkdir(parents=True, exist_ok=True); history=[]; best=-1
    for epoch in range(1, cfg["training"]["epochs"] + 1):
        model.train(); loss_sum = count = 0
        for images, labels in loader:
            optimizer.zero_grad(); loss = criterion(model(images.to(device)), labels.to(device)); loss.backward(); optimizer.step(); loss_sum += loss.item()*len(labels); count += len(labels)
        score = accuracy(model, valid_loader, device); scheduler.step(); history.append({"epoch":epoch,"train_loss":loss_sum/count,"validation_accuracy":score})
        if score > best: best=score; torch.save({"model":model.state_dict(),"config":cfg,"epoch":epoch,"validation_accuracy":score}, output/"best.pt")
        print(f"Epoch {epoch:03d}: validation accuracy={score:.4f}")
    with (output/"history.csv").open("w",newline="",encoding="utf-8") as f: writer=csv.DictWriter(f,fieldnames=history[0]); writer.writeheader(); writer.writerows(history)
if __name__ == "__main__": main()
