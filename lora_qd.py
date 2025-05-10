#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  2 15:05:35 2025

@author: junqi

ResNet + LoRA head for quantum-dot classification
Freezes ResNet backbone and fine-tunes only low-rank LoRA adapters.
"""

import argparse, numpy as np, torch
import torch.nn as nn, torch.optim as optim
import torchvision
from torch.utils.data import Dataset, DataLoader

# ──────────────────────────────────────────────────────────────────────────────
# 1) LoRA adapter for a single Linear layer
# ──────────────────────────────────────────────────────────────────────────────
class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=4, alpha=32, bias=True):
        super().__init__()
        # the frozen base weight
        self.base = nn.Linear(in_features, out_features, bias=bias)
        self.base.weight.requires_grad = False
        if bias:
            self.base.bias.requires_grad = False

        # LoRA factors
        self.r, self.alpha = r, alpha
        self.lora_A = nn.Parameter(torch.randn(r, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.randn(out_features, r) * 0.01)
        self.scaling = alpha / r

    def forward(self, x):
        # base output + low-rank update
        return self.base(x) + self.scaling * (x @ self.lora_A.T @ self.lora_B.T)

# ──────────────────────────────────────────────────────────────────────────────
# 2) Build ResNet + LoRA head
# ──────────────────────────────────────────────────────────────────────────────
def build_model(model_kind, lora_r, lora_alpha, num_classes=2):
    # load and freeze backbone
    if model_kind == 'ResNet18':
        backbone = torchvision.models.resnet18(weights=None)
        feat_dim = 512
    else:
        backbone = torchvision.models.resnet50(weights=None)
        feat_dim = 2048
    for p in backbone.parameters():
        p.requires_grad = False

    # adapt to grayscale 50×50
    backbone.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
    # freeze the new conv1 so its weights aren’t trainable
    backbone.conv1.weight.requires_grad = True

    # replace fc with a LoRA-wrapped linear classifier
    lora_fc = LoRALinear(feat_dim, num_classes, r=lora_r, alpha=lora_alpha, bias=True)
    backbone.fc = lora_fc

    return backbone

# ──────────────────────────────────────────────────────────────────────────────
# 3) Dataset + train/eval utilities
# ──────────────────────────────────────────────────────────────────────────────
class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.x = torch.Tensor(data)       # (N,50,50)
        self.y = torch.LongTensor(labels) # (N,)
    def __len__(self): return len(self.x)
    def __getitem__(self, i):
        return self.x[i].unsqueeze(0), self.y[i]

def train_epoch(model, loader, opt, loss_fn, device, log_interval=100):
    model.train()
    tot_loss, tot_samples, corr = 0.0, 0, 0
    for i,(X,y) in enumerate(loader,1):
        X,y = X.to(device), y.to(device)
        logits = model(X)
        loss   = loss_fn(logits, y)
        opt.zero_grad(); loss.backward(); opt.step()

        b = y.size(0)
        tot_loss    += loss.item()*b
        tot_samples += b
        corr        += (logits.argmax(1)==y).sum().item()

        if i % log_interval == 0:
            print(f"[Batch {i:4d}] "
                  f"Avg Loss: {tot_loss/tot_samples:.4f} "
                  f"Acc: {corr/tot_samples:.3f}")

    print(f"TRAIN ▶ Loss: {tot_loss/tot_samples:.4f} Acc: {corr/tot_samples:.3f}")

def eval_epoch(model, loader, loss_fn, device):
    model.eval()
    tot_loss, tot_samples, corr = 0.0, 0, 0
    with torch.no_grad():
        for X,y in loader:
            X,y = X.to(device), y.to(device)
            logits = model(X)
            loss   = loss_fn(logits,y)
            b = y.size(0)
            tot_loss    += loss.item()*b
            tot_samples += b
            corr        += (logits.argmax(1)==y).sum().item()

    print(f" TEST ▶ Loss: {tot_loss/tot_samples:.4f} Acc: {corr/tot_samples:.3f}")

# ──────────────────────────────────────────────────────────────────────────────
# 4) Main: data load, model, training loop
# ──────────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_path', default='../data/mlqe_2023_edx/week1/dataset')
    p.add_argument('--model_kind', choices=['ResNet18','ResNet50'], default='ResNet18')
    p.add_argument('--lora_r',     type=int, default=4)
    p.add_argument('--lora_alpha', type=int, default=32)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--lr',         type=float, default=1e-3)
    p.add_argument('--epochs',     type=int, default=15)
    p.add_argument('--log_interval', type=int, default=100)
    p.add_argument('--test_kind',  choices=['gen','rep'], default='gen')
    args = p.parse_args()

    torch.manual_seed(1234)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)

    # load data
    data = np.load(f"{args.data_path}/csds{'' if args.test_kind=='gen' else '_noiseless'}.npy")
    labels = np.load(f"{args.data_path}/labels.npy")
    ds = CustomDataset(data, labels)
    n = int(0.8*len(ds))
    train_ds, test_ds = torch.utils.data.random_split(ds, [n, len(ds)-n])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size)

    # model
    model = build_model(args.model_kind, args.lora_r, args.lora_alpha, num_classes=2)
    model = model.to(device)
    tp = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {tp}")

    # optimizer & loss
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    loss_fn   = nn.CrossEntropyLoss()

    # training
    for ep in range(1, args.epochs+1):
        print(f"\nEpoch {ep}/{args.epochs}")
        train_epoch(model, train_loader, optimizer, loss_fn, device, args.log_interval)
        eval_epoch( model, test_loader,   loss_fn, device)

if __name__=='__main__':
    main()
