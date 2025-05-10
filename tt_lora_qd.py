#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TT-LoRA head on ResNet for quantum-dot classification.
Backbone is frozen; only two TT-cores are trained to generate W1 and W2.
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import Dataset, DataLoader

# ──────────────────────────────────────────────────────────────────────────────
# 1) Tiny Tensor–Train layer (no bias)
# ──────────────────────────────────────────────────────────────────────────────
class TensorTrainLayer(nn.Module):
    def __init__(self, input_dims, output_dims, tt_ranks):
        super().__init__()
        d = len(input_dims)
        assert len(output_dims)==d and len(tt_ranks)==d+1, "TT shapes mismatch"
        self.input_dims, self.output_dims, self.tt_ranks = input_dims, output_dims, tt_ranks

        self.tt_cores = nn.ParameterList()
        for k in range(d):
            r1, n, m, r2 = tt_ranks[k], input_dims[k], output_dims[k], tt_ranks[k+1]
            # core shape = (r1, n, m, r2)
            core = nn.Parameter(torch.randn(r1, n, m, r2)*0.1)
            self.tt_cores.append(core)

    def forward(self, z):
        # z: (1, ∏ input_dims)
        b = z.size(0)
        x = z.view(b, *self.input_dims)
        # build einsum string 
        letters = "abcdefghijklmnopqrstuvwxyz"
        batch_l = letters[0]
        d = len(self.input_dims)
        nL = letters[1:1+d]
        mL = letters[1+d:1+2*d]
        rL = letters[1+2*d:1+3*d+1]

        inp  = batch_l + "".join(nL)
        cores = [rL[k]+nL[k]+mL[k]+rL[k+1] for k in range(d)]
        out  = batch_l + "".join(mL)
        eins_str = inp + "," + ",".join(cores) + "->" + out

        y = torch.einsum(eins_str, x, *self.tt_cores)
        return y.reshape(b, -1)  # (b, ∏output_dims)

# ──────────────────────────────────────────────────────────────────────────────
# 2) TT-LoRA adapter for a single Linear layer
# ──────────────────────────────────────────────────────────────────────────────
class TTLoRALinear(nn.Module):
    def __init__(self, in_features, hidden_dim, out_features, bias=True):
        super().__init__()
        # frozen base
        self.base = nn.Linear(in_features, out_features, bias=bias)
        self.base.weight.requires_grad = False
        if bias: self.base.bias.requires_grad = False

        self.feat_dim   = in_features    # D = 512
        self.hidden_dim = hidden_dim     # H = 10
        self.num_classes= out_features   # Q = 2

        # ─── TT for W1: shape (D × H) = (512×10)=5120
        # factor 5120 = 8×8×8×10
        self.tt1 = TensorTrainLayer(
            input_dims  = [8, 8, 8],
            output_dims = [8, 8, 80],   # 8⋅8⋅80 =5120
            tt_ranks    = [1, 2, 2, 1]
        )
        self.register_buffer('z1', torch.randn(1, 8*8*8))

        # ─── TT for W2: shape (H × Q) = (10×2)=20
        # factor 20 = 4×5
        self.tt2 = TensorTrainLayer(
            input_dims  = [4, 5],
            output_dims = [10, 2],      # 4⋅5 gives latent dims
            tt_ranks    = [1, 2, 1]
        )
        self.register_buffer('z2', torch.randn(1, 4*5))

    def forward(self, x):
        # compute base
        out = self.base(x)

        # generate W1
        flat1 = self.tt1(self.z1).squeeze(0)     # (5120,)
        W1    = flat1.view(self.feat_dim, self.hidden_dim)  # (512,10)

        # generate W2
        flat2 = self.tt2(self.z2).squeeze(0)     # (20,)
        W2    = flat2.view(self.hidden_dim, self.num_classes) # (10,2)

        # adapt
        h = x @ W1                              # (batch, 10)
        delta = h @ W2                         # (batch, 2)
        return out + delta

# ──────────────────────────────────────────────────────────────────────────────
# 3) Build ResNet + TT-LoRA head
# ──────────────────────────────────────────────────────────────────────────────
def build_model(model_kind:str, hidden_dim:int, num_classes:int):
    if model_kind=='ResNet18':
        backbone, feat_dim = torchvision.models.resnet18(weights=None), 512
    else:
        backbone, feat_dim = torchvision.models.resnet50(weights=None), 2048

    # freeze all
    for p in backbone.parameters():
        p.requires_grad = False

    # adapt conv1 for 1-channel input and freeze it too
    backbone.conv1 = nn.Conv2d(1,
                               backbone.conv1.out_channels,
                               kernel_size=backbone.conv1.kernel_size,
                               stride=backbone.conv1.stride,
                               padding=backbone.conv1.padding,
                               bias=False)
    backbone.conv1.weight.requires_grad = True

    # replace fc
    backbone.fc = TTLoRALinear(feat_dim, hidden_dim, num_classes, bias=True)
    return backbone

# ──────────────────────────────────────────────────────────────────────────────
# 4) Dataset & train/eval
# ──────────────────────────────────────────────────────────────────────────────
class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.x = torch.Tensor(data)       # (N,50,50)
        self.y = torch.LongTensor(labels) # (N,)
    def __len__(self): return len(self.x)
    def __getitem__(self,i):
        return self.x[i].unsqueeze(0), self.y[i]

def train_epoch(model, loader, opt, loss_fn, device, log_interval=100):
    model.train()
    tot_loss, tot, corr = 0.0,0,0
    for i,(X,y) in enumerate(loader,1):
        X,y = X.to(device), y.to(device)
        logits = model(X)
        loss   = loss_fn(logits,y)
        opt.zero_grad(); loss.backward(); opt.step()

        bs = y.size(0)
        tot_loss += loss.item()*bs
        tot     += bs
        corr    += (logits.argmax(1)==y).sum().item()
        if i%log_interval==0:
            print(f"[{i:4d}] Loss={tot_loss/tot:.4f} Acc={corr/tot:.3f}")
    print(f"TRAIN ▶ Loss={tot_loss/tot:.4f} Acc={corr/tot:.3f}")

def eval_epoch(model, loader, loss_fn, device):
    model.eval()
    tot_loss, tot, corr = 0.0,0,0
    with torch.no_grad():
        for X,y in loader:
            X,y = X.to(device), y.to(device)
            logits = model(X)
            loss   = loss_fn(logits,y)
            bs = y.size(0)
            tot_loss += loss.item()*bs
            tot     += bs
            corr    += (logits.argmax(1)==y).sum().item()
    print(f" TEST ▶ Loss={tot_loss/tot:.4f} Acc={corr/tot:.3f}")

# ──────────────────────────────────────────────────────────────────────────────
# 5) Main
# ──────────────────────────────────────────────────────────────────────────────
if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data_path',   default='../data/mlqe_2023_edx/week1/dataset')
    p.add_argument('--model_kind',  choices=['ResNet18','ResNet50'], default='ResNet18')
    p.add_argument('--hidden_dim',  type=int, default=10)
    p.add_argument('--batch_size',  type=int, default=8)
    p.add_argument('--lr',          type=float, default=3e-3)
    p.add_argument('--epochs',      type=int, default=15)
    p.add_argument('--log_interval',type=int, default=100)
    p.add_argument('--test_kind',   choices=['gen','rep'], default='gen')
    args = p.parse_args()

    torch.manual_seed(1234)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # build
    model = build_model(
        args.model_kind,
        hidden_dim  = args.hidden_dim,
        num_classes = 2
    ).to(device)

    tp = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Trainable TT-LoRA parameters:", tp)   # should report ~1764

    # load data
    fn = 'csds.npy' if args.test_kind=='gen' else 'csds_noiseless.npy'
    X = np.load(f"{args.data_path}/{fn}")
    Y = np.load(f"{args.data_path}/labels.npy")
    ds = CustomDataset(X,Y)
    n  = int(0.8*len(ds))
    tr,te = torch.utils.data.random_split(ds,[n,len(ds)-n])
    trl = DataLoader(tr, batch_size=args.batch_size, shuffle=True)
    tel = DataLoader(te, batch_size=args.batch_size)

    optimizer = optim.Adam(filter(lambda p:p.requires_grad, model.parameters()), lr=args.lr)
    loss_fn   = nn.CrossEntropyLoss()

    for ep in range(1, args.epochs+1):
        print(f"\nEpoch {ep}/{args.epochs}")
        train_epoch(model, trl, optimizer, loss_fn, device, args.log_interval)
        eval_epoch( model, tel, loss_fn, device)
