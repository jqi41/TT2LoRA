#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ResNet18/50 + TT-MLP head (single TT generates W1 and W2).
Backbone is frozen; only TT‐cores are trained; hidden layer is linear (no activation).
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple

# Helper function for parsing list of integers from command line
def parse_int_list(s: str) -> List[int]:
    """Parses a string like '1,2,3' or '[1,2,3]' into a list of integers."""
    s = s.strip()
    if not s:
        return []
    if s.startswith('[') and s.endswith(']'):
        s = s[1:-1]
    return [int(item.strip()) for item in s.split(',') if item.strip()]

# ──────────────────────────────────────────────────────────────────────────────
# 1) Tiny Tensor–Train layer (no bias)
# ──────────────────────────────────────────────────────────────────────────────
class TensorTrainLayer(nn.Module):
    """
    Implements a Tensor-Train (TT) layer without bias.
    The layer reshapes the input into a tensor, performs a TT contraction
    with the learnable TT-cores, and then reshapes the output.
    """
    def __init__(self, input_dims: List[int], output_dims: List[int], tt_ranks: List[int]):
        """
        Initializes the TensorTrainLayer.

        Args:
            input_dims (List[int]): List of input mode dimensions.
                                    The product of these is the total input feature dimension.
            output_dims (List[int]): List of output mode dimensions.
                                     The product of these is the total output feature dimension.
            tt_ranks (List[int]): List of TT-ranks. Must have length len(input_dims) + 1.
                                  tt_ranks[0] and tt_ranks[-1] are typically 1.
        """
        super().__init__()
        d = len(input_dims)
        if not (len(output_dims) == d and len(tt_ranks) == d + 1):
            raise ValueError(
                f"TT shapes mismatch: len(input_dims)={d}, len(output_dims)={len(output_dims)}, len(tt_ranks)={len(tt_ranks)}"
            )

        self.input_dims, self.output_dims, self.tt_ranks = input_dims, output_dims, tt_ranks

        self.tt_cores = nn.ParameterList()
        for k in range(d):
            r1, n_k, m_k, r2 = tt_ranks[k], input_dims[k], output_dims[k], tt_ranks[k+1]
            # Initialize cores with small random values
            core = nn.Parameter(torch.randn(r1, n_k, m_k, r2) * 0.1)
            self.tt_cores.append(core)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the TT layer.

        Args:
            z (torch.Tensor): Input tensor of shape (batch_size, prod(input_dims)).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, prod(output_dims)).
        """
        # z: (batch_size, prod(input_dims))
        batch_size = z.size(0)
        # Reshape input to (batch_size, n1, n2, ..., nd)
        x = z.view(batch_size, *self.input_dims)

        # Einstein summation string construction for TT contraction
        # Example: For d=3
        # Input x: (b, n1, n2, n3)
        # Core 0: (r0, n1, m1, r1)
        # Core 1: (r1, n2, m2, r2)
        # Core 2: (r2, n3, m3, r3)
        # Output y: (b, m1, m2, m3)
        # einsum_str: "bijk,aipx,xjqy,yklr -> bplr" (approx, letters vary)

        letters = "abcdefghijklmnopqrstuvwxyz"
        batch_letter = letters[0] # 'a' for batch dimension
        d = len(self.input_dims)

        # Letters for input modes (n_k), output modes (m_k), and ranks (r_k)
        input_mode_letters = letters[1 : 1+d]                 # e.g., "bcd" for d=3
        output_mode_letters = letters[1+d : 1+2*d]            # e.g., "efg" for d=3
        rank_letters = letters[1+2*d : 1+3*d+1]             # e.g., "hijk" for d=3 (ranks r0, r1, r2, r3)

        # Input tensor einsum string: batch_letter + input_mode_letters (e.g., "abcd")
        einsum_input_str = batch_letter + "".join(input_mode_letters)

        # Core tensor einsum strings
        einsum_cores_str_list = []
        for k in range(d):
            # Core_k: (rank_letters[k], input_mode_letters[k], output_mode_letters[k], rank_letters[k+1])
            # e.g., Core0: "hbip", Core1: "icjq", Core2: "jkgr"
            core_str = rank_letters[k] + input_mode_letters[k] + \
                       output_mode_letters[k] + rank_letters[k+1]
            einsum_cores_str_list.append(core_str)

        # Output tensor einsum string: batch_letter + output_mode_letters (e.g., "aefg")
        einsum_output_str = batch_letter + "".join(output_mode_letters)

        einsum_str = f"{einsum_input_str},{','.join(einsum_cores_str_list)}->{einsum_output_str}"

        y = torch.einsum(einsum_str, x, *self.tt_cores)
        # Reshape output to (batch_size, prod(output_dims))
        return y.reshape(batch_size, -1)

# ──────────────────────────────────────────────────────────────────────────────
# 2) TT-MLP head (single TT → W1 and W2, linear hidden)
# ──────────────────────────────────────────────────────────────────────────────
class TTPostNN(nn.Module):
    """
    A Post-Hoc Neural Network head using a TensorTrainLayer to generate
    the weights for a two-layer MLP (W1 and W2).
    The hidden layer of the MLP is linear (no activation).
    """
    def __init__(self,
                 feat_dim: int,
                 hidden_dim: int,
                 num_classes: int,
                 tt_in_dims: List[int], tt_out_dims: List[int], tt_ranks: List[int],
                 scale: float = 0.01):
        """
        Initializes the TTPostNN head.

        Args:
            feat_dim (int): Input feature dimension to the MLP (output of backbone).
            hidden_dim (int): Dimension of the hidden layer in the MLP.
            num_classes (int): Number of output classes.
            tt_in_dims (List[int]): Input dimensions for the TensorTrainLayer.
            tt_out_dims (List[int]): Output dimensions for the TensorTrainLayer.
                                     prod(tt_out_dims) must equal feat_dim * hidden_dim + hidden_dim * num_classes.
            tt_ranks (List[int]): TT-ranks for the TensorTrainLayer.
            scale (float): Scaling factor applied to the final output logits.
        """
        super().__init__()
        # Total number of parameters for W1 (feat_dim * hidden_dim) and W2 (hidden_dim * num_classes)
        total_mlp_params = feat_dim * hidden_dim + hidden_dim * num_classes
        prod_tt_out_dims = int(np.prod(tt_out_dims))

        if total_mlp_params != prod_tt_out_dims:
            raise ValueError(
                f"Product of tt_out_dims ({prod_tt_out_dims}) must match "
                f"total MLP parameters ({total_mlp_params})."
            )

        self.tt_layer = TensorTrainLayer(tt_in_dims, tt_out_dims, tt_ranks)
        # Fixed latent input 'z' for the TT layer to generate weights.
        # Registered as a buffer, so it's part of the model's state but not a trainable parameter.
        self.register_buffer('z_latent', torch.randn(1, int(np.prod(tt_in_dims))))

        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the TT-MLP head.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, feat_dim).

        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes).
        """
        # x: (batch_size, feat_dim)

        # Generate the flattened weights W1 and W2 using the TT layer
        # self.z_latent has shape (1, prod(tt_in_dims))
        flat_weights = self.tt_layer(self.z_latent).squeeze(0)  # Shape: (total_mlp_params,)

        # Split point for W1 and W2
        split_idx = self.feat_dim * self.hidden_dim

        # Extract W1 and W2
        w1_flat = flat_weights[:split_idx]
        w2_flat = flat_weights[split_idx:]

        w1 = w1_flat.view(self.feat_dim, self.hidden_dim)    # Shape: (feat_dim, hidden_dim)
        w2 = w2_flat.view(self.hidden_dim, self.num_classes) # Shape: (hidden_dim, num_classes)

        # MLP forward pass: x @ W1 (linear hidden layer) -> h @ W2
        h = x @ w1                                        # Shape: (batch_size, hidden_dim)
        out_logits = (h @ w2) * self.scale                # Shape: (batch_size, num_classes)
        return out_logits

# ──────────────────────────────────────────────────────────────────────────────
# 3) Build model
# ──────────────────────────────────────────────────────────────────────────────
def build_model(model_kind: str,
                in_ch: int,
                hidden_dim: int,
                num_classes: int,
                tt_in_dims: List[int], tt_out_dims: List[int], tt_ranks: List[int],
                scale: float) -> nn.Sequential:
    """
    Builds the full model: a ResNet backbone followed by the TT-MLP head.
    The backbone is frozen, and only the TT-cores in the head are trainable.

    Args:
        model_kind (str): Type of ResNet backbone ('ResNet18' or 'ResNet50').
        in_ch (int): Number of input channels for the first convolutional layer.
        hidden_dim (int): Hidden dimension for the TT-MLP head.
        num_classes (int): Number of output classes.
        tt_in_dims (List[int]): Input dimensions for the TensorTrainLayer in the head.
        tt_out_dims (List[int]): Output dimensions for the TensorTrainLayer in the head.
        tt_ranks (List[int]): TT-ranks for the TensorTrainLayer in the head.
        scale (float): Scaling factor for the output logits in the TT-MLP head.

    Returns:
        nn.Sequential: The constructed PyTorch model.
    """
    # Load backbone (ResNet18 or ResNet50) with random weights
    if model_kind == 'ResNet18':
        backbone, feat_dim = torchvision.models.resnet18(weights=None), 512
    elif model_kind == 'ResNet50':
        backbone, feat_dim = torchvision.models.resnet50(weights=None), 2048
    else:
        raise ValueError(f"Unsupported model_kind: {model_kind}")

    # Freeze all parameters in the backbone
    for param in backbone.parameters():
        param.requires_grad = False

    # Adapt the first convolutional layer to the specified number of input channels (in_ch)
    # The original conv1 is designed for 3-channel (RGB) images.
    original_conv1 = backbone.conv1
    backbone.conv1 = nn.Conv2d(
        in_channels=in_ch,
        out_channels=original_conv1.out_channels,
        kernel_size=original_conv1.kernel_size,
        stride=original_conv1.stride,
        padding=original_conv1.padding,
        bias=original_conv1.bias # Usually False in ResNets as BN follows
    )
    # Ensure the new conv1 layer's weights are also frozen, consistent with freezing the backbone.
    # If this layer were to be trainable, requires_grad should be True.
    # However, the script's goal is to train *only* TT-cores.
    backbone.conv1.weight.requires_grad = True
    if backbone.conv1.bias is not None:
        backbone.conv1.bias.requires_grad = False

    # Replace the original fully connected layer (classifier) with an Identity layer,
    # as we will use our custom TT-MLP head.
    backbone.fc = nn.Identity()

    # Stack the backbone and the TT-MLP head
    model = nn.Sequential(
        backbone,
        nn.Flatten(start_dim=1), # Flatten the output of the backbone
        TTPostNN(feat_dim, hidden_dim, num_classes,
                 tt_in_dims, tt_out_dims, tt_ranks,
                 scale)
    )
    return model

# ──────────────────────────────────────────────────────────────────────────────
# 4) Dataset & train/eval
# ──────────────────────────────────────────────────────────────────────────────
class CustomDataset(Dataset):
    """
    Custom PyTorch Dataset.
    Assumes data is a NumPy array of images and labels is a NumPy array of class indices.
    """
    def __init__(self, data: np.ndarray, labels: np.ndarray):
        """
        Args:
            data (np.ndarray): Data, e.g., images, of shape (N, H, W) or (N, C, H, W).
                               Expected shape for this script: (N, 50, 50)
            labels (np.ndarray): Labels, of shape (N,).
        """
        # Convert NumPy data to PyTorch tensors.
        # Using .float() for image data, .long() for labels (class indices).
        self.x = torch.from_numpy(data).float()  # Shape: (N, 50, 50)
        self.y = torch.from_numpy(labels).long() # Shape: (N,)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Unsqueeze to add channel dimension: (50, 50) -> (1, 50, 50)
        # Assumes single-channel input for the model's first conv layer.
        return self.x[i].unsqueeze(0), self.y[i]

def train_epoch(model: nn.Module, loader: DataLoader,
                optimizer: optim.Optimizer, loss_fn: nn.Module,
                device: torch.device, log_interval: int = 100):
    """Trains the model for one epoch."""
    model.train()
    total_loss = 0.0
    total_samples = 0
    correct_predictions = 0

    for i, (X_batch, y_batch) in enumerate(loader, 1):
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        logits = model(X_batch)
        loss = loss_fn(logits, y_batch)
        loss.backward()
        optimizer.step()

        batch_size = y_batch.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        correct_predictions += (logits.argmax(1) == y_batch).sum().item()

        if i % log_interval == 0 or i == len(loader):
            avg_loss = total_loss / total_samples
            accuracy = correct_predictions / total_samples
            print(f"[Batch {i:4d}/{len(loader)}]  Loss={avg_loss:.4f}  Acc={accuracy:.3f}")

    final_avg_loss = total_loss / total_samples
    final_accuracy = correct_predictions / total_samples
    print(f"TRAIN Epoch Summary ▶ Loss={final_avg_loss:.4f}  Acc={final_accuracy:.3f}")

def eval_epoch(model: nn.Module, loader: DataLoader,
               loss_fn: nn.Module, device: torch.device):
    """Evaluates the model on the provided data loader."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    correct_predictions = 0

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            loss = loss_fn(logits, y_batch)

            batch_size = y_batch.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            correct_predictions += (logits.argmax(1) == y_batch).sum().item()

    avg_loss = total_loss / total_samples
    accuracy = correct_predictions / total_samples
    print(f" TEST Epoch Summary ▶ Loss={avg_loss:.4f}  Acc={accuracy:.3f}")

# ──────────────────────────────────────────────────────────────────────────────
# 5) Main
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    p = argparse.ArgumentParser(description="ResNet with TT-MLP Head Training")
    p.add_argument('--data_path', default='../data/mlqe_2023_edx/week1/dataset',
                   help="Path to the dataset directory.")
    p.add_argument('--model_kind', choices=['ResNet18', 'ResNet50'], default='ResNet18',
                   help="Type of ResNet backbone to use.")
    p.add_argument('--hidden_dim', type=int, default=1024,
                   help="Hidden dimension of the TT-generated MLP head.")
    p.add_argument('--batch_size', type=int, default=16,
                   help="Batch size for training and evaluation.")
    p.add_argument('--lr', type=float, default=3e-3,
                   help="Learning rate for the optimizer.")
    p.add_argument('--epochs', type=int, default=15,
                   help="Number of training epochs.")
    p.add_argument('--log_interval', type=int, default=100,
                   help="Interval for logging training progress (in batches).")
    p.add_argument('--test_kind', choices=['gen', 'rep'], default='gen',
                   help="Dataset variant ('gen' for csds.npy, 'rep' for csds_noiseless.npy).")

    # TT-related arguments
    p.add_argument('--tt_in_dims', type=parse_int_list, default=[1, 2, 2, 1],
                   help="Input mode dimensions for TT-layer, e.g., '[1,2,2,1]' or '1,2,2,1'. Product is latent_dim.")
    p.add_argument('--tt_out_dims', type=parse_int_list, default=[16, 8, 257, 16],
                   help="Output mode dimensions for TT-layer, e.g., '[16,8,257,16]'. Product is total MLP params.")
    p.add_argument('--tt_ranks', type=parse_int_list, default=[1, 2, 2, 1, 1],
                   help="TT-ranks for TT-layer, e.g., '[1,2,2,1,1]'. Length = len(tt_in_dims)+1.")
    p.add_argument('--tt_scale', type=float, default=0.01,
                   help="Scaling factor for the final output logits from TT-MLP head.")
    p.add_argument('--seed', type=int, default=1234, help="Random seed for reproducibility.")

    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed) # For numpy used in dataset or other parts
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # TT dimensions & MLP structure derived from them:
    # Latent 'z' input to TT-layer has shape (1, prod(tt_in_dims)).
    # Default prod(tt_in_dims) = 1*2*2*1 = 4.
    # The TT-layer output (flat_weights) must have prod(tt_out_dims) elements.
    # This prod(tt_out_dims) must equal feat_dim * hidden_dim + hidden_dim * num_classes.
    # For ResNet18 (feat_dim=512), hidden_dim=1024 (default), num_classes=2:
    # total_mlp_params = 512*1024 + 1024*2 = 524288 + 2048 = 526336.
    # Default prod(tt_out_dims) = 16*8*257*16 = 526336. This matches.
    # Ensure your tt_out_dims are compatible with model_kind and hidden_dim.

    model = build_model(
        model_kind=args.model_kind,
        in_ch=1,  # Assuming single-channel input data
        hidden_dim=args.hidden_dim,
        num_classes=2, # Hardcoded for binary classification as per original
        tt_in_dims=args.tt_in_dims,
        tt_out_dims=args.tt_out_dims,
        tt_ranks=args.tt_ranks,
        scale=args.tt_scale
    ).to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.model_kind} with TT-MLP Head")
    print(f"Trainable parameters: {trainable_params}")
    print(f"Total parameters: {total_params}")
    print(f"  Input TT latent dim: {int(np.prod(args.tt_in_dims))}")
    feat_dim_actual = 512 if args.model_kind == 'ResNet18' else 2048
    expected_mlp_params = feat_dim_actual * args.hidden_dim + args.hidden_dim * 2
    print(f"  Generated MLP params: {int(np.prod(args.tt_out_dims))} (expected: {expected_mlp_params})")


    # Load data
    data_filename = 'csds.npy' if args.test_kind == 'gen' else 'csds_noiseless.npy'
    X_data = np.load(f"{args.data_path}/{data_filename}")
    Y_labels = np.load(f"{args.data_path}/labels.npy")

    dataset = CustomDataset(X_data, Y_labels)

    # Split dataset into training and testing sets
    # For more robust split reproducibility, especially if other random ops occur,
    # use a dedicated generator.
    split_generator = torch.Generator().manual_seed(args.seed)
    num_total_samples = len(dataset)
    num_train_samples = int(0.8 * num_total_samples)
    num_test_samples = num_total_samples - num_train_samples

    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [num_train_samples, num_test_samples], generator=split_generator
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    print(f"Dataset loaded: {len(dataset)} samples")
    print(f"Training set: {len(train_dataset)} samples")
    print(f"Test set: {len(test_dataset)} samples")

    # Optimizer: Adam, only on parameters that require gradients (i.e., TT-cores)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    loss_fn = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs} (LR: {scheduler.get_last_lr()[0]:.2e})")
        train_epoch(model, train_loader, optimizer, loss_fn, device, args.log_interval)
        eval_epoch(model, test_loader, loss_fn, device)
        scheduler.step() # Step the scheduler after each epoch

    print("\nTraining complete.")