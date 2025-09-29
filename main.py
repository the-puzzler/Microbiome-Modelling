#%%
"""
main.py - Simple training script for microbiome transformer EBM
"""

import torch
import torch.optim as optim
from torch.utils.data import random_split
import h5py
import pickle
import numpy as np

from dataloader_simplified import PerturbedMicrobiomeDataset, collate_microbiome_batch
from model import MicrobiomeTransformer #model 2 is text conditioning, model is text as context.
from train_funcs import train_epoch, eval_epoch #train_funcs 2 is for text conditiniong, train_funcs is for text as context.
import pickle


# =============================================================================
# CONFIGURATION
# =============================================================================

# Data
BATCH_SIZE = 32  # 2
NUM_WORKERS = 2
TRAIN_SPLIT = 0.8
MAX_OTUS_PER_SAMPLE = 600 # 5000
#reducing dimff really hurt performance.

# Model
D_MODEL = 100 #50, 20
NHEAD = 5
NUM_LAYERS = 5 # 3
DROPOUT = 0.1
DIM_FF = 400 #80 default # 2048, this is much too big in all lilklihood.
OTU_EMB = 384
TXT_EMB = 1536

# Loss
W = 1 # 1, 50, 500, 5000
T = 0.15 # 0.15, 0.33
M = 1 # 5

# Training
EPOCHS = 3 # 1
LR = 1e-4
WEIGHT_DECAY = 1e-5
ACCUMULATION_STEPS = 1 #8

# System
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42
#%%
# =============================================================================
# MAIN
# =============================================================================


def main():
    torch.manual_seed(SEED)
    print(f"Using device: {DEVICE}")

    filtered_dataset = PerturbedMicrobiomeDataset(
        'microbiome_dataset_clean.pkl',
        blacklist_path='blacklist.txt',
        M=M,
        min_otus=10,
        max_otus_per_sample=MAX_OTUS_PER_SAMPLE,
        otu_perturbation_rate=T, 
        text_perturbation_rate=T,
        otu_topk_range=(2000,90000),
        text_topk_range=(50,6000)
    )

    # Create dataloader dataset
    full_dataloader = torch.utils.data.DataLoader(
        filtered_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_microbiome_batch,
        num_workers=0
    )

    # Split train/val
    dataset = full_dataloader.dataset
    train_size = int(TRAIN_SPLIT * len(dataset))
    val_size = len(dataset) - train_size
    torch.manual_seed(42)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
        num_workers=NUM_WORKERS, collate_fn=full_dataloader.collate_fn
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, collate_fn=full_dataloader.collate_fn
    )

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    
    model = MicrobiomeTransformer(
        d_model=D_MODEL, nhead=NHEAD, input_dim_type1 = OTU_EMB,
        input_dim_type2 = TXT_EMB, num_layers=NUM_LAYERS,
        dropout=DROPOUT, dim_feedforward=DIM_FF, use_output_activation=False #true for EBM
    ).to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=500, verbose=True
    )

    loss_params = {'w': W, 't': T, 'M': M}

    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Loss params: {loss_params}")

    # Training loop
    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, loss_params, DEVICE, accumulation_steps=ACCUMULATION_STEPS)
        val_loss, val_preds = eval_epoch(model, val_loader, loss_params, DEVICE)
        
        print(f"Epoch {epoch+1:3d}: Train {train_loss:.4f}, Val {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pt')

    print(f"Done! Best val loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    main()