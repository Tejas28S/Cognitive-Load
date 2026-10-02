# ============================================================
#  train_gnn.py  –  Graph Attention Network (Transformer) for Cognitive Load
# ============================================================

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

from config import Config
from data_loader import CognitiveLoadDataLoader
from trainer import train_model

class GraphAttentionHybrid(nn.Module):
    def __init__(self, eeg_nodes=30, eeg_feats=18, nirs_nodes=36, nirs_feats=26, hidden_dim=64, num_classes=2):
        super().__init__()
        self.eeg_nodes = eeg_nodes
        self.eeg_feats = eeg_feats
        self.nirs_nodes = nirs_nodes
        self.nirs_feats = nirs_feats
        
        # Project both node types to the same embedding dimension
        self.eeg_proj = nn.Linear(eeg_feats, hidden_dim)
        self.nirs_proj = nn.Linear(nirs_feats, hidden_dim)
        
        # A fully connected Graph Attention Network (GAT) is mathematically equivalent 
        # to a Transformer Encoder. This avoids memorizing a static adjacency matrix.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=4, 
            dim_feedforward=hidden_dim * 2, 
            dropout=0.5, 
            batch_first=True
        )
        self.gat = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, eeg, fnirs):
        B = eeg.size(0)
        
        # Reshape to (B, Nodes, Features)
        eeg_nodes = eeg.view(B, self.eeg_nodes, self.eeg_feats)
        nirs_nodes = fnirs.view(B, self.nirs_nodes, self.nirs_feats)
        
        # Embed nodes
        h_eeg = self.eeg_proj(eeg_nodes)
        h_nirs = self.nirs_proj(nirs_nodes)
        
        # Concatenate all 66 nodes: (B, 66, hidden_dim)
        h = torch.cat([h_eeg, h_nirs], dim=1)
        
        # Apply Graph Attention
        h = self.gat(h)
        
        # Global Average Pooling over all nodes
        h_pool = h.mean(dim=1)
        
        # Classification head
        return self.fc(h_pool)

def main():
    cfg = Config()
    cfg.BINARY_CLASSIFICATION = True
    cfg.USE_DANN = False
    cfg.USE_CLASS_WEIGHTS = False
    
    # Critical Fix 1: Enable Per-Subject Normalization to prevent inter-subject variability from ruining test accuracy
    cfg.PER_SUBJECT_NORMALIZE = True
    cfg.PER_SUBJECT_NORMALIZE_STAGE = "after_log"
    
    # Critical Fix 2: Stronger regularization
    cfg.EPOCHS = 50
    cfg.BATCH_SIZE = 64
    cfg.LEARNING_RATE = 1e-3
    cfg.WEIGHT_DECAY = 0.05
    cfg.MODEL_PATH = "outputs/gnn_attention_model.pt"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n============================================================")
    print(f"  Training Graph Attention Network on {device}")
    print(f"============================================================")
    
    loader = CognitiveLoadDataLoader(cfg)
    train_loader, val_loader, test_loader, meta = loader.prepare()
    
    y_test_array = meta['y_test']
    num_classes = len(np.unique(y_test_array))
    
    model = GraphAttentionHybrid(num_classes=num_classes).to(device)
    
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        meta=meta,
        cfg=cfg,
        verbose=True
    )
    
    print("\nEvaluating on Test Set...")
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            eeg, fnirs, y = batch[:3]
            eeg, fnirs = eeg.to(device), fnirs.to(device)
            logits = model(eeg, fnirs)
            preds = logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y.numpy())
            
    acc = accuracy_score(all_targets, all_preds)
    print(f"\n[RESULTS] Graph Attention Network Test Accuracy: {acc:.4f} ({acc:.2%})")
    
    target_names = ["Low (0-back)", "High (3-back)"] if cfg.BINARY_CLASSIFICATION else cfg.CLASS_NAMES
    print("\nClassification Report:")
    print(classification_report(all_targets, all_preds, target_names=target_names))

if __name__ == "__main__":
    main()
