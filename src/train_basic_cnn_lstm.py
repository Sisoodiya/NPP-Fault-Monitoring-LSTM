# src/train_backprop.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
from models import CNN_LSTM_Model
from features import compute_basic_stats, compute_wks, compute_kurtosis_skewness

# Enhanced Configuration - More epochs
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_EPOCHS = 60  # Increased from 30
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
WINDOW_SIZE = 100

def main():
    print(f"Using device: {DEVICE}")
    
    # 2. Load preprocessed windows & labels
    if not os.path.exists("../windows.npy") or not os.path.exists("../labels.npy"):
        print("Windows or labels files not found. Please run data_preprocessing.py first.")
        return
    
    windows = np.load("../windows.npy")   # shape: (N_windows, window_size, num_features)
    labels = np.load("../labels.npy")     # shape: (N_windows,)
    
    print(f"Loaded data: {windows.shape}, Labels: {labels.shape}")
    print(f"Label distribution: {np.bincount(labels)}")
    
    num_windows, ws, num_features = windows.shape
    if ws != WINDOW_SIZE:
        print(f"Warning: Expected window size {WINDOW_SIZE}, got {ws}")
    
    # 3. Compute statistical features for each window
    print("Computing statistical features...")
    stat_feats_list = []
    for i in range(num_windows):
        if i % 1000 == 0:
            print(f"Processing window {i}/{num_windows}")
        
        w = windows[i, :, :]  # (window_size, num_features)
        stats_basic = compute_basic_stats(w)       # (5, num_features)
        wks = compute_wks(w, omega=1.0)            # (num_features,)
        
        # Flatten and combine: [mean, median, std, var, entropy, wks] = (6*F,)
        combined = np.concatenate([stats_basic.reshape(-1), wks])  # shape: (6 * num_features,)
        stat_feats_list.append(combined)
    
    stat_feats = np.vstack(stat_feats_list)  # (num_windows, stat_feature_dim)
    print(f"Statistical features shape: {stat_feats.shape}")
    
    # 4. Create PyTorch Datasets
    X_windows = torch.tensor(windows, dtype=torch.float32)        # (N, W, F)
    X_stats = torch.tensor(stat_feats, dtype=torch.float32)       # (N, 6*F)
    y = torch.tensor(labels, dtype=torch.long)                    # (N,)
    
    # 5. Train/Validation Split (80/20)
    dataset = TensorDataset(X_windows, X_stats, y)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size], 
                                   generator=torch.Generator().manual_seed(42))
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Train size: {train_size}, Val size: {val_size}")
    
    # 6. Initialize Model
    stat_dim = X_stats.shape[-1]
    num_classes = len(np.unique(labels))
    print(f"Model config: {num_features} features, {stat_dim} stat features, {num_classes} classes")
    
    model = CNN_LSTM_Model(num_features=num_features,
                           window_size=ws,  # Use actual window size
                           stat_feature_dim=stat_dim,
                           num_classes=num_classes)
    model = model.to(DEVICE)
    
    # 7. Optimizer, Loss, Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     patience=3, factor=0.5, verbose=True)
    
    # 8. Training Functions
    def train_one_epoch(epoch):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (X_win, X_stat, y_true) in enumerate(train_loader):
            X_win = X_win.to(DEVICE)
            X_stat = X_stat.to(DEVICE)
            y_true = y_true.to(DEVICE)
            
            optimizer.zero_grad()
            logits = model(X_win, X_stat)  # (batch, num_classes)
            loss = criterion(logits, y_true)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            running_loss += loss.item() * X_win.size(0)
            _, predicted = logits.max(1)
            correct += (predicted == y_true).sum().item()
            total += y_true.size(0)
            
        train_loss = running_loss / total
        train_acc = correct / total
        print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        return train_loss, train_acc
    
    def validate(epoch):
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for X_win, X_stat, y_true in val_loader:
                X_win = X_win.to(DEVICE)
                X_stat = X_stat.to(DEVICE)
                y_true = y_true.to(DEVICE)
                
                logits = model(X_win, X_stat)
                loss = criterion(logits, y_true)
                
                running_loss += loss.item() * X_win.size(0)
                _, predicted = logits.max(1)
                correct += (predicted == y_true).sum().item()
                total += y_true.size(0)
                
        val_loss = running_loss / total
        val_acc = correct / total
        print(f"[Epoch {epoch}] Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        return val_loss, val_acc
    
    # 9. Training Loop
    print("Starting training...")
    best_val_loss = float("inf")
    patience_counter = 0
    
    # Track training history
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(epoch)
        val_loss, val_acc = validate(epoch)
        
        # Store history
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        scheduler.step(val_loss)
        
        # Save checkpoint if val_loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "../models/best_cnn_lstm.pth")
            print(f"Saved best model (epoch {epoch}) with val_loss: {val_loss:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            
        # Early stopping with increased patience
        if patience_counter >= 15:  # Increased from 10
            print(f"Early stopping at epoch {epoch}")
            break
    
    print(f"Training complete. Best validation loss: {best_val_loss:.4f}")
    
    # 10. Plot training history
    from utils import plot_training_history
    plot_training_history(train_losses, val_losses, train_accs, val_accs, 
                         save_path="../results/backprop_training_history.png")

if __name__ == "__main__":
    main()
