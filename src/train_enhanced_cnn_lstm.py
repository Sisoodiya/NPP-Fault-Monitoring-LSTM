import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
from models import Enhanced_CNN_LSTM_Model
from features import compute_basic_stats, compute_wks, compute_sample_entropy, compute_rate_of_change

# Enhanced Configuration - More epochs and parameters
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_EPOCHS = 80  # Increased from 35
BATCH_SIZE = 12  # Slightly reduced for larger models
LEARNING_RATE = 5e-4  # Slightly lower for stability
WINDOW_SIZE = 100
PATIENCE = 20  # Increased patience for longer training

def main():
    print(f"Using device: {DEVICE}")
    
    # Load preprocessed data
    if not os.path.exists("../windows.npy") or not os.path.exists("../labels.npy"):
        print("Windows or labels files not found. Please run data_preprocessing.py first.")
        return
    
    windows = np.load("../windows.npy")
    labels = np.load("../labels.npy")
    
    print(f"Loaded data: {windows.shape}, Labels: {labels.shape}")
    print(f"Label distribution: {np.bincount(labels)}")
    
    num_windows, ws, num_features = windows.shape
    
    # Compute enhanced statistical features
    print("Computing enhanced statistical features...")
    stat_feats_list = []
    for i in range(num_windows):
        if i % 50 == 0:
            print(f"Processing window {i}/{num_windows}")
        
        w = windows[i, :, :]
        
        # Basic features
        stats_basic = compute_basic_stats(w)
        wks = compute_wks(w, omega=1.0)
        
        # Enhanced features
        sample_entropies = []
        for ch in range(num_features):
            try:
                se = compute_sample_entropy(w[:, ch], m=2, r=0.2)
                if np.isnan(se) or np.isinf(se):
                    se = 0.0
            except:
                se = 0.0
            sample_entropies.append(se)
        sample_entropy = np.array(sample_entropies)
        
        rate_of_change = compute_rate_of_change(w)
        
        # Combine all features
        combined = np.concatenate([
            stats_basic.reshape(-1),
            wks,
            sample_entropy,
            rate_of_change.reshape(-1)
        ])
        stat_feats_list.append(combined)
    
    stat_feats = np.vstack(stat_feats_list)
    print(f"Enhanced statistical features shape: {stat_feats.shape}")
    
    # Create PyTorch datasets
    X_windows = torch.tensor(windows, dtype=torch.float32)
    X_stats = torch.tensor(stat_feats, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)
    
    # Train/validation split
    dataset = TensorDataset(X_windows, X_stats, y)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size], 
                                   generator=torch.Generator().manual_seed(42))
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Train size: {train_size}, Val size: {val_size}")
    
    # Initialize enhanced model
    stat_dim = X_stats.shape[-1]
    num_classes = len(np.unique(labels))
    print(f"Enhanced model config: {num_features} features, {stat_dim} stat features, {num_classes} classes")
    
    model = Enhanced_CNN_LSTM_Model(
        num_features=num_features,
        window_size=ws,
        stat_feature_dim=stat_dim,
        num_classes=num_classes,
        use_cnn_attention=True,
        use_lstm_attention=True,
        bidirectional=True
    )
    model = model.to(DEVICE)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Enhanced model parameters: {total_params:,}")
    
    # Optimizer and loss
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     patience=4, factor=0.7, verbose=True)
    
    # Training functions
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
            logits = model(X_win, X_stat)
            loss = criterion(logits, y_true)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total += y_true.size(0)
            correct += (predicted == y_true).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total
        return epoch_loss, epoch_acc
    
    def validate():
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for X_win, X_stat, y_true in val_loader:
                X_win = X_win.to(DEVICE)
                X_stat = X_stat.to(DEVICE)
                y_true = y_true.to(DEVICE)
                
                logits = model(X_win, X_stat)
                loss = criterion(logits, y_true)
                
                val_loss += loss.item()
                _, predicted = torch.max(logits.data, 1)
                total += y_true.size(0)
                correct += (predicted == y_true).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = correct / total
        return val_loss, val_acc
    
    # Training loop with history tracking
    print("\nStarting enhanced model training...")
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Track training history
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    epochs_list = []
    
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(epoch)
        val_loss, val_acc = validate()
        
        # Record history
        epochs_list.append(epoch)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"[Epoch {epoch}] Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "../models/enhanced_cnn_lstm.pth")
            print(f"Saved best enhanced model (epoch {epoch}) with val_loss: {val_loss:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping
        if patience_counter >= 8:
            print(f"Early stopping after epoch {epoch}")
            break
    
    # Save training history
    import matplotlib.pyplot as plt
    
    # Create training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Loss curves
    ax1.plot(epochs_list, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs_list, val_losses, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_title('Enhanced CNN-LSTM Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy curves
    ax2.plot(epochs_list, train_accuracies, 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs_list, val_accuracies, 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_title('Enhanced CNN-LSTM Training Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("../results/enhanced_cnn_lstm_training_curves.png", dpi=200, bbox_inches='tight')
    print("Enhanced training curves saved to ../results/enhanced_cnn_lstm_training_curves.png")
    
    # Save training history data
    training_history = {
        'epochs': epochs_list,
        'train_loss': train_losses,
        'train_acc': train_accuracies,
        'val_loss': val_losses,
        'val_acc': val_accuracies
    }
    np.save("../results/enhanced_cnn_lstm_training_history.npy", training_history)
    
    print(f"\nEnhanced model training complete. Best validation loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    main()
