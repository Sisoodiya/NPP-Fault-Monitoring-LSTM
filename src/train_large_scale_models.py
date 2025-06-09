import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
from models import LargeCNN_LSTM_Model, UltraLargeEnhanced_CNN_LSTM_Model
from features import compute_basic_stats, compute_wks, compute_sample_entropy, compute_rate_of_change
import matplotlib.pyplot as plt

# Enhanced Configuration for Large Models
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_EPOCHS = 100  # Increased from 30-35
BATCH_SIZE = 8    # Reduced for larger models
LEARNING_RATE = 5e-4  # Slightly lower for stability
WINDOW_SIZE = 100
PATIENCE = 15     # Increased patience for longer training
WEIGHT_DECAY = 1e-4  # Stronger regularization

def main():
    print(f"🚀 Training Large NPP Models with Enhanced Parameters")
    print(f"Using device: {DEVICE}")
    print(f"Target epochs: {NUM_EPOCHS}")
    print("=" * 60)
    
    # Load preprocessed data
    if not os.path.exists("../windows.npy") or not os.path.exists("../labels.npy"):
        print("❌ Data files not found. Please run data_preprocessing.py first.")
        return
    
    windows = np.load("../windows.npy")
    labels = np.load("../labels.npy")
    
    print(f"📊 Dataset: {windows.shape}, Labels: {labels.shape}")
    print(f"📈 Label distribution: {np.bincount(labels)}")
    
    num_windows, ws, num_features = windows.shape
    
    # Compute enhanced statistical features
    print("🔧 Computing enhanced statistical features...")
    stat_feats_list = []
    for i in range(num_windows):
        if i % 50 == 0:
            print(f"   Processing window {i}/{num_windows}")
        
        w = windows[i, :, :]
        
        # Enhanced feature extraction
        stats_basic = compute_basic_stats(w)
        wks = compute_wks(w, omega=1.0)
        
        # Sample entropy for each channel
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
        
        # Rate of change features
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
    print(f"✅ Enhanced features shape: {stat_feats.shape}")
    
    # Create datasets
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
    
    print(f"📚 Train size: {train_size}, Val size: {val_size}")
    
    # Train both large models
    models_to_train = [
        ("Large CNN-LSTM", LargeCNN_LSTM_Model),
        ("Ultra-Large Enhanced", UltraLargeEnhanced_CNN_LSTM_Model)
    ]
    
    for model_name, model_class in models_to_train:
        print(f"\n🎯 Training {model_name} Model")
        print("=" * 50)
        
        # Initialize model
        if model_name == "Large CNN-LSTM":
            model = model_class(
                num_features=num_features,
                window_size=ws,
                stat_feature_dim=stat_feats.shape[1],
                num_classes=len(np.unique(labels)),
                cnn_channels=[64, 128, 256, 512],
                lstm_hidden=256,
                lstm_layers=3
            )
        else:
            model = model_class(
                num_features=num_features,
                window_size=ws,
                stat_feature_dim=stat_feats.shape[1],
                num_classes=len(np.unique(labels))
            )
        
        model = model.to(DEVICE)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"📐 Model parameters: {total_params:,}")
        
        # Optimizer and scheduler
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=8, factor=0.5, verbose=True, min_lr=1e-6
        )
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        best_val_loss = float("inf")
        patience_counter = 0
        train_losses, val_losses = [], []
        train_accs, val_accs = [], []
        
        print(f"🏃 Starting training for {NUM_EPOCHS} epochs...")
        
        for epoch in range(1, NUM_EPOCHS + 1):
            # Training phase
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for X_win, X_stat, y_true in train_loader:
                X_win, X_stat, y_true = X_win.to(DEVICE), X_stat.to(DEVICE), y_true.to(DEVICE)
                
                optimizer.zero_grad()
                logits = model(X_win, X_stat)
                loss = criterion(logits, y_true)
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                running_loss += loss.item() * X_win.size(0)
                _, predicted = logits.max(1)
                correct += (predicted == y_true).sum().item()
                total += y_true.size(0)
            
            train_loss = running_loss / total
            train_acc = correct / total
            
            # Validation phase
            model.eval()
            val_running_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for X_win, X_stat, y_true in val_loader:
                    X_win, X_stat, y_true = X_win.to(DEVICE), X_stat.to(DEVICE), y_true.to(DEVICE)
                    
                    logits = model(X_win, X_stat)
                    loss = criterion(logits, y_true)
                    
                    val_running_loss += loss.item() * X_win.size(0)
                    _, predicted = logits.max(1)
                    val_correct += (predicted == y_true).sum().item()
                    val_total += y_true.size(0)
            
            val_loss = val_running_loss / val_total
            val_acc = val_correct / val_total
            
            # Store metrics
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            
            # Print progress
            if epoch % 5 == 0 or epoch <= 10:
                print(f"[Epoch {epoch:3d}] Train: {train_loss:.4f} ({train_acc:.1%}) | "
                      f"Val: {val_loss:.4f} ({val_acc:.1%})")
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                model_filename = f"../models/large_{model_name.lower().replace(' ', '_').replace('-', '_')}.pth"
                torch.save(model.state_dict(), model_filename)
                print(f"✅ Saved best model (epoch {epoch}) with val_loss: {val_loss:.4f}")
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= PATIENCE:
                print(f"⏹️ Early stopping at epoch {epoch}")
                break
        
        print(f"🏁 {model_name} training complete!")
        print(f"📊 Best validation loss: {best_val_loss:.4f}")
        print(f"📈 Final validation accuracy: {val_accs[-1]:.1%}")
        
        # Plot training history
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.title(f'{model_name} - Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(train_accs, label='Train Acc')
        plt.plot(val_accs, label='Val Acc')
        plt.title(f'{model_name} - Training Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plot_filename = f"../results/large_{model_name.lower().replace(' ', '_').replace('-', '_')}_training.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📈 Training plots saved to: {plot_filename}")
        print()

if __name__ == "__main__":
    main()
