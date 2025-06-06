#!/usr/bin/env python3

import torch
import numpy as np
from models import CNN_LSTM_Model, Enhanced_CNN_LSTM_Model, LargeCNN_LSTM_Model, UltraLargeEnhanced_CNN_LSTM_Model
from features import compute_basic_stats, compute_wks, compute_sample_entropy, compute_rate_of_change
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def plot_confusion_matrix(cm, class_names, model_name, save_path):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_model_comparison_plot(results, save_path):
    """Create comparison plots for all models"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    models = [r['name'] for r in results]
    accuracies = [r['metrics']['accuracy'] for r in results]
    losses = [r['metrics']['loss'] for r in results]
    f1_scores = [r['metrics']['f1_score'] for r in results]
    params = [r['parameters'] for r in results]
    
    # Accuracy comparison
    bars1 = ax1.bar(models, [acc * 100 for acc in accuracies], color='skyblue')
    ax1.set_title('Model Accuracy Comparison')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_ylim(90, 101)
    ax1.tick_params(axis='x', rotation=45)
    for i, v in enumerate(accuracies):
        ax1.text(i, v * 100 + 0.1, f'{v:.1%}', ha='center', va='bottom')
    
    # Loss comparison
    bars2 = ax2.bar(models, losses, color='lightcoral')
    ax2.set_title('Model Loss Comparison')
    ax2.set_ylabel('Loss')
    ax2.tick_params(axis='x', rotation=45)
    for i, v in enumerate(losses):
        ax2.text(i, v + 0.001, f'{v:.4f}', ha='center', va='bottom')
    
    # F1-Score comparison
    bars3 = ax3.bar(models, [f1 * 100 for f1 in f1_scores], color='lightgreen')
    ax3.set_title('Model F1-Score Comparison')
    ax3.set_ylabel('F1-Score (%)')
    ax3.set_ylim(90, 101)
    ax3.tick_params(axis='x', rotation=45)
    for i, v in enumerate(f1_scores):
        ax3.text(i, v * 100 + 0.1, f'{v:.1%}', ha='center', va='bottom')
    
    # Parameter count comparison (log scale)
    bars4 = ax4.bar(models, params, color='gold')
    ax4.set_title('Model Parameter Count Comparison')
    ax4.set_ylabel('Parameters (log scale)')
    ax4.set_yscale('log')
    ax4.tick_params(axis='x', rotation=45)
    for i, v in enumerate(params):
        ax4.text(i, v * 1.1, f'{v:,}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def save_detailed_report(results, save_path):
    """Save detailed evaluation report to text file"""
    with open(save_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("NPP FAULT MONITORING SYSTEM - DETAILED EVALUATION REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        for result in results:
            f.write(f"Model: {result['name']}\n")
            f.write("-" * 40 + "\n")
            f.write(f"Parameters: {result['parameters']:,}\n")
            f.write(f"Accuracy: {result['metrics']['accuracy']:.4f} ({result['metrics']['accuracy']:.1%})\n")
            f.write(f"Loss: {result['metrics']['loss']:.6f}\n")
            f.write(f"Precision: {result['metrics']['precision']:.4f}\n")
            f.write(f"Recall: {result['metrics']['recall']:.4f}\n")
            f.write(f"F1-Score: {result['metrics']['f1_score']:.4f}\n\n")
            
            f.write("Classification Report:\n")
            f.write(result['metrics']['classification_report'])
            f.write("\n" + "=" * 80 + "\n\n")

def evaluate_model(model, test_loader):
    """Evaluate a single model and return comprehensive metrics"""
    model.eval()
    all_preds = []
    all_trues = []
    test_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for X_win, X_stat, y_true in test_loader:
            X_win = X_win.to(DEVICE)
            X_stat = X_stat.to(DEVICE)
            y_true = y_true.to(DEVICE)
            
            logits = model(X_win, X_stat)
            loss = criterion(logits, y_true)
            test_loss += loss.item()
            
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_trues.extend(y_true.cpu().numpy())
    
    test_loss /= len(test_loader)
    test_acc = accuracy_score(all_trues, all_preds)
    
    # Compute additional metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        all_trues, all_preds, average='weighted', zero_division=0
    )
    
    # Confusion matrix
    cm = confusion_matrix(all_trues, all_preds)
    
    # Classification report
    class_report = classification_report(all_trues, all_preds, zero_division=0)
    
    return {
        'accuracy': test_acc,
        'loss': test_loss,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm,
        'classification_report': class_report,
        'predictions': all_preds,
        'true_labels': all_trues
    }

def main():
    print("🚀 COMPREHENSIVE NPP FAULT MONITORING MODEL EVALUATION")
    print("=" * 80)
    
    # Create results directory if it doesn't exist
    results_dir = "../results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Load data
    windows = np.load("../windows.npy")
    labels = np.load("../labels.npy")
    num_windows, ws, num_features = windows.shape
    
    print(f"📊 Dataset: {windows.shape}, Labels: {labels.shape}")
    
    # Define class names (assuming 5 classes based on your data)
    class_names = ['Normal', 'Fault_1', 'Fault_2', 'Fault_3', 'Fault_4']
    if len(np.unique(labels)) != len(class_names):
        class_names = [f'Class_{i}' for i in range(len(np.unique(labels)))]
    
    print(f"📝 Classes: {class_names}")
    
    # Compute basic statistical features
    print("🔧 Computing statistical features...")
    basic_stat_feats_list = []
    enhanced_stat_feats_list = []
    
    for i in range(num_windows):
        w = windows[i, :, :]
        stats_basic = compute_basic_stats(w)
        wks = compute_wks(w, omega=1.0)
        basic_combined = np.concatenate([stats_basic.reshape(-1), wks])
        basic_stat_feats_list.append(basic_combined)
        
        # Enhanced features for larger models
        entropy_feats = []
        roc_feats = []
        
        for j in range(w.shape[1]):
            try:
                entropy_val = compute_sample_entropy(w[:, j])
                entropy_feats.append(entropy_val)
                roc_val = compute_rate_of_change(w[:, j].reshape(-1, 1))
                roc_feats.extend(roc_val.reshape(-1))
            except:
                entropy_feats.append(np.mean(w[:, j]))
                roc_feats.append(np.std(w[:, j]))
        
        entropy_array = np.array(entropy_feats)
        roc_array = np.array(roc_feats)
        enhanced_combined = np.concatenate([
            stats_basic.reshape(-1), wks, entropy_array.reshape(-1), roc_array.reshape(-1)
        ])
        enhanced_stat_feats_list.append(enhanced_combined)
    
    basic_stat_feats = np.vstack(basic_stat_feats_list)
    enhanced_stat_feats = np.vstack(enhanced_stat_feats_list)
    
    print(f"📈 Basic features: {basic_stat_feats.shape[1]}, Enhanced features: {enhanced_stat_feats.shape[1]}")
    
    # Create datasets
    X_windows = torch.tensor(windows, dtype=torch.float32)
    X_basic_stats = torch.tensor(basic_stat_feats, dtype=torch.float32)
    X_enhanced_stats = torch.tensor(enhanced_stat_feats, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)
    
    basic_dataset = TensorDataset(X_windows, X_basic_stats, y)
    enhanced_dataset = TensorDataset(X_windows, X_enhanced_stats, y)
    
    basic_loader = DataLoader(basic_dataset, batch_size=BATCH_SIZE, shuffle=False)
    enhanced_loader = DataLoader(enhanced_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    num_classes = len(np.unique(labels))
    results = []
    
    # Available models
    model_configs = [
        ("CNN-LSTM (Backprop)", "../models/best_cnn_lstm.pth", "basic", CNN_LSTM_Model),
        ("CNN-LSTM (SIAO)", "../models/cnn_lstm_siao_tuned.pth", "basic", CNN_LSTM_Model),
        ("Enhanced CNN-LSTM", "../models/enhanced_cnn_lstm.pth", "enhanced", Enhanced_CNN_LSTM_Model),
        ("Large CNN-LSTM", "../models/large_large_cnn_lstm.pth", "enhanced", LargeCNN_LSTM_Model),
        ("Ultra-Large Enhanced", "../models/large_ultra_large_enhanced.pth", "enhanced", UltraLargeEnhanced_CNN_LSTM_Model),
    ]
    
    print("\n🔍 Evaluating available models...")
    
    for model_name, model_path, feature_type, model_class in model_configs:
        if not os.path.exists(model_path):
            print(f"❌ {model_name}: Model file not found - {model_path}")
            continue
        
        try:
            print(f"\n🔧 Loading {model_name}...")
            
            # Initialize model based on type
            if model_class == CNN_LSTM_Model:
                model = model_class(
                    num_features=num_features, 
                    window_size=ws,
                    stat_feature_dim=basic_stat_feats.shape[1], 
                    num_classes=num_classes
                )
            elif model_class == Enhanced_CNN_LSTM_Model:
                model = model_class(
                    num_features=num_features, 
                    window_size=ws,
                    stat_feature_dim=enhanced_stat_feats.shape[1],
                    num_classes=num_classes,
                    use_cnn_attention=True,
                    use_lstm_attention=True,
                    bidirectional=True
                )
            elif model_class == LargeCNN_LSTM_Model:
                model = model_class(
                    num_features=num_features,
                    window_size=ws,
                    stat_feature_dim=enhanced_stat_feats.shape[1],
                    num_classes=num_classes,
                    cnn_channels=[64, 128, 256, 512],
                    lstm_hidden=256,
                    lstm_layers=3
                )
            elif model_class == UltraLargeEnhanced_CNN_LSTM_Model:
                model = model_class(
                    num_features=num_features,
                    window_size=ws,
                    stat_feature_dim=enhanced_stat_feats.shape[1],
                    num_classes=num_classes
                )
            
            # Load weights
            model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
            model = model.to(DEVICE)
            
            # Select appropriate loader
            loader = basic_loader if feature_type == "basic" else enhanced_loader
            
            # Evaluate
            metrics = evaluate_model(model, loader)
            param_count = count_parameters(model)
            
            print(f"✅ {model_name}")
            print(f"   📊 Parameters: {param_count:,}")
            print(f"   🎯 Accuracy: {metrics['accuracy']:.1%}")
            print(f"   📉 Loss: {metrics['loss']:.4f}")
            print(f"   🎖️  F1-Score: {metrics['f1_score']:.3f}")
            print(f"   ⚡ Precision: {metrics['precision']:.3f}")
            print(f"   🔄 Recall: {metrics['recall']:.3f}")
            
            # Save confusion matrix
            cm_save_path = os.path.join(results_dir, f"confusion_matrix_{model_name.replace(' ', '_').replace('(', '').replace(')', '').lower()}.png")
            plot_confusion_matrix(metrics['confusion_matrix'], class_names, model_name, cm_save_path)
            print(f"   💾 Confusion matrix saved: {cm_save_path}")
            
            results.append({
                'name': model_name,
                'parameters': param_count,
                'metrics': metrics
            })
            
        except Exception as e:
            print(f"❌ {model_name} failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Generate comprehensive results
    if results:
        print("\n" + "=" * 100)
        print("🏆 COMPREHENSIVE PERFORMANCE SUMMARY")
        print("=" * 100)
        print(f"{'Model':<25} {'Parameters':<15} {'Accuracy':<12} {'Loss':<12} {'F1-Score':<12} {'Param Ratio'}")
        print("-" * 100)
        
        baseline_params = results[0]['parameters'] if results else 1
        for result in results:
            model_name = result['name']
            params = result['parameters']
            metrics = result['metrics']
            ratio = f"{params / baseline_params:.1f}x"
            print(f"{model_name:<25} {params:>14,} {metrics['accuracy']:>11.1%} {metrics['loss']:>11.4f} {metrics['f1_score']:>11.3f} {ratio:>11}")
        
        # Create comparison visualizations
        comparison_plot_path = os.path.join(results_dir, "comprehensive_model_comparison.png")
        create_model_comparison_plot(results, comparison_plot_path)
        print(f"\n📊 Model comparison plots saved: {comparison_plot_path}")
        
        # Save detailed report
        report_path = os.path.join(results_dir, "detailed_evaluation_report.txt")
        save_detailed_report(results, report_path)
        print(f"📄 Detailed report saved: {report_path}")
        
        print("\n🎖️  Best performing models:")
        # Best accuracy
        best_accuracy = max(results, key=lambda x: x['metrics']['accuracy'])
        print(f"   🎯 Highest Accuracy: {best_accuracy['name']} ({best_accuracy['metrics']['accuracy']:.1%})")
        
        # Best F1-score
        best_f1 = max(results, key=lambda x: x['metrics']['f1_score'])
        print(f"   🎖️  Highest F1-Score: {best_f1['name']} ({best_f1['metrics']['f1_score']:.3f})")
        
        # Lowest loss
        best_loss = min(results, key=lambda x: x['metrics']['loss'])
        print(f"   📉 Lowest Loss: {best_loss['name']} ({best_loss['metrics']['loss']:.4f})")
        
        # Most efficient (highest accuracy per parameter)
        efficiency_scores = [(r['metrics']['accuracy'] / (r['parameters'] / 1000000), r) for r in results]
        most_efficient = max(efficiency_scores, key=lambda x: x[0])
        print(f"   ⚡ Most Efficient: {most_efficient[1]['name']} ({most_efficient[0]:.2f} acc/M-params)")
        
        print(f"\n📊 Total models evaluated: {len(results)}")
        perfect_accuracy = sum(1 for r in results if r['metrics']['accuracy'] >= 0.999)
        print(f"🎯 Models with ≥99.9% accuracy: {perfect_accuracy}")
        high_f1 = sum(1 for r in results if r['metrics']['f1_score'] >= 0.99)
        print(f"🎖️  Models with F1-Score ≥0.99: {high_f1}")
        
        print(f"\n💾 All evaluation results saved to: {results_dir}")
        print("   📈 Confusion matrices for each model")
        print("   📊 Comprehensive comparison plots") 
        print("   📄 Detailed evaluation report")
        
    else:
        print("❌ No models were successfully evaluated!")

if __name__ == "__main__":
    main()
