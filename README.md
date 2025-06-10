# NPP Fault Monitoring System - Source Code

This directory contains the clean, organized source code for the NPP Fault Monitoring System.

## 📁 File Structure

### Core Modules
- **`models.py`** - All neural network model architectures (CNN-LSTM, Enhanced, Large-scale, Ultra-large)
- **`features.py`** - Feature extraction functions (statistical features, WKS, entropy, etc.)
- **`utils.py`** - Utility functions for data processing and visualisation
- **`data_preprocessing.py`** - Data preprocessing and windowing functions

### Training Scripts
- **`train_basic_cnn_lstm.py`** - Train the basic CNN-LSTM model with backpropagation
- **`train_enhanced_cnn_lstm.py`** - Train the enhanced CNN-LSTM model with attention mechanisms
- **`train_large_scale_models.py`** - Train large-scale models (11M and 37M parameters)
- **`train_siao_optimization.py`** - Train models using SIAO optimization

### Optimization
- **`siao_optimizer.py`** - SIAO (Self Aquila Optimization) implementation

### Evaluation
- **`evaluate_all_models.py`** - Comprehensive evaluation script for all trained models

## 🚀 Usage

### Quick Model Evaluation
```bash
cd src
python evaluate_all_models.py
```

### Training Models
```bash
# Train basic model
python train_basic_cnn_lstm.py

# Train enhanced model  
python train_enhanced_cnn_lstm.py

# Train large-scale models
python train_large_scale_models.py

# Train with SIAO optimization
python train_siao_optimization.py
```

## 📊 Model Performance Summary

| Model | Parameters | Accuracy | Features |
|-------|------------|----------|----------|
| CNN-LSTM (Backprop) | 289,509 | 100.0% | Baseline model |
| CNN-LSTM (SIAO) | 289,509 | 93.0% | SIAO optimization |
| Enhanced CNN-LSTM | 1,281,157 | 100.0% | Attention + bidirectional |
| Large CNN-LSTM | 11,243,141 | 99.5% | 4-layer CNN + 3-layer LSTM |
| Ultra-Large Enhanced | 37,636,869 | 97.0% | Multi-scale + 8-head attention |

## 🔧 Dependencies

All required packages are listed in `../requirements.txt`.
