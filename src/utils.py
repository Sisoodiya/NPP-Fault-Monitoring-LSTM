# src/utils.py

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import torch
import joblib
from scipy import stats
from scipy.optimize import minimize
from scipy.special import gamma

def save_model(model, filepath):
    """Save PyTorch model state dict"""
    torch.save(model.state_dict(), filepath)
    print(f"Model saved to {filepath}")

def load_model(model, filepath, device='cpu'):
    """Load PyTorch model state dict"""
    model.load_state_dict(torch.load(filepath, map_location=device))
    model.eval()
    print(f"Model loaded from {filepath}")
    return model

def save_scaler(scaler, filepath):
    """Save sklearn scaler"""
    joblib.dump(scaler, filepath)
    print(f"Scaler saved to {filepath}")

def load_scaler(filepath):
    """Load sklearn scaler"""
    scaler = joblib.load(filepath)
    print(f"Scaler loaded from {filepath}")
    return scaler

def plot_confusion_matrix(y_true, y_pred, class_names=None, figsize=(8, 6), save_path=None):
    """Plot confusion matrix with nice formatting"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.show()

def plot_training_history(train_losses, val_losses, train_accs, val_accs, save_path=None):
    """Plot training and validation metrics"""
    epochs = range(1, len(train_losses) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss plot
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(epochs, train_accs, 'b-', label='Training Accuracy')
    ax2.plot(epochs, val_accs, 'r-', label='Validation Accuracy')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Training history saved to {save_path}")
    
    plt.show()

def print_classification_metrics(y_true, y_pred, class_names=None):
    """Print detailed classification metrics"""
    print("Classification Report:")
    print("=" * 50)
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    
    # Calculate per-class accuracy
    class_accuracies = cm.diagonal() / cm.sum(axis=1)
    print("\nPer-class Accuracy:")
    for i, acc in enumerate(class_accuracies):
        class_name = class_names[i] if class_names else f"Class {i}"
        print(f"{class_name}: {acc:.4f}")

def create_directory(path):
    """Create directory if it doesn't exist"""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")

def save_results(results_dict, filepath):
    """Save results dictionary to pickle file"""
    with open(filepath, 'wb') as f:
        pickle.dump(results_dict, f)
    print(f"Results saved to {filepath}")

def load_results(filepath):
    """Load results dictionary from pickle file"""
    with open(filepath, 'rb') as f:
        results = pickle.load(f)
    print(f"Results loaded from {filepath}")
    return results

def count_parameters(model):
    """Count trainable parameters in a PyTorch model"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    return total_params, trainable_params

def plot_feature_importance(feature_names, importance_scores, top_k=20, save_path=None):
    """Plot feature importance scores"""
    # Sort features by importance
    sorted_idx = np.argsort(importance_scores)[::-1][:top_k]
    sorted_features = [feature_names[i] for i in sorted_idx]
    sorted_scores = importance_scores[sorted_idx]
    
    plt.figure(figsize=(10, max(6, top_k * 0.3)))
    bars = plt.barh(range(len(sorted_scores)), sorted_scores)
    plt.yticks(range(len(sorted_scores)), sorted_features)
    plt.xlabel('Importance Score')
    plt.title(f'Top {top_k} Feature Importance')
    plt.gca().invert_yaxis()
    
    # Color bars
    colors = plt.cm.viridis(np.linspace(0, 1, len(bars)))
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Feature importance plot saved to {save_path}")
    
    plt.show()

def setup_logging(log_file='training.log'):
    """Setup logging configuration"""
    import logging
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

class EarlyStopping:
    """Early stopping utility class"""
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False
        
    def save_checkpoint(self, model):
        """Save model weights"""
        self.best_weights = model.state_dict().copy()

def set_random_seeds(seed=42):
    """Set random seeds for reproducibility"""
    import random
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"Random seeds set to {seed}")

def get_device():
    """Get the best available device (CUDA, MPS, or CPU)"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA: {torch.cuda.get_device_name()}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon)")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    return device

def format_time(seconds):
    """Format seconds into human readable time"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"

def fit_weibull_distribution(data, method='mle'):
    """
    Fit Weibull distribution to data and extract parameters.
    
    Args:
        data: 1D array of positive values
        method: 'mle' for maximum likelihood estimation or 'lsq' for least squares
    
    Returns:
        dict: Dictionary containing Weibull parameters and goodness of fit
    """
    # Remove zeros and negative values
    data = data[data > 0]
    
    if len(data) < 3:
        return {'shape': 1.0, 'scale': 1.0, 'location': 0.0, 'aic': np.inf, 'bic': np.inf}
    
    try:
        # Fit Weibull distribution using scipy
        if method == 'mle':
            # Method of moments for initial guess
            mean_data = np.mean(data)
            var_data = np.var(data)
            
            # Initial parameter estimates
            scale_init = mean_data
            shape_init = 1.2  # Common initial guess
            
            # Fit using MLE
            shape, loc, scale = stats.weibull_min.fit(data, floc=0, f0=shape_init, scale=scale_init)
            
        else:  # Least squares method
            def weibull_cdf(x, shape, scale):
                return 1 - np.exp(-(x/scale)**shape)
            
            def objective(params):
                shape, scale = params
                if shape <= 0 or scale <= 0:
                    return np.inf
                
                # Empirical CDF
                sorted_data = np.sort(data)
                empirical_cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
                
                # Theoretical CDF
                theoretical_cdf = weibull_cdf(sorted_data, shape, scale)
                
                # Sum of squared differences
                return np.sum((empirical_cdf - theoretical_cdf)**2)
            
            # Initial guess
            initial_guess = [1.2, np.mean(data)]
            bounds = [(0.1, 10), (0.1, np.max(data) * 2)]
            
            result = minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B')
            
            if result.success:
                shape, scale = result.x
                loc = 0
            else:
                # Fallback to MLE
                shape, loc, scale = stats.weibull_min.fit(data, floc=0)
        
        # Calculate goodness of fit metrics
        # Log-likelihood
        log_likelihood = np.sum(stats.weibull_min.logpdf(data, shape, loc, scale))
        
        # AIC and BIC
        k = 2  # Number of parameters (shape and scale, location fixed at 0)
        n = len(data)
        aic = 2 * k - 2 * log_likelihood
        bic = k * np.log(n) - 2 * log_likelihood
        
        # Kolmogorov-Smirnov test
        ks_statistic, ks_p_value = stats.kstest(data, 
                                               lambda x: stats.weibull_min.cdf(x, shape, loc, scale))
        
        return {
            'shape': shape,
            'scale': scale,
            'location': loc,
            'log_likelihood': log_likelihood,
            'aic': aic,
            'bic': bic,
            'ks_statistic': ks_statistic,
            'ks_p_value': ks_p_value,
            'mean': scale * gamma(1 + 1/shape),
            'variance': scale**2 * (gamma(1 + 2/shape) - gamma(1 + 1/shape)**2)
        }
        
    except Exception as e:
        print(f"Weibull fitting failed: {e}")
        return {'shape': 1.0, 'scale': 1.0, 'location': 0.0, 'aic': np.inf, 'bic': np.inf}

def weibull_reliability_analysis(data, time_points=None):
    """
    Perform reliability analysis using Weibull distribution.
    
    Args:
        data: Failure time data
        time_points: Time points for reliability calculation (optional)
    
    Returns:
        dict: Reliability analysis results
    """
    # Fit Weibull distribution
    weibull_params = fit_weibull_distribution(data)
    
    if time_points is None:
        time_points = np.linspace(0, np.max(data), 100)
    
    shape = weibull_params['shape']
    scale = weibull_params['scale']
    loc = weibull_params['location']
    
    # Calculate reliability function R(t) = exp(-(t/scale)^shape)
    reliability = np.exp(-((time_points - loc) / scale) ** shape)
    reliability[time_points <= loc] = 1.0  # Before location parameter
    
    # Failure rate (hazard function) h(t) = (shape/scale) * ((t-loc)/scale)^(shape-1)
    hazard_rate = np.zeros_like(time_points)
    valid_idx = time_points > loc
    if np.any(valid_idx):
        hazard_rate[valid_idx] = (shape / scale) * (((time_points[valid_idx] - loc) / scale) ** (shape - 1))
    
    # Mean time to failure (MTTF)
    mttf = loc + scale * gamma(1 + 1/shape)
    
    return {
        'weibull_params': weibull_params,
        'time_points': time_points,
        'reliability': reliability,
        'hazard_rate': hazard_rate,
        'mttf': mttf,
        'characteristic_life': scale  # Time at which 63.2% fail
    }

def plot_weibull_analysis(data, save_path=None):
    """
    Create comprehensive Weibull analysis plots.
    
    Args:
        data: Failure time data
        save_path: Path to save the plot
    """
    import matplotlib.pyplot as plt
    
    # Perform analysis
    analysis = weibull_reliability_analysis(data)
    weibull_params = analysis['weibull_params']
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Histogram with fitted PDF
    ax1.hist(data, bins=20, density=True, alpha=0.7, color='skyblue', label='Data')
    
    x_pdf = np.linspace(0, np.max(data), 200)
    pdf_values = stats.weibull_min.pdf(x_pdf, weibull_params['shape'], 
                                      weibull_params['location'], weibull_params['scale'])
    ax1.plot(x_pdf, pdf_values, 'r-', linewidth=2, label=f"Weibull PDF\n(β={weibull_params['shape']:.2f}, η={weibull_params['scale']:.2f})")
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Probability Density')
    ax1.set_title('Weibull Distribution Fit')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Q-Q Plot
    theoretical_quantiles = stats.weibull_min.ppf(np.linspace(0.01, 0.99, len(data)), 
                                                 weibull_params['shape'], 
                                                 weibull_params['location'], 
                                                 weibull_params['scale'])
    sample_quantiles = np.sort(data)
    
    ax2.scatter(theoretical_quantiles, sample_quantiles, alpha=0.6)
    min_val = min(np.min(theoretical_quantiles), np.min(sample_quantiles))
    max_val = max(np.max(theoretical_quantiles), np.max(sample_quantiles))
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    ax2.set_xlabel('Theoretical Quantiles')
    ax2.set_ylabel('Sample Quantiles')
    ax2.set_title('Q-Q Plot')
    ax2.grid(True, alpha=0.3)
    
    # 3. Reliability Function
    ax3.plot(analysis['time_points'], analysis['reliability'], 'b-', linewidth=2)
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Reliability R(t)')
    ax3.set_title('Reliability Function')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1)
    
    # 4. Hazard Rate
    ax4.plot(analysis['time_points'], analysis['hazard_rate'], 'g-', linewidth=2)
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Hazard Rate h(t)')
    ax4.set_title('Hazard Rate Function')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Add summary text
    summary_text = f"""Weibull Parameters:
Shape (β): {weibull_params['shape']:.3f}
Scale (η): {weibull_params['scale']:.3f}
MTTF: {analysis['mttf']:.3f}
AIC: {weibull_params['aic']:.2f}
KS p-value: {weibull_params.get('ks_p_value', 0):.3f}"""
    
    fig.text(0.02, 0.98, summary_text, transform=fig.transFigure, 
             verticalalignment='top', fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Weibull analysis plot saved to {save_path}")
    
    plt.show()
    
    return analysis

def advanced_signal_analysis(signal, fs=1.0):
    """
    Perform advanced signal analysis including spectral and time-domain features.
    
    Args:
        signal: 1D time series signal
        fs: Sampling frequency
    
    Returns:
        dict: Advanced signal features
    """
    from scipy import signal as scipy_signal
    from scipy.stats import kurtosis, skew
    
    features = {}
    
    # Time domain features
    features['mean'] = np.mean(signal)
    features['std'] = np.std(signal)
    features['rms'] = np.sqrt(np.mean(signal**2))
    features['peak_to_peak'] = np.ptp(signal)
    features['crest_factor'] = np.max(np.abs(signal)) / features['rms'] if features['rms'] > 0 else 0
    features['kurtosis'] = kurtosis(signal)
    features['skewness'] = skew(signal)
    
    # Zero crossing rate
    zero_crossings = np.where(np.diff(np.signbit(signal)))[0]
    features['zero_crossing_rate'] = len(zero_crossings) / len(signal)
    
    # Frequency domain features
    fft = np.fft.fft(signal)
    freqs = np.fft.fftfreq(len(signal), 1/fs)
    magnitude = np.abs(fft[:len(fft)//2])
    freqs_positive = freqs[:len(freqs)//2]
    
    if np.sum(magnitude) > 0:
        # Spectral centroid
        features['spectral_centroid'] = np.sum(freqs_positive * magnitude) / np.sum(magnitude)
        
        # Spectral spread
        features['spectral_spread'] = np.sqrt(
            np.sum(((freqs_positive - features['spectral_centroid'])**2) * magnitude) / np.sum(magnitude)
        )
        
        # Spectral rolloff (95% energy)
        cumsum_mag = np.cumsum(magnitude)
        rolloff_idx = np.where(cumsum_mag >= 0.95 * cumsum_mag[-1])[0]
        features['spectral_rolloff'] = freqs_positive[rolloff_idx[0]] if len(rolloff_idx) > 0 else 0
        
        # Spectral flatness (Wiener entropy) - manual calculation
        # Geometric mean / arithmetic mean of power spectrum
        if np.all(magnitude > 0):
            geometric_mean = np.exp(np.mean(np.log(magnitude + 1e-10)))
            arithmetic_mean = np.mean(magnitude)
            features['spectral_flatness'] = geometric_mean / arithmetic_mean if arithmetic_mean > 0 else 0
        else:
            features['spectral_flatness'] = 0
    else:
        features['spectral_centroid'] = 0
        features['spectral_spread'] = 0
        features['spectral_rolloff'] = 0
        features['spectral_flatness'] = 0
    
    # Envelope analysis
    analytic_signal = scipy_signal.hilbert(signal)
    envelope = np.abs(analytic_signal)
    features['envelope_mean'] = np.mean(envelope)
    features['envelope_std'] = np.std(envelope)
    
    return features