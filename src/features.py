# src/features.py

import numpy as np
import scipy.stats as stats
from scipy.stats import entropy

def compute_sample_entropy(data, m=2, r=None):
    """
    Compute sample entropy for a time series.
    
    Args:
        data: 1D time series
        m: pattern length (default 2)
        r: tolerance for matching (default 0.2 * std)
    
    Returns:
        Sample entropy value
    """
    if r is None:
        r = 0.2 * np.std(data)
    
    def _maxdist(xi, xj, m):
        return max([abs(ua - va) for ua, va in zip(xi, xj)])
    
    def _phi(m):
        patterns = np.array([data[i:i + m] for i in range(len(data) - m + 1)])
        C = np.zeros(len(patterns))
        for i in range(len(patterns)):
            template = patterns[i]
            for j in range(len(patterns)):
                if _maxdist(template, patterns[j], m) <= r:
                    C[i] += 1.0
        phi = np.sum([np.log(c / len(patterns)) for c in C if c > 0]) / len(patterns)
        return phi
    
    return _phi(m) - _phi(m + 1)

def compute_rate_of_change(window_array):
    """
    Compute rate of change (first derivative approximation) for each channel.
    
    Args:
        window_array: (window_size, num_features)
    
    Returns:
        roc_features: (num_features,) - statistical measures of rate of change
    """
    # Compute first differences for each channel
    diff = np.diff(window_array, axis=0)  # (window_size-1, num_features)
    
    # Compute statistical measures of the rate of change
    roc_mean = np.mean(diff, axis=0)
    roc_std = np.std(diff, axis=0)
    roc_max = np.max(np.abs(diff), axis=0)
    
    # Combine into feature vector
    roc_features = np.concatenate([roc_mean, roc_std, roc_max])
    return roc_features

def compute_spectral_features(window_array, fs=1.0):
    """
    Compute spectral domain features for each channel.
    
    Args:
        window_array: (window_size, num_features)
        fs: sampling frequency
    
    Returns:
        spectral_features: (num_features * 4,) - spectral centroid, bandwidth, rolloff, flux
    """
    spectral_features = []
    
    for ch in range(window_array.shape[1]):
        signal = window_array[:, ch]
        
        # Compute FFT
        fft = np.fft.fft(signal)
        magnitude = np.abs(fft[:len(fft)//2])
        freqs = np.fft.fftfreq(len(signal), 1/fs)[:len(fft)//2]
        
        # Spectral centroid
        if np.sum(magnitude) > 0:
            spectral_centroid = np.sum(freqs * magnitude) / np.sum(magnitude)
        else:
            spectral_centroid = 0
        
        # Spectral bandwidth
        if np.sum(magnitude) > 0:
            spectral_bandwidth = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * magnitude) / np.sum(magnitude))
        else:
            spectral_bandwidth = 0
        
        # Spectral rolloff (frequency below which 85% of energy is contained)
        cumsum_mag = np.cumsum(magnitude)
        if cumsum_mag[-1] > 0:
            rolloff_idx = np.where(cumsum_mag >= 0.85 * cumsum_mag[-1])[0]
            spectral_rolloff = freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else 0
        else:
            spectral_rolloff = 0
        
        # Spectral flux (change in magnitude spectrum)
        if len(signal) > 1:
            prev_magnitude = np.abs(np.fft.fft(np.roll(signal, 1))[:len(fft)//2])
            spectral_flux = np.sum((magnitude - prev_magnitude) ** 2)
        else:
            spectral_flux = 0
        
        spectral_features.extend([spectral_centroid, spectral_bandwidth, spectral_rolloff, spectral_flux])
    
    return np.array(spectral_features)

def compute_advanced_stats(window_array):
    """
    Compute advanced statistical features including sample entropy and rate of change.
    
    Args:
        window_array: (window_size, num_features)
    
    Returns:
        advanced_features: concatenated advanced statistical features
    """
    features = []
    
    for ch in range(window_array.shape[1]):
        signal = window_array[:, ch]
        
        # Sample entropy
        try:
            samp_ent = compute_sample_entropy(signal)
            if np.isnan(samp_ent) or np.isinf(samp_ent):
                samp_ent = 0.0
        except:
            samp_ent = 0.0
        
        # Approximate entropy
        try:
            # Simple approximate entropy calculation
            def _approximate_entropy(data, m=2, r=None):
                if r is None:
                    r = 0.2 * np.std(data)
                
                def _maxdist(xi, xj):
                    return max([abs(ua - va) for ua, va in zip(xi, xj)])
                
                def _phi(m):
                    patterns = [data[i:i + m] for i in range(len(data) - m + 1)]
                    C = []
                    for i in range(len(patterns)):
                        template = patterns[i]
                        matches = sum(1 for pattern in patterns if _maxdist(template, pattern) <= r)
                        C.append(matches / len(patterns))
                    phi = sum(np.log(c) for c in C if c > 0) / len(C)
                    return phi
                
                return _phi(m) - _phi(m + 1)
            
            app_ent = _approximate_entropy(signal)
            if np.isnan(app_ent) or np.isinf(app_ent):
                app_ent = 0.0
        except:
            app_ent = 0.0
        
        # Hurst exponent (simplified calculation)
        try:
            def _hurst_exponent(data):
                lags = range(2, min(20, len(data)//4))
                tau = []
                for lag in lags:
                    if lag < len(data):
                        # Calculate rescaled range
                        y = np.cumsum(data - np.mean(data))
                        R = np.max(y[:lag]) - np.min(y[:lag])
                        S = np.std(data[:lag])
                        if S > 0:
                            tau.append(R / S)
                        else:
                            tau.append(0)
                
                if len(tau) > 1:
                    # Linear regression to find Hurst exponent
                    log_lags = np.log(lags[:len(tau)])
                    log_tau = np.log(np.array(tau))
                    hurst = np.polyfit(log_lags, log_tau, 1)[0]
                else:
                    hurst = 0.5
                
                return hurst
            
            hurst = _hurst_exponent(signal)
            if np.isnan(hurst) or np.isinf(hurst):
                hurst = 0.5
        except:
            hurst = 0.5
        
        features.extend([samp_ent, app_ent, hurst])
    
    # Add rate of change features
    roc_features = compute_rate_of_change(window_array)
    features.extend(roc_features)
    
    # Add spectral features
    spectral_features = compute_spectral_features(window_array)
    features.extend(spectral_features)
    
    return np.array(features)

def compute_basic_stats(window_array):
    """
    Compute basic statistical features for each channel in a window.
    
    window_array: np.ndarray shape (window_size, num_features)
    Returns:
      stats_vector: np.ndarray shape (5, num_features)
       Order: [ mean, median, std, var, entropy ]
    """
    x = window_array  # (W, F)
    mean = x.mean(axis=0)                      # (F,)
    median = np.median(x, axis=0)               # (F,)
    std = x.std(axis=0, ddof=0)                 # (F,)
    var = x.var(axis=0, ddof=0)                 # (F,)
    
    # Entropy: Use histogram‐based Shannon entropy per channel
    entropy = []
    for ch_index in range(x.shape[1]):
        vals = x[:, ch_index]
        # Adaptive bin count based on window size, minimum 10, maximum 50
        n_bins = min(50, max(10, int(np.sqrt(len(vals)))))
        
        # Handle edge case where all values are the same
        if np.all(vals == vals[0]):
            entropy.append(0.0)
        else:
            hist, bin_edges = np.histogram(vals, bins=n_bins, density=True)
            # Normalize to get probabilities
            hist = hist / (hist.sum() + 1e-12)
            hist = hist + 1e-12  # avoid log(0)
            ent = -np.sum(hist * np.log2(hist))  # Shannon entropy
            entropy.append(ent)
    
    entropy = np.array(entropy)  # (F,)
    stats_vector = np.vstack([mean, median, std, var, entropy])  # (5, F)
    return stats_vector

def compute_kurtosis_skewness(window_array):
    """
    Compute kurtosis and skewness for each feature channel.
    
    Returns:
       kurtosis: (num_features,)
       skewness: (num_features,)
    """
    # Use scipy.stats with bias correction
    kurtosis = stats.kurtosis(window_array, axis=0, fisher=False, bias=False)  # Pearson kurtosis
    skewness = stats.skew(window_array, axis=0, bias=False)
    
    # Handle any NaN values that might occur with constant data
    kurtosis = np.nan_to_num(kurtosis, nan=0.0, posinf=0.0, neginf=0.0)
    skewness = np.nan_to_num(skewness, nan=0.0, posinf=0.0, neginf=0.0)
    
    return kurtosis, skewness  # each shape: (F,)

def compute_wks(window_array, omega):
    """
    Compute Weighted Kurtosis + Skewness per channel.
    WKS = omega * Kurtosis + Skewness
    
    Args:
        window_array: (window_size, num_features)
        omega: weight factor for kurtosis
    
    Returns:
        wks: (num_features,) - weighted combination of kurtosis and skewness
    """
    kurtosis, skewness = compute_kurtosis_skewness(window_array)
    wks = omega * kurtosis + skewness  # shape: (F,)
    return wks
