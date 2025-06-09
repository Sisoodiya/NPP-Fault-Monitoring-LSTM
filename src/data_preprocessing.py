import os
import glob
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import signal
import joblib  # for saving scalers to disk

DATA_DIR = "../data"
WINDOW_SIZE = 100       # Further reduced to accommodate more files
WINDOW_STRIDE = 50      # 50% overlap for more training samples
CHANNEL_COLUMNS = None  # set this after loading the first CSV

# Label mapping based on fault types observed in the data
LABEL_MAPPING = {
    # Normal/Steady State operations (Label 0)
    "Steady State at 10 power.csv": 0,
    "Steady State at 20 power.csv": 0,
    "Steady State at 30 power.csv": 0,
    "Steady State at 40 power.csv": 0,
    "Steady State at 50 power.csv": 0,
    "Steady State at 60 power.csv": 0,
    "Steady State at 70 power.csv": 0,
    "Steady State at 80 power.csv": 0,
    "Steady State at 90 power.csv": 0,
    "Steady State at 100 power.csv": 0,
    "Power Change 60 to 80.csv": 0,
    
    # Pressurizer PORV faults (Label 1)
    "Pressurizer PORV opening 100.csv": 1,
    "IP200-pzrv1p2.csv": 1, "IP200-pzrv1p4.csv": 1, "IP200-pzrv1p6.csv": 1,
    "IP200-pzrv1p8.csv": 1, "IP200-pzrv1p10.csv": 1,
    "IP200-pzrv2p2.csv": 1, "IP200-pzrv2p4.csv": 1, "IP200-pzrv2p6.csv": 1,
    "IP200-pzrv2p8.csv": 1, "IP200-pzrv2p10.csv": 1,
    "IP200-pzrv3p2.csv": 1, "IP200-pzrv3p4.csv": 1, "IP200-pzrv3p6.csv": 1,
    "IP200-pzrv3p8.csv": 1, "IP200-pzrv3p10.csv": 1,
    "IP200-pzrv4p2.csv": 1, "IP200-pzrv4p4.csv": 1, "IP200-pzrv4p6.csv": 1,
    "IP200-pzrv4p8.csv": 1, "IP200-pzrv4p10.csv": 1,
    "IP200-pzrv5p2.csv": 1, "IP200-pzrv5p4.csv": 1, "IP200-pzrv5p6.csv": 1,
    "IP200-pzrv5p8.csv": 1, "IP200-pzrv5p10.csv": 1,
    "IP200-pzrv40p.csv": 1, "IP200-pzrv80p.csv": 1, "IP200-pzrv100p.csv": 1,
    "pzrvTestFault.csv": 1, "PZRVPredictedResults.csv": 1,
    
    # Steam Generator Tube Rupture (Label 2)
    "SG tube rupture 928.csv": 2,
    "IP200-sgtr1p2.csv": 2, "IP200-sgtr1p4.csv": 2, "IP200-sgtr1p6.csv": 2,
    "IP200-sgtr1p8.csv": 2, "IP200-sgtr1p10.csv": 2,
    "IP200-sgtr2p2.csv": 2, "IP200-sgtr2p4.csv": 2, "IP200-sgtr2p6.csv": 2,
    "IP200-sgtr2p8.csv": 2, "IP200-sgtr2p10.csv": 2,
    "IP200-sgtr3p2.csv": 2, "IP200-sgtr3p4.csv": 2, "IP200-sgtr3p6.csv": 2,
    "IP200-sgtr3p8.csv": 2, "IP200-sgtr3p10.csv": 2,
    "IP200-sgtr4p2.csv": 2, "IP200-sgtr4p4.csv": 2, "IP200-sgtr4p6.csv": 2,
    "IP200-sgtr4p8.csv": 2, "IP200-sgtr4p10.csv": 2,
    "IP200-sgtr5p2.csv": 2, "IP200-sgtr5p4.csv": 2, "IP200-sgtr5p6.csv": 2,
    "IP200-sgtr5p8.csv": 2, "IP200-sgtr5p10.csv": 2,
    "IP200-sgtr30p.csv": 2, "IP200-sgtr70p.csv": 2, "IP200-sgtr100p.csv": 2,
    "SGTRFinalFault.csv": 2, "SGTRPredictedResults.csv": 2,
    
    # Feed Water Break (Label 3)
    "FeedWater Break 50.csv": 3,
    "IP200-fwb20p.csv": 3, "IP200-fwb50p.csv": 3, "IP200-fwb100p.csv": 3,
    
    # Pump Failure and other faults (Label 4)
    "Pump Failure 1.csv": 4,
    "IP200-pp50p.csv": 4, "IP200-pp60p.csv": 4, "IP200-pp100p.csv": 4,
}

# 1. Gather all CSV file paths
def get_csv_paths(data_dir):
    pattern = os.path.join(data_dir, "*.csv")
    return sorted(glob.glob(pattern))

# 2. Load a single CSV, clean it
def load_and_clean(csv_path):
    """Load and clean a single CSV file"""
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded {csv_path}: {df.shape}")
        
        # 2a. Drop exact duplicates
        df = df.drop_duplicates().reset_index(drop=True)
        
        # 2b. Handle missing values
        # Get numeric columns (exclude time column which starts with 'time')
        numeric_cols = df.select_dtypes(include=["float", "int"]).columns.to_list()
        if any(col.startswith('time') for col in df.columns):
            time_cols = [col for col in df.columns if col.startswith('time')]
            for time_col in time_cols:
                if time_col in numeric_cols:
                    numeric_cols.remove(time_col)
        
        # Impute missing values with mean
        if len(numeric_cols) > 0:
            imputer = SimpleImputer(strategy="mean")
            df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
        
        # 2c. Remove any completely NaN columns
        df = df.dropna(axis=1, how='all')
        
        return df
    except Exception as e:
        print(f"Error loading {csv_path}: {e}")
        return None

# 3. Window the data
def create_windows(df, window_size, stride, label):
    """
    Create overlapping windows from time series data.
    
    Input:
      df: cleaned Pandas DataFrame with columns [time, ch1, ch2, ..., chN]
      window_size: number of rows per window
      stride: how many rows to shift between windows
      label: integer class label to attach to each window
    Returns:
      A list of tuples: (window_array, label)
      where window_array has shape (window_size, num_features)
    """
    # Get time column name (starts with 'time')
    time_cols = [col for col in df.columns if col.startswith('time')]
    if time_cols:
        features = df.drop(columns=time_cols).values
    else:
        # If no time column, use all numeric columns
        features = df.select_dtypes(include=[np.number]).values
    
    num_rows, num_features = features.shape
    windows = []
    
    for start in range(0, num_rows - window_size + 1, stride):
        end = start + window_size
        window_data = features[start:end, :]  # shape (window_size, num_features)
        windows.append((window_data, label))
    
    return windows

def augment_data(window_array, augment_factor=0.2, noise_level=0.01):
    """
    Apply data augmentation techniques to increase dataset diversity.
    
    Args:
        window_array: (window_size, num_features)
        augment_factor: probability of applying each augmentation
        noise_level: standard deviation of gaussian noise
    
    Returns:
        List of augmented windows
    """
    augmented_windows = [window_array]  # Always include original
    
    # 1. Add Gaussian noise
    if np.random.rand() < augment_factor:
        noise = np.random.normal(0, noise_level, window_array.shape)
        augmented_windows.append(window_array + noise)
    
    # 2. Time shifting (circular shift)
    if np.random.rand() < augment_factor:
        shift_amount = np.random.randint(1, min(20, window_array.shape[0]//10))
        shifted = np.roll(window_array, shift_amount, axis=0)
        augmented_windows.append(shifted)
    
    # 3. Amplitude scaling
    if np.random.rand() < augment_factor:
        scale_factor = np.random.uniform(0.9, 1.1)
        scaled = window_array * scale_factor
        augmented_windows.append(scaled)
    
    # 4. Signal smoothing (mild low-pass filter)
    if np.random.rand() < augment_factor:
        smoothed = np.copy(window_array)
        for i in range(window_array.shape[1]):
            smoothed[:, i] = signal.savgol_filter(window_array[:, i], 
                                                 window_length=min(11, window_array.shape[0]//4), 
                                                 polyorder=3)
        augmented_windows.append(smoothed)
    
    return augmented_windows

def apply_pca_reduction(windows, labels, n_components=0.95, save_pca=True):
    """
    Apply PCA for dimensionality reduction on flattened windows.
    
    Args:
        windows: (num_windows, window_size, num_features)
        labels: (num_windows,)
        n_components: number of components or variance ratio to keep
        save_pca: whether to save the PCA transformer
    
    Returns:
        reduced_windows: PCA-transformed windows
        pca_transformer: fitted PCA object
    """
    num_windows, window_size, num_features = windows.shape
    
    # Flatten windows for PCA
    flattened = windows.reshape(num_windows, -1)
    
    # Apply PCA
    pca = PCA(n_components=n_components, random_state=42)
    reduced_flattened = pca.fit_transform(flattened)
    
    # Reshape back to 3D (treating PCA components as new "features")
    n_components_actual = reduced_flattened.shape[1]
    
    # Redistribute PCA components across time steps
    if n_components_actual >= window_size:
        new_features = n_components_actual // window_size
        remainder = n_components_actual % window_size
        
        # Reshape to (num_windows, window_size, new_features)
        reduced_windows = reduced_flattened[:, :window_size * new_features].reshape(
            num_windows, window_size, new_features)
        
        if remainder > 0:
            # Handle remainder by zero-padding
            padding = np.zeros((num_windows, window_size, 1))
            padding[:, :remainder, 0] = reduced_flattened[:, -remainder:]
            reduced_windows = np.concatenate([reduced_windows, padding], axis=2)
    else:
        # If fewer components than window_size, replicate across time
        new_features = 1
        reduced_windows = np.zeros((num_windows, window_size, new_features))
        for i in range(window_size):
            comp_idx = i % n_components_actual
            reduced_windows[:, i, 0] = reduced_flattened[:, comp_idx]
    
    if save_pca:
        joblib.dump(pca, "../pca_transformer.pkl")
        print(f"PCA transformer saved. Reduced from {flattened.shape[1]} to {n_components_actual} components")
        print(f"Explained variance ratio: {pca.explained_variance_ratio_.sum():.4f}")
    
    return reduced_windows, pca

# 4. Main preprocessing function
def preprocess_all(data_dir, window_size=WINDOW_SIZE, stride=WINDOW_STRIDE, 
                  enable_augmentation=False, enable_pca=False, pca_components=0.95):
    """Main preprocessing function that processes all CSV files"""
    csv_paths = get_csv_paths(data_dir)
    print(f"Found {len(csv_paths)} CSV files")
    
    # First pass: collect all raw window data
    raw_windows = []
    window_labels = []
    global CHANNEL_COLUMNS
    
    for csv_path in csv_paths:
        filename = os.path.basename(csv_path)
        
        # Skip files not in our label mapping
        if filename not in LABEL_MAPPING:
            print(f"Skipping {filename} - not in label mapping")
            continue
            
        label = LABEL_MAPPING[filename]
        df = load_and_clean(csv_path)
        
        if df is None or df.empty:
            print(f"Skipping {filename} - failed to load or empty")
            continue
            
        # Capture column names on first successful iteration
        if CHANNEL_COLUMNS is None:
            time_cols = [col for col in df.columns if col.startswith('time')]
            if time_cols:
                CHANNEL_COLUMNS = df.drop(columns=time_cols).columns.to_list()
            else:
                CHANNEL_COLUMNS = df.select_dtypes(include=[np.number]).columns.to_list()
            print(f"Using {len(CHANNEL_COLUMNS)} features: {CHANNEL_COLUMNS[:5]}...")
        
        windows = create_windows(df, window_size, stride, label)
        print(f"Created {len(windows)} windows from {filename}")
        
        for w, lab in windows:
            # Apply data augmentation if enabled
            if enable_augmentation and label != 0:  # Don't augment normal class too much
                augmented = augment_data(w, augment_factor=0.3)
                for aug_w in augmented:
                    raw_windows.append(aug_w)
                    window_labels.append(lab)
            else:
                raw_windows.append(w)
                window_labels.append(lab)
    
    if not raw_windows:
        raise ValueError("No windows created. Check data directory and label mapping.")
    
    # Convert to numpy arrays
    raw_windows = np.stack(raw_windows, axis=0)
    labels = np.array(window_labels)
    
    print(f"Total windows: {raw_windows.shape}, Labels distribution: {np.bincount(labels)}")
    
    # Fit scaler on flattened data
    num_windows, ws, nf = raw_windows.shape
    flattened = raw_windows.reshape(num_windows * ws, nf)
    
    scaler = StandardScaler()
    flattened_scaled = scaler.fit_transform(flattened)
    joblib.dump(scaler, "../scaler.pkl")
    print("Saved scaler to ../scaler.pkl")
    
    # Reshape back to windows
    all_windows_scaled = flattened_scaled.reshape(num_windows, ws, nf)
    
    # Apply PCA reduction if enabled
    if enable_pca:
        all_windows_scaled, pca_transformer = apply_pca_reduction(
            all_windows_scaled, labels, n_components=pca_components)
        print(f"Applied PCA reduction: {all_windows_scaled.shape}")
    
    return all_windows_scaled, labels, scaler

if __name__ == "__main__":
    windows, labels, scaler = preprocess_all(DATA_DIR)
    print(f"Total windows: {windows.shape}, Labels: {labels.shape}")
    # Save windows and labels to disk to speed up training
    np.save("../windows.npy", windows)
    np.save("../labels.npy", labels)
