import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TemporalAttention(nn.Module):
    """
    Temporal attention mechanism for time series data.
    """
    def __init__(self, hidden_size, num_heads=4):
        super(TemporalAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, hidden_size)
        Returns:
            attended: (batch_size, seq_len, hidden_size)
        """
        batch_size, seq_len, _ = x.size()
        
        # Generate queries, keys, values
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attended = torch.matmul(attention_weights, V)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        
        # Final linear transformation
        output = self.out(attended)
        
        # Residual connection
        return output + x

class EnhancedCNNEncoder(nn.Module):
    """
    Enhanced CNN with residual connections and attention.
    """
    def __init__(self, num_channels, cnn_out_channels=32, kernel_sizes=[5,5], pool_sizes=[2,2], use_attention=True):
        super(EnhancedCNNEncoder, self).__init__()
        self.use_attention = use_attention
        
        layers = []
        in_channels = num_channels
        
        for idx, (k, p, out_c) in enumerate(zip(kernel_sizes, pool_sizes, [cnn_out_channels, cnn_out_channels*2])):
            # Convolutional block
            conv = nn.Conv1d(in_channels=in_channels,
                             out_channels=out_c,
                             kernel_size=k,
                             stride=1,
                             padding=k//2)
            bn = nn.BatchNorm1d(out_c)
            relu = nn.ReLU()
            pool = nn.MaxPool1d(kernel_size=p, stride=p)
            
            # Add residual connection if dimensions match
            if in_channels == out_c:
                layers += [ResidualBlock1D(conv, bn, relu), pool]
            else:
                layers += [conv, bn, relu, pool]
            
            in_channels = out_c
        
        self.cnn = nn.Sequential(*layers)
        self.flatten = nn.Flatten()
        
        # Add temporal attention if enabled
        if self.use_attention:
            # Calculate output size after CNN
            dummy_input = torch.zeros((1, num_channels, 100))  # Dummy for size calculation
            with torch.no_grad():
                dummy_out = self.cnn(dummy_input)
                self.feature_size = dummy_out.size(-1)
                self.out_channels = dummy_out.size(1)
            
            self.temporal_attention = TemporalAttention(self.out_channels)

    def forward(self, x):
        # x: (batch_size, window_size, num_features) → transpose to (batch, num_features, window_size)
        x = x.permute(0, 2, 1)
        out = self.cnn(x)  # shape: (batch, out_channels, new_length)
        
        if self.use_attention:
            # Apply temporal attention
            out = out.permute(0, 2, 1)  # (batch, new_length, out_channels)
            out = self.temporal_attention(out)
            out = out.permute(0, 2, 1)  # Back to (batch, out_channels, new_length)
        
        out = self.flatten(out)  # shape: (batch, out_channels * new_length)
        return out

class ResidualBlock1D(nn.Module):
    """
    1D Residual block for CNN.
    """
    def __init__(self, conv, bn, activation):
        super(ResidualBlock1D, self).__init__()
        self.conv = conv
        self.bn = bn
        self.activation = activation
        
    def forward(self, x):
        residual = x
        out = self.conv(x)
        out = self.bn(out)
        out = self.activation(out)
        
        # Add residual connection
        if x.size() == out.size():
            out += residual
        
        return out

class BidirectionalLSTMClassifier(nn.Module):
    """
    Enhanced LSTM classifier with bidirectional option and attention.
    """
    def __init__(self, cnn_feature_dim, stat_feature_dim, lstm_hidden_size=32, 
                 num_classes=5, dropout=0.2, bidirectional=False, use_attention=True):
        super(BidirectionalLSTMClassifier, self).__init__()
        self.input_dim = cnn_feature_dim + stat_feature_dim
        self.lstm_hidden_size = lstm_hidden_size
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size=self.input_dim,
                            hidden_size=self.lstm_hidden_size,
                            num_layers=2,  # Increased depth
                            batch_first=True,
                            bidirectional=self.bidirectional,
                            dropout=dropout if dropout > 0 else 0)
        
        # Calculate LSTM output size
        lstm_output_size = self.lstm_hidden_size * (2 if self.bidirectional else 1)
        
        # Add attention mechanism
        if self.use_attention:
            self.attention = TemporalAttention(lstm_output_size)
        
        self.dropout = nn.Dropout(dropout)
        
        # Multi-layer classifier
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_size, lstm_output_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_output_size // 2, num_classes)
        )

    def forward(self, cnn_feats, stat_feats):
        """
        cnn_feats: shape (batch_size, seq_len, cnn_feature_dim)
        stat_feats: shape (batch_size, seq_len, stat_feature_dim)
        """
        # Concatenate features along last dimension
        x = torch.cat([cnn_feats, stat_feats], dim=-1)  # shape: (batch, seq_len, input_dim)
        
        # Pass through LSTM
        out, (hn, cn) = self.lstm(x)  # out: (batch, seq_len, lstm_output_size)
        
        # Apply temporal attention if enabled
        if self.use_attention:
            out = self.attention(out)
        
        # Use the last time step or global average pooling
        if self.use_attention:
            # Global average pooling after attention
            last = torch.mean(out, dim=1)  # (batch, lstm_output_size)
        else:
            # Use last time step
            last = out[:, -1, :]  # (batch, lstm_output_size)
        
        last = self.dropout(last)
        logits = self.classifier(last)  # (batch, num_classes)
        return logits

class CNN_LSTM_Model(nn.Module):
    def __init__(self, num_features, window_size, stat_feature_dim, num_classes=5):
        super(CNN_LSTM_Model, self).__init__()
        # 1. Build CNN encoder
        self.cnn_encoder = EnhancedCNNEncoder(num_channels=num_features,
                                      cnn_out_channels=32,
                                      kernel_sizes=[5,5],
                                      pool_sizes=[2,2])
        # 2. Determine cnn_feature_dim dynamically
        #    We’ll run a dummy pass
        dummy_input = torch.zeros((1, window_size, num_features))
        dummy_cnn_out = self.cnn_encoder(dummy_input)
        cnn_feature_dim = dummy_cnn_out.shape[-1]  # e.g., 32 * (window_size/4)
        # 3. LSTM classifier
        self.classifier = BidirectionalLSTMClassifier(cnn_feature_dim=cnn_feature_dim,
                                         stat_feature_dim=stat_feature_dim,
                                         lstm_hidden_size=32,
                                         num_classes=num_classes,
                                         dropout=0.2)

    def forward(self, window_batch, stat_batch):
        """
        window_batch: (batch_size, window_size, num_features)
        stat_batch: (batch_size, stat_feature_dim)
        We want to treat each window as a sequence of length 1.
        If you want to do multiple windows in time, you can expand dims.
        """
        # 1. CNN features (for each window) → (batch, cnn_feature_dim)
        cnn_feats = self.cnn_encoder(window_batch)  # (batch, cnn_feature_dim)
        # 2. “Sequence” dimension: reshape to (batch, seq_len=1, cnn_feature_dim)
        cnn_feats = cnn_feats.unsqueeze(1)
        # 3. Stats: (batch, stat_feature_dim) → (batch, seq_len=1, stat_feature_dim)
        stat_feats = stat_batch.unsqueeze(1)
        # 4. Pass to classifier
        logits = self.classifier(cnn_feats, stat_feats)  # (batch, num_classes)
        return logits

class Enhanced_CNN_LSTM_Model(nn.Module):
    """
    Enhanced CNN-LSTM model with attention and bidirectional options.
    """
    def __init__(self, num_features, window_size, stat_feature_dim, num_classes=5, 
                 use_cnn_attention=True, use_lstm_attention=True, bidirectional=False):
        super(Enhanced_CNN_LSTM_Model, self).__init__()
        
        # 1. Build enhanced CNN encoder
        self.cnn_encoder = EnhancedCNNEncoder(num_channels=num_features,
                                              cnn_out_channels=32,
                                              kernel_sizes=[5,5],
                                              pool_sizes=[2,2],
                                              use_attention=use_cnn_attention)
        
        # 2. Determine cnn_feature_dim dynamically
        dummy_input = torch.zeros((1, window_size, num_features))
        dummy_cnn_out = self.cnn_encoder(dummy_input)
        cnn_feature_dim = dummy_cnn_out.shape[-1]
        
        # 3. Enhanced LSTM classifier
        self.classifier = BidirectionalLSTMClassifier(cnn_feature_dim=cnn_feature_dim,
                                                      stat_feature_dim=stat_feature_dim,
                                                      lstm_hidden_size=64,  # Increased size
                                                      num_classes=num_classes,
                                                      dropout=0.3,
                                                      bidirectional=bidirectional,
                                                      use_attention=use_lstm_attention)

    def forward(self, window_batch, stat_batch):
        """
        window_batch: (batch_size, window_size, num_features)
        stat_batch: (batch_size, stat_feature_dim)
        """
        # 1. CNN features
        cnn_feats = self.cnn_encoder(window_batch)  # (batch, cnn_feature_dim)
        cnn_feats = cnn_feats.unsqueeze(1)  # (batch, seq_len=1, cnn_feature_dim)
        
        # 2. Stats features
        stat_feats = stat_batch.unsqueeze(1)  # (batch, seq_len=1, stat_feature_dim)
        
        # 3. Pass to classifier
        logits = self.classifier(cnn_feats, stat_feats)  # (batch, num_classes)
        return logits

class LargeCNN_LSTM_Model(nn.Module):
    """
    Large CNN-LSTM model with significantly more parameters.
    Target: ~2-3M parameters
    """
    def __init__(self, num_features, window_size, stat_feature_dim, num_classes, 
                 cnn_channels=[64, 128, 256, 512], lstm_hidden=256, lstm_layers=3):
        super(LargeCNN_LSTM_Model, self).__init__()
        
        # Large CNN encoder with more channels and layers
        self.cnn = nn.Sequential(
            # Layer 1: 43 -> 64
            nn.Conv1d(num_features, cnn_channels[0], kernel_size=7, padding=3),
            nn.BatchNorm1d(cnn_channels[0]),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),
            
            # Layer 2: 64 -> 128
            nn.Conv1d(cnn_channels[0], cnn_channels[1], kernel_size=5, padding=2),
            nn.BatchNorm1d(cnn_channels[1]),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),
            
            # Layer 3: 128 -> 256
            nn.Conv1d(cnn_channels[1], cnn_channels[2], kernel_size=5, padding=2),
            nn.BatchNorm1d(cnn_channels[2]),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),
            
            # Layer 4: 256 -> 512
            nn.Conv1d(cnn_channels[2], cnn_channels[3], kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels[3]),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),
        )
        
        # Calculate CNN output size
        with torch.no_grad():
            dummy_input = torch.zeros(1, num_features, window_size)
            cnn_out = self.cnn(dummy_input)
            cnn_out_size = cnn_out.size(1) * cnn_out.size(2)
        
        # Large LSTM with more layers and hidden units
        self.lstm = nn.LSTM(
            input_size=cnn_out_size,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )
        
        # Large dense layers
        lstm_out_size = lstm_hidden * 2  # bidirectional
        combined_size = lstm_out_size + stat_feature_dim
        
        self.classifier = nn.Sequential(
            nn.Linear(combined_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, num_classes)
        )
        
        self.flatten = nn.Flatten()
        
    def forward(self, x_windows, x_stats):
        # CNN processing
        x = x_windows.permute(0, 2, 1)  # (batch, features, time)
        cnn_out = self.cnn(x)  # (batch, channels, reduced_time)
        cnn_flat = self.flatten(cnn_out)  # (batch, features)
        
        # LSTM processing
        cnn_flat = cnn_flat.unsqueeze(1)  # (batch, 1, features)
        lstm_out, _ = self.lstm(cnn_flat)
        lstm_features = lstm_out[:, -1, :]  # Take last output
        
        # Combine with statistical features
        combined = torch.cat([lstm_features, x_stats], dim=1)
        
        # Classification
        output = self.classifier(combined)
        return output

class UltraLargeEnhanced_CNN_LSTM_Model(nn.Module):
    """
    Ultra-large enhanced model with maximum parameters.
    Target: ~5-8M parameters
    """
    def __init__(self, num_features, window_size, stat_feature_dim, num_classes):
        super(UltraLargeEnhanced_CNN_LSTM_Model, self).__init__()
        
        # Multi-scale CNN with residual connections
        self.cnn_scale1 = nn.Sequential(
            nn.Conv1d(num_features, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        
        self.cnn_scale2 = nn.Sequential(
            nn.Conv1d(num_features, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        
        self.cnn_scale3 = nn.Sequential(
            nn.Conv1d(num_features, 128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        
        # Combined multi-scale features: 128*3 = 384 channels
        self.cnn_deep = nn.Sequential(
            nn.Conv1d(384, 512, kernel_size=5, padding=2),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),
            
            nn.Conv1d(512, 768, kernel_size=5, padding=2),
            nn.BatchNorm1d(768),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),
            
            nn.Conv1d(768, 1024, kernel_size=3, padding=1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.3),
        )
        
        # Calculate CNN output size
        with torch.no_grad():
            dummy_input = torch.zeros(1, num_features, window_size)
            # Multi-scale processing
            scale1 = self.cnn_scale1(dummy_input)
            scale2 = self.cnn_scale2(dummy_input)
            scale3 = self.cnn_scale3(dummy_input)
            combined_scales = torch.cat([scale1, scale2, scale3], dim=1)
            
            # Deep processing
            deep_out = self.cnn_deep(combined_scales)
            # Feature dimension is the channel dimension, not the flattened size
            self.cnn_out_size = deep_out.size(1)  # Channel dimension
        
        # Multi-head attention with more heads - ensure divisibility
        # Adjust CNN output size to be divisible by num_heads
        num_heads = 8
        if self.cnn_out_size % num_heads != 0:
            # Pad to make it divisible
            pad_size = num_heads - (self.cnn_out_size % num_heads)
            self.cnn_projection = nn.Linear(self.cnn_out_size, self.cnn_out_size + pad_size)
            self.cnn_out_size = self.cnn_out_size + pad_size
        else:
            self.cnn_projection = nn.Identity()
            
        self.temporal_attention = TemporalAttention(self.cnn_out_size, num_heads=num_heads)
        
        # Large bidirectional LSTM with multiple layers
        self.lstm = nn.LSTM(
            input_size=self.cnn_out_size,
            hidden_size=512,
            num_layers=4,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )
        
        # Ultra-large classification head
        lstm_out_size = 512 * 2  # bidirectional
        combined_size = lstm_out_size + stat_feature_dim
        
        self.classifier = nn.Sequential(
            nn.Linear(combined_size, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            nn.Linear(1024, 768),
            nn.BatchNorm1d(768),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            nn.Linear(768, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, num_classes)
        )
        
        self.flatten = nn.Flatten()
        
    def forward(self, x_windows, x_stats):
        # Multi-scale CNN processing
        x = x_windows.permute(0, 2, 1)  # (batch, features, time)
        
        # Extract features at different scales
        scale1 = self.cnn_scale1(x)
        scale2 = self.cnn_scale2(x)
        scale3 = self.cnn_scale3(x)
        
        # Combine multi-scale features
        multi_scale = torch.cat([scale1, scale2, scale3], dim=1)
        
        # Deep CNN processing
        deep_features = self.cnn_deep(multi_scale)
        
        # Flatten for sequence processing
        batch_size = deep_features.size(0)
        seq_len = deep_features.size(2)
        features = deep_features.permute(0, 2, 1).contiguous()  # (batch, time, features)
        
        # Project features to ensure correct dimensionality
        features = features.reshape(batch_size * seq_len, -1)
        features = self.cnn_projection(features)
        features = features.reshape(batch_size, seq_len, -1)
        
        # Temporal attention
        attended = self.temporal_attention(features)
        
        # LSTM processing
        lstm_out, _ = self.lstm(attended)
        lstm_features = lstm_out[:, -1, :]  # Take last output
        
        # Combine with statistical features
        combined = torch.cat([lstm_features, x_stats], dim=1)
        
        # Classification
        output = self.classifier(combined)
        return output
