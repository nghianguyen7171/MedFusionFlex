import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier, XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import numpy as np

class LSTMBaseline(nn.Module):
    """LSTM baseline model"""
    def __init__(self, input_size, hidden_size=128, num_layers=2, num_classes=1):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, 
            batch_first=True, dropout=0.2
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        # Only use timeseries data
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use last hidden state
        last_hidden = lstm_out[:, -1, :]
        
        # Classification
        output = self.classifier(last_hidden)
        
        return output

class CNNBaseline(nn.Module):
    """CNN baseline model"""
    def __init__(self, input_size, num_classes=1):
        super().__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_size, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        x = x.transpose(1, 2)  # (batch_size, input_size, sequence_length)
        
        # CNN feature extraction
        features = self.conv_layers(x)
        features = features.squeeze(-1)  # Remove last dimension
        
        # Classification
        output = self.classifier(features)
        
        return output

class TransformerBaseline(nn.Module):
    """Transformer baseline model"""
    def __init__(self, input_size, d_model=256, nhead=8, num_layers=6, num_classes=1):
        super().__init__()
        
        self.input_projection = nn.Linear(input_size, d_model)
        self.positional_encoding = nn.Parameter(torch.randn(1000, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        batch_size, seq_len, _ = x.shape
        
        # Project input
        x = self.input_projection(x)
        
        # Add positional encoding
        x = x + self.positional_encoding[:seq_len, :].unsqueeze(0)
        
        # Transformer encoding
        transformer_out = self.transformer(x)
        
        # Use mean pooling
        pooled = transformer_out.mean(dim=1)
        
        # Classification
        output = self.classifier(pooled)
        
        return output

class MLBaseline:
    """Machine Learning baseline models"""
    def __init__(self, model_type='rf'):
        self.model_type = model_type
        
        if model_type == 'rf':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_type == 'xgb':
            self.model = XGBClassifier(n_estimators=100, random_state=42)
        elif model_type == 'lr':
            self.model = LogisticRegression(random_state=42)
        elif model_type == 'svm':
            self.model = SVC(probability=True, random_state=42)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def fit(self, X, y):
        """Fit the model"""
        # Flatten time-series data
        if len(X.shape) == 3:
            X = X.reshape(X.shape[0], -1)
        
        self.model.fit(X, y)
    
    def predict(self, X):
        """Make predictions"""
        if len(X.shape) == 3:
            X = X.reshape(X.shape[0], -1)
        
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Make probability predictions"""
        if len(X.shape) == 3:
            X = X.reshape(X.shape[0], -1)
        
        return self.model.predict_proba(X)[:, 1]

class MEWSScore:
    """Modified Early Warning Score baseline"""
    def __init__(self):
        # MEWS thresholds cho vital signs
        self.thresholds = {
            'heart_rate': [(0, 40, 3), (40, 51, 1), (51, 100, 0), (100, 110, 1), (110, 130, 2), (130, float('inf'), 3)],
            'systolic_bp': [(0, 70, 3), (70, 81, 2), (81, 100, 1), (100, 200, 0), (200, float('inf'), 2)],
            'respiratory_rate': [(0, 9, 2), (9, 15, 0), (15, 21, 1), (21, 30, 2), (30, float('inf'), 3)],
            'temperature': [(0, 35, 2), (35, 38.5, 0), (38.5, float('inf'), 2)],
            'consciousness': [(0, 1, 0), (1, float('inf'), 3)]  # 0 = alert, 1 = confused/agitated
        }
    
    def calculate_mews(self, vital_signs):
        """Tính MEWS score"""
        score = 0
        
        for vital, ranges in self.thresholds.items():
            if vital in vital_signs:
                value = vital_signs[vital]
                for min_val, max_val, points in ranges:
                    if min_val <= value < max_val:
                        score += points
                        break
        
        return score
    
    def predict(self, X):
        """Predict using MEWS threshold"""
        # Giả sử X có các cột vital signs
        predictions = []
        
        for sample in X:
            # Lấy vital signs từ sample (giả sử là last time point)
            last_vitals = sample[-1]  # Last time point
            
            vital_dict = {
                'heart_rate': last_vitals[0],
                'systolic_bp': last_vitals[1],
                'respiratory_rate': last_vitals[2],
                'temperature': last_vitals[3],
                'consciousness': 0  # Giả sử alert
            }
            
            mews_score = self.calculate_mews(vital_dict)
            
            # Threshold: MEWS >= 5 indicates high risk
            prediction = 1 if mews_score >= 5 else 0
            predictions.append(prediction)
        
        return np.array(predictions)
    
    def predict_proba(self, X):
        """Predict probabilities using MEWS"""
        probabilities = []
        
        for sample in X:
            last_vitals = sample[-1]
            
            vital_dict = {
                'heart_rate': last_vitals[0],
                'systolic_bp': last_vitals[1],
                'respiratory_rate': last_vitals[2],
                'temperature': last_vitals[3],
                'consciousness': 0
            }
            
            mews_score = self.calculate_mews(vital_dict)
            
            # Convert score to probability (0-15 scale)
            probability = min(mews_score / 15.0, 1.0)
            probabilities.append(probability)
        
        return np.array(probabilities)
