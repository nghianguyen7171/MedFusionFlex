import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

class TimeSeriesEncoder(nn.Module):
    """CNN-LSTM encoder cho time-series data"""
    def __init__(self, input_size, hidden_size=128, output_size=256):
        super().__init__()
        
        # 1D CNN layers
        self.conv1 = nn.Conv1d(input_size, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        
        # LSTM layers
        self.lstm = nn.LSTM(128, hidden_size, batch_first=True, bidirectional=True)
        
        # Projection layer
        self.projection = nn.Linear(hidden_size * 2, output_size)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        x = x.transpose(1, 2)  # (batch_size, input_size, sequence_length)
        
        # CNN feature extraction
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        
        # Transpose back for LSTM
        x = x.transpose(1, 2)
        
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use last hidden state
        last_hidden = lstm_out[:, -1, :]
        
        # Project to output size
        output = self.projection(last_hidden)
        output = self.dropout(output)
        
        return output

class CategoricalEncoder(nn.Module):
    """Embedding encoder cho categorical data"""
    def __init__(self, categorical_dims, embedding_dim=32, output_size=128):
        super().__init__()
        
        # Embedding layers
        self.embeddings = nn.ModuleDict({
            name: nn.Embedding(num_classes, embedding_dim)
            for name, num_classes in categorical_dims.items()
        })
        
        # MLP layers
        total_embedding_dim = len(categorical_dims) * embedding_dim
        self.mlp = nn.Sequential(
            nn.Linear(total_embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, output_size),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
    def forward(self, categorical_inputs):
        # Get embeddings
        embeddings = []
        for name, input_tensor in categorical_inputs.items():
            emb = self.embeddings[name](input_tensor)
            embeddings.append(emb)
        
        # Concatenate embeddings
        concatenated = torch.cat(embeddings, dim=1)
        
        # Pass through MLP
        output = self.mlp(concatenated)
        
        return output

class TextEncoder(nn.Module):
    """BERT encoder cho clinical notes"""
    def __init__(self, model_name='bert-base-uncased', output_size=256):
        super().__init__()
        
        # Load pre-trained BERT
        self.bert = AutoModel.from_pretrained(model_name)
        
        # Freeze BERT parameters (optional)
        for param in self.bert.parameters():
            param.requires_grad = False
        
        # Projection layer
        self.projection = nn.Linear(self.bert.config.hidden_size, output_size)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, input_ids, attention_mask):
        # Get BERT outputs
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use [CLS] token
        cls_output = outputs.last_hidden_state[:, 0, :]
        
        # Project to output size
        output = self.projection(cls_output)
        output = self.dropout(output)
        
        return output

class MultimodalFusion(nn.Module):
    """Fusion module với attention mechanism"""
    def __init__(self, input_size=256, hidden_size=128):
        super().__init__()
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(input_size, num_heads=8, batch_first=True)
        
        # Projection layers
        self.projection = nn.Linear(input_size * 3, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, timeseries_features, categorical_features, text_features):
        # Stack features for attention
        stacked_features = torch.stack([timeseries_features, categorical_features, text_features], dim=1)
        
        # Apply self-attention
        attended_features, _ = self.attention(stacked_features, stacked_features, stacked_features)
        
        # Flatten and project
        flattened = attended_features.flatten(start_dim=1)
        
        # Project to fusion space
        fused = self.projection(flattened)
        fused = self.layer_norm(fused)
        fused = F.relu(fused)
        fused = self.dropout(fused)
        
        return fused

class FocalLoss(nn.Module):
    """Focal Loss cho imbalanced data"""
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

class DiceLoss(nn.Module):
    """Dice Loss cho imbalanced data"""
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
        
    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        
        # Flatten
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        
        return 1 - dice
