import torch
import torch.nn as nn
from .components import TimeSeriesEncoder, CategoricalEncoder, TextEncoder, MultimodalFusion, FocalLoss, DiceLoss
from .text_encoders import FlexibleTextEncoder

class MultimodalClinicalAI(nn.Module):
    """Mô hình AI multimodal cho dự đoán suy giảm lâm sàng"""
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        # Feature encoders
        self.timeseries_encoder = TimeSeriesEncoder(
            config.timeseries_input_size, 
            config.lstm_hidden_size, 
            config.feature_size
        )
        
        self.categorical_encoder = CategoricalEncoder(
            config.categorical_dims, 
            output_size=config.feature_size
        )
        
        self.text_encoder = FlexibleTextEncoder(
            encoder_type=config.text_encoder_type,
            output_size=config.feature_size
        )
        
        # Fusion module
        self.fusion = MultimodalFusion(config.feature_size, config.fusion_hidden_size)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(config.fusion_hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(64, config.num_classes)
        )
        
        # Loss functions
        self.focal_loss = FocalLoss(config.focal_loss_alpha, config.focal_loss_gamma)
        self.dice_loss = DiceLoss()
        
    def forward(self, timeseries_data, categorical_data, text_input_ids, text_attention_mask):
        # Extract features from each modality
        timeseries_features = self.timeseries_encoder(timeseries_data)
        categorical_features = self.categorical_encoder(categorical_data)
        text_features = self.text_encoder(text_input_ids, text_attention_mask)
        
        # Fuse features
        fused_features = self.fusion(timeseries_features, categorical_features, text_features)
        
        # Final prediction
        output = self.classifier(fused_features)
        
        return output
    
    def compute_loss(self, predictions, targets):
        """Tính combined loss"""
        focal_loss = self.focal_loss(predictions, targets)
        dice_loss = self.dice_loss(predictions, targets)
        
        total_loss = focal_loss + dice_loss
        
        return total_loss, {
            'focal_loss': focal_loss.item(),
            'dice_loss': dice_loss.item(),
            'total_loss': total_loss.item()
        }

class ContrastiveLearningModule(nn.Module):
    """Module cho contrastive learning"""
    def __init__(self, feature_dim, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, features_1, features_2):
        """Tính contrastive loss giữa hai sets của features"""
        # Normalize features
        features_1 = F.normalize(features_1, dim=1)
        features_2 = F.normalize(features_2, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.mm(features_1, features_2.t()) / self.temperature
        
        # Create labels (positive pairs are on diagonal)
        batch_size = features_1.size(0)
        labels = torch.arange(batch_size).to(features_1.device)
        
        # Compute contrastive loss
        loss = F.cross_entropy(similarity_matrix, labels)
        
        return loss

class MultimodalContrastiveAI(MultimodalClinicalAI):
    """Mô hình với contrastive learning"""
    def __init__(self, config):
        super().__init__(config)
        
        # Contrastive learning module
        self.contrastive_module = ContrastiveLearningModule(
            config.feature_size, 
            config.contrastive_temperature
        )
        
    def forward(self, timeseries_data, categorical_data, text_input_ids, text_attention_mask):
        # Extract features
        timeseries_features = self.timeseries_encoder(timeseries_data)
        categorical_features = self.categorical_encoder(categorical_data)
        text_features = self.text_encoder(text_input_ids, text_attention_mask)
        
        # Fuse features
        fused_features = self.fusion(timeseries_features, categorical_features, text_features)
        
        # Classification
        output = self.classifier(fused_features)
        
        return output, (timeseries_features, categorical_features, text_features)
    
    def compute_loss(self, predictions, targets, features=None):
        """Tính combined loss với contrastive learning"""
        # Classification loss
        classification_loss, loss_dict = super().compute_loss(predictions, targets)
        
        # Contrastive loss
        if features is not None:
            timeseries_features, categorical_features, text_features = features
            
            # Contrastive learning giữa các modalities
            contrastive_loss_1 = self.contrastive_module(timeseries_features, categorical_features)
            contrastive_loss_2 = self.contrastive_module(timeseries_features, text_features)
            contrastive_loss_3 = self.contrastive_module(categorical_features, text_features)
            
            contrastive_loss = (contrastive_loss_1 + contrastive_loss_2 + contrastive_loss_3) / 3
            
            total_loss = classification_loss + self.config.contrastive_weight * contrastive_loss
            
            loss_dict['contrastive_loss'] = contrastive_loss.item()
            loss_dict['total_loss'] = total_loss.item()
            
            return total_loss, loss_dict
        
        return classification_loss, loss_dict
