import torch
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from typing import List


@dataclass
class ModelConfig:
    # Model dimensions
    timeseries_input_size: int = 32  # số lượng vital signs + blood tests
    categorical_dims: Dict[str, int] = None
    text_max_length: int = 512
    feature_size: int = 256
    fusion_hidden_size: int = 128
    num_classes: int = 1
    
    # Model architecture
    lstm_hidden_size: int = 128
    lstm_num_layers: int = 2
    mlp_hidden_sizes: List[int] = None
    dropout_rate: float = 0.2
    
    # Training parameters
    batch_size: int = 32
    learning_rate: float = 1e-4
    num_epochs: int = 100
    weight_decay: float = 1e-5
    
    # Loss parameters
    focal_loss_alpha: float = 0.25
    focal_loss_gamma: float = 2.0
    contrastive_temperature: float = 0.07
    contrastive_weight: float = 1.0
    
    # Data parameters
    sequence_length: int = 24  # 24 hours of data
    prediction_window: int = 8  # predict next 8 hours
    
    # Text encoder options
    text_encoder_type: str = 'clinicalbert'
    available_encoders: List[str] = None          # list for batch testing
    encoder_comparison_mode: bool = False         # toggle grid training

    def __post_init__(self):
        if self.categorical_dims is None:
            self.categorical_dims = {
                'age_group': 5,
                'gender': 2,
                'admission_type': 3,
                'department': 10
            }
        if self.mlp_hidden_sizes is None:
            self.mlp_hidden_sizes = [256, 128, 64]
        if self.available_encoders is None:
            self.available_encoders = [
                'clinicalbert', 'biobert', 'bluebert',
                'pubmedbert', 'roberta_clin', 'deberta_med'
            ]
@dataclass
class TrainingConfig:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    early_stopping_patience: int = 10
    save_best_model: bool = True
    model_save_path: str = "checkpoints/"
    log_dir: str = "logs/"
    
    # Cross-validation
    cv_folds: int = 5
    test_size: float = 0.2
    val_size: float = 0.2

# Global config instances
model_config = ModelConfig()
training_config = TrainingConfig()
