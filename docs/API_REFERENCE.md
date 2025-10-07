# MediFusion-Flex API Reference

## Model Classes

### MultimodalClinicalAI

The main multimodal model class that integrates time-series, categorical, and text data.

```python
from models.multimodal_model import MultimodalClinicalAI

model = MultimodalClinicalAI(config)
```

#### Parameters
- `config`: ModelConfig object containing model hyperparameters

#### Methods

##### `forward(timeseries_data, categorical_data, text_input_ids, text_attention_mask)`
Forward pass through the model.

**Parameters:**
- `timeseries_data`: Tensor of shape `(batch_size, sequence_length, features)`
- `categorical_data`: Dictionary of categorical tensors
- `text_input_ids`: Tokenized text input
- `text_attention_mask`: Attention mask for text input

**Returns:**
- `output`: Model predictions

##### `compute_loss(predictions, targets)`
Compute combined loss (Focal + Dice).

**Parameters:**
- `predictions`: Model predictions
- `targets`: Ground truth labels

**Returns:**
- `total_loss`: Combined loss tensor
- `loss_dict`: Dictionary of individual loss components

### MultimodalContrastiveAI

Extended model with contrastive learning capabilities.

```python
from models.multimodal_model import MultimodalContrastiveAI

model = MultimodalContrastiveAI(config)
```

#### Additional Methods

##### `forward(timeseries_data, categorical_data, text_input_ids, text_attention_mask)`
Returns both predictions and feature representations for contrastive learning.

**Returns:**
- `output`: Model predictions
- `features`: Tuple of (timeseries_features, categorical_features, text_features)

##### `compute_loss(predictions, targets, features=None)`
Compute combined loss including contrastive learning component.

## Text Encoders

### FlexibleTextEncoder

Factory class for different text encoders.

```python
from models.text_encoders import FlexibleTextEncoder

encoder = FlexibleTextEncoder(encoder_type='clinicalbert', output_size=256)
```

#### Available Encoder Types
- `'clinicalbert'`: ClinicalBERT
- `'biobert'`: BioBERT
- `'bluebert'`: BlueBERT
- `'pubmedbert'`: PubMedBERT
- `'roberta_clin'`: Clinical RoBERTa
- `'deberta_med'`: Medical DeBERTa

## Configuration Classes

### ModelConfig

Main configuration class for model parameters.

```python
from config.config import ModelConfig

config = ModelConfig(
    timeseries_input_size=32,
    feature_size=256,
    text_encoder_type='clinicalbert',
    batch_size=32,
    learning_rate=1e-4
)
```

#### Key Parameters
- `timeseries_input_size`: Number of time-series features
- `categorical_dims`: Dictionary of categorical feature dimensions
- `text_max_length`: Maximum text sequence length
- `feature_size`: Size of feature representations
- `fusion_hidden_size`: Hidden size for fusion layer
- `text_encoder_type`: Type of text encoder to use
- `focal_loss_alpha`: Alpha parameter for Focal Loss
- `focal_loss_gamma`: Gamma parameter for Focal Loss
- `contrastive_temperature`: Temperature for contrastive learning
- `contrastive_weight`: Weight for contrastive loss

### TrainingConfig

Configuration for training parameters.

```python
from config.config import TrainingConfig

training_config = TrainingConfig(
    device='cuda',
    num_epochs=100,
    early_stopping_patience=10
)
```

## Utility Classes

### ModelEvaluator

Comprehensive evaluation metrics for clinical prediction.

```python
from utils.evaluation import ModelEvaluator

evaluator = ModelEvaluator()
metrics = evaluator.evaluate(y_true, y_pred, y_prob)
```

#### Methods

##### `evaluate(y_true, y_pred, y_prob)`
Compute comprehensive evaluation metrics.

**Parameters:**
- `y_true`: Ground truth labels
- `y_pred`: Predicted labels
- `y_prob`: Prediction probabilities

**Returns:**
- Dictionary containing ROC AUC, PR AUC, F1 Score, Sensitivity, Specificity, etc.

### ResultVisualizer

Visualization utilities for results and comparisons.

```python
from utils.visualization import ResultVisualizer

visualizer = ResultVisualizer()
visualizer.create_interactive_dashboard(results)
```

## Data Loading

### create_data_loaders

Create PyTorch data loaders for training, validation, and testing.

```python
from utils.data_loader import create_data_loaders

train_loader, val_loader, test_loader = create_data_loaders(
    train_dataset, val_dataset, test_dataset, config
)
```

### ClinicalDataset

Custom dataset class for clinical data.

```python
from utils.preprocessing import ClinicalDataset

dataset = ClinicalDataset(
    timeseries_data,
    categorical_data,
    text_data,
    labels
)
```

## Training Functions

### train_model

Main training function with early stopping and logging.

```python
from train import train_model

model, history = train_model(
    model, train_loader, val_loader, config, device, model_name
)
```

### train_single_encoder

Train model with a specific text encoder.

```python
from train import train_single_encoder

result = train_single_encoder(
    encoder_type, train_loader, val_loader, test_loader, config, device
)
```

## Loss Functions

### FocalLoss

Focal loss for handling class imbalance.

```python
from models.components import FocalLoss

focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
loss = focal_loss(predictions, targets)
```

### DiceLoss

Dice loss for segmentation-like tasks.

```python
from models.components import DiceLoss

dice_loss = DiceLoss()
loss = dice_loss(predictions, targets)
```

## Example Usage

### Basic Training

```python
import torch
from config.config import model_config, training_config
from models.multimodal_model import MultimodalContrastiveAI
from utils.data_loader import create_data_loaders

# Create model
model = MultimodalContrastiveAI(model_config)

# Create data loaders (assuming datasets are already prepared)
train_loader, val_loader, test_loader = create_data_loaders(
    train_dataset, val_dataset, test_dataset, model_config
)

# Training loop
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

for epoch in range(100):
    model.train()
    for batch in train_loader:
        # Move to device
        timeseries = batch['timeseries'].to(device)
        categorical = {k: v.to(device) for k, v in batch['categorical'].items()}
        text_input_ids = batch['text_input_ids'].to(device)
        text_attention_mask = batch['text_attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs, features = model(timeseries, categorical, text_input_ids, text_attention_mask)
        loss, loss_dict = model.compute_loss(outputs, labels, features)
        
        # Backward pass
        loss.backward()
        optimizer.step()
```

### Model Evaluation

```python
from utils.evaluation import ModelEvaluator, evaluate_model_on_loader

# Create evaluator
evaluator = ModelEvaluator()

# Evaluate on test set
test_metrics = evaluate_model_on_loader(model, test_loader, device, evaluator)

# Print results
for metric, value in test_metrics.items():
    if metric != 'confusion_matrix':
        print(f"{metric}: {value:.4f}")
```
