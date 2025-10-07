#!/usr/bin/env python3
"""
Training script for Multimodal Clinical AI System
Supports multiple text encoders and comparison mode
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import sys
import json
import argparse
import logging
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Sklearn imports
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split

# Project imports
from config.config import model_config, training_config
from config.encoder_configs import ENCODER_CONFIGS
from models.multimodal_model import MultimodalClinicalAI, MultimodalContrastiveAI
from models.baseline_models import LSTMBaseline, CNNBaseline, TransformerBaseline
from models.text_encoders import FlexibleTextEncoder
from utils.data_loader import create_data_loaders, collate_fn
from utils.preprocessing import load_and_preprocess_data, ClinicalDataPreprocessor
from utils.evaluation import ModelEvaluator, evaluate_model_on_loader
from utils.visualization import ResultVisualizer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def set_seed(seed: int = 42):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Set random seed to {seed}")

class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience: int = 10, min_delta: float = 0.001, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, val_score: float) -> bool:
        if self.best_score is None:
            self.best_score = val_score
        elif self._is_better(val_score):
            self.best_score = val_score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                logger.info(f"Early stopping triggered after {self.counter} epochs")
        
        return self.early_stop
    
    def _is_better(self, score: float) -> bool:
        if self.mode == 'min':
            return score < self.best_score - self.min_delta
        else:
            return score > self.best_score + self.min_delta

class TrainingMetrics:
    """Track training metrics"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.train_loss = []
        self.train_acc = []
        self.train_f1 = []
        self.train_auc = []
        self.val_loss = []
        self.val_acc = []
        self.val_f1 = []
        self.val_auc = []
        self.learning_rates = []
    
    def update(self, train_metrics: dict, val_metrics: dict, lr: float):
        self.train_loss.append(train_metrics['loss'])
        self.train_acc.append(train_metrics['accuracy'])
        self.train_f1.append(train_metrics['f1'])
        self.train_auc.append(train_metrics.get('auc', 0))
        
        self.val_loss.append(val_metrics['loss'])
        self.val_acc.append(val_metrics['accuracy'])
        self.val_f1.append(val_metrics['f1'])
        self.val_auc.append(val_metrics.get('auc', 0))
        
        self.learning_rates.append(lr)
    
    def get_history(self) -> dict:
        return {
            'train_loss': self.train_loss,
            'train_acc': self.train_acc,
            'train_f1': self.train_f1,
            'train_auc': self.train_auc,
            'val_loss': self.val_loss,
            'val_acc': self.val_acc,
            'val_f1': self.val_f1,
            'val_auc': self.val_auc,
            'learning_rates': self.learning_rates
        }

def train_epoch(model, data_loader, optimizer, device, epoch: int, writer=None):
    """Train model for one epoch"""
    model.train()
    total_loss = 0
    all_predictions = []
    all_probabilities = []
    all_labels = []
    
    pbar = tqdm(data_loader, desc=f'Training Epoch {epoch}', leave=False)
    
    for batch_idx, batch in enumerate(pbar):
        # Move batch to device
        timeseries = batch['timeseries'].to(device)
        categorical = {k: v.to(device) for k, v in batch['categorical'].items()}
        text_input_ids = batch['text_input_ids'].to(device)
        text_attention_mask = batch['text_attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        
        try:
            if isinstance(model, MultimodalContrastiveAI):
                outputs, features = model(timeseries, categorical, text_input_ids, text_attention_mask)
                loss, loss_dict = model.compute_loss(outputs, labels, features)
            else:
                outputs = model(timeseries, categorical, text_input_ids, text_attention_mask)
                loss, loss_dict = model.compute_loss(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            
            # Get predictions and probabilities
            with torch.no_grad():
                probabilities = torch.sigmoid(outputs).cpu().numpy()
                predictions = (probabilities > 0.5).astype(int)
                
                all_predictions.extend(predictions.flatten())
                all_probabilities.extend(probabilities.flatten())
                all_labels.extend(labels.cpu().numpy().flatten())
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Focal': f'{loss_dict.get("focal_loss", 0):.4f}',
                'Dice': f'{loss_dict.get("dice_loss", 0):.4f}'
            })
            
            # Log to tensorboard
            if writer and batch_idx % 10 == 0:
                global_step = epoch * len(data_loader) + batch_idx
                writer.add_scalar('Batch/Loss', loss.item(), global_step)
                for loss_name, loss_value in loss_dict.items():
                    writer.add_scalar(f'Batch/{loss_name}', loss_value, global_step)
        
        except Exception as e:
            logger.error(f"Error in batch {batch_idx}: {str(e)}")
            continue
    
    # Calculate epoch metrics
    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions, average='binary', zero_division=0)
    
    try:
        auc = roc_auc_score(all_labels, all_probabilities)
    except:
        auc = 0.0
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'f1': f1,
        'auc': auc
    }

def validate_epoch(model, data_loader, device, epoch: int):
    """Validate model for one epoch"""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_probabilities = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(data_loader, desc=f'Validation Epoch {epoch}', leave=False)
        
        for batch in pbar:
            # Move batch to device
            timeseries = batch['timeseries'].to(device)
            categorical = {k: v.to(device) for k, v in batch['categorical'].items()}
            text_input_ids = batch['text_input_ids'].to(device)
            text_attention_mask = batch['text_attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            try:
                # Forward pass
                if isinstance(model, MultimodalContrastiveAI):
                    outputs, features = model(timeseries, categorical, text_input_ids, text_attention_mask)
                    loss, loss_dict = model.compute_loss(outputs, labels, features)
                else:
                    outputs = model(timeseries, categorical, text_input_ids, text_attention_mask)
                    loss, loss_dict = model.compute_loss(outputs, labels)
                
                total_loss += loss.item()
                
                # Get predictions and probabilities
                probabilities = torch.sigmoid(outputs).cpu().numpy()
                predictions = (probabilities > 0.5).astype(int)
                
                all_predictions.extend(predictions.flatten())
                all_probabilities.extend(probabilities.flatten())
                all_labels.extend(labels.cpu().numpy().flatten())
                
                # Update progress bar
                pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
                
            except Exception as e:
                logger.error(f"Error in validation batch: {str(e)}")
                continue
    
    # Calculate validation metrics
    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions, average='binary', zero_division=0)
    
    try:
        auc = roc_auc_score(all_labels, all_probabilities)
        pr_auc = average_precision_score(all_labels, all_probabilities)
    except:
        auc = 0.0
        pr_auc = 0.0
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'f1': f1,
        'auc': auc,
        'pr_auc': pr_auc
    }

def train_model(model, train_loader, val_loader, config, device, model_name: str = "model"):
    """Main training loop"""
    logger.info(f"Starting training for {model_name}")
    
    # Setup optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5, verbose=True
    )
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=config.early_stopping_patience,
        mode='max'  # maximize F1 score
    )
    
    # Tensorboard writer
    log_dir = os.path.join(config.log_dir, model_name)
    writer = SummaryWriter(log_dir=log_dir)
    
    # Training metrics
    metrics = TrainingMetrics()
    
    best_f1 = 0.0
    best_model_state = None
    
    logger.info(f"Training for {config.num_epochs} epochs")
    
    for epoch in range(config.num_epochs):
        logger.info(f"\n=== Epoch {epoch + 1}/{config.num_epochs} ===")
        
        # Training
        train_metrics = train_epoch(model, train_loader, optimizer, device, epoch, writer)
        
        # Validation
        val_metrics = validate_epoch(model, val_loader, device, epoch)
        
        # Update metrics
        current_lr = optimizer.param_groups[0]['lr']
        metrics.update(train_metrics, val_metrics, current_lr)
        
        # Log to tensorboard
        writer.add_scalar('Epoch/Train_Loss', train_metrics['loss'], epoch)
        writer.add_scalar('Epoch/Val_Loss', val_metrics['loss'], epoch)
        writer.add_scalar('Epoch/Train_Accuracy', train_metrics['accuracy'], epoch)
        writer.add_scalar('Epoch/Val_Accuracy', val_metrics['accuracy'], epoch)
        writer.add_scalar('Epoch/Train_F1', train_metrics['f1'], epoch)
        writer.add_scalar('Epoch/Val_F1', val_metrics['f1'], epoch)
        writer.add_scalar('Epoch/Val_AUC', val_metrics['auc'], epoch)
        writer.add_scalar('Epoch/Learning_Rate', current_lr, epoch)
        
        # Print progress
        logger.info(f"Train - Loss: {train_metrics['loss']:.4f}, "
                   f"Acc: {train_metrics['accuracy']:.4f}, "
                   f"F1: {train_metrics['f1']:.4f}, "
                   f"AUC: {train_metrics['auc']:.4f}")
        
        logger.info(f"Val   - Loss: {val_metrics['loss']:.4f}, "
                   f"Acc: {val_metrics['accuracy']:.4f}, "
                   f"F1: {val_metrics['f1']:.4f}, "
                   f"AUC: {val_metrics['auc']:.4f}")
        
        # Save best model
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            best_model_state = model.state_dict().copy()
            
            if config.save_best_model:
                save_path = os.path.join(config.model_save_path, f'best_{model_name}.pth')
                torch.save(best_model_state, save_path)
                logger.info(f"Saved best model to {save_path}")
        
        # Scheduler step
        scheduler.step(val_metrics['loss'])
        
        # Early stopping
        if early_stopping(val_metrics['f1']):
            logger.info(f"Early stopping triggered at epoch {epoch + 1}")
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logger.info(f"Loaded best model with F1 score: {best_f1:.4f}")
    
    writer.close()
    
    return model, metrics.get_history()

def train_single_encoder(encoder_type: str, train_loader, val_loader, test_loader, config, device):
    """Train model with a single encoder"""
    logger.info(f"Training with encoder: {encoder_type}")
    
    # Update config
    config.text_encoder_type = encoder_type
    
    # Create model
    model = MultimodalContrastiveAI(config).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Train model
    model, history = train_model(
        model, train_loader, val_loader, training_config, device, encoder_type
    )
    
    # Evaluate on test set
    evaluator = ModelEvaluator()
    test_metrics = evaluate_model_on_loader(model, test_loader, device, evaluator)
    
    return {
        'model': model,
        'history': history,
        'test_metrics': test_metrics,
        'encoder_type': encoder_type
    }

def train_with_multiple_encoders(train_loader, val_loader, test_loader, config, device):
    """Train and compare multiple encoders"""
    logger.info("Starting multi-encoder comparison")
    
    results = {}
    comparison_metrics = []
    
    for encoder_type in config.available_encoders:
        logger.info(f"\n{'='*50}")
        logger.info(f"Training with encoder: {encoder_type}")
        logger.info(f"{'='*50}")
        
        try:
            result = train_single_encoder(
                encoder_type, train_loader, val_loader, test_loader, config, device
            )
            
            results[encoder_type] = result
            
            # Extract key metrics for comparison
            test_metrics = result['test_metrics']
            comparison_metrics.append({
                'encoder': encoder_type,
                'roc_auc': test_metrics.get('roc_auc', 0),
                'pr_auc': test_metrics.get('pr_auc', 0),
                'f1_score': test_metrics.get('f1_score', 0),
                'accuracy': test_metrics.get('accuracy', 0),
                'sensitivity': test_metrics.get('sensitivity', 0),
                'specificity': test_metrics.get('specificity', 0),
                'false_alarm_rate': test_metrics.get('false_alarm_rate', 0),
                'late_alarm_rate': test_metrics.get('late_alarm_rate', 0)
            })
            
            logger.info(f"Completed training with {encoder_type}")
            logger.info(f"Test F1: {test_metrics.get('f1_score', 0):.4f}, "
                       f"AUC: {test_metrics.get('roc_auc', 0):.4f}")
            
        except Exception as e:
            logger.error(f"Error training with {encoder_type}: {str(e)}")
            continue
    
    # Save comparison results
    comparison_df = pd.DataFrame(comparison_metrics)
    comparison_df = comparison_df.sort_values('f1_score', ascending=False)
    
    # Save results
    results_dir = Path('experiments/encoder_results')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    comparison_df.to_csv(results_dir / 'encoder_comparison.csv', index=False)
    
    with open(results_dir / 'full_results.json', 'w') as f:
        # Convert results to JSON-serializable format
        json_results = {}
        for encoder, result in results.items():
            json_results[encoder] = {
                'test_metrics': result['test_metrics'],
                'encoder_type': result['encoder_type']
            }
        json.dump(json_results, f, indent=2)
    
    logger.info(f"\n{'='*50}")
    logger.info("ENCODER COMPARISON RESULTS")
    logger.info(f"{'='*50}")
    print(comparison_df.round(4))
    
    # Create visualizations
    visualizer = ResultVisualizer()
    visualizer.create_interactive_dashboard(json_results)
    
    return results

def create_synthetic_data(num_samples: int = 1000):
    """Create synthetic data for testing"""
    logger.info(f"Creating synthetic data with {num_samples} samples")
    
    np.random.seed(42)
    
    # Time-series data (24 time points, 32 features)
    timeseries_data = np.random.randn(num_samples, 24, 32)
    
    # Categorical data
    categorical_data = {
        'age_group': np.random.randint(0, 5, num_samples),
        'gender': np.random.randint(0, 2, num_samples),
        'admission_type': np.random.randint(0, 3, num_samples),
        'department': np.random.randint(0, 10, num_samples)
    }
    
    # Text data (clinical notes)
    clinical_notes = [
        f"Patient {i} presents with condition {np.random.choice(['stable', 'deteriorating', 'critical'])}"
        for i in range(num_samples)
    ]
    
    # Labels (imbalanced: 5% positive)
    labels = np.random.choice([0, 1], size=num_samples, p=[0.95, 0.05])
    
    # Create datasets
    from utils.preprocessing import ClinicalDataset
    
    # Split data
    train_size = int(0.6 * num_samples)
    val_size = int(0.2 * num_samples)
    
    train_indices = list(range(train_size))
    val_indices = list(range(train_size, train_size + val_size))
    test_indices = list(range(train_size + val_size, num_samples))
    
    def create_dataset(indices):
        return ClinicalDataset(
            timeseries_data[indices],
            {k: v[indices] for k, v in categorical_data.items()},
            {'input_ids': torch.randint(0, 1000, (len(indices), 512)),
             'attention_mask': torch.ones(len(indices), 512)},
            labels[indices]
        )
    
    train_dataset = create_dataset(train_indices)
    val_dataset = create_dataset(val_indices)
    test_dataset = create_dataset(test_indices)
    
    return train_dataset, val_dataset, test_dataset

def setup_directories(config):
    """Setup necessary directories"""
    directories = [
        config.model_save_path,
        config.log_dir,
        'experiments/encoder_results',
        'experiments/comparison_reports',
        'experiments/trained_models'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train Multimodal Clinical AI')
    
    parser.add_argument('--encoder', type=str, default='clinicalbert',
                       help='Text encoder to use')
    parser.add_argument('--compare-all', action='store_true',
                       help='Compare all available encoders')
    parser.add_argument('--data-path', type=str, default='data/clinical_data.csv',
                       help='Path to clinical data')
    parser.add_argument('--synthetic-data', action='store_true',
                       help='Use synthetic data for testing')
    parser.add_argument('--num-epochs', type=int, default=None,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=None,
                       help='Learning rate')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda/cpu/auto)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    
    return parser.parse_args()

def main():
    """Main training function"""
    args = parse_arguments()
    
    # Set logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Set random seed
    set_seed(args.seed)
    
    # Update configs based on arguments
    if args.num_epochs:
        training_config.num_epochs = args.num_epochs
    if args.batch_size:
        model_config.batch_size = args.batch_size
    if args.learning_rate:
        training_config.learning_rate = args.learning_rate
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    training_config.device = device
    logger.info(f"Using device: {device}")
    
    # Setup directories
    setup_directories(training_config)
    
    # Load or create data
    if args.synthetic_data:
        logger.info("Using synthetic data")
        train_dataset, val_dataset, test_dataset = create_synthetic_data()
    else:
        logger.info(f"Loading data from {args.data_path}")
        try:
            train_dataset, val_dataset, test_dataset = load_and_preprocess_data(
                args.data_path, model_config
            )
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            logger.info("Falling back to synthetic data")
            train_dataset, val_dataset, test_dataset = create_synthetic_data()
    
    # Create data loaders
    logger.info("Creating data loaders")
    train_loader, val_loader, test_loader = create_data_loaders(
        train_dataset, val_dataset, test_dataset, model_config
    )
    
    logger.info(f"Dataset sizes - Train: {len(train_dataset)}, "
               f"Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Training mode
    if args.compare_all:
        logger.info("Running multi-encoder comparison")
        model_config.encoder_comparison_mode = True
        results = train_with_multiple_encoders(
            train_loader, val_loader, test_loader, model_config, device
        )
    else:
        logger.info(f"Training with single encoder: {args.encoder}")
        model_config.text_encoder_type = args.encoder
        result = train_single_encoder(
            args.encoder, train_loader, val_loader, test_loader, model_config, device
        )
        
        logger.info("\nFinal Results:")
        logger.info(f"Encoder: {result['encoder_type']}")
        for metric, value in result['test_metrics'].items():
            if metric != 'confusion_matrix':
                logger.info(f"{metric}: {value:.4f}")
    
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        sys.exit(1)
