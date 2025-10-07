#!/usr/bin/env python3
"""
Basic usage example for MediFusion-Flex
This example demonstrates how to train and evaluate the model with synthetic data.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from config.config import model_config, training_config
from models.multimodal_model import MultimodalContrastiveAI
from utils.data_loader import create_data_loaders
from utils.preprocessing import ClinicalDataset
from utils.evaluation import ModelEvaluator, evaluate_model_on_loader

def create_synthetic_clinical_data(num_samples=1000):
    """Create synthetic clinical data for demonstration"""
    print(f"Creating synthetic clinical data with {num_samples} samples...")
    
    np.random.seed(42)
    
    # Time-series data (24 hours, 32 features: vital signs + lab results)
    timeseries_data = np.random.randn(num_samples, 24, 32)
    
    # Categorical data
    categorical_data = {
        'age_group': np.random.randint(0, 5, num_samples),
        'gender': np.random.randint(0, 2, num_samples),
        'admission_type': np.random.randint(0, 3, num_samples),
        'department': np.random.randint(0, 10, num_samples)
    }
    
    # Simulate clinical notes (tokenized)
    text_data = {
        'input_ids': torch.randint(0, 1000, (num_samples, 512)),
        'attention_mask': torch.ones(num_samples, 512)
    }
    
    # Labels (5% positive cases - typical for clinical deterioration)
    labels = np.random.choice([0, 1], size=num_samples, p=[0.95, 0.05])
    
    return timeseries_data, categorical_data, text_data, labels

def create_datasets(timeseries_data, categorical_data, text_data, labels):
    """Create train/validation/test datasets"""
    num_samples = len(timeseries_data)
    
    # Split data (60% train, 20% val, 20% test)
    train_size = int(0.6 * num_samples)
    val_size = int(0.2 * num_samples)
    
    train_indices = list(range(train_size))
    val_indices = list(range(train_size, train_size + val_size))
    test_indices = list(range(train_size + val_size, num_samples))
    
    def create_dataset(indices):
        return ClinicalDataset(
            timeseries_data[indices],
            {k: v[indices] for k, v in categorical_data.items()},
            {k: v[indices] for k, v in text_data.items()},
            labels[indices]
        )
    
    train_dataset = create_dataset(train_indices)
    val_dataset = create_dataset(val_indices)
    test_dataset = create_dataset(test_indices)
    
    print(f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    return train_dataset, val_dataset, test_dataset

def quick_train_example():
    """Quick training example with synthetic data"""
    print("=" * 60)
    print("MediFusion-Flex Basic Usage Example")
    print("=" * 60)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create synthetic data
    timeseries_data, categorical_data, text_data, labels = create_synthetic_clinical_data(1000)
    
    # Create datasets
    train_dataset, val_dataset, test_dataset = create_datasets(
        timeseries_data, categorical_data, text_data, labels
    )
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        train_dataset, val_dataset, test_dataset, model_config
    )
    
    # Create model
    print("Creating MediFusion-Flex model...")
    model = MultimodalContrastiveAI(model_config).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Quick training (just a few epochs for demo)
    print("\nStarting quick training (5 epochs)...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    model.train()
    for epoch in range(5):
        total_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device
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
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}')
    
    # Evaluation
    print("\nEvaluating model...")
    model.eval()
    evaluator = ModelEvaluator()
    test_metrics = evaluate_model_on_loader(model, test_loader, device, evaluator)
    
    print("\nTest Results:")
    print("-" * 40)
    for metric, value in test_metrics.items():
        if metric != 'confusion_matrix':
            print(f"{metric}: {value:.4f}")
    
    print("\nBasic usage example completed successfully!")
    return model, test_metrics

if __name__ == "__main__":
    model, metrics = quick_train_example()
