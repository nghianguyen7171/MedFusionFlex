#!/usr/bin/env python3
"""
Text Encoder Comparison Example
This example demonstrates how to compare different text encoders.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pandas as pd
from config.config import model_config, training_config
from models.multimodal_model import MultimodalContrastiveAI
from utils.data_loader import create_data_loaders
from utils.preprocessing import ClinicalDataset
from utils.evaluation import ModelEvaluator, evaluate_model_on_loader

def compare_text_encoders():
    """Compare different text encoders"""
    print("=" * 60)
    print("MediFusion-Flex Text Encoder Comparison")
    print("=" * 60)
    
    # Available encoders
    encoders = ['clinicalbert', 'biobert', 'bluebert', 'pubmedbert']
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create synthetic data (smaller for quick comparison)
    from examples.basic_usage import create_synthetic_clinical_data, create_datasets
    
    timeseries_data, categorical_data, text_data, labels = create_synthetic_clinical_data(500)
    train_dataset, val_dataset, test_dataset = create_datasets(
        timeseries_data, categorical_data, text_data, labels
    )
    
    train_loader, val_loader, test_loader = create_data_loaders(
        train_dataset, val_dataset, test_dataset, model_config
    )
    
    results = []
    
    for encoder_type in encoders:
        print(f"\nTraining with {encoder_type}...")
        
        # Update config
        model_config.text_encoder_type = encoder_type
        
        try:
            # Create model
            model = MultimodalContrastiveAI(model_config).to(device)
            
            # Quick training (3 epochs for demo)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            
            model.train()
            for epoch in range(3):
                for batch in train_loader:
                    timeseries = batch['timeseries'].to(device)
                    categorical = {k: v.to(device) for k, v in batch['categorical'].items()}
                    text_input_ids = batch['text_input_ids'].to(device)
                    text_attention_mask = batch['text_attention_mask'].to(device)
                    labels = batch['labels'].to(device)
                    
                    optimizer.zero_grad()
                    outputs, features = model(timeseries, categorical, text_input_ids, text_attention_mask)
                    loss, loss_dict = model.compute_loss(outputs, labels, features)
                    loss.backward()
                    optimizer.step()
            
            # Evaluate
            model.eval()
            evaluator = ModelEvaluator()
            test_metrics = evaluate_model_on_loader(model, test_loader, device, evaluator)
            
            # Store results
            result = {
                'encoder': encoder_type,
                'roc_auc': test_metrics.get('roc_auc', 0),
                'pr_auc': test_metrics.get('pr_auc', 0),
                'f1_score': test_metrics.get('f1_score', 0),
                'accuracy': test_metrics.get('accuracy', 0)
            }
            results.append(result)
            
            print(f"{encoder_type} - ROC AUC: {result['roc_auc']:.4f}, F1: {result['f1_score']:.4f}")
            
        except Exception as e:
            print(f"Error with {encoder_type}: {str(e)}")
            continue
    
    # Display comparison results
    print("\n" + "=" * 60)
    print("ENCODER COMPARISON RESULTS")
    print("=" * 60)
    
    if results:
        df = pd.DataFrame(results)
        df = df.sort_values('roc_auc', ascending=False)
        print(df.round(4))
        
        best_encoder = df.iloc[0]['encoder']
        print(f"\nBest performing encoder: {best_encoder}")
    else:
        print("No results obtained")

if __name__ == "__main__":
    compare_text_encoders()
