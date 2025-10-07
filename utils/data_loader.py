import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from collections import Counter
import numpy as np

def create_data_loaders(train_dataset, val_dataset, test_dataset, config):
    """Tạo data loaders với class balancing"""
    
    # Tính class weights cho imbalanced data
    train_labels = [train_dataset[i]['labels'].item() for i in range(len(train_dataset))]
    class_counts = Counter(train_labels)
    
    # Tạo weighted sampler
    weights = [1.0 / class_counts[label] for label in train_labels]
    sampler = WeightedRandomSampler(weights, len(weights))
    
    # Tạo data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

def collate_fn(batch):
    """Custom collate function for variable length sequences"""
    timeseries = torch.stack([item['timeseries'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    
    # Categorical data
    categorical = {}
    for key in batch[0]['categorical'].keys():
        categorical[key] = torch.stack([item['categorical'][key] for item in batch])
    
    # Text data
    text_input_ids = torch.stack([item['text_input_ids'] for item in batch])
    text_attention_mask = torch.stack([item['text_attention_mask'] for item in batch])
    
    return {
        'timeseries': timeseries,
        'categorical': categorical,
        'text_input_ids': text_input_ids,
        'text_attention_mask': text_attention_mask,
        'labels': labels
    }
