import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple

class ModelEvaluator:
    def __init__(self):
        self.metrics = {}
    
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray = None) -> Dict:
        """Tính toán các metrics đánh giá"""
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='binary'),
            'recall': recall_score(y_true, y_pred, average='binary'),
            'f1_score': f1_score(y_true, y_pred, average='binary'),
            'specificity': self._calculate_specificity(y_true, y_pred),
            'sensitivity': recall_score(y_true, y_pred, average='binary'),
        }
        
        if y_prob is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
            metrics['pr_auc'] = average_precision_score(y_true, y_prob)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm
        
        # False alarm rate và Late alarm rate
        tn, fp, fn, tp = cm.ravel()
        metrics['false_alarm_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
        metrics['late_alarm_rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        return metrics
    
    def _calculate_specificity(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Tính specificity"""
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        return tn / (tn + fp) if (tn + fp) > 0 else 0
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                            title: str = "Confusion Matrix"):
        """Vẽ confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['No Event', 'Event'],
                   yticklabels=['No Event', 'Event'])
        plt.title(title)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()
    
    def plot_roc_curve(self, y_true: np.ndarray, y_prob: np.ndarray, 
                      title: str = "ROC Curve"):
        """Vẽ ROC curve"""
        from sklearn.metrics import roc_curve
        
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def plot_precision_recall_curve(self, y_true: np.ndarray, y_prob: np.ndarray,
                                   title: str = "Precision-Recall Curve"):
        """Vẽ Precision-Recall curve"""
        from sklearn.metrics import precision_recall_curve
        
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = average_precision_score(y_true, y_prob)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=f'PR Curve (AUC = {pr_auc:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def compare_models(self, results: Dict[str, Dict]) -> None:
        """So sánh kết quả của các models"""
        import pandas as pd
        
        # Tạo DataFrame từ results
        df = pd.DataFrame(results).T
        
        # Sắp xếp theo F1-score
        df = df.sort_values('f1_score', ascending=False)
        
        print("Model Comparison:")
        print("=" * 80)
        print(df.round(4))
        
        # Vẽ biểu đồ so sánh
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # ROC AUC
        axes[0, 0].bar(df.index, df['roc_auc'])
        axes[0, 0].set_title('ROC AUC')
        axes[0, 0].set_ylim(0, 1)
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # PR AUC
        axes[0, 1].bar(df.index, df['pr_auc'])
        axes[0, 1].set_title('PR AUC')
        axes[0, 1].set_ylim(0, 1)
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # F1 Score
        axes[1, 0].bar(df.index, df['f1_score'])
        axes[1, 0].set_title('F1 Score')
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Late Alarm Rate
        axes[1, 1].bar(df.index, df['late_alarm_rate'])
        axes[1, 1].set_title('Late Alarm Rate (Lower is Better)')
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()

def evaluate_model_on_loader(model, data_loader, device, evaluator):
    """Evaluate model trên data loader"""
    model.eval()
    
    all_predictions = []
    all_probabilities = []
    all_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            # Move batch to device
            timeseries = batch['timeseries'].to(device)
            categorical = {k: v.to(device) for k, v in batch['categorical'].items()}
            text_input_ids = batch['text_input_ids'].to(device)
            text_attention_mask = batch['text_attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            if hasattr(model, 'forward') and len(model.forward.__code__.co_varnames) > 5:
                # Multimodal model
                outputs = model(timeseries, categorical, text_input_ids, text_attention_mask)
            else:
                # Baseline model (only timeseries)
                outputs = model(timeseries)
            
            # Get predictions and probabilities
            probabilities = torch.sigmoid(outputs).cpu().numpy()
            predictions = (probabilities > 0.5).astype(int)
            
            all_predictions.extend(predictions.flatten())
            all_probabilities.extend(probabilities.flatten())
            all_labels.extend(labels.cpu().numpy().flatten())
    
    # Calculate metrics
    metrics = evaluator.evaluate(
        np.array(all_labels),
        np.array(all_predictions),
        np.array(all_probabilities)
    )
    
    return metrics
