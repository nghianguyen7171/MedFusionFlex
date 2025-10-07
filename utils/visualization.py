import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from typing import Dict, List

class ResultVisualizer:
    def __init__(self):
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def plot_training_history(self, history: Dict[str, List], title: str = "Training History"):
        """Vẽ quá trình training"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        axes[0, 0].plot(history['train_loss'], label='Train Loss')
        axes[0, 0].plot(history['val_loss'], label='Validation Loss')
        axes[0, 0].set_title('Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy
        axes[0, 1].plot(history['train_acc'], label='Train Accuracy')
        axes[0, 1].plot(history['val_acc'], label='Validation Accuracy')
        axes[0, 1].set_title('Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # F1 Score
        axes[1, 0].plot(history['train_f1'], label='Train F1')
        axes[1, 0].plot(history['val_f1'], label='Validation F1')
        axes[1, 0].set_title('F1 Score')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # ROC AUC
        axes[1, 1].plot(history['train_auc'], label='Train AUC')
        axes[1, 1].plot(history['val_auc'], label='Validation AUC')
        axes[1, 1].set_title('ROC AUC')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('AUC')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.show()
    
    def plot_feature_importance(self, feature_names: List[str], importance_scores: List[float],
                               title: str = "Feature Importance"):
        """Vẽ feature importance"""
        # Sắp xếp theo importance
        sorted_idx = np.argsort(importance_scores)[::-1]
        
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(importance_scores)), [importance_scores[i] for i in sorted_idx])
        plt.yticks(range(len(importance_scores)), [feature_names[i] for i in sorted_idx])
        plt.xlabel('Importance Score')
        plt.title(title)
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
    
    def plot_prediction_distribution(self, y_true: np.ndarray, y_prob: np.ndarray,
                                   title: str = "Prediction Distribution"):
        """Vẽ phân phối predictions"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Phân phối probability cho từng class
        event_probs = y_prob[y_true == 1]
        no_event_probs = y_prob[y_true == 0]
        
        axes[0].hist(no_event_probs, bins=50, alpha=0.7, label='No Event', color='blue')
        axes[0].hist(event_probs, bins=50, alpha=0.7, label='Event', color='red')
        axes[0].set_xlabel('Predicted Probability')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Prediction Distribution by Class')
        axes[0].legend()
        axes[0].grid(True)
        
        # Box plot
        data = [no_event_probs, event_probs]
        axes[1].boxplot(data, labels=['No Event', 'Event'])
        axes[1].set_ylabel('Predicted Probability')
        axes[1].set_title('Prediction Distribution Box Plot')
        axes[1].grid(True)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.show()
    
    def plot_calibration_curve(self, y_true: np.ndarray, y_prob: np.ndarray,
                              title: str = "Calibration Curve"):
        """Vẽ calibration curve"""
        from sklearn.calibration import calibration_curve
        
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_prob, n_bins=10
        )
        
        plt.figure(figsize=(8, 6))
        plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Model")
        plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        plt.xlabel("Mean Predicted Probability")
        plt.ylabel("Fraction of Positives")
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def create_interactive_dashboard(self, results: Dict[str, Dict]):
        """Tạo interactive dashboard với Plotly"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('ROC AUC', 'PR AUC', 'F1 Score', 'Late Alarm Rate'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        models = list(results.keys())
        
        # ROC AUC
        fig.add_trace(
            go.Bar(x=models, y=[results[m]['roc_auc'] for m in models], name='ROC AUC'),
            row=1, col=1
        )
        
        # PR AUC
        fig.add_trace(
            go.Bar(x=models, y=[results[m]['pr_auc'] for m in models], name='PR AUC'),
            row=1, col=2
        )
        
        # F1 Score
        fig.add_trace(
            go.Bar(x=models, y=[results[m]['f1_score'] for m in models], name='F1 Score'),
            row=2, col=1
        )
        
        # Late Alarm Rate
        fig.add_trace(
            go.Bar(x=models, y=[results[m]['late_alarm_rate'] for m in models], name='Late Alarm Rate'),
            row=2, col=2
        )
        
        fig.update_layout(
            title="Model Performance Comparison",
            showlegend=False,
            height=600,
            width=1000
        )
        
        fig.show()
    
    def plot_time_series_prediction(self, timeseries_data: np.ndarray, 
                                   predictions: np.ndarray, 
                                   labels: np.ndarray,
                                   feature_names: List[str],
                                   sample_idx: int = 0):
        """Vẽ time series prediction cho một sample"""
        sample_data = timeseries_data[sample_idx]
        sample_pred = predictions[sample_idx]
        sample_label = labels[sample_idx]
        
        fig, axes = plt.subplots(len(feature_names), 1, figsize=(15, 12))
        
        for i, feature_name in enumerate(feature_names):
            axes[i].plot(sample_data[:, i], label=feature_name)
            axes[i].set_ylabel(feature_name)
            axes[i].grid(True)
            
            # Highlight prediction point
            axes[i].axvline(x=len(sample_data)-1, color='red', linestyle='--', 
                          label=f'Prediction Point\nPred: {sample_pred:.3f}, True: {sample_label}')
            axes[i].legend()
        
        axes[-1].set_xlabel('Time Steps')
        plt.suptitle(f'Time Series Prediction for Sample {sample_idx}', fontsize=16)
        plt.tight_layout()
        plt.show()
