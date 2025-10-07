import torch
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
import json
import matplotlib.pyplot as plt

from config.config import model_config, training_config
from models.multimodal_model import MultimodalContrastiveAI
from models.baseline_models import LSTMBaseline, CNNBaseline, TransformerBaseline, MLBaseline, MEWSScore
from utils.data_loader import create_data_loaders
from utils.preprocessing import load_and_preprocess_data
from utils.evaluation import ModelEvaluator, evaluate_model_on_loader
from utils.visualization import ResultVisualizer

def evaluate_multimodal_model(model_path, test_loader, device):
    """Evaluate multimodal model"""
    model = MultimodalContrastiveAI(model_config)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    
    evaluator = ModelEvaluator()
    metrics = evaluate_model_on_loader(model, test_loader, device, evaluator)
    
    return metrics

def evaluate_baseline_models(train_dataset, test_dataset, device):
    """Evaluate baseline models"""
    results = {}
    
    # Prepare data for baseline models
    def prepare_data(dataset):
        X, y = [], []
        for i in range(len(dataset)):
            sample = dataset[i]
            X.append(sample['timeseries'].numpy())
            y.append(sample['labels'].numpy().flatten()[0])
        return np.array(X), np.array(y)
    
    X_train, y_train = prepare_data(train_dataset)
    X_test, y_test = prepare_data(test_dataset)
    
    # LSTM Baseline
    print("Evaluating LSTM Baseline...")
    lstm_model = LSTMBaseline(input_size=model_config.timeseries_input_size)
    lstm_model.to(device)
    
    # Train LSTM baseline (simplified training)
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.001)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    lstm_model.train()
    for epoch in range(50):  # Quick training
        total_loss = 0
        for i in range(0, len(X_train), 32):
            batch_X = torch.FloatTensor(X_train[i:i+32]).to(device)
            batch_y = torch.FloatTensor(y_train[i:i+32]).unsqueeze(1).to(device)
            
            optimizer.zero_grad()
            outputs = lstm_model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if epoch % 10 == 0:
            print(f'LSTM Epoch {epoch}, Loss: {total_loss:.4f}')
    
    # Evaluate LSTM
    lstm_model.eval()
    with torch.no_grad():
        test_outputs = lstm_model(torch.FloatTensor(X_test).to(device))
        test_probs = torch.sigmoid(test_outputs).cpu().numpy().flatten()
        test_preds = (test_probs > 0.5).astype(int)
    
    evaluator = ModelEvaluator()
    results['LSTM'] = evaluator.evaluate(y_test, test_preds, test_probs)
    
    # CNN Baseline
    print("Evaluating CNN Baseline...")
    cnn_model = CNNBaseline(input_size=model_config.timeseries_input_size)
    cnn_model.to(device)
    
    # Train CNN baseline (simplified)
    optimizer = torch.optim.Adam(cnn_model.parameters(), lr=0.001)
    
    cnn_model.train()
    for epoch in range(50):
        total_loss = 0
        for i in range(0, len(X_train), 32):
            batch_X = torch.FloatTensor(X_train[i:i+32]).to(device)
            batch_y = torch.FloatTensor(y_train[i:i+32]).unsqueeze(1).to(device)
            
            optimizer.zero_grad()
            outputs = cnn_model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if epoch % 10 == 0:
            print(f'CNN Epoch {epoch}, Loss: {total_loss:.4f}')
    
    # Evaluate CNN
    cnn_model.eval()
    with torch.no_grad():
        test_outputs = cnn_model(torch.FloatTensor(X_test).to(device))
        test_probs = torch.sigmoid(test_outputs).cpu().numpy().flatten()
        test_preds = (test_probs > 0.5).astype(int)
    
    results['CNN'] = evaluator.evaluate(y_test, test_preds, test_probs)
    
    # Machine Learning Baselines
    print("Evaluating ML Baselines...")
    
    # Random Forest
    rf_model = MLBaseline('rf')
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)
    rf_probs = rf_model.predict_proba(X_test)
    results['Random Forest'] = evaluator.evaluate(y_test, rf_preds, rf_probs)
    
    # XGBoost
    xgb_model = MLBaseline('xgb')
    xgb_model.fit(X_train, y_train)
    xgb_preds = xgb_model.predict(X_test)
    xgb_probs = xgb_model.predict_proba(X_test)
    results['XGBoost'] = evaluator.evaluate(y_test, xgb_preds, xgb_probs)
    
    # Logistic Regression
    lr_model = MLBaseline('lr')
    lr_model.fit(X_train, y_train)
    lr_preds = lr_model.predict(X_test)
    lr_probs = lr_model.predict_proba(X_test)
    results['Logistic Regression'] = evaluator.evaluate(y_test, lr_preds, lr_probs)
    
    # MEWS Score
    print("Evaluating MEWS Score...")
    mews_model = MEWSScore()
    mews_preds = mews_model.predict(X_test)
    mews_probs = mews_model.predict_proba(X_test)
    results['MEWS'] = evaluator.evaluate(y_test, mews_preds, mews_probs)
    
    return results

def main():
    """Main evaluation function"""
    print("Loading test data...")
    
    # Load data
    train_dataset, val_dataset, test_dataset = load_and_preprocess_data(
        'data/clinical_data.csv', model_config
    )
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        train_dataset, val_dataset, test_dataset, model_config
    )
    
    # Evaluate multimodal model
    print("Evaluating Multimodal Model...")
    multimodal_results = evaluate_multimodal_model(
        f'{training_config.model_save_path}/best_model.pth',
        test_loader,
        training_config.device
    )
    
    # Evaluate baseline models
    print("Evaluating Baseline Models...")
    baseline_results = evaluate_baseline_models(
        train_dataset, test_dataset, training_config.device
    )
    
    # Combine results
    all_results = {'Multimodal AI': multimodal_results, **baseline_results}
    
    # Remove confusion matrix for JSON serialization
    for model_name in all_results:
        if 'confusion_matrix' in all_results[model_name]:
            del all_results[model_name]['confusion_matrix']
    
    # Save results
    with open(f'{training_config.model_save_path}/evaluation_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print results
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    
    results_df = pd.DataFrame(all_results).T
    print(results_df.round(4))
    
    # Visualize results
    evaluator = ModelEvaluator()
    evaluator.compare_models(all_results)
    
    visualizer = ResultVisualizer()
    visualizer.create_interactive_dashboard(all_results)
    
    # Plot ROC curves for all models
    plt.figure(figsize=(10, 8))
    
    # This would require storing y_prob for each model
    # For demonstration, we'll show the structure
    print("\nModel Performance Summary:")
    print("-" * 50)
    
    for model_name, metrics in all_results.items():
        print(f"{model_name}:")
        print(f"  ROC AUC: {metrics['roc_auc']:.4f}")
        print(f"  PR AUC: {metrics['pr_auc']:.4f}")
        print(f"  F1 Score: {metrics['f1_score']:.4f}")
        print(f"  Sensitivity: {metrics['sensitivity']:.4f}")
        print(f"  Specificity: {metrics['specificity']:.4f}")
        print(f"  Late Alarm Rate: {metrics['late_alarm_rate']:.4f}")
        print()
    
    print("Evaluation completed!")

if __name__ == "__main__":
    main()
