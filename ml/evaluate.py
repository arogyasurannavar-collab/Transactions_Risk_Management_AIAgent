import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score

def evaluate_model(data_path='data/transactions_engineered.csv', model_path='models/fraud_model.pkl'):
    print(f"Loading data from {data_path}...")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Engineered dataset not found at {data_path}. Run ml/feature_engineering.py first.")
        
    df = pd.read_csv(data_path)
    
    print(f"Loading model from {model_path}...")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Run ml/train.py first.")
        
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
        
    model = model_data['model']
    feature_cols = model_data['features']
    
    # Stratified split to get the test dataset (same as train.py)
    X = df[feature_cols]
    y = df['is_fraud']
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Evaluating model on test dataset of shape {X_test.shape}...")
    
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    
    thresholds = [0.4, 0.5, 0.7]
    
    print("\n" + "="*50)
    print("MODEL PERFORMANCE METRICS")
    print("="*50)
    
    for thresh in thresholds:
        y_pred = (y_pred_prob >= thresh).astype(int)
        
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
        print(f"\n--- Threshold: {thresh} ---")
        print(f"Confusion Matrix:")
        print(f"  Predicted Legit  Predicted Fraud")
        print(f"Legit:  TN={tn:5d}    FP={fp:5d}")
        print(f"Fraud:  FN={fn:5d}    TP={tp:5d}")
        print(f"Precision : {precision:.4f} (Of all flagged transactions, how many were actually fraud)")
        print(f"Recall    : {recall:.4f} (Of all actual fraud, how many did we catch)")
        print(f"F1-Score  : {f1:.4f}")
        print(f"False Positive Rate (FPR): {fpr*100:.2f}% (Percentage of good transactions blocked)")
        
    print("\n" + "="*50)
    print("Top Feature Importances:")
    importances = model.feature_importances_
    feat_imp = pd.Series(importances, index=feature_cols).sort_values(ascending=False)
    for feat, imp in feat_imp.head(7).items():
        print(f" - {feat:25}: {imp:.4f}")
    print("="*50)

if __name__ == '__main__':
    evaluate_model()
