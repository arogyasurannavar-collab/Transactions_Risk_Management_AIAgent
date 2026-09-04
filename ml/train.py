import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

def train_model(data_path='data/transactions_engineered.csv', model_dir='models'):
    print(f"Loading engineered dataset from {data_path}...")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Engineered dataset not found at {data_path}. Run ml/feature_engineering.py first.")
        
    df = pd.read_csv(data_path)
    
    exclude_cols = [
        'transaction_id', 
        'customer_id', 
        'device_id', 
        'timestamp', 
        'transaction_location', 
        'customer_location', 
        'payment_method', 
        'merchant_category', 
        'is_fraud'
    ]
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols]
    y = df['is_fraud']
    
    print(f"Features used for training ({len(feature_cols)}):")
    print(feature_cols)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    print(f"Training fraud instances: {y_train.sum()} ({y_train.mean()*100:.2f}%)")
    
    model = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        scale_pos_weight=5,
        random_state=42,
        eval_metric='logloss'
    )
    
    print("Training XGBoost model...")
    model.fit(X_train, y_train)
    
    os.makedirs(model_dir, exist_ok=True)
    
    model_data = {
        'model': model,
        'features': feature_cols
    }
    
    model_path = os.path.join(model_dir, 'fraud_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
        
    print(f"Model saved successfully to {model_path}!")
    
if __name__ == '__main__':
    train_model()
