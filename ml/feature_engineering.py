import pandas as pd
import numpy as np
import os

def engineer_features(input_path='data/transactions.csv', output_path='data/transactions_engineered.csv'):
    print(f"Loading raw transaction data from {input_path}...")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Raw transaction dataset not found at {input_path}. Run generate_dataset.py first.")
        
    df = pd.read_csv(input_path)
    
    print("Engineering risk features...")
    
    df['amount_ratio'] = df['amount'] / df['customer_avg_amount']
    
    df['location_mismatch'] = (df['customer_location'] != df['transaction_location']).astype(int)
    
    df['high_velocity'] = (df['transactions_last_10min'] >= 8).astype(int)
    
    df['high_failed_attempts'] = (df['failed_attempts_last_10min'] >= 3).astype(int)
    
    df['new_account'] = (df['account_age_days'] < 30).astype(int)
    
    df['payment_method_raw'] = df['payment_method']
    df['merchant_category_raw'] = df['merchant_category']
    
    print("Performing one-hot encoding for categorical variables...")
    df = pd.get_dummies(df, columns=['payment_method', 'merchant_category'], prefix=['pay', 'cat'], dtype=int)
    
    df = df.rename(columns={
        'payment_method_raw': 'payment_method',
        'merchant_category_raw': 'merchant_category'
    })
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    df.to_csv(output_path, index=False)
    print(f"Feature engineering complete! Saved to {output_path}")
    print(f"Dataset shape: {df.shape}")
    print("\nEngineered Feature Summary (Fraud vs Legitimate averages):")
    
    summary_cols = ['amount_ratio', 'location_mismatch', 'high_velocity', 'high_failed_attempts', 'new_account']
    for col in summary_cols:
        legit_mean = df[df['is_fraud'] == 0][col].mean()
        fraud_mean = df[df['is_fraud'] == 1][col].mean()
        print(f" - {col:25} | Legit: {legit_mean:.4f} | Fraud: {fraud_mean:.4f}")

if __name__ == '__main__':
    engineer_features()
