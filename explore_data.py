import pandas as pd
import os

def explore():
    data_path = 'data/transactions.csv'
    if not os.path.exists(data_path):
        print(f"Data file not found at {data_path}. Run generate_dataset.py first.")
        return
        
    df = pd.read_csv(data_path)
    print("="*50)
    print("RAW TRANSACTION DATA EXPLORATION")
    print("="*50)
    print(f"Total Rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nFraud Rate: {df['is_fraud'].mean() * 100:.2f}% ({df['is_fraud'].sum()} flagged / {len(df) - df['is_fraud'].sum()} normal)")
    
    print("\nAmount Distribution Summary:")
    print(df['amount'].describe())
    
    print("\nTransactions by Payment Method:")
    print(df['payment_method'].value_counts())
    
    print("\nTransactions by Merchant Category:")
    print(df['merchant_category'].value_counts())
    
    print("\nNew Device Usage:")
    print(df['is_new_device'].value_counts())
    print("="*50)

if __name__ == '__main__':
    explore()
