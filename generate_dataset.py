import pandas as pd
import numpy as np
import os
import random
from datetime import datetime, timedelta

def generate_data(num_transactions=50000, num_customers=1500, random_seed=42):
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    print(f"Generating customer profiles for {num_customers} customers...")
    
    locations = ['Bengaluru', 'Mumbai', 'Delhi', 'Chennai', 'Hyderabad', 'Kolkata', 'Pune', 'Ahmedabad']
    categories = ['Food & Beverage', 'Retail', 'Electronics', 'Travel', 'Entertainment', 'Services']
    payment_methods = ['UPI', 'Credit Card', 'Debit Card', 'Net Banking']
    
    customers = []
    for i in range(num_customers):
        cust_id = f"C{10000 + i}"
        home_loc = random.choice(locations)
        avg_amt = round(float(np.random.exponential(scale=3000) + 500), 2)  # Avg between 500 and ~15000
        tx_count = int(np.random.randint(10, 300))
        acc_age = int(np.random.randint(10, 1200)) # age in days
        primary_device = f"D{20000 + i}"
        
        customers.append({
            'customer_id': cust_id,
            'customer_location': home_loc,
            'customer_avg_amount': avg_amt,
            'customer_transaction_count': tx_count,
            'account_age_days': acc_age,
            'primary_device_id': primary_device
        })
        
    df_customers = pd.DataFrame(customers)
    
    transactions = []
    start_date = datetime.now() - timedelta(days=30)
    
    print(f"Generating {num_transactions} transactions...")
    
    fraud_rate = 0.02
    
    for i in range(num_transactions):
        tx_id = f"TX{100000 + i}"
        cust = df_customers.iloc[random.randint(0, num_customers - 1)]
        
        is_fraud = 1 if random.random() < fraud_rate else 0
        
        time_offset = random.random() * 30 # days
        tx_time = start_date + timedelta(days=time_offset)
        
        if is_fraud == 0:

            amount = round(float(cust['customer_avg_amount'] * np.random.lognormal(mean=0, sigma=0.35)), 2)
            amount = max(100.0, amount)
            
            is_new_device = 1 if random.random() < 0.05 else 0
            if is_new_device:
                device_id = f"D{random.randint(30000, 99999)}"
                device_age = int(np.random.randint(0, 10))
            else:
                device_id = cust['primary_device_id']
                device_age = int(np.random.randint(30, 365))
                
            if random.random() < 0.93:
                tx_loc = cust['customer_location']
            else:
                tx_loc = random.choice([loc for loc in locations if loc != cust['customer_location']])
                
            tx_last_10min = int(np.random.choice([1, 2, 3], p=[0.8, 0.15, 0.05]))
            failed_attempts = int(np.random.choice([0, 1], p=[0.95, 0.05]))
            
            category = random.choice(categories)
            pay_method = np.random.choice(payment_methods, p=[0.6, 0.2, 0.1, 0.1]) # UPI is dominant
            
        else:

            scenario = random.choice(['velocity_attack', 'large_amount_spike', 'new_device_mismatch'])
            
            if scenario == 'velocity_attack':
                amount = round(float(cust['customer_avg_amount'] * random.uniform(1.5, 3.5)), 2)
                is_new_device = 1 if random.random() < 0.8 else 0
                device_id = f"D{random.randint(30000, 99999)}" if is_new_device else cust['primary_device_id']
                device_age = int(np.random.randint(0, 3)) if is_new_device else int(np.random.randint(10, 100))
                tx_loc = random.choice(locations) 
                tx_last_10min = int(np.random.randint(8, 15))
                failed_attempts = int(np.random.randint(3, 7))
                category = random.choice(['Electronics', 'Travel', 'Services'])
                pay_method = random.choice(payment_methods)
                
            elif scenario == 'large_amount_spike':
                amount = round(float(cust['customer_avg_amount'] * random.uniform(12.0, 45.0)), 2)
                is_new_device = 1
                device_id = f"D{random.randint(30000, 99999)}"
                device_age = 0
                tx_loc = random.choice([loc for loc in locations if loc != cust['customer_location']])
                tx_last_10min = int(np.random.choice([1, 2], p=[0.8, 0.2]))
                failed_attempts = int(np.random.choice([0, 1, 2], p=[0.6, 0.3, 0.1]))
                category = random.choice(['Electronics', 'Travel']) 
                pay_method = 'Credit Card' 
                
            else: 
                amount = round(float(cust['customer_avg_amount'] * random.uniform(3.0, 8.0)), 2)
                is_new_device = 1
                device_id = f"D{random.randint(30000, 99999)}"
                device_age = 0
                tx_loc = random.choice([loc for loc in locations if loc != cust['customer_location']])
                tx_last_10min = int(np.random.randint(3, 7))
                failed_attempts = int(np.random.randint(2, 4))
                category = random.choice(categories)
                pay_method = random.choice(payment_methods)
        
        transactions.append({
            'transaction_id': tx_id,
            'customer_id': cust['customer_id'],
            'amount': amount,
            'timestamp': tx_time.strftime('%Y-%m-%d %H:%M:%S'),
            'payment_method': pay_method,
            'merchant_category': category, 
            'customer_avg_amount': cust['customer_avg_amount'],
            'customer_transaction_count': cust['customer_transaction_count'],
            'account_age_days': cust['account_age_days'],
            'device_id': device_id,
            'device_age_days': device_age,
            'is_new_device': is_new_device,
            'transaction_location': tx_loc,
            'customer_location': cust['customer_location'],
            'transactions_last_10min': tx_last_10min,
            'failed_attempts_last_10min': failed_attempts,
            'is_fraud': is_fraud
        })
        
    df_transactions = pd.DataFrame(transactions)
    
    df_transactions = df_transactions.sort_values(by='timestamp').reset_index(drop=True)
    
    story_tx = {
        'transaction_id': 'TX10091',
        'customer_id': 'C1023',
        'amount': 85000.0,
        'timestamp': (start_date + timedelta(days=15)).strftime('%Y-%m-%d %H:%M:%S'),
        'payment_method': 'Credit Card',
        'merchant_category': 'Electronics',
        'customer_avg_amount': 2500.0,
        'customer_transaction_count': 42,
        'account_age_days': 150,
        'device_id': 'D9182',
        'device_age_days': 1,
        'is_new_device': 1,
        'transaction_location': 'Mumbai',
        'customer_location': 'Bengaluru',
        'transactions_last_10min': 12,
        'failed_attempts_last_10min': 4,
        'is_fraud': 1
    }
    
    df_transactions.loc[25000] = story_tx
    df_transactions = df_transactions.sort_values(by='timestamp').reset_index(drop=True)
    
    os.makedirs('data', exist_ok=True)
    
    file_path = os.path.join('data', 'transactions.csv')
    df_transactions.to_csv(file_path, index=False)
    print(f"Dataset generated successfully! Saved to {file_path}")
    print(f"Total rows: {len(df_transactions)}")
    print(f"Legitimate transactions: {len(df_transactions[df_transactions['is_fraud'] == 0])}")
    print(f"Fraud transactions: {len(df_transactions[df_transactions['is_fraud'] == 1])} ({len(df_transactions[df_transactions['is_fraud'] == 1])/len(df_transactions)*100:.2f}%)")
    
if __name__ == '__main__':
    generate_data()
