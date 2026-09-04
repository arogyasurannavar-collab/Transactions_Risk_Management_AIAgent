import pandas as pd
import numpy as np
import os
from datetime import datetime
import sys

# Add parent directory to path so we can import from ml
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml.predict import FraudPredictor

DATA_PATH = 'data/transactions_engineered.csv'
CASES_PATH = 'data/review_cases.csv'

# Initialize the predictor once
_predictor = None
def get_predictor():
    global _predictor
    if _predictor is None:
        _predictor = FraudPredictor()
    return _predictor

def get_transaction(transaction_id: str) -> dict:
    
    if not os.path.exists(DATA_PATH):
        return {"error": "Data file not found. Run feature engineering first."}
        
    df = pd.read_csv(DATA_PATH)
    tx = df[df['transaction_id'] == transaction_id]
    
    if len(tx) == 0:
        return {"error": f"Transaction {transaction_id} not found."}
        
    row = tx.iloc[0]
    return {
        "transaction_id": str(row['transaction_id']),
        "customer_id": str(row['customer_id']),
        "amount": float(row['amount']),
        "timestamp": str(row['timestamp']),
        "payment_method": str(row['payment_method']),
        "merchant_category": str(row['merchant_category']),
        "location": str(row['transaction_location'])
    }

def get_customer_history(customer_id: str) -> dict:
    
    if not os.path.exists(DATA_PATH):
        return {"error": "Data file not found. Run feature engineering first."}
        
    df = pd.read_csv(DATA_PATH)
    cust_txs = df[df['customer_id'] == customer_id]
    
    if len(cust_txs) == 0:
        return {"error": f"Customer {customer_id} not found."}
        
    row = cust_txs.iloc[0]
    return {
        "customer_id": str(customer_id),
        "customer_location": str(row['customer_location']),
        "customer_avg_amount": float(row['customer_avg_amount']),
        "customer_transaction_count": int(row['customer_transaction_count']),
        "account_age_days": int(row['account_age_days'])
    }

def check_device(device_id: str) -> dict:
    
    if not os.path.exists(DATA_PATH):
        return {"error": "Data file not found. Run feature engineering first."}
        
    df = pd.read_csv(DATA_PATH)
    dev_txs = df[df['device_id'] == device_id]
    
    if len(dev_txs) == 0:
        return {
            "device_id": str(device_id),
            "device_age_days": 0,
            "is_new_device": 1,
            "total_transactions_with_device": 0
        }
        
    row = dev_txs.iloc[0]
    return {
        "device_id": str(device_id),
        "device_age_days": int(row['device_age_days']),
        "is_new_device": int(row['is_new_device']),
        "total_transactions_with_device": len(dev_txs)
    }

def check_velocity(customer_id: str) -> dict:
    
    if not os.path.exists(DATA_PATH):
        return {"error": "Data file not found. Run feature engineering first."}
        
    df = pd.read_csv(DATA_PATH)
    cust_txs = df[df['customer_id'] == customer_id]
    
    if len(cust_txs) == 0:
        return {"error": f"Customer {customer_id} not found."}
        
    row = cust_txs.iloc[0]
    return {
        "customer_id": str(customer_id),
        "transactions_last_10min": int(row['transactions_last_10min']),
        "failed_attempts_last_10min": int(row['failed_attempts_last_10min'])
    }

def run_fraud_model(transaction_id: str) -> dict:
    
    if not os.path.exists(DATA_PATH):
        return {"error": "Data file not found. Run feature engineering first."}
        
    df = pd.read_csv(DATA_PATH)
    tx = df[df['transaction_id'] == transaction_id]
    
    if len(tx) == 0:
        return {"error": f"Transaction {transaction_id} not found."}
        
    try:
        predictor = get_predictor()
        res = predictor.predict_transaction(tx)
        return {
            "transaction_id": transaction_id,
            "fraud_probability": float(res['fraud_probability'])
        }
    except Exception as e:
        return {"error": f"Error running fraud model: {str(e)}"}

def get_shap_explanation(transaction_id: str) -> dict:
    
    if not os.path.exists(DATA_PATH):
        return {"error": "Data file not found. Run feature engineering first."}
        
    df = pd.read_csv(DATA_PATH)
    tx = df[df['transaction_id'] == transaction_id]
    
    if len(tx) == 0:
        return {"error": f"Transaction {transaction_id} not found."}
        
    try:
        predictor = get_predictor()
        res = predictor.predict_transaction(tx)
        
        # Convert SHAP dictionary values to standard float for serialization
        serializable_shap = {k: float(v) for k, v in res['shap_values'].items()}
        
        return {
            "transaction_id": transaction_id,
            "shap_values": serializable_shap,
            "human_explanation": res['human_explanation']
        }
    except Exception as e:
        return {"error": f"Error generating SHAP explanation: {str(e)}"}

def create_review_case(transaction_id: str, risk_score: float, reasoning: str, recommendation: str) -> dict:
    
    tx_details = get_transaction(transaction_id)
    if "error" in tx_details:
        return tx_details
        
    if not os.path.exists(CASES_PATH):
        df_cases = pd.DataFrame(columns=[
            'case_id', 'transaction_id', 'amount', 'risk_score', 
            'reasoning', 'recommendation', 'status', 'human_decision', 'created_at'
        ])
        df_cases.to_csv(CASES_PATH, index=False)
    else:
        df_cases = pd.read_csv(CASES_PATH)
        
    if len(df_cases[df_cases['transaction_id'] == transaction_id]) > 0:
        existing_case = df_cases[df_cases['transaction_id'] == transaction_id].iloc[0]
        return {
            "message": "Review case already exists for this transaction.",
            "case_id": str(existing_case['case_id']),
            "status": str(existing_case['status'])
        }
        
    case_num = 10001 + len(df_cases)
    case_id = f"CASE{case_num}"
    
    new_case = {
        'case_id': case_id,
        'transaction_id': transaction_id,
        'amount': float(tx_details['amount']),
        'risk_score': float(risk_score),
        'reasoning': reasoning,
        'recommendation': recommendation,
        'status': 'PENDING',
        'human_decision': 'UNRESOLVED',
        'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    df_new = pd.DataFrame([new_case])
    df_cases = pd.concat([df_cases, df_new], ignore_index=True)
    df_cases.to_csv(CASES_PATH, index=False)
    
    return {
        "message": "Review case successfully created.",
        "case_id": case_id,
        "status": "PENDING"
    }
