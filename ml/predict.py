import pandas as pd
import numpy as np
import os
import pickle
import shap

class FraudPredictor:
    def __init__(self, model_path='models/fraud_model.pkl'):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}. Train the model first.")
            
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
            
        self.model = model_data['model']
        self.feature_cols = model_data['features']
        

        self.explainer = shap.TreeExplainer(self.model)
        
    def predict_transaction(self, transaction_row):
        """
        transaction_row: pandas DataFrame with 1 row, containing all feature columns.
        """
        X = transaction_row[self.feature_cols]
        
        prob = float(self.model.predict_proba(X)[0, 1])
        
        shap_values = self.explainer.shap_values(X)[0]
        
        shap_dict = dict(zip(self.feature_cols, shap_values))
        
        sorted_shap = sorted(shap_dict.items(), key=lambda x: x[1], reverse=True)
        
        explanation_phrases = []
        for feature, val in sorted_shap:
            if val > 0.05:  
                if feature == 'amount_ratio':
                    ratio_val = float(transaction_row['amount_ratio'].iloc[0])
                    explanation_phrases.append(f"Spending is {ratio_val:.1f}x higher than average")
                elif feature == 'device_age_days':
                    age = int(transaction_row['device_age_days'].iloc[0])
                    explanation_phrases.append(f"Device is very new ({age} days old)")
                elif feature == 'is_new_device':
                    explanation_phrases.append("Transaction initiated from a new device")
                elif feature == 'transactions_last_10min':
                    vel = int(transaction_row['transactions_last_10min'].iloc[0])
                    explanation_phrases.append(f"Unusually high velocity ({vel} transactions in 10 minutes)")
                elif feature == 'failed_attempts_last_10min':
                    fails = int(transaction_row['failed_attempts_last_10min'].iloc[0])
                    explanation_phrases.append(f"Multiple failed transaction attempts ({fails} attempts in 10 minutes)")
                elif feature == 'location_mismatch':
                    explanation_phrases.append("Transaction location mismatch from home location")
                elif feature == 'account_age_days':
                    acc_age = int(transaction_row['account_age_days'].iloc[0])
                    explanation_phrases.append(f"Account is relatively new ({acc_age} days old)")
                elif feature.startswith('pay_'):
                    pay_m = feature.split('_')[1]
                    explanation_phrases.append(f"Using payment method: {pay_m}")
                elif feature.startswith('cat_'):
                    cat = feature.split('_')[1]
                    explanation_phrases.append(f"Using merchant category: {cat}")
                    
        if not explanation_phrases:
            explanation = "No significant risk indicators identified. Normal transaction pattern."
        else:
            explanation = "The transaction is flagged as risky because: " + ", ".join(explanation_phrases) + "."
            
        return {
            'fraud_probability': prob,
            'shap_values': shap_dict,
            'top_contributors': [k for k, v in sorted_shap if v > 0.05],
            'human_explanation': explanation
        }

if __name__ == '__main__':
    data_path = 'data/transactions_engineered.csv'
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        tx_test = df[df['transaction_id'] == 'TX10091']
        
        if len(tx_test) > 0:
            print("Found test transaction TX10091. Running prediction and SHAP explanation...")
            predictor = FraudPredictor()
            result = predictor.predict_transaction(tx_test)
            
            print("\n" + "="*50)
            print(f"Transaction ID : TX10091")
            print(f"Amount         : INR {tx_test['amount'].iloc[0]:,.2f}")
            print(f"Fraud Risk Score: {result['fraud_probability']*100:.2f}%")
            print(f"AI Explanation : {result['human_explanation']}")
            print("="*50)
            print("\nDetailed SHAP Contributions (Raw log-odds scale):")
            for feat, val in sorted(result['shap_values'].items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f" - {feat:25}: {val:.4f}")
        else:
            print("Transaction TX10091 not found in dataset. Run feature engineering first.")
    else:
        print("Engineered dataset file not found.")
