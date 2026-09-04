import os
import sys
import google.generativeai as genai


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.tools import (
    get_transaction,
    get_customer_history,
    check_device,
    check_velocity,
    run_fraud_model,
    get_shap_explanation,
    create_review_case
)
from agent.prompts import SYSTEM_PROMPT
from agent.decision_engine import determine_action

DATA_PATH = 'data/transactions_engineered.csv'






def investigate_transaction(transaction_id: str, api_key: str = None) -> dict:
    if api_key and api_key.strip():
        try:
            return investigate_transaction_live(transaction_id, api_key.strip())
        except Exception as e:
            print(f"Failed to run live agent (falling back to mock): {str(e)}")
            result = investigate_transaction_mock(transaction_id)
            result['warning'] = f"Live agent failed: {str(e)}. Switched to local mode."
            return result
    else:
        result = investigate_transaction_mock(transaction_id)
        result['warning'] = "No Gemini API key provided. Running in offline/simulated mode."
        return result

def investigate_transaction_live(transaction_id: str, api_key: str) -> dict:
    
    genai.configure(api_key=api_key)
    
    tools_list = [
        get_transaction,
        get_customer_history,
        check_device,
        check_velocity,
        run_fraud_model,
        get_shap_explanation,
        create_review_case
    ]
    
    model = genai.GenerativeModel(
        model_name='gemini-1.5-flash',
        tools=tools_list,
        system_instruction=SYSTEM_PROMPT
    )
    
    chat = model.start_chat(enable_automatic_function_calling=True)
    
    prompt = f"Please investigate the transaction with ID {transaction_id}."
    response = chat.send_message(prompt)
    
    timeline = []
    
    for message in chat.history:
        role = message.role
        for part in message.parts:
            if hasattr(part, 'text') and part.text:
                if role == 'user' and transaction_id in part.text:
                    continue
                timeline.append({
                    "type": "thought",
                    "role": "Agent",
                    "content": part.text
                })
            
            elif hasattr(part, 'function_call') and part.function_call:
                timeline.append({
                    "type": "call",
                    "name": part.function_call.name,
                    "args": dict(part.function_call.args)
                })
                
            elif hasattr(part, 'function_response') and part.function_response:
                resp_dict = {}
                if hasattr(part.function_response, 'response'):
                    try:
                        fields = part.function_response.response.fields
                        for k, v in fields.items():
                            if hasattr(v, 'string_value') and v.string_value:
                                resp_dict[k] = v.string_value
                            elif hasattr(v, 'number_value') and v.number_value:
                                resp_dict[k] = v.number_value
                            else:
                                resp_dict[k] = str(v)
                    except:
                        resp_dict = {"raw": str(part.function_response.response)}
                timeline.append({
                    "type": "response",
                    "name": part.function_response.name,
                    "response": resp_dict
                })
                
    prob_info = run_fraud_model(transaction_id)
    prob = prob_info.get('fraud_probability', 0.0)
    decision = determine_action(prob)
    
    return {
        "mode": "LIVE_GEMINI",
        "transaction_id": transaction_id,
        "risk_score": prob,
        "action": decision['action'],
        "timeline": timeline,
        "report": response.text
    }

def investigate_transaction_mock(transaction_id: str) -> dict:
    """
    Offline mock agent that simulates the logical execution trace 
    and generates the corresponding Markdown report.
    """
    timeline = []
    
    timeline.append({
        "type": "thought",
        "role": "Agent",
        "content": f"Initializing investigation for transaction {transaction_id}. Retrieving core transaction details..."
    })
    
    timeline.append({
        "type": "call",
        "name": "get_transaction",
        "args": {"transaction_id": transaction_id}
    })
    tx_details = get_transaction(transaction_id)
    timeline.append({
        "type": "response",
        "name": "get_transaction",
        "response": tx_details
    })
    
    if "error" in tx_details:
        return {
            "mode": "MOCK_LOCAL",
            "transaction_id": transaction_id,
            "error": tx_details["error"],
            "timeline": timeline,
            "report": f"Investigation failed: {tx_details['error']}"
        }
        
    cust_id = tx_details['customer_id']
    amount = tx_details['amount']
    
    timeline.append({
        "type": "thought",
        "role": "Agent",
        "content": f"Transaction details loaded. Customer ID is {cust_id}. Checking spending history and location profile..."
    })
    
    timeline.append({
        "type": "call",
        "name": "get_customer_history",
        "args": {"customer_id": cust_id}
    })
    cust_history = get_customer_history(cust_id)
    timeline.append({
        "type": "response",
        "name": "get_customer_history",
        "response": cust_history
    })
    

    import pandas as pd
    df = pd.read_csv(DATA_PATH)
    device_id = str(df[df['transaction_id'] == transaction_id]['device_id'].iloc[0])
    
    timeline.append({
        "type": "thought",
        "role": "Agent",
        "content": f"Customer history retrieved. Spending average is INR {cust_history['customer_avg_amount']:,.2f}. Checking device profile for device {device_id}..."
    })
    
    timeline.append({
        "type": "call",
        "name": "check_device",
        "args": {"device_id": device_id}
    })
    device_info = check_device(device_id)
    timeline.append({
        "type": "response",
        "name": "check_device",
        "response": device_info
    })
    
    
    timeline.append({
        "type": "thought",
        "role": "Agent",
        "content": "Device profile checked. Running transaction velocity and failed attempts checks in the last 10 minutes..."
    })
    
    timeline.append({
        "type": "call",
        "name": "check_velocity",
        "args": {"customer_id": cust_id}
    })
    velocity_info = check_velocity(cust_id)
    timeline.append({
        "type": "response",
        "name": "check_velocity",
        "response": velocity_info
    })
    
    timeline.append({
        "type": "thought",
        "role": "Agent",
        "content": "Velocity metrics complete. Querying the XGBoost model to calculate statistical risk probability..."
    })
    
    timeline.append({
        "type": "call",
        "name": "run_fraud_model",
        "args": {"transaction_id": transaction_id}
    })
    model_info = run_fraud_model(transaction_id)
    timeline.append({
        "type": "response",
        "name": "run_fraud_model",
        "response": model_info
    })
    
    prob = model_info['fraud_probability']
    decision = determine_action(prob)
    
    timeline.append({
        "type": "thought",
        "role": "Agent",
        "content": f"XGBoost risk score is {prob*100:.2f}%. Generating SHAP explainability insights to identify primary risk drivers..."
    })
    
    timeline.append({
        "type": "call",
        "name": "get_shap_explanation",
        "args": {"transaction_id": transaction_id}
    })
    shap_info = get_shap_explanation(transaction_id)
    timeline.append({
        "type": "response",
        "name": "get_shap_explanation",
        "response": shap_info
    })
    
    case_created = False
    case_details = {}
    if decision['action'] == "HUMAN REVIEW":
        timeline.append({
            "type": "thought",
            "role": "Agent",
            "content": f"Risk score ({prob*100:.2f}%) exceeds the 70% threshold. Initiating support ticket in review database..."
        })
        
        reasoning = shap_info['human_explanation']
        reco = "Hold transaction and route to specialized human risk analyst."
        
        timeline.append({
            "type": "call",
            "name": "create_review_case",
            "args": {
                "transaction_id": transaction_id,
                "risk_score": prob,
                "reasoning": reasoning,
                "recommendation": reco
            }
        })
        case_details = create_review_case(transaction_id, prob, reasoning, reco)
        timeline.append({
            "type": "response",
            "name": "create_review_case",
            "response": case_details
        })
        case_created = True
        
    timeline.append({
        "type": "thought",
        "role": "Agent",
        "content": "Investigation finalized. Compiling final Markdown report."
    })
    
    case_str = f" Created review case {case_details.get('case_id')}." if case_created else ""
    report = f"""### RISK INVESTIGATION SUMMARY
- **Transaction ID**: {transaction_id}
- **Fraud Risk Score**: {prob*100:.2f}%
- **Recommended Bounded Action**: {decision['action']}

### KEY EVIDENCE GATHERED
- **Spending Behavior**: Transaction amount is INR {amount:,.2f} vs customer average of INR {cust_history['customer_avg_amount']:,.2f} (amount ratio is {amount/cust_history['customer_avg_amount']:.2f}x).
- **Device Risk**: Device ID is {device_id}. Device age is {device_info['device_age_days']} days (New Device: {"YES" if device_info['is_new_device'] else "NO"}).
- **Velocity & Attempts**: {velocity_info['transactions_last_10min']} transactions in the last 10 minutes with {velocity_info['failed_attempts_last_10min']} failed payment attempts.
- **Location Status**: Home location is {cust_history['customer_location']}; transaction was made in {tx_details['location']} (Mismatch: {"YES" if tx_details['location'] != cust_history['customer_location'] else "NO"}).

### MACHINE LEARNING & EXPLAINABLE AI ANALYSIS
- {shap_info['human_explanation']}

### FINAL INVESTIGATION ACTION
- The transaction risk is evaluated as {decision['risk_level']} ({prob*100:.2f}%), resolving in bounded decision {decision['action']}.{case_str}
"""
    
    return {
        "mode": "MOCK_LOCAL",
        "transaction_id": transaction_id,
        "risk_score": prob,
        "action": decision['action'],
        "timeline": timeline,
        "report": report
    }
