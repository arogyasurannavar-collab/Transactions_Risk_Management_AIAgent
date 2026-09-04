import os

def determine_action(risk_score: float) -> dict:
    
    percentage_score = risk_score * 100
    
    if percentage_score < 40.0:
        return {
            "action": "ALLOW",
            "risk_level": "LOW",
            "threshold_range": "0% - 39.9%",
            "color": "green",
            "description": "Transaction approved. Risk score is within normal boundaries."
        }
    elif percentage_score < 70.0:
        return {
            "action": "VERIFY",
            "risk_level": "MEDIUM",
            "threshold_range": "40% - 69.9%",
            "color": "orange",
            "description": "Verification required. Trigger secondary authentication (OTP/3DS)."
        }
    else:
        return {
            "action": "HUMAN REVIEW",
            "risk_level": "HIGH",
            "threshold_range": "70% - 100%",
            "color": "red",
            "description": "Transaction held for human review. Risk score exceeds acceptable thresholds."
        }
