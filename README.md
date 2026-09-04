# RazorGuard: AI Transaction Risk Investigation Agent 🛡️

**RazorGuard** is an AI-powered transaction risk management system built for the Razorpay buildathon. It combines behavioral fraud detection, explainable machine learning (XGBoost + SHAP), and tool-using AI (Gemini Agent) to investigate suspicious payments and recommend bounded actions while keeping humans in the loop for high-risk decisions.

---

## 🏗️ Project Architecture

```
                        PAYMENT TRANSACTION
                                |
                                v
                       +------------------+
                       |   AI RISK AGENT  |
                       +--------+---------+
                                |
              +-----------------+-----------------+
              |                 |                 |
              v                 v                 v
       Customer History    Device Check     Velocity Check
              |                 |                 |
              +-----------------+-----------------+
                                |
                                v
                       Feature Engineering
                                |
                                v
                           XGBoost Model
                                |
                                v
                       Fraud Probability
                                |
                                v
                              SHAP
                                |
                                v
                        AI Reasoning
                                |
                                v
                        Risk Assessment
                                |
                 +--------------+--------------+
                 |              |              |
                 v              v              v
               ALLOW         VERIFY         REVIEW
                                               |
                                               v
                                        Human Decision
                                               |
                                               v
                                         Store Result
                                               |
                                               v
                                          Monitoring
```

---

## 📂 Project Folder Structure

- `data/`
  - `transactions.csv`: Raw synthetic transaction dataset (50,000 rows).
  - `transactions_engineered.csv`: Dataset with engineered risk and categorical features.
  - `review_cases.csv`: Database tracking AI recommendations and human review decisions.
- `models/`
  - `fraud_model.pkl`: Trained XGBoost model with feature index lists.
- `ml/`
  - `feature_engineering.py`: Computes risk ratios, location checks, and velocities.
  - `train.py`: Trains the stratified XGBoost model.
  - `evaluate.py`: Generates confusion matrix and precision/recall stats.
  - `predict.py`: Runs predictions and calculates SHAP value explanations.
- `agent/`
  - `agent.py`: Orchestrates live Gemini tool-calling and offline local loops.
  - `tools.py`: Exposes database and model prediction helper functions.
  - `prompts.py`: Holds instructions, rules, and report format rules.
  - `decision_engine.py`: Encapsulates static threshold bounds (ALLOW / VERIFY / REVIEW).
- `app.py`: Streamlit-based web dashboard.
- `generate_dataset.py`: Python synthetic dataset generator.
- `explore_data.py`: Basic data inspection script.
- `requirements.txt`: Python package requirements list.

---

## ⚡ Quick Start

### 1. Prerequisites
Ensure you have Python 3.11+ installed. Run the command to install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Generate Data and Train Model
Run these steps sequentially to setup your local workspace:
```bash
# Generate the synthetic dataset (50,000 rows)
python generate_dataset.py

# Compute engineered features
python ml/feature_engineering.py

# Train the XGBoost model
python ml/train.py

# Run diagnostics and view metrics
python ml/evaluate.py
```

### 3. Launch Dashboard
Start the Streamlit dashboard by running:
```bash
streamlit run app.py
```

---

## 🎭 This agent Demo (`TX10091`)

To demonstrate the power of risk management live:
1. Navigate to the **Investigate Transaction** tab on the left sidebar.
2. Select **`TX10091`** from the Quick Select dropdown list.
3. Click **RUN AI RISK AGENT INVESTIGATION**.
4. The system will demonstrate:
   - **Agent Tool Execution**: The agent automatically queries customer average spending (`INR 2,500.00`), current transaction amount (`INR 85,000.00`), device status (`New Device D9182`), velocity (`12 transactions/10min`), and failed attempts (`4`).
   - **XGBoost Risk Prediction**: Calculated fraud probability is **99.98%**.
   - **SHAP Explanation**: An interactive Plotly chart explains that the risk was driven by velocity spikes, the new device, and the spending ratio.
   - **Decision Engine Boundary**: The risk exceeds the 70% threshold, mapping to **HUMAN REVIEW** and creating a support case ticket `CASE10001`.
5. Go to the **Review Cases** tab:
   - Select Case ID `CASE10001` and review the AI-provided evidence summary.
   - Click **REJECT (Block Fraud)** to resolve the ticket as a human risk analyst.
   - Observe how the **Human Validation Analytics** dashboard instantly logs and metrics the decision.

replace the razoguard to Transfer ai risk management
