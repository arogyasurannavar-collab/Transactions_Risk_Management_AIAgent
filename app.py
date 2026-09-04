import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import sys
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agent.agent import investigate_transaction
from agent.decision_engine import determine_action
from agent.tools import run_fraud_model, get_shap_explanation

st.set_page_config(
    page_title="RazorGuard | AI Transaction Risk Investigation Agent",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .main-title {
        font-size: 2.6rem;
        font-weight: 700;
        background: linear-gradient(90deg, #10b981, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    
    .subtitle {
        color: #6b7280;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 1.2rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        border: 1px solid #e5e7eb;
        text-align: center;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 0.2rem;
    }
    
    .metric-label {
        font-size: 0.9rem;
        font-weight: 600;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .status-badge {
        padding: 0.3rem 0.8rem;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
    }
    
    .status-allow {
        background-color: #d1fae5;
        color: #065f46;
    }
    
    .status-verify {
        background-color: #fef3c7;
        color: #92400e;
    }
    
    .status-review {
        background-color: #fee2e2;
        color: #991b1b;
    }
    
    .agent-thought {
        background-color: #f3f4f6;
        border-left: 4px solid #10b981;
        padding: 0.8rem 1.2rem;
        border-radius: 0 8px 8px 0;
        font-style: italic;
        margin: 0.8rem 0;
    }
    
    .agent-tool-call {
        background-color: #eff6ff;
        border-left: 4px solid #3b82f6;
        padding: 0.8rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin: 0.8rem 0;
    }
</style>
""", unsafe_allow_html=True)

DATA_PATH = 'data/transactions_engineered.csv'
CASES_PATH = 'data/review_cases.csv'

def check_data_files():
    if not os.path.exists(DATA_PATH):
        st.error("Error: `data/transactions_engineered.csv` is missing. Please run `generate_dataset.py` and `ml/feature_engineering.py` first.")
        st.stop()

check_data_files()

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df

df = load_data()

with st.sidebar:
    st.image("https://img.icons8.com/color/96/shield.png", width=65)
    st.markdown("## **RazorGuard Control**")
    st.markdown("AI Transaction Risk Investigation & Management System")
    st.write("---")
    
    # Navigation
    page = st.radio(
        "Navigation",
        ["📊 Dashboard", "🔍 Investigate Transaction", "💼 Review Cases (Human-in-the-Loop)", "📈 Model Performance"]
    )
    
    st.write("---")
    st.markdown("### **LLM Reasoning Layer**")
    api_key = st.text_input("Gemini API Key", type="password", help="Enter your Google Gemini API Key to enable live Agentic tool calling and reasoning. If empty, the app runs in local simulation mode.")
    
    if api_key:
        st.success("Gemini API Key configured! Running in LIVE Agent mode.")
    else:
        st.info("No API Key entered. Running in LOCAL Simulated Agent mode (Fully functional offline).")
        
    st.write("---")
    st.caption("Developed for Razorpay Buildathon | Aug 2026")

if page == "📊 Dashboard":
    st.markdown('<div class="main-title">Fraud Monitoring Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Real-time payment transaction metrics and risk statistics</div>', unsafe_allow_html=True)
    
    num_pending_cases = 0
    if os.path.exists(CASES_PATH):
        df_cases = pd.read_csv(CASES_PATH)
        num_pending_cases = len(df_cases[df_cases['status'] == 'PENDING'])
    
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(df):,}</div>
            <div class="metric-label">Total Transactions</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        fraud_count = df['is_fraud'].sum()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color: #ef4444;">{fraud_count:,}</div>
            <div class="metric-label">Fraud Detected (ML)</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        fraud_rate = (fraud_count / len(df)) * 100
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{fraud_rate:.2f}%</div>
            <div class="metric-label">Fraud Rate</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color: #f59e0b;">{num_pending_cases}</div>
            <div class="metric-label">Pending Reviews</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.write("")
    st.write("")
    
    ch1, ch2 = st.columns([2, 1])
    with ch1:
        st.subheader("Transaction Volume over Time")
        df_time = df.copy()
        df_time['timestamp'] = pd.to_datetime(df_time['timestamp'])
        df_time_grouped = df_time.resample('D', on='timestamp').size().reset_index(name='Transaction Count')
        
        fig_vol = px.line(
            df_time_grouped, x='timestamp', y='Transaction Count',
            color_discrete_sequence=['#3b82f6'],
            labels={'timestamp': 'Date', 'Transaction Count': 'Volume'}
        )
        fig_vol.update_layout(height=350, margin=dict(l=20, r=20, t=10, b=20), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_vol, use_container_width=True)
        
    with ch2:
        st.subheader("Payment Method Distribution")
        df_pay = df.groupby('payment_method').size().reset_index(name='count')
        fig_pay = px.pie(
            df_pay, values='count', names='payment_method',
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_pay.update_layout(height=350, margin=dict(l=20, r=20, t=10, b=20))
        st.plotly_chart(fig_pay, use_container_width=True)
        
    st.write("---")
    
    st.subheader("Latest Transactions")
    recent_cols = ['transaction_id', 'customer_id', 'amount', 'timestamp', 'payment_method', 'transaction_location', 'is_fraud']
    df_recent = df[recent_cols].tail(10).iloc[::-1]
    
    formatted_recent = df_recent.copy()
    formatted_recent['amount'] = formatted_recent['amount'].apply(lambda x: f"INR {x:,.2f}")
    
    st.dataframe(
        formatted_recent,
        column_config={
            "transaction_id": "Transaction ID",
            "customer_id": "Customer ID",
            "amount": "Amount",
            "timestamp": "Timestamp",
            "payment_method": "Method",
            "transaction_location": "Location",
            "is_fraud": st.column_config.CheckboxColumn("Fraud flagged?")
        },
        use_container_width=True,
        hide_index=True
    )

elif page == "🔍 Investigate Transaction":
    st.markdown('<div class="main-title">AI Transaction Risk Investigator</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Select or input a transaction to launch the AI risk agent investigation</div>', unsafe_allow_html=True)
    
    col_input, col_info = st.columns([1, 2])
    with col_input:
        st.subheader("Select Transaction")
        story_tx = "TX10091" 
        high_risk_sample = df[df['is_fraud'] == 1]['transaction_id'].iloc[5]
        low_risk_sample = df[df['is_fraud'] == 0]['transaction_id'].iloc[5]
        
        tx_options = [story_tx, high_risk_sample, low_risk_sample, "Custom Search..."]
        selected_option = st.selectbox(
            "Quick Select Test Case",
            tx_options,
            index=0,
            help="Select one of our preset transactions to run, or choose 'Custom Search' to enter a specific Transaction ID."
        )
        
        if selected_option == "Custom Search...":
            tx_id_input = st.text_input("Enter Transaction ID (e.g. TX10002)", value="TX10091")
        else:
            tx_id_input = selected_option
            
        investigate_btn = st.button("RUN AI RISK AGENT INVESTIGATION", type="primary")
        
    with col_info:
        st.info("""
        💡 **How to Test the Story:**
        1. Select **`TX10091`** (this is our Buildathon Story transaction).
        2. Click the **Run Investigation** button.
        3. Observe how the AI Risk Agent executes multiple sequential tools, generates risk thresholds, maps SHAP values to natural explanations, and registers a human-in-the-loop review ticket.
        """)

    if investigate_btn or tx_id_input:
        st.write("---")
        with st.spinner(f"Agent starting investigation on {tx_id_input}..."):
            res = investigate_transaction(tx_id_input, api_key=api_key)
            
        if "error" in res:
            st.error(f"Error: {res['error']}")
        else:
            if "warning" in res:
                st.warning(res['warning'])
                
            st.subheader(f"Investigation Results for {tx_id_input}")
            
            r1, r2, r3 = st.columns([1, 1, 2])
            with r1:
                risk_pct = res['risk_score'] * 100
                st.metric(
                    label="Model Risk Score",
                    value=f"{risk_pct:.2f}%",
                    delta=f"{risk_pct - 2.0:.2f}% vs system base",
                    delta_color="inverse"
                )
                
            with r2:
                action = res['action']
                action_desc = determine_action(res['risk_score'])['description']
                badge_class = "status-allow"
                if action == "VERIFY":
                    badge_class = "status-verify"
                elif action == "HUMAN REVIEW":
                    badge_class = "status-review"
                    
                st.markdown(f"**Recommended Action:**")
                st.markdown(f'<div class="status-badge {badge_class}" style="font-size: 1.1rem; padding: 0.5rem 1rem;">{action}</div>', unsafe_allow_html=True)
                st.caption(action_desc)
                
            with r3:
                tx_details = df[df['transaction_id'] == tx_id_input].iloc[0]
                st.markdown(f"""
                - **Customer ID**: `{tx_details['customer_id']}`
                - **Amount**: `INR {tx_details['amount']:,.2f}`
                - **Location**: `{tx_details['transaction_location']}` (Home: `{tx_details['customer_location']}`)
                - **Device**: `{tx_details['device_id']}` (New Device: `{"YES" if tx_details['is_new_device'] else "NO"}`)
                """)
                
            st.write("---")
            
            col_timeline, col_shap = st.columns([1, 1])
            with col_timeline:
                st.subheader("🤖 AI Agent Execution Log (Timeline)")
                st.write("Below is the step-by-step trace showing the Agent's decision path, tool invocations, and observations:")
                
                for idx, step in enumerate(res['timeline']):
                    if step['type'] == 'thought':
                        st.markdown(f"""
                        <div class="agent-thought">
                            <strong>Step {idx+1}: Agent Reasoning</strong><br>
                            {step['content']}
                        </div>
                        """, unsafe_allow_html=True)
                    elif step['type'] == 'call':
                        st.markdown(f"""
                        <div class="agent-tool-call">
                            <strong>🛠️ Tool Call: <code>{step['name']}()</code></strong><br>
                            Arguments: <code>{step['args']}</code>
                        </div>
                        """, unsafe_allow_html=True)
                    elif step['type'] == 'response':
                        with st.expander(f"📥 Response from {step['name']}()", expanded=False):
                            st.json(step['response'])
                            
            with col_shap:
                st.subheader("🔍 Explainable AI (SHAP Contributions)")
                st.write("SHAP values explain why the XGBoost model predicted this risk score, showing which features pushed the risk up (red) or down (blue):")
                
                shap_res = get_shap_explanation(tx_id_input)
                
                if "error" not in shap_res:
                    sh_vals = shap_res['shap_values']
                    sorted_shap = sorted(sh_vals.items(), key=lambda x: abs(x[1]), reverse=True)
                    top_shap = sorted_shap[:7] 
                    
                    features = [k for k, v in top_shap]
                    values = [v for k, v in top_shap]
                    colors = ['#ef4444' if v > 0 else '#3b82f6' for v in values] # Red for risk increase, Blue for decrease
                    
                    fig_shap = go.Figure(go.Bar(
                        x=values,
                        y=features,
                        orientation='h',
                        marker_color=colors,
                        text=[f"+{v:.2f}" if v > 0 else f"{v:.2f}" for v in values],
                        textposition='auto'
                    ))
                    
                    fig_shap.update_layout(
                        xaxis_title="SHAP Value (Log-odds Impact)",
                        yaxis=dict(autorange="reversed"),
                        margin=dict(l=20, r=20, t=10, b=20),
                        height=350,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)'
                    )
                    st.plotly_chart(fig_shap, use_container_width=True)
                    
                    st.success(f"**Explainable AI Context:**\n{shap_res['human_explanation']}")
                else:
                    st.error("Could not load SHAP details.")
                    
            st.write("---")
            
            st.subheader("📄 AI Risk Investigation Report")
            st.markdown(res['report'])

elif page == "💼 Review Cases (Human-in-the-Loop)":
    st.markdown('<div class="main-title">Human-in-the-Loop Review Center</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Specialized space for risk analysts to review and make decisions on high-risk transaction hold alerts</div>', unsafe_allow_html=True)
    
    if not os.path.exists(CASES_PATH):
        df_cases = pd.DataFrame(columns=[
            'case_id', 'transaction_id', 'amount', 'risk_score', 
            'reasoning', 'recommendation', 'status', 'human_decision', 'created_at'
        ])
        df_cases.to_csv(CASES_PATH, index=False)
    else:
        df_cases = pd.read_csv(CASES_PATH)
        
    pending_cases = df_cases[df_cases['status'] == 'PENDING']
    resolved_cases = df_cases[df_cases['status'] == 'RESOLVED']
    
    tab_pending, tab_resolved = st.tabs([f"Pending Cases ({len(pending_cases)})", f"Resolved Cases ({len(resolved_cases)})"])
    
    with tab_pending:
        if len(pending_cases) == 0:
            st.info("🎉 Excellent work! No pending high-risk review cases found.")
        else:
            st.write("Select a case from the table below to review details and take action:")
            
            st.dataframe(
                pending_cases[['case_id', 'transaction_id', 'amount', 'risk_score', 'recommendation', 'created_at']],
                column_config={
                    "case_id": "Case ID",
                    "transaction_id": "Transaction ID",
                    "amount": "Amount",
                    "risk_score": st.column_config.ProgressColumn("Model Risk Score", format="%.2f", min_value=0, max_value=1),
                    "recommendation": "Agent Reco",
                    "created_at": "Alert Time"
                },
                use_container_width=True,
                hide_index=True
            )
            
            st.write("")
            st.markdown("### **Execute Human Decision**")
            
            case_to_resolve = st.selectbox(
                "Select Case ID to Resolve",
                pending_cases['case_id'].tolist()
            )
            
            if case_to_resolve:
                case_row = pending_cases[pending_cases['case_id'] == case_to_resolve].iloc[0]
                
                col_d1, col_d2 = st.columns(2)
                with col_d1:
                    st.write(f"**Case ID**: `{case_to_resolve}`")
                    st.write(f"**Transaction ID**: `{case_row['transaction_id']}`")
                    st.write(f"**Amount**: `INR {case_row['amount']:,.2f}`")
                    st.write(f"**Risk Probability**: `{case_row['risk_score']*100:.2f}%`")
                with col_d2:
                    st.write(f"**AI Agent Reasoning**:")
                    st.warning(case_row['reasoning'])
                    
                col_btn1, col_btn2, _ = st.columns([1, 1, 3])
                
                with col_btn1:
                    approve_btn = st.button("✅ APPROVE (Allow Payment)", use_container_width=True)
                with col_btn2:
                    reject_btn = st.button("❌ REJECT (Block Fraud)", use_container_width=True)
                    
                if approve_btn or reject_btn:
                    decision = "APPROVED" if approve_btn else "REJECTED"
                    
                    df_cases.loc[df_cases['case_id'] == case_to_resolve, 'status'] = 'RESOLVED'
                    df_cases.loc[df_cases['case_id'] == case_to_resolve, 'human_decision'] = decision
                    df_cases.to_csv(CASES_PATH, index=False)
                    
                    st.success(f"Case {case_to_resolve} resolved! Decision: **{decision}**.")
                    st.rerun()
                    
    with tab_resolved:
        if len(resolved_cases) == 0:
            st.info("No resolved cases in history yet.")
        else:
            st.dataframe(
                resolved_cases[['case_id', 'transaction_id', 'amount', 'risk_score', 'human_decision', 'created_at']],
                column_config={
                    "case_id": "Case ID",
                    "transaction_id": "Transaction ID",
                    "amount": "Amount",
                    "risk_score": "Risk Score",
                    "human_decision": "Decision",
                    "created_at": "Reviewed At"
                },
                use_container_width=True,
                hide_index=True
            )
            
            st.write("")
            st.subheader("Human Validation Analytics")
            
            total_resolved = len(resolved_cases)
            approvals = len(resolved_cases[resolved_cases['human_decision'] == 'APPROVED'])
            rejections = len(resolved_cases[resolved_cases['human_decision'] == 'REJECTED'])
            
            c_s1, c_s2, c_s3 = st.columns(3)
            with c_s1:
                st.metric("Total human audits", total_resolved)
            with c_s2:
                st.metric("Fraud blocks confirmed", rejections)
            with c_s3:
                st.metric("False alarms released", approvals, help="High-risk triggers overridden by analysts to allow legitimate customer transactions.")

elif page == "📈 Model Performance":
    st.markdown('<div class="main-title">Model Diagnostics & Performance</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Detailed evaluation metrics and data distributions for the RazorGuard classifier</div>', unsafe_allow_html=True)
    
    st.write("Our XGBoost classification model is evaluated on a stratified test subset (10,000 transactions).")
    
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric(label="Precision Score", value="98.05%", help="Of all flagged transactions, how many were actually fraud")
    with m2:
        st.metric(label="Recall Score (FDR)", value="99.50%", help="Of all actual fraud, how many did we catch")
    with m3:
        st.metric(label="F1 Score", value="98.77%", help="Harmonic mean of precision and recall")
    with m4:
        st.metric(label="False Positive Rate (FPR)", value="0.04%", help="Percentage of legitimate transactions blocked")
        
    st.write("---")
    
    ch_col1, ch_col2 = st.columns([1, 1])
    with ch_col1:
        st.subheader("Confusion Matrix")
        z = [[9794, 4], [1, 201]]
        x = ['Predicted Legit', 'Predicted Fraud']
        y = ['Actual Legit', 'Actual Fraud']
        
        txt = [['TN: 9,794', 'FP: 4'], ['FN: 1', 'TP: 201']]
        
        fig_cm = px.imshow(
            z, x=x, y=y, 
            color_continuous_scale='Blues',
            text_auto=True
        )
        fig_cm.update_layout(
            margin=dict(l=20, r=20, t=10, b=20),
            height=300,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_cm, use_container_width=True)
        st.caption("FPR of 0.04% is highly optimized, ensuring only 4 out of 9,798 legitimate customers are flagged.")
        
    with ch_col2:
        st.subheader("Top Feature Importance Drivers")
        features = [
            "device_age_days",
            "amount_ratio",
            "transactions_last_10min",
            "customer_avg_amount",
            "location_mismatch",
            "failed_attempts_last_10min"
        ]
        importances = [0.7300, 0.1859, 0.0739, 0.0064, 0.0021, 0.0015]
        
        fig_imp = px.bar(
            x=importances, y=features, orientation='h',
            labels={'x': 'Relative Importance', 'y': 'Feature'},
            color_discrete_sequence=['#3b82f6']
        )
        fig_imp.update_layout(
            yaxis=dict(autorange="reversed"),
            margin=dict(l=20, r=20, t=10, b=20),
            height=300,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_imp, use_container_width=True)
        st.caption("Device age is the single highest predictor of risk, reflecting the new device fraud pattern.")
