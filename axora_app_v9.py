"""
AXORA — Body Performance Analytics & Intelligent Classification System
Version 9.0 | Team Axora
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="AXORA | Body Performance",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────── Global Styles ───────────────────────
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;800&family=Inter:wght@300;400;500;600&display=swap');

    .stApp { background-color: #0B1621; color: #C8D8EA; }

    section[data-testid="stSidebar"] {
        background-color: #0D1B2A !important;
        border-right: 1px solid #1A3A5C;
    }

    h1 { color: #3B8BD4 !important; font-family: 'Orbitron', sans-serif !important; }
    h2, h3 { color: #FFFFFF !important; font-family: 'Inter', sans-serif !important; }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #0F2236 0%, #162840 100%);
        border: 1px solid #1E4D7B;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .metric-card .label {
        font-size: 0.78rem;
        color: #7FB3D3;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 4px;
        font-family: 'Inter', sans-serif;
    }
    .metric-card .value {
        font-size: 2rem;
        font-weight: 700;
        color: #4CC9F0;
        font-family: 'Orbitron', sans-serif;
        line-height: 1.1;
    }
    .metric-card .sub {
        font-size: 0.75rem;
        color: #5A8EA8;
        margin-top: 4px;
    }

    /* Section headers */
    .section-header {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 0.6rem 0;
        border-bottom: 1px solid #1E4D7B;
        margin-bottom: 1.2rem;
    }
    .section-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #4CC9F0;
        font-family: 'Inter', sans-serif;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    /* Table styling */
    .styled-table {
        width: 100%;
        border-collapse: collapse;
        font-family: 'Inter', sans-serif;
        font-size: 0.88rem;
        background: #0F2236;
        border-radius: 10px;
        overflow: hidden;
        margin-bottom: 1rem;
    }
    .styled-table thead tr {
        background: linear-gradient(90deg, #1A3A5C, #0D2340);
        color: #4CC9F0;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .styled-table th, .styled-table td {
        padding: 10px 14px;
        text-align: center;
        border-bottom: 1px solid #162840;
    }
    .styled-table tbody tr:hover { background-color: #132335; }
    .styled-table .best-row { background-color: #0D2D1A; color: #4ADE80; font-weight: 600; }
    .styled-table .highlight { color: #4CC9F0; font-weight: 600; }

    /* GitHub link button */
    .github-btn {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: linear-gradient(135deg, #1A3A5C, #0D2340);
        border: 1px solid #3B8BD4;
        border-radius: 8px;
        padding: 8px 16px;
        color: #4CC9F0 !important;
        text-decoration: none;
        font-size: 0.83rem;
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        transition: all 0.2s;
        margin-bottom: 1.2rem;
    }
    .github-btn:hover {
        background: linear-gradient(135deg, #1E4D7B, #162840);
        border-color: #4CC9F0;
    }

    /* Badge */
    .badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        font-family: 'Inter', sans-serif;
    }
    .badge-green  { background: #0D2D1A; color: #4ADE80; border: 1px solid #166534; }
    .badge-blue   { background: #0D1F3A; color: #60A5FA; border: 1px solid #1E4D7B; }
    .badge-purple { background: #1A0D3A; color: #C084FC; border: 1px solid #6B21A8; }

    /* Divider */
    .axora-divider {
        border: none;
        border-top: 1px solid #1E3A5C;
        margin: 1.5rem 0;
    }

    /* Info box */
    .info-box {
        background: #0F2236;
        border-left: 3px solid #3B8BD4;
        border-radius: 0 8px 8px 0;
        padding: 0.9rem 1.2rem;
        font-size: 0.85rem;
        color: #A8C8E8;
        margin-bottom: 1rem;
        font-family: 'Inter', sans-serif;
    }

    .stRadio [data-testid="stWidgetLabel"] { color: #3B8BD4; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

plt.style.use('dark_background')
AXORA_COLORS = ["#3B8BD4", "#9B51E0", "#4CC9F0", "#7209B7"]
sns.set_palette(sns.color_palette(AXORA_COLORS))

# ─────────────────────── Sidebar ───────────────────────
with st.sidebar:
    try:
        st.image("axora_team_logo.svg", width=150)
    except Exception:
        st.markdown("### 🧠 AXORA")
    st.markdown("### Navigation")
    menu = st.radio(
        "",
        ["🏠 Home", "📊 Data Overview", "📈 EDA", "🤖 Model Analysis", "🎯 Prediction"]
    )

    if menu == "🤖 Model Analysis":
        st.markdown("---")
        model_choice = st.selectbox(
            "Select Model to View",
            ["KNN", "Decision Tree", "SVM", "Neural Network", "Linear Regression"]
        )

# ═══════════════════════════════════════════════════════════════
# HOME
# ═══════════════════════════════════════════════════════════════
if menu == "🏠 Home":
    st.markdown('<div style="display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center;padding-top:4rem;">', unsafe_allow_html=True)
    try:
        st.image("axora_team_logo.svg", width=420)
    except Exception:
        pass
    st.markdown('<h1 style="font-size:3.5rem;margin-bottom:0;">AXORA</h1>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:1.2rem;opacity:0.8;color:#A8C8E8;">Intelligent Body Performance Analytics System</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# DATA OVERVIEW
# ═══════════════════════════════════════════════════════════════
elif menu == "📊 Data Overview":
    st.header("📋 Dataset Overview")
    st.write("Add dataset preview here.")

# ═══════════════════════════════════════════════════════════════
# EDA
# ═══════════════════════════════════════════════════════════════
elif menu == "📈 EDA":
    st.header("📊 Exploratory Data Analysis")
    st.write("Add EDA plots here.")

# ═══════════════════════════════════════════════════════════════
# MODEL ANALYSIS
# ═══════════════════════════════════════════════════════════════
elif menu == "🤖 Model Analysis":

    # ─────── DECISION TREE ───────
    if model_choice == "Decision Tree":
        st.markdown("## 🌳 Decision Tree — Model Analysis")

        # GitHub link
        st.markdown(
            '<a class="github-btn" href="https://github.com/ayaemad10/Body-Performance-Analytics-and-Intelligent/tree/main/5_MODEL%20_AI" target="_blank">'
            '📂 &nbsp; View Decision Tree Notebook on GitHub</a>',
            unsafe_allow_html=True
        )

        st.markdown('<div class="info-box">Optimal pruning via Cost-Complexity Pruning (CCP). Best alpha: <strong>0.000412</strong> — balances tree depth (13 levels, 419 nodes) against generalisation.</div>', unsafe_allow_html=True)

        # ── Classification ──
        st.markdown('<div class="section-header"><span class="section-title">📊 Classification — Body Class (A/B/C/D)</span></div>', unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown('<div class="metric-card"><div class="label">Best Test Accuracy</div><div class="value">70.24%</div><div class="sub">Best Alpha 0.000412</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="metric-card"><div class="label">CV Accuracy (5-Fold)</div><div class="value">70.52%</div><div class="sub">σ = 0.57%</div></div>', unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="metric-card"><div class="label">Tree Depth</div><div class="value">13</div><div class="sub">419 nodes</div></div>', unsafe_allow_html=True)
        with col4:
            st.markdown('<div class="metric-card"><div class="label">Macro F1</div><div class="value">0.70</div><div class="sub">Weighted avg</div></div>', unsafe_allow_html=True)

        # Per-class table
        st.markdown("#### Per-Class Classification Report")
        clf_data = {
            "Class": ["A", "B", "C", "D", "Macro Avg", "Weighted Avg"],
            "Precision": ["0.73", "0.57", "0.67", "0.86", "0.71", "0.71"],
            "Recall":    ["0.80", "0.59", "0.63", "0.79", "0.70", "0.70"],
            "F1-Score":  ["0.76", "0.58", "0.65", "0.83", "0.70", "0.70"],
            "Support":   ["837",  "836",  "837",  "837",  "3347", "3347"],
        }
        clf_df = pd.DataFrame(clf_data)
        best_idx = 3  # Class D

        rows_html = ""
        for i, row in clf_df.iterrows():
            row_class = "best-row" if row["Class"] == "D" else ""
            badge = '<span class="badge badge-green">Best</span> ' if row["Class"] == "D" else ""
            rows_html += f'<tr class="{row_class}"><td>{badge}{row["Class"]}</td><td class="highlight">{row["Precision"]}</td><td>{row["Recall"]}</td><td>{row["F1-Score"]}</td><td>{row["Support"]}</td></tr>'

        st.markdown(f"""
        <table class="styled-table">
          <thead><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1-Score</th><th>Support</th></tr></thead>
          <tbody>{rows_html}</tbody>
        </table>""", unsafe_allow_html=True)

        # Split experiments
        st.markdown("#### Split Ratio Experiments")
        split_data = {
            "Split (Test %)": ["20%", "30%", "50%"],
            "Train Accuracy": ["76.29%", "77.43%", "82.55%"],
            "Test Accuracy":  ["70.27%", "70.52%", "66.85%"],
        }
        split_rows = ""
        for i, (s, tr, te) in enumerate(zip(split_data["Split (Test %)"], split_data["Train Accuracy"], split_data["Test Accuracy"])):
            highlight = "best-row" if s == "30%" else ""
            split_rows += f'<tr class="{highlight}"><td>{s}</td><td>{tr}</td><td class="highlight">{te}</td></tr>'
        st.markdown(f"""
        <table class="styled-table">
          <thead><tr><th>Split (Test %)</th><th>Train Accuracy</th><th>Test Accuracy</th></tr></thead>
          <tbody>{split_rows}</tbody>
        </table>""", unsafe_allow_html=True)

        st.markdown('<hr class="axora-divider">', unsafe_allow_html=True)

        # ── Regression ──
        st.markdown('<div class="section-header"><span class="section-title">📈 Regression — Broad Jump Distance (cm)</span></div>', unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown('<div class="metric-card"><div class="label">Best R² Score</div><div class="value">0.693</div><div class="sub">Best Alpha 16.18</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="metric-card"><div class="label">RMSE</div><div class="value">22.07</div><div class="sub">cm</div></div>', unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="metric-card"><div class="label">MSE</div><div class="value">486.97</div><div class="sub">cm²</div></div>', unsafe_allow_html=True)
        with col4:
            st.markdown('<div class="metric-card"><div class="label">Tree Depth</div><div class="value">3</div><div class="sub">13 nodes</div></div>', unsafe_allow_html=True)

        st.markdown("#### Regression Split Experiments")
        rsplit_data = {
            "Split (Test %)": ["20%", "30%", "50%"],
            "Train R²": ["0.6943", "0.6939", "0.6890"],
            "Test R²":  ["0.6930", "0.6993", "0.7038"],
        }
        rsplit_rows = ""
        for s, tr, te in zip(rsplit_data["Split (Test %)"], rsplit_data["Train R²"], rsplit_data["Test R²"]):
            highlight = "best-row" if s == "50%" else ""
            rsplit_rows += f'<tr class="{highlight}"><td>{s}</td><td>{tr}</td><td class="highlight">{te}</td></tr>'
        st.markdown(f"""
        <table class="styled-table">
          <thead><tr><th>Split (Test %)</th><th>Train R²</th><th>Test R²</th></tr></thead>
          <tbody>{rsplit_rows}</tbody>
        </table>""", unsafe_allow_html=True)

        st.markdown("#### Key Hyperparameters Used")
        st.markdown("""
        <table class="styled-table">
          <thead><tr><th>Hyperparameter</th><th>Classification Value</th><th>Regression Value</th></tr></thead>
          <tbody>
            <tr><td>Algorithm</td><td class="highlight">CART (DecisionTreeClassifier)</td><td class="highlight">CART (DecisionTreeRegressor)</td></tr>
            <tr><td>Pruning (alpha)</td><td>0.000412</td><td>16.175524</td></tr>
            <tr><td>Best Test Accuracy / R²</td><td>70.24%</td><td>0.6933</td></tr>
            <tr><td>Tree Depth</td><td>13</td><td>3</td></tr>
            <tr><td>Node Count</td><td>419</td><td>13</td></tr>
            <tr><td>CV Strategy</td><td colspan="2">5-Fold Cross Validation</td></tr>
          </tbody>
        </table>""", unsafe_allow_html=True)

    # ─────── SVM ───────
    elif model_choice == "SVM":
        st.markdown("## ⚡ SVM — Model Analysis")

        # GitHub link
        st.markdown(
            '<a class="github-btn" href="https://github.com/ayaemad10/Body-Performance-Analytics-and-Intelligent/tree/main/5_MODEL%20_AI" target="_blank">'
            '📂 &nbsp; View SVM Notebook on GitHub</a>',
            unsafe_allow_html=True
        )

        st.markdown('<div class="info-box">Best kernel: <strong>RBF</strong> with C=1, γ=0.1. Grid search over C∈{1,10} × γ∈{0.1,0.01}. Results shown for 70/30 split (highest accuracy) and 50/50 split for comparison.</div>', unsafe_allow_html=True)

        # ── Classification ──
        st.markdown('<div class="section-header"><span class="section-title">📊 Classification — Body Class (A/B/C/D)</span></div>', unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown('<div class="metric-card"><div class="label">Best Test Accuracy</div><div class="value">84.67%</div><div class="sub">70/30 split · RBF kernel</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="metric-card"><div class="label">CV Best Score</div><div class="value">77.58%</div><div class="sub">3-Fold Grid Search CV</div></div>', unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="metric-card"><div class="label">Macro F1</div><div class="value">0.85</div><div class="sub">Weighted avg</div></div>', unsafe_allow_html=True)
        with col4:
            st.markdown('<div class="metric-card"><div class="label">Improvement vs Basic</div><div class="value">+19.3%</div><div class="sub">Linear → RBF tuned</div></div>', unsafe_allow_html=True)

        # Per-class table – best split (70/30)
        st.markdown("#### Per-Class Report — 70/30 Split (Best)")
        clf_data = {
            "Class": ["A", "B", "C", "D", "Macro Avg", "Weighted Avg"],
            "Precision": ["0.82", "0.88", "0.82", "0.88", "0.85", "0.85"],
            "Recall":    ["0.83", "0.79", "0.91", "0.87", "0.85", "0.85"],
            "F1-Score":  ["0.82", "0.83", "0.86", "0.87", "0.85", "0.85"],
            "Support":   ["75",   "75",   "75",   "75",   "300",  "300"],
        }
        clf_df = pd.DataFrame(clf_data)
        rows_html = ""
        for _, row in clf_df.iterrows():
            is_best = row["Class"] == "D"
            row_class = "best-row" if is_best else ""
            badge = '<span class="badge badge-green">Best F1</span> ' if is_best else ""
            rows_html += f'<tr class="{row_class}"><td>{badge}{row["Class"]}</td><td class="highlight">{row["Precision"]}</td><td>{row["Recall"]}</td><td>{row["F1-Score"]}</td><td>{row["Support"]}</td></tr>'

        st.markdown(f"""
        <table class="styled-table">
          <thead><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1-Score</th><th>Support</th></tr></thead>
          <tbody>{rows_html}</tbody>
        </table>""", unsafe_allow_html=True)

        # Model comparison table
        st.markdown("#### Model Comparison Across Splits")
        st.markdown("""
        <table class="styled-table">
          <thead><tr><th>Split</th><th>Basic SVM (Linear)</th><th>Tuned SVM (RBF)</th><th>Improvement</th></tr></thead>
          <tbody>
            <tr class="best-row"><td>70 / 30</td><td>65.33%</td><td class="highlight">84.67%</td><td>+19.33%</td></tr>
            <tr><td>50 / 50</td><td>60.80%</td><td class="highlight">79.60%</td><td>+18.80%</td></tr>
          </tbody>
        </table>""", unsafe_allow_html=True)

        st.markdown('<hr class="axora-divider">', unsafe_allow_html=True)

        # ── Regression ──
        st.markdown('<div class="section-header"><span class="section-title">📈 Regression — Broad Jump Distance (cm)</span></div>', unsafe_allow_html=True)

        st.markdown('<div class="info-box">SVR (RBF) compared against Random Forest and Gradient Boosting. Best overall model: <strong>Gradient Boosting</strong> (R² 0.7953). SVR-RBF achieves competitive R² 0.7925.</div>', unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown('<div class="metric-card"><div class="label">SVR R² Score</div><div class="value">0.793</div><div class="sub">RBF kernel</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="metric-card"><div class="label">SVR MAE</div><div class="value">13.56</div><div class="sub">cm avg error</div></div>', unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="metric-card"><div class="label">SVR RMSE</div><div class="value">18.17</div><div class="sub">cm</div></div>', unsafe_allow_html=True)
        with col4:
            st.markdown('<div class="metric-card"><div class="label">Best Model R²</div><div class="value">0.795</div><div class="sub">Gradient Boosting</div></div>', unsafe_allow_html=True)

        st.markdown("#### Regression Model Comparison")
        st.markdown("""
        <table class="styled-table">
          <thead><tr><th>Model</th><th>R² Score</th><th>MAE (cm)</th><th>RMSE (cm)</th></tr></thead>
          <tbody>
            <tr class="best-row"><td>🏆 Gradient Boosting</td><td class="highlight">0.7953</td><td>13.56</td><td>18.05</td></tr>
            <tr><td>SVR (RBF)</td><td>0.7925</td><td>13.56</td><td>18.17</td></tr>
            <tr><td>Random Forest</td><td>0.7900</td><td>13.76</td><td>18.28</td></tr>
          </tbody>
        </table>""", unsafe_allow_html=True)

        st.markdown("#### Top Feature Importances (Gradient Boosting)")
        st.markdown("""
        <table class="styled-table">
          <thead><tr><th>Feature</th><th>Importance</th></tr></thead>
          <tbody>
            <tr class="best-row"><td>Gender</td><td class="highlight">42.3%</td></tr>
            <tr><td>Sit-ups Count</td><td>20.2%</td></tr>
            <tr><td>Grip Force</td><td>18.8%</td></tr>
            <tr><td>Age</td><td>7.4%</td></tr>
            <tr><td>Body Fat %</td><td>4.4%</td></tr>
          </tbody>
        </table>""", unsafe_allow_html=True)

        st.markdown("#### Key Hyperparameters Used")
        st.markdown("""
        <table class="styled-table">
          <thead><tr><th>Hyperparameter</th><th>Classification</th><th>Regression (SVR)</th></tr></thead>
          <tbody>
            <tr><td>Kernel</td><td class="highlight">RBF</td><td class="highlight">RBF</td></tr>
            <tr><td>C</td><td>1</td><td>1</td></tr>
            <tr><td>Gamma (γ)</td><td>0.1</td><td>0.1</td></tr>
            <tr><td>Best CV Accuracy / R²</td><td>77.58%</td><td>0.7925</td></tr>
            <tr><td>Search Strategy</td><td colspan="2">Grid Search (3-Fold CV)</td></tr>
            <tr><td>Features</td><td>20</td><td>14</td></tr>
          </tbody>
        </table>""", unsafe_allow_html=True)

    # ─────── Other models (placeholder) ───────
    else:
        st.header(f"Model Performance: {model_choice}")
        st.info("Analysis for this model coming soon.")

# ═══════════════════════════════════════════════════════════════
# PREDICTION
# ═══════════════════════════════════════════════════════════════
elif menu == "🎯 Prediction":
    st.header("🔮 Intelligent Prediction")
    st.write("Add prediction inputs here.")
