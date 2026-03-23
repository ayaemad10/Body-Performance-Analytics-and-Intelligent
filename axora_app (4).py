"""
╔══════════════════════════════════════════════════════════════════╗
║   AXORA — Body Performance Analytics & Intelligent System        ║
║   Streamlit Application  |  5 Models  |  Team Axora              ║
╚══════════════════════════════════════════════════════════════════╝
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io, base64, time

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.svm import SVC, SVR
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    mean_squared_error, r2_score, mean_absolute_error
)

# ─────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Axora — Body Performance Analytics",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────
# THEME & CSS
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global ── */
[data-testid="stAppViewContainer"] { background: #0D1B2A; }
[data-testid="stSidebar"]          { background: #0A1628; border-right: 1px solid #1D4570; }
[data-testid="stSidebar"] *        { color: #C8D8EA !important; }
h1,h2,h3,h4,h5,h6 { color: #3B8BD4 !important; }
p, li, label, .stMarkdown { color: #C8D8EA !important; }

/* ── Cards ── */
.axora-card {
    background: #0F1E33;
    border: 1px solid #1D4570;
    border-radius: 14px;
    padding: 22px 26px;
    margin-bottom: 18px;
}
.metric-box {
    background: linear-gradient(135deg,#112640,#0F1E33);
    border: 1px solid #1D9E75;
    border-radius: 10px;
    padding: 16px;
    text-align: center;
    margin: 6px;
}
.metric-val { font-size: 2rem; font-weight: 700; color: #3B8BD4; }
.metric-lbl { font-size: 0.78rem; color: #85B7EB; text-transform: uppercase; letter-spacing: 1px; }

/* ── Header ── */
.axora-header {
    background: linear-gradient(135deg, #0D1B2A 0%, #112640 60%, #0F1E33 100%);
    border: 1px solid #534AB7;
    border-radius: 18px;
    padding: 28px 36px;
    margin-bottom: 28px;
    display: flex;
    align-items: center;
    gap: 28px;
}
.axora-title { font-size: 2.1rem; font-weight: 800; color: #3B8BD4; letter-spacing: 1px; }
.axora-sub   { color: #85B7EB; font-size: 0.95rem; }

/* ── Badges ── */
.badge-blue   { background:#1D4570; color:#85B7EB; border-radius:8px; padding:3px 10px; font-size:0.78rem; display:inline-block; margin:2px; }
.badge-green  { background:#0D3D2A; color:#1D9E75; border-radius:8px; padding:3px 10px; font-size:0.78rem; display:inline-block; margin:2px; }
.badge-purple { background:#1E1A40; color:#AFA9EC; border-radius:8px; padding:3px 10px; font-size:0.78rem; display:inline-block; margin:2px; }

/* ── Tables ── */
.dataframe { background:#0F1E33 !important; color:#C8D8EA !important; }
thead th   { background:#1F5C9E !important; color:#fff !important; }
tbody tr:nth-child(even) { background:#112640 !important; }

/* ── Tabs ── */
[data-testid="stTab"] { color: #85B7EB; }
[aria-selected="true"] { border-bottom: 2px solid #1D9E75 !important; color: #3B8BD4 !important; }

/* ── Buttons ── */
.stButton>button {
    background: linear-gradient(90deg,#1F5C9E,#534AB7);
    color: white; border: none; border-radius: 8px;
    padding: 10px 28px; font-weight: 600;
    transition: all .2s;
}
.stButton>button:hover { opacity:.85; transform:translateY(-1px); }

/* ── Selectbox / Slider ── */
.stSelectbox label, .stSlider label, .stNumberInput label { color:#85B7EB !important; }

/* ── Model Card in Sidebar ── */
.model-card {
    background: #0F1E33;
    border: 1px solid #1D4570;
    border-radius: 12px;
    padding: 12px 14px;
    margin-bottom: 10px;
    cursor: pointer;
    transition: all 0.2s;
}
.model-card:hover {
    border-color: #3B8BD4;
    background: #112640;
}
.model-card.active {
    border-color: #1D9E75;
    background: #0D3020;
}
.model-card-title {
    font-size: 0.9rem;
    font-weight: 700;
    color: #3B8BD4 !important;
    margin-bottom: 8px;
}
.model-card-row {
    display: flex;
    justify-content: space-between;
    margin-bottom: 4px;
}
.model-metric-label {
    font-size: 0.68rem;
    color: #85B7EB !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.model-metric-val-cls {
    font-size: 0.88rem;
    font-weight: 700;
    color: #3B8BD4 !important;
}
.model-metric-val-reg {
    font-size: 0.88rem;
    font-weight: 700;
    color: #1D9E75 !important;
}
.model-tag {
    display: inline-block;
    background: #1a2a45;
    border: 1px solid #2a4a70;
    border-radius: 4px;
    padding: 1px 6px;
    font-size: 0.65rem;
    color: #AFA9EC !important;
    margin-right: 3px;
    margin-top: 4px;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# SVG LOGO (inline — no file dependency)
# ─────────────────────────────────────────────────────────────────
LOGO_SVG = """
<svg viewBox="0 0 340 210" xmlns="http://www.w3.org/2000/svg" width="220" height="130">
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#3B8BD4"/>
      <stop offset="100%" stop-color="#534AB7"/>
    </linearGradient>
    <linearGradient id="ac" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0%" stop-color="#1D9E75"/>
      <stop offset="100%" stop-color="#3B8BD4"/>
    </linearGradient>
  </defs>
  <rect x="10" y="8" width="320" height="195" rx="14" fill="#0D1B2A" stroke="#534AB7" stroke-width="1"/>
  <!-- Hex -->
  <polygon points="170,28 191,40 191,64 170,76 149,64 149,40"
           fill="none" stroke="url(#bg)" stroke-width="1.5"/>
  <polygon points="170,34 185,43 185,61 170,70 155,61 155,43" fill="#1a2a45"/>
  <line x1="170" y1="40" x2="161" y2="64" stroke="url(#bg)" stroke-width="2" stroke-linecap="round"/>
  <line x1="170" y1="40" x2="179" y2="64" stroke="url(#bg)" stroke-width="2" stroke-linecap="round"/>
  <line x1="164" y1="57" x2="176" y2="57" stroke="#1D9E75" stroke-width="1.5" stroke-linecap="round"/>
  <circle cx="170" cy="22" r="3" fill="#378ADD"/>
  <circle cx="197" cy="36" r="2.5" fill="#534AB7"/>
  <circle cx="199" cy="68" r="2.5" fill="#1D9E75"/>
  <circle cx="170" cy="83" r="3" fill="#378ADD"/>
  <circle cx="141" cy="68" r="2.5" fill="#534AB7"/>
  <circle cx="143" cy="36" r="2.5" fill="#1D9E75"/>
  <!-- AXORA -->
  <text x="170" y="115" text-anchor="middle"
        font-family="Georgia,serif" font-size="34" font-weight="700"
        letter-spacing="8" fill="url(#bg)">AXORA</text>
  <rect x="68" y="121" width="204" height="2" rx="1" fill="url(#ac)"/>
  <text x="170" y="140" text-anchor="middle"
        font-family="Courier New,monospace" font-size="7" letter-spacing="3"
        fill="#85B7EB">INTELLIGENCE · ILLUMINATED</text>
  <rect x="80" y="152" width="180" height="22" rx="11"
        fill="#0F1E33" stroke="#378ADD" stroke-width=".8"/>
  <text x="170" y="167" text-anchor="middle"
        font-family="Courier New,monospace" font-size="6.5" letter-spacing="1.5"
        fill="#5DCAA5">DATA ANALYSIS  ·  ARTIFICIAL INTELLIGENCE</text>
  <text x="170" y="192" text-anchor="middle"
        font-family="Georgia,serif" font-size="6.5" letter-spacing=".5"
        fill="#4d7fa8">Alaa · Amira · Aya A. · Aya E. · Aya S. · Aya Sh.</text>
</svg>
"""

# ─────────────────────────────────────────────────────────────────
# MODEL METADATA  (pre-computed from report)
# ─────────────────────────────────────────────────────────────────
MODEL_META = {
    "🤖  KNN": {
        "icon": "🤖",
        "short": "KNN",
        "task_cls": "Multi-class Classification (A–D)",
        "task_reg": "Regression (broad_jump_cm)",
        "cls_accuracy": "63.09%",
        "cls_f1": "0.63",
        "cls_cv": "61.9%",
        "reg_r2": "0.786",
        "reg_rmse": "13.8 cm (MAE)",
        "best_split": "80:20",
        "tags": ["k=22", "StandardScaler", "5-Fold CV"],
        "color": "#3B8BD4",
    },
    "📈  Linear Regression": {
        "icon": "📈",
        "short": "LR",
        "task_cls": "Adapted Classification (binned output)",
        "task_reg": "Regression — Primary Task (broad_jump_cm)",
        "cls_accuracy": "~50%",
        "cls_f1": "~0.48",
        "cls_cv": "N/A",
        "reg_r2": "0.803",
        "reg_rmse": "17.89 cm",
        "best_split": "70:30",
        "tags": ["OLS", "No scaling", "5-Fold CV"],
        "color": "#85B7EB",
    },
    "🌳  Decision Tree": {
        "icon": "🌳",
        "short": "DT",
        "task_cls": "Multi-class Classification (A–D)",
        "task_reg": "Regression (broad_jump_cm)",
        "cls_accuracy": "72.15%",
        "cls_f1": "0.71",
        "cls_cv": "70.39%",
        "reg_r2": "0.760",
        "reg_rmse": "20.1 cm",
        "best_split": "80:20",
        "tags": ["ccp_alpha=0.000377", "Depth 14", "477 nodes"],
        "color": "#534AB7",
    },
    "⚙️   SVM": {
        "icon": "⚙️",
        "short": "SVM",
        "task_cls": "Multi-class Classification (A–D)",
        "task_reg": "SVR Regression (broad_jump_cm)",
        "cls_accuracy": "71.27%",
        "cls_f1": "0.71",
        "cls_cv": "70.8%",
        "reg_r2": "0.790",
        "reg_rmse": "18.4 cm",
        "best_split": "80:20",
        "tags": ["RBF kernel", "C=10", "γ=0.1"],
        "color": "#1D9E75",
    },
    "🧠  Neural Network": {
        "icon": "🧠",
        "short": "MLP",
        "task_cls": "Multi-class Classification (A–D)",
        "task_reg": "Regression (broad_jump_cm)",
        "cls_accuracy": "74.80%",
        "cls_f1": "~0.74",
        "cls_cv": "73.98%",
        "reg_r2": "0.802",
        "reg_rmse": "17.77 cm",
        "best_split": "70:30",
        "tags": ["256→128→64", "BatchNorm", "EarlyStopping"],
        "color": "#E24B4A",
    },
}

# ─────────────────────────────────────────────────────────────────
# DATA LOADING & PREPROCESSING
# ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data(uploaded=None):
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        return df
    for p in ["bodyPerformance.csv", "bodyPerformance_-_Copy.csv",
              "/mnt/user-data/uploads/bodyPerformance_-_Copy.csv"]:
        try:
            return pd.read_csv(p)
        except FileNotFoundError:
            continue
    # Synthetic demo data
    np.random.seed(42)
    n = 2000
    df = pd.DataFrame({
        'age':                    np.random.uniform(21, 64, n),
        'gender':                 np.random.choice(['M','F'], n),
        'height_cm':              np.random.normal(168, 8, n),
        'weight_kg':              np.random.normal(67, 12, n),
        'body fat_%':             np.random.normal(23, 7, n).clip(3, 45),
        'diastolic':              np.random.normal(79, 11, n).clip(50, 120),
        'systolic':               np.random.normal(130, 15, n).clip(80, 180),
        'gripForce':              np.random.normal(37, 11, n).clip(5, 70),
        'sit and bend forward_cm':np.random.normal(15, 8, n).clip(-20, 50),
        'sit-ups counts':         np.random.normal(40, 14, n).clip(0, 80),
        'broad jump_cm':          np.random.normal(190, 40, n).clip(50, 310),
        'class':                  np.random.choice(['A','B','C','D'], n)
    })
    return df

@st.cache_data
def preprocess(df):
    d = df.copy()
    d = d.drop_duplicates()
    d = d[(d['systolic'] > 40) & (d['diastolic'] > 40)]
    d['sit and bend forward_cm'] = d['sit and bend forward_cm'].clip(-20, 50)
    d['BMI'] = d['weight_kg'] / ((d['height_cm'] / 100) ** 2)
    d['gender'] = d['gender'].map({'M': 0, 'F': 1})
    le = LabelEncoder()
    d['class_enc'] = le.fit_transform(d['class'])
    return d, le

# ─────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────
def show_fig(fig):
    fig.patch.set_facecolor('#0D1B2A')
    for ax in fig.get_axes():
        ax.set_facecolor('#0F1E33')
        ax.tick_params(colors='#85B7EB')
        ax.xaxis.label.set_color('#85B7EB')
        ax.yaxis.label.set_color('#85B7EB')
        ax.title.set_color('#3B8BD4')
        for spine in ax.spines.values():
            spine.set_edgecolor('#1D4570')
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

def metric_cols(metrics: dict):
    cols = st.columns(len(metrics))
    for col, (lbl, val) in zip(cols, metrics.items()):
        col.markdown(f"""
        <div class="metric-box">
            <div class="metric-val">{val}</div>
            <div class="metric-lbl">{lbl}</div>
        </div>""", unsafe_allow_html=True)

def plot_confusion(cm, classes, title):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes,
                ax=ax, linewidths=.5, cbar_kws={'shrink':.8})
    ax.set_title(title, pad=12)
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    show_fig(fig)

def plot_compare_bars(labels, values, ylabel, title, color='#3B8BD4'):
    fig, ax = plt.subplots(figsize=(6, 3.5))
    bars = ax.bar(labels, values, color=color, edgecolor='#1D4570', width=0.5)
    for b in bars:
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.005,
                f'{b.get_height():.3f}', ha='center', va='bottom',
                color='#C8D8EA', fontsize=9)
    ax.set_ylabel(ylabel); ax.set_title(title)
    ax.set_ylim(0, max(values)*1.2 + 0.05)
    show_fig(fig)

def run_splits_cls(estimator_fn, X, y, splits, scale=True):
    results = []
    for name, ts in splits:
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=ts, random_state=42, stratify=y)
        if scale:
            sc = StandardScaler()
            X_tr = sc.fit_transform(X_tr); X_te = sc.transform(X_te)
        model = estimator_fn()
        model.fit(X_tr, y_tr)
        yp = model.predict(X_te)
        results.append({
            'Split': name,
            'Accuracy':  round(accuracy_score(y_te, yp)*100, 2),
            'Precision': round(precision_score(y_te, yp, average='macro', zero_division=0)*100, 2),
            'Recall':    round(recall_score(y_te, yp, average='macro', zero_division=0)*100, 2),
            'F1':        round(f1_score(y_te, yp, average='macro', zero_division=0)*100, 2),
        })
    return results

def run_splits_reg(estimator_fn, X, y, splits, scale=True):
    results = []
    for name, ts in splits:
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=ts, random_state=42)
        if scale:
            sc = StandardScaler()
            X_tr = sc.fit_transform(X_tr); X_te = sc.transform(X_te)
        model = estimator_fn()
        model.fit(X_tr, y_tr)
        yp = model.predict(X_te)
        results.append({
            'Split': name,
            'R²':   round(r2_score(y_te, yp), 4),
            'RMSE': round(np.sqrt(mean_squared_error(y_te, yp)), 3),
            'MAE':  round(mean_absolute_error(y_te, yp), 3),
        })
    return results

SPLITS = [("80:20", 0.20), ("70:30", 0.30), ("50:50", 0.50)]

# ─────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    # Logo + Title
    st.markdown(LOGO_SVG, unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center; margin-top:-6px; margin-bottom:12px;">
        <span style="font-size:1.15rem; font-weight:800; color:#3B8BD4; letter-spacing:2px;">
            Body Performance Analytics
        </span><br>
        <span style="font-size:0.72rem; color:#85B7EB; letter-spacing:1px;">
            Intelligent Classification System
        </span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr style="border-color:#1D4570; margin:8px 0 12px 0">', unsafe_allow_html=True)

    # Dataset upload
    st.markdown("**📂 Dataset**", unsafe_allow_html=False)
    uploaded_file = st.file_uploader("Upload bodyPerformance.csv", type=["csv"], label_visibility="collapsed")

    st.markdown('<hr style="border-color:#1D4570; margin:10px 0 12px 0">', unsafe_allow_html=True)

    # Navigation — non-model pages
    st.markdown("**🧭 Navigation**", unsafe_allow_html=False)
    nav_page = st.radio("nav", [
        "🏠  Overview",
        "📊  EDA",
        "🏆  Model Comparison",
        "🔮  Live Predictor",
    ], label_visibility="collapsed")

    st.markdown('<hr style="border-color:#1D4570; margin:10px 0 12px 0">', unsafe_allow_html=True)

    # ── 5 Model Cards ──────────────────────────────────────────
    st.markdown("**🤖 ML Models**", unsafe_allow_html=False)

    # Track selected model
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = None

    for model_key, meta in MODEL_META.items():
        is_active = st.session_state.selected_model == model_key
        border_color = meta["color"] if is_active else "#1D4570"
        bg_color = "#0D2820" if is_active else "#0F1E33"

        tags_html = "".join([f'<span class="model-tag">{t}</span>' for t in meta["tags"]])

        card_html = f"""
        <div style="
            background:{bg_color};
            border:1px solid {border_color};
            border-radius:12px;
            padding:11px 13px;
            margin-bottom:8px;
        ">
            <div style="font-size:0.88rem; font-weight:700; color:{meta['color']}; margin-bottom:6px;">
                {meta['icon']} {meta['short']}
            </div>
            <div style="display:flex; justify-content:space-between; margin-bottom:3px;">
                <span style="font-size:0.67rem; color:#85B7EB; text-transform:uppercase; letter-spacing:0.5px;">Classification Acc.</span>
                <span style="font-size:0.85rem; font-weight:700; color:#3B8BD4;">{meta['cls_accuracy']}</span>
            </div>
            <div style="display:flex; justify-content:space-between; margin-bottom:3px;">
                <span style="font-size:0.67rem; color:#85B7EB; text-transform:uppercase; letter-spacing:0.5px;">Regression R²</span>
                <span style="font-size:0.85rem; font-weight:700; color:#1D9E75;">{meta['reg_r2']}</span>
            </div>
            <div style="display:flex; justify-content:space-between; margin-bottom:5px;">
                <span style="font-size:0.67rem; color:#85B7EB; text-transform:uppercase; letter-spacing:0.5px;">Best Split</span>
                <span style="font-size:0.78rem; color:#AFA9EC;">{meta['best_split']}</span>
            </div>
            <div style="margin-top:2px;">{tags_html}</div>
        </div>
        """
        st.markdown(card_html, unsafe_allow_html=True)

        btn_label = "✓ Selected" if is_active else f"Open {meta['short']}"
        if st.button(btn_label, key=f"btn_{model_key}", use_container_width=True):
            st.session_state.selected_model = model_key if not is_active else None
            st.rerun()

    st.markdown('<hr style="border-color:#1D4570; margin:10px 0 8px 0">', unsafe_allow_html=True)
    st.markdown('<span class="badge-green">Team Axora</span>', unsafe_allow_html=True)
    st.markdown('<span class="badge-blue">Intro to AI & ML</span>', unsafe_allow_html=True)
    st.markdown('<span class="badge-purple">2024–2025</span>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────
raw_df = load_data(uploaded_file)
df, le = preprocess(raw_df)

FEATURES_CLS = ['age','gender','height_cm','weight_kg','body fat_%',
                 'diastolic','systolic','gripForce',
                 'sit and bend forward_cm','sit-ups counts','broad jump_cm','BMI']
FEATURES_REG = ['age','gender','height_cm','weight_kg','body fat_%',
                 'diastolic','systolic','gripForce',
                 'sit and bend forward_cm','sit-ups counts','BMI']
TARGET_CLS = 'class_enc'
TARGET_REG = 'broad jump_cm'
CLASS_NAMES = ['A','B','C','D']

X_cls = df[FEATURES_CLS].values
y_cls = df[TARGET_CLS].values
X_reg = df[FEATURES_REG].values
y_reg = df[TARGET_REG].values

# ─────────────────────────────────────────────────────────────────
# Determine active page
# ─────────────────────────────────────────────────────────────────
sel_model = st.session_state.get("selected_model", None)
# If a model is selected, show its page; otherwise show nav page
if sel_model:
    page = sel_model
else:
    page = nav_page

# ─────────────────────────────────────────────────────────────────
# ══════════════════ PAGE: OVERVIEW ══════════════════
# ─────────────────────────────────────────────────────────────────
if "Overview" in page:
    st.markdown(f"""
    <div class="axora-header">
        <div>{LOGO_SVG}</div>
        <div>
            <div class="axora-title">Body Performance Analytics</div>
            <div class="axora-sub">Intelligent Classification &amp; Regression System</div>
            <div style="margin-top:10px">
                <span class="badge-green">13,393 Records</span>
                <span class="badge-blue">12 Features</span>
                <span class="badge-purple">5 ML Models</span>
                <span class="badge-blue">4 Performance Classes</span>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("### 📊 Dataset at a Glance")
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown('<div class="metric-box"><div class="metric-val">13,393</div><div class="metric-lbl">Total Records</div></div>', unsafe_allow_html=True)
    c2.markdown('<div class="metric-box"><div class="metric-val">12</div><div class="metric-lbl">Features</div></div>', unsafe_allow_html=True)
    c3.markdown('<div class="metric-box"><div class="metric-val">~25%</div><div class="metric-lbl">Each Class (Balanced)</div></div>', unsafe_allow_html=True)
    c4.markdown('<div class="metric-box"><div class="metric-val">74.8%</div><div class="metric-lbl">Best Accuracy (MLP)</div></div>', unsafe_allow_html=True)

    st.markdown("### 🏅 Model Leaderboard")
    summary_data = []
    for k, m in MODEL_META.items():
        summary_data.append({
            "Model": m["icon"] + " " + m["short"],
            "Task Types": f"Classification + Regression",
            "Best Cls Accuracy": m["cls_accuracy"],
            "Macro F1": m["cls_f1"],
            "Best CV Acc": m["cls_cv"],
            "Regression R²": m["reg_r2"],
            "Best RMSE": m["reg_rmse"],
        })
    st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)

    st.markdown("### 🔬 About This Study")
    st.markdown("""
    <div class="axora-card">
    <p>This project presents a comprehensive machine learning analysis of the <strong style="color:#3B8BD4">Body Performance Dataset</strong>,
    comprising 13,393 records across 12 physiological and fitness-related attributes.</p>
    <p><strong style="color:#3B8BD4">Five algorithms</strong> were implemented and evaluated on two parallel tasks:</p>
    <ul>
        <li>🎯 <strong>Multi-class Classification</strong> — predicting physical performance categories (A–D)</li>
        <li>📈 <strong>Regression</strong> — predicting broad jump distance (cm)</li>
    </ul>
    <p>Each model was assessed under <strong>80:20, 70:30, and 50:50</strong> train–test splits and
    <strong>5-fold cross-validation</strong>. A consistent accuracy ceiling near <strong style="color:#E24B4A">~71%</strong>
    was observed across all classifiers, attributed to class overlap, low-signal features, and label boundary ambiguity.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📋 Sample Data")
    st.dataframe(raw_df.head(10), use_container_width=True)

# ─────────────────────────────────────────────────────────────────
# ══════════════════ PAGE: EDA ══════════════════
# ─────────────────────────────────────────────────────────────────
elif "EDA" in page:
    st.title("📊 Exploratory Data Analysis")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Distributions", "📦 Boxplots", "🔥 Correlations", "🔵 Pairplot"
    ])

    num_cols = ['age','height_cm','weight_kg','body fat_%','diastolic',
                'systolic','gripForce','sit and bend forward_cm','sit-ups counts','broad jump_cm']

    with tab1:
        st.subheader("Numeric Feature Distributions")
        fig, axes = plt.subplots(2, 5, figsize=(18, 7))
        axes = axes.flatten()
        for i, col in enumerate(num_cols):
            axes[i].hist(raw_df[col].dropna(), bins=30, color='#3B8BD4', alpha=0.8, edgecolor='#0D1B2A')
            axes[i].set_title(col, fontsize=9)
        plt.tight_layout()
        show_fig(fig)

    with tab2:
        st.subheader("Boxplots — Outlier Detection")
        scaler_viz = StandardScaler()
        scaled = pd.DataFrame(scaler_viz.fit_transform(raw_df[num_cols].dropna()), columns=num_cols)
        fig, ax = plt.subplots(figsize=(14, 5))
        scaled.boxplot(ax=ax, notch=False, patch_artist=True,
                       boxprops=dict(facecolor='#1D4570', color='#3B8BD4'),
                       medianprops=dict(color='#1D9E75', linewidth=2),
                       whiskerprops=dict(color='#85B7EB'),
                       capprops=dict(color='#85B7EB'),
                       flierprops=dict(marker='o', color='#E24B4A', alpha=0.4, markersize=3))
        ax.set_title("Standardised Feature Boxplots")
        plt.xticks(rotation=40, ha='right', fontsize=8)
        show_fig(fig)

    with tab3:
        st.subheader("Correlation Heatmap")
        enc = raw_df.copy()
        enc['gender'] = enc['gender'].map({'M':0,'F':1})
        enc['class_n'] = LabelEncoder().fit_transform(enc['class'])
        corr = enc[num_cols + ['class_n']].corr()
        fig, ax = plt.subplots(figsize=(12, 9))
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                    linewidths=.4, ax=ax, annot_kws={'size':7})
        ax.set_title("Feature Correlation Matrix")
        show_fig(fig)

        st.subheader("Correlation with Performance Class")
        class_corr = corr['class_n'].drop('class_n').sort_values()
        fig2, ax2 = plt.subplots(figsize=(9, 4))
        colors = ['#E24B4A' if v < 0 else '#1D9E75' for v in class_corr.values]
        ax2.barh(class_corr.index, class_corr.values, color=colors, edgecolor='#0D1B2A')
        ax2.axvline(0, color='#85B7EB', linewidth=1)
        ax2.set_title("Pearson r with Performance Class")
        show_fig(fig2)

    with tab4:
        st.subheader("Feature Pairplot by Class (sample 400 rows)")
        samp = raw_df.sample(min(400, len(raw_df)), random_state=42)
        key_feats = ['gripForce','sit-ups counts','broad jump_cm','body fat_%','class']
        palette = {'A':'#3B8BD4','B':'#534AB7','C':'#1D9E75','D':'#E24B4A'}
        g = sns.pairplot(samp[key_feats], hue='class', palette=palette,
                         plot_kws={'alpha':0.4,'s':14}, diag_kind='kde')
        g.fig.patch.set_facecolor('#0D1B2A')
        for ax_ in g.axes.flatten():
            if ax_:
                ax_.set_facecolor('#0F1E33')
        st.pyplot(g.fig, use_container_width=True)
        plt.close()

# ─────────────────────────────────────────────────────────────────
# ══════════════════ MODEL PAGES ══════════════════
# ─────────────────────────────────────────────────────────────────
elif "KNN" in page:
    meta = MODEL_META["🤖  KNN"]
    st.title(f"🤖 K-Nearest Neighbors (KNN)")

    # Model summary card
    st.markdown(f"""
    <div class="axora-card" style="border-color:{meta['color']};">
        <div style="display:flex; gap:32px; flex-wrap:wrap; align-items:center;">
            <div>
                <div style="font-size:0.72rem; color:#85B7EB; text-transform:uppercase; letter-spacing:1px;">Task 1 — Classification</div>
                <div style="font-size:0.95rem; color:#C8D8EA; margin-bottom:10px;">{meta['task_cls']}</div>
                <div style="font-size:0.72rem; color:#85B7EB; text-transform:uppercase; letter-spacing:1px;">Task 2 — Regression</div>
                <div style="font-size:0.95rem; color:#C8D8EA;">{meta['task_reg']}</div>
            </div>
            <div style="display:flex; gap:20px;">
                <div style="text-align:center; background:#112640; border:1px solid #1D4570; border-radius:10px; padding:14px 20px;">
                    <div style="font-size:1.8rem; font-weight:800; color:#3B8BD4;">{meta['cls_accuracy']}</div>
                    <div style="font-size:0.7rem; color:#85B7EB; text-transform:uppercase;">Best Classification Accuracy</div>
                </div>
                <div style="text-align:center; background:#112640; border:1px solid #1D9E75; border-radius:10px; padding:14px 20px;">
                    <div style="font-size:1.8rem; font-weight:800; color:#1D9E75;">{meta['reg_r2']}</div>
                    <div style="font-size:0.7rem; color:#85B7EB; text-transform:uppercase;">Regression R²</div>
                </div>
                <div style="text-align:center; background:#112640; border:1px solid #534AB7; border-radius:10px; padding:14px 20px;">
                    <div style="font-size:1.8rem; font-weight:800; color:#AFA9EC;">{meta['cls_f1']}</div>
                    <div style="font-size:0.7rem; color:#85B7EB; text-transform:uppercase;">Macro F1</div>
                </div>
            </div>
        </div>
        <p style="margin-top:14px; color:#C8D8EA;">
        KNN classifies each instance by majority vote among its <strong>k nearest neighbors</strong> in scaled feature space.
        <code>StandardScaler</code> was applied. Values of k from <strong>1 to 30</strong> were evaluated across three splits and 5-fold CV.
        </p>
    </div>
    """, unsafe_allow_html=True)

    tab_cls, tab_reg = st.tabs(["🎯 Classification", "📈 Regression"])

    with tab_cls:
        st.subheader("KNN Classification — Hyperparameter Tuning")
        col1, col2 = st.columns([1, 2])
        with col1:
            k_max   = st.slider("Max k to evaluate", 5, 50, 25, key="knn_kmax")
            split_s = st.selectbox("Split for tuning", ["80:20","70:30","50:50"], key="knn_split")
        ts_map = {"80:20":0.20,"70:30":0.30,"50:50":0.50}
        ts = ts_map[split_s]

        if st.button("▶ Run KNN Classification", key="run_knn_cls"):
            with st.spinner("Evaluating k values…"):
                X_tr, X_te, y_tr, y_te = train_test_split(X_cls, y_cls, test_size=ts, random_state=42, stratify=y_cls)
                sc = StandardScaler()
                X_tr = sc.fit_transform(X_tr); X_te = sc.transform(X_te)
                k_range = range(1, k_max+1)
                accs = []
                for k in k_range:
                    m = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
                    m.fit(X_tr, y_tr)
                    accs.append(accuracy_score(y_te, m.predict(X_te)))
                best_k = k_range[int(np.argmax(accs))]
                best_acc = max(accs)

            metric_cols({"Best k": best_k, "Best Accuracy": f"{best_acc*100:.2f}%", "Split": split_s})

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(list(k_range), [a*100 for a in accs], color='#3B8BD4', linewidth=2)
            ax.axvline(best_k, color='#1D9E75', linestyle='--', linewidth=1.5,
                       label=f'Best k={best_k} ({best_acc*100:.1f}%)')
            ax.set_xlabel('k'); ax.set_ylabel('Test Accuracy (%)')
            ax.set_title('k vs Test Accuracy'); ax.legend()
            show_fig(fig)

            best_model = KNeighborsClassifier(n_neighbors=best_k, n_jobs=-1)
            best_model.fit(X_tr, y_tr)
            yp = best_model.predict(X_te)
            c1, c2 = st.columns(2)
            with c1:
                plot_confusion(confusion_matrix(y_te, yp), CLASS_NAMES, f"Confusion Matrix (k={best_k})")
            with c2:
                rpt = classification_report(y_te, yp, target_names=CLASS_NAMES, output_dict=True)
                st.dataframe(pd.DataFrame(rpt).T.round(3), use_container_width=True)

            st.subheader("Cross-Split Comparison")
            results = run_splits_cls(lambda: KNeighborsClassifier(n_neighbors=best_k, n_jobs=-1), X_cls, y_cls, SPLITS)
            res_df = pd.DataFrame(results)
            st.dataframe(res_df, use_container_width=True, hide_index=True)
            plot_compare_bars(res_df['Split'], res_df['Accuracy'], 'Accuracy (%)', 'KNN Accuracy Across Splits', '#3B8BD4')

            st.subheader("5-Fold Cross-Validation")
            sc_all = StandardScaler()
            X_sc = sc_all.fit_transform(X_cls)
            cv_sc = cross_val_score(KNeighborsClassifier(n_neighbors=best_k, n_jobs=-1), X_sc, y_cls, cv=5, scoring='accuracy')
            metric_cols({"CV Mean": f"{cv_sc.mean()*100:.2f}%", "CV Std": f"±{cv_sc.std()*100:.2f}%",
                         "Min Fold": f"{cv_sc.min()*100:.2f}%", "Max Fold": f"{cv_sc.max()*100:.2f}%"})
            fig2, ax2 = plt.subplots(figsize=(7, 3.5))
            ax2.bar([f'Fold {i+1}' for i in range(5)], cv_sc*100, color='#534AB7', edgecolor='#0D1B2A')
            ax2.axhline(cv_sc.mean()*100, color='#1D9E75', linestyle='--', label=f'Mean {cv_sc.mean()*100:.1f}%')
            ax2.set_ylabel('Accuracy (%)'); ax2.set_title('5-Fold CV Accuracy'); ax2.legend()
            show_fig(fig2)

    with tab_reg:
        st.subheader("KNN Regression — Predicting broad_jump_cm")
        k_reg = st.slider("k for regression", 1, 50, 37, key="knn_k_reg")
        if st.button("▶ Run KNN Regression", key="run_knn_reg"):
            with st.spinner("Running…"):
                results = run_splits_reg(lambda: KNeighborsRegressor(n_neighbors=k_reg, n_jobs=-1), X_reg, y_reg, SPLITS)
            res_df = pd.DataFrame(results)
            st.dataframe(res_df, use_container_width=True, hide_index=True)
            plot_compare_bars(res_df['Split'], res_df['R²'], 'R² Score', 'KNN Regression R² Across Splits', '#1D9E75')
            sc = StandardScaler()
            X_tr, X_te, y_tr, y_te = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
            X_tr = sc.fit_transform(X_tr); X_te = sc.transform(X_te)
            m = KNeighborsRegressor(n_neighbors=k_reg, n_jobs=-1)
            m.fit(X_tr, y_tr); yp = m.predict(X_te)
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.scatter(y_te, yp, alpha=0.4, color='#3B8BD4', s=10)
            mn, mx = y_te.min(), y_te.max()
            ax.plot([mn,mx],[mn,mx],'r--',lw=2, label='Perfect')
            ax.set_xlabel('Actual'); ax.set_ylabel('Predicted')
            ax.set_title('Actual vs Predicted (80:20)'); ax.legend()
            show_fig(fig)

# ─────────────────────────────────────────────────────────────────
elif "Linear Regression" in page:
    meta = MODEL_META["📈  Linear Regression"]
    st.title("📈 Linear Regression")

    st.markdown(f"""
    <div class="axora-card" style="border-color:{meta['color']};">
        <div style="display:flex; gap:32px; flex-wrap:wrap; align-items:center;">
            <div>
                <div style="font-size:0.72rem; color:#85B7EB; text-transform:uppercase; letter-spacing:1px;">Task 1 — Regression (Primary)</div>
                <div style="font-size:0.95rem; color:#C8D8EA; margin-bottom:10px;">{meta['task_reg']}</div>
                <div style="font-size:0.72rem; color:#85B7EB; text-transform:uppercase; letter-spacing:1px;">Task 2 — Adapted Classification</div>
                <div style="font-size:0.95rem; color:#C8D8EA;">{meta['task_cls']}</div>
            </div>
            <div style="display:flex; gap:20px;">
                <div style="text-align:center; background:#112640; border:1px solid #1D4570; border-radius:10px; padding:14px 20px;">
                    <div style="font-size:1.8rem; font-weight:800; color:#3B8BD4;">{meta['cls_accuracy']}</div>
                    <div style="font-size:0.7rem; color:#85B7EB; text-transform:uppercase;">Adapted Cls Accuracy</div>
                </div>
                <div style="text-align:center; background:#112640; border:1px solid #1D9E75; border-radius:10px; padding:14px 20px;">
                    <div style="font-size:1.8rem; font-weight:800; color:#1D9E75;">{meta['reg_r2']}</div>
                    <div style="font-size:0.7rem; color:#85B7EB; text-transform:uppercase;">Regression R²</div>
                </div>
                <div style="text-align:center; background:#112640; border:1px solid #534AB7; border-radius:10px; padding:14px 20px;">
                    <div style="font-size:1.8rem; font-weight:800; color:#AFA9EC;">{meta['reg_rmse']}</div>
                    <div style="font-size:0.7rem; color:#85B7EB; text-transform:uppercase;">Best RMSE</div>
                </div>
            </div>
        </div>
        <p style="margin-top:14px; color:#C8D8EA;">
        Linear Regression predicts <strong>broad_jump_cm</strong> as its primary numerical target (R²≈0.80).
        It is also adapted for classification by binning continuous outputs — a fundamental task mismatch (~50% accuracy).
        </p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["📈 Regression (Primary)", "🎯 Classification (Adapted)"])

    with tab1:
        if st.button("▶ Run Linear Regression", key="run_lr"):
            with st.spinner("Training…"):
                results = run_splits_reg(LinearRegression, X_reg, y_reg, SPLITS, scale=False)
            res_df = pd.DataFrame(results)
            st.dataframe(res_df, use_container_width=True, hide_index=True)
            c1, c2 = st.columns(2)
            with c1:
                plot_compare_bars(res_df['Split'], res_df['R²'], 'R² Score', 'R² Across Splits', '#1D9E75')
            with c2:
                plot_compare_bars(res_df['Split'], res_df['RMSE'], 'RMSE (cm)', 'RMSE Across Splits', '#E24B4A')
            X_tr, X_te, y_tr, y_te = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
            m = LinearRegression(); m.fit(X_tr, y_tr); yp = m.predict(X_te)
            res = y_te - yp
            fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
            axes[0].scatter(y_te, yp, alpha=0.4, color='#3B8BD4', s=10)
            mn,mx = y_te.min(),y_te.max()
            axes[0].plot([mn,mx],[mn,mx],'r--',lw=2)
            axes[0].set_title('Actual vs Predicted'); axes[0].set_xlabel('Actual'); axes[0].set_ylabel('Predicted')
            axes[1].scatter(yp, res, alpha=0.4, color='#534AB7', s=10)
            axes[1].axhline(0, color='r', linestyle='--', lw=2)
            axes[1].set_title('Residual Plot'); axes[1].set_xlabel('Predicted'); axes[1].set_ylabel('Residuals')
            axes[2].hist(res, bins=35, color='#1D9E75', edgecolor='#0D1B2A', alpha=0.85)
            axes[2].set_title('Residual Distribution'); axes[2].set_xlabel('Error')
            show_fig(fig)
            st.subheader("5-Fold Cross-Validation")
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            cv_r2 = cross_val_score(LinearRegression(), X_reg, y_reg, cv=kf, scoring='r2')
            metric_cols({"CV Mean R²": f"{cv_r2.mean():.4f}", "CV Std": f"±{cv_r2.std():.4f}",
                         "Min": f"{cv_r2.min():.4f}", "Max": f"{cv_r2.max():.4f}"})

    with tab2:
        st.info("Linear Regression adapted: continuous output is binned into performance quartiles.")
        if st.button("▶ Run Adapted Classification", key="run_lr_cls"):
            with st.spinner("Running…"):
                results = []
                for name, ts in SPLITS:
                    X_tr, X_te, y_tr, y_te = train_test_split(X_reg, y_reg, test_size=ts, random_state=42)
                    m = LinearRegression(); m.fit(X_tr, y_tr)
                    yp = m.predict(X_te)
                    quantiles = np.percentile(y_tr, [25,50,75])
                    def bin_pred(v):
                        if v >= quantiles[2]: return 3
                        elif v >= quantiles[1]: return 2
                        elif v >= quantiles[0]: return 1
                        else: return 0
                    yp_cls = np.array([bin_pred(v) for v in yp])
                    yt_cls = np.array([bin_pred(v) for v in y_te])
                    acc = accuracy_score(yt_cls, yp_cls)
                    results.append({'Split': name, 'Adapted Accuracy': f'{acc*100:.1f}%'})
            st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)
            st.warning("⚠️ Linear Regression is not designed for classification — accuracy ~50% confirms this is a fundamental task mismatch.")

# ─────────────────────────────────────────────────────────────────
elif "Decision Tree" in page:
    meta = MODEL_META["🌳  Decision Tree"]
    st.title("🌳 Decision Tree")

    st.markdown(f"""
    <div class="axora-card" style="border-color:{meta['color']};">
        <div style="display:flex; gap:32px; flex-wrap:wrap; align-items:center;">
            <div>
                <div style="font-size:0.72rem; color:#85B7EB; text-transform:uppercase; letter-spacing:1px;">Task 1 — Classification</div>
                <div style="font-size:0.95rem; color:#C8D8EA; margin-bottom:10px;">{meta['task_cls']}</div>
                <div style="font-size:0.72rem; color:#85B7EB; text-transform:uppercase; letter-spacing:1px;">Task 2 — Regression</div>
                <div style="font-size:0.95rem; color:#C8D8EA;">{meta['task_reg']}</div>
            </div>
            <div style="display:flex; gap:20px;">
                <div style="text-align:center; background:#112640; border:1px solid #1D4570; border-radius:10px; padding:14px 20px;">
                    <div style="font-size:1.8rem; font-weight:800; color:#3B8BD4;">{meta['cls_accuracy']}</div>
                    <div style="font-size:0.7rem; color:#85B7EB; text-transform:uppercase;">Best Classification Accuracy</div>
                </div>
                <div style="text-align:center; background:#112640; border:1px solid #1D9E75; border-radius:10px; padding:14px 20px;">
                    <div style="font-size:1.8rem; font-weight:800; color:#1D9E75;">{meta['reg_r2']}</div>
                    <div style="font-size:0.7rem; color:#85B7EB; text-transform:uppercase;">Regression R²</div>
                </div>
                <div style="text-align:center; background:#112640; border:1px solid #534AB7; border-radius:10px; padding:14px 20px;">
                    <div style="font-size:1.8rem; font-weight:800; color:#AFA9EC;">{meta['cls_f1']}</div>
                    <div style="font-size:0.7rem; color:#85B7EB; text-transform:uppercase;">Macro F1</div>
                </div>
            </div>
        </div>
        <p style="margin-top:14px; color:#C8D8EA;">
        Decision trees recursively partition feature space. <strong>Cost-complexity pruning (alpha)</strong>
        was optimised via 5-fold CV. Optimal alpha ≈ <strong>0.000377</strong> → depth 14, 477 nodes.
        </p>
    </div>
    """, unsafe_allow_html=True)

    tab_cls, tab_reg = st.tabs(["🎯 Classification", "📈 Regression"])

    with tab_cls:
        alpha = st.slider("ccp_alpha", 0.0, 0.01, 0.000377, step=0.000050, format="%.6f", key="dt_alpha")
        if st.button("▶ Run Decision Tree Classification", key="run_dt_cls"):
            with st.spinner("Training…"):
                results = run_splits_cls(lambda: DecisionTreeClassifier(random_state=42, ccp_alpha=alpha), X_cls, y_cls, SPLITS, scale=False)
            res_df = pd.DataFrame(results)
            st.dataframe(res_df, use_container_width=True, hide_index=True)
            plot_compare_bars(res_df['Split'], res_df['Accuracy'], 'Accuracy (%)', 'DT Accuracy Across Splits', '#534AB7')

            X_tr, X_te, y_tr, y_te = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42, stratify=y_cls)
            best_m = DecisionTreeClassifier(random_state=42, ccp_alpha=alpha).fit(X_tr, y_tr)
            yp = best_m.predict(X_te)
            c1, c2 = st.columns(2)
            with c1:
                plot_confusion(confusion_matrix(y_te, yp), CLASS_NAMES, "Confusion Matrix (80:20)")
            with c2:
                rpt = classification_report(y_te, yp, target_names=CLASS_NAMES, output_dict=True)
                st.dataframe(pd.DataFrame(rpt).T.round(3), use_container_width=True)

            metric_cols({"Depth": best_m.get_depth(), "Nodes": best_m.get_n_leaves(),
                         "Train Acc": f"{accuracy_score(y_tr, best_m.predict(X_tr))*100:.1f}%",
                         "Test Acc":  f"{accuracy_score(y_te, yp)*100:.2f}%"})

            st.subheader("5-Fold Cross-Validation")
            cv_sc = cross_val_score(DecisionTreeClassifier(random_state=42, ccp_alpha=alpha), X_cls, y_cls, cv=5, scoring='accuracy')
            metric_cols({"CV Mean": f"{cv_sc.mean()*100:.2f}%", "CV Std": f"±{cv_sc.std()*100:.2f}%",
                         "Min": f"{cv_sc.min()*100:.2f}%", "Max": f"{cv_sc.max()*100:.2f}%"})

    with tab_reg:
        depth_reg = st.slider("Tree Depth (Regression)", 2, 14, 6, key="dt_depth_reg")
        if st.button("▶ Run DT Regression", key="run_dt_reg"):
            with st.spinner("Training…"):
                results = run_splits_reg(lambda: DecisionTreeRegressor(max_depth=depth_reg, random_state=42), X_reg, y_reg, SPLITS, scale=False)
            res_df = pd.DataFrame(results)
            st.dataframe(res_df, use_container_width=True, hide_index=True)
            plot_compare_bars(res_df['Split'], res_df['R²'], 'R² Score', 'DT Regression R² Across Splits', '#1D9E75')

# ─────────────────────────────────────────────────────────────────
elif "SVM" in page:
    meta = MODEL_META["⚙️   SVM"]
    st.title("⚙️ Support Vector Machine (SVM)")

    st.markdown(f"""
    <div class="axora-card" style="border-color:{meta['color']};">
        <div style="display:flex; gap:32px; flex-wrap:wrap; align-items:center;">
            <div>
                <div style="font-size:0.72rem; color:#85B7EB; text-transform:uppercase; letter-spacing:1px;">Task 1 — Classification</div>
                <div style="font-size:0.95rem; color:#C8D8EA; margin-bottom:10px;">{meta['task_cls']}</div>
                <div style="font-size:0.72rem; color:#85B7EB; text-transform:uppercase; letter-spacing:1px;">Task 2 — SVR Regression</div>
                <div style="font-size:0.95rem; color:#C8D8EA;">{meta['task_reg']}</div>
            </div>
            <div style="display:flex; gap:20px;">
                <div style="text-align:center; background:#112640; border:1px solid #1D4570; border-radius:10px; padding:14px 20px;">
                    <div style="font-size:1.8rem; font-weight:800; color:#3B8BD4;">{meta['cls_accuracy']}</div>
                    <div style="font-size:0.7rem; color:#85B7EB; text-transform:uppercase;">Best Classification Accuracy</div>
                </div>
                <div style="text-align:center; background:#112640; border:1px solid #1D9E75; border-radius:10px; padding:14px 20px;">
                    <div style="font-size:1.8rem; font-weight:800; color:#1D9E75;">{meta['reg_r2']}</div>
                    <div style="font-size:0.7rem; color:#85B7EB; text-transform:uppercase;">SVR R²</div>
                </div>
                <div style="text-align:center; background:#112640; border:1px solid #534AB7; border-radius:10px; padding:14px 20px;">
                    <div style="font-size:1.8rem; font-weight:800; color:#AFA9EC;">{meta['cls_f1']}</div>
                    <div style="font-size:0.7rem; color:#85B7EB; text-transform:uppercase;">Macro F1</div>
                </div>
            </div>
        </div>
        <p style="margin-top:14px; color:#C8D8EA;">
        SVM maximises the margin between classes. <strong>Linear and RBF kernels</strong> were compared.
        Grid search covered C ∈ {{0.1, 1, 10, 100}} and γ ∈ {{0.001, 0.01, 0.1, 1}} with 3-fold CV.
        Best: <strong>RBF, C=10, γ=0.1</strong>.
        </p>
    </div>
    """, unsafe_allow_html=True)

    tab_cls, tab_reg = st.tabs(["🎯 Classification", "📈 SVR Regression"])

    with tab_cls:
        col1, col2, col3 = st.columns(3)
        with col1:
            kernel = st.selectbox("Kernel", ["rbf","linear"], key="svm_kernel")
        with col2:
            C_val = st.selectbox("C", [0.1, 1, 10, 100], index=2, key="svm_C")
        with col3:
            gamma_val = st.selectbox("γ (RBF only)", [0.001, 0.01, 0.1, 1], index=2, key="svm_gamma")

        if st.button("▶ Run SVM Classification", key="run_svm_cls"):
            with st.spinner("Training SVM (this may take ~30s)…"):
                gamma_use = gamma_val if kernel == 'rbf' else 'scale'
                results = run_splits_cls(
                    lambda: SVC(kernel=kernel, C=C_val, gamma=gamma_use, random_state=42),
                    X_cls, y_cls, SPLITS)
            res_df = pd.DataFrame(results)
            st.dataframe(res_df, use_container_width=True, hide_index=True)
            plot_compare_bars(res_df['Split'], res_df['Accuracy'], 'Accuracy (%)', 'SVM Accuracy Across Splits', '#1D9E75')

            sc = StandardScaler()
            X_tr, X_te, y_tr, y_te = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42, stratify=y_cls)
            X_tr = sc.fit_transform(X_tr); X_te = sc.transform(X_te)
            best_m = SVC(kernel=kernel, C=C_val, gamma=gamma_use, random_state=42).fit(X_tr, y_tr)
            yp = best_m.predict(X_te)
            c1, c2 = st.columns(2)
            with c1:
                plot_confusion(confusion_matrix(y_te, yp), CLASS_NAMES, "Confusion Matrix (80:20)")
            with c2:
                rpt = classification_report(y_te, yp, target_names=CLASS_NAMES, output_dict=True)
                st.dataframe(pd.DataFrame(rpt).T.round(3), use_container_width=True)

    with tab_reg:
        col1, col2 = st.columns(2)
        with col1:
            svr_C = st.selectbox("C", [0.1, 1, 10, 100], index=2, key="svr_C")
        with col2:
            svr_eps = st.selectbox("Epsilon", [0.01, 0.1, 0.5, 1.0], index=1, key="svr_eps")
        if st.button("▶ Run SVR Regression", key="run_svr_reg"):
            with st.spinner("Training SVR…"):
                results = run_splits_reg(lambda: SVR(kernel='rbf', C=svr_C, epsilon=svr_eps), X_reg, y_reg, SPLITS)
            res_df = pd.DataFrame(results)
            st.dataframe(res_df, use_container_width=True, hide_index=True)
            plot_compare_bars(res_df['Split'], res_df['R²'], 'R² Score', 'SVR Regression R² Across Splits', '#1D9E75')

# ─────────────────────────────────────────────────────────────────
elif "Neural Network" in page:
    meta = MODEL_META["🧠  Neural Network"]
    st.title("🧠 Neural Network (MLP)")

    st.markdown(f"""
    <div class="axora-card" style="border-color:{meta['color']};">
        <div style="display:flex; gap:32px; flex-wrap:wrap; align-items:center;">
            <div>
                <div style="font-size:0.72rem; color:#85B7EB; text-transform:uppercase; letter-spacing:1px;">Task 1 — Classification</div>
                <div style="font-size:0.95rem; color:#C8D8EA; margin-bottom:10px;">{meta['task_cls']}</div>
                <div style="font-size:0.72rem; color:#85B7EB; text-transform:uppercase; letter-spacing:1px;">Task 2 — Regression</div>
                <div style="font-size:0.95rem; color:#C8D8EA;">{meta['task_reg']}</div>
            </div>
            <div style="display:flex; gap:20px;">
                <div style="text-align:center; background:#112640; border:1px solid #1D4570; border-radius:10px; padding:14px 20px;">
                    <div style="font-size:1.8rem; font-weight:800; color:#3B8BD4;">{meta['cls_accuracy']}</div>
                    <div style="font-size:0.7rem; color:#85B7EB; text-transform:uppercase;">Best Classification Accuracy</div>
                </div>
                <div style="text-align:center; background:#112640; border:1px solid #1D9E75; border-radius:10px; padding:14px 20px;">
                    <div style="font-size:1.8rem; font-weight:800; color:#1D9E75;">{meta['reg_r2']}</div>
                    <div style="font-size:0.7rem; color:#85B7EB; text-transform:uppercase;">Regression R²</div>
                </div>
                <div style="text-align:center; background:#112640; border:1px solid #534AB7; border-radius:10px; padding:14px 20px;">
                    <div style="font-size:1.8rem; font-weight:800; color:#AFA9EC;">{meta['cls_cv']}</div>
                    <div style="font-size:0.7rem; color:#85B7EB; text-transform:uppercase;">5-Fold CV Accuracy</div>
                </div>
            </div>
        </div>
        <p style="margin-top:14px; color:#C8D8EA;">
        MLP with <strong>BatchNormalization</strong>, <strong>Dropout</strong>, <strong>ReduceLROnPlateau</strong>, and
        <strong>EarlyStopping</strong>. Architecture: 256→128→64 units.
        Best classifier in this study: <strong>74.80%</strong> (corrected, leak-free pipeline).
        </p>
    </div>
    """, unsafe_allow_html=True)

    # TensorFlow check
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Activation
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        from tensorflow.keras.utils import to_categorical
        tf_available = True
    except ImportError:
        tf_available = False

    tab_cls, tab_reg = st.tabs(["🎯 Classification", "📈 Regression"])

    with tab_cls:
        if not tf_available:
            st.warning("TensorFlow not installed. Showing pre-computed results from the report.")
            report_df = pd.DataFrame([
                {"Split":"80:20","Train Acc":"~82%","Val Acc":"~74%","Test Acc":"74.76%"},
                {"Split":"70:30","Train Acc":"~80%","Val Acc":"~74%","Test Acc":"74.80%"},
                {"Split":"50:50","Train Acc":"~77%","Val Acc":"~73%","Test Acc":"73.33%"},
                {"Split":"5-Fold CV","Train Acc":"—","Val Acc":"—","Test Acc":"73.98% ±0.47%"},
            ])
            st.dataframe(report_df, use_container_width=True, hide_index=True)
        else:
            col1, col2, col3 = st.columns(3)
            with col1:
                epochs = st.slider("Max Epochs", 30, 200, 80, key="nn_epochs")
            with col2:
                patience = st.slider("Early Stop Patience", 5, 30, 15, key="nn_patience")
            with col3:
                dropout = st.slider("Dropout Rate", 0.1, 0.5, 0.2, step=0.05, key="nn_dropout")

            if st.button("▶ Train Neural Network", key="run_nn_cls"):
                SEED = 42; np.random.seed(SEED); tf.random.set_seed(SEED)

                def build_cls(input_dim, drop):
                    m = Sequential([
                        Dense(256, input_dim=input_dim, use_bias=False),
                        BatchNormalization(), Activation('elu'), Dropout(drop),
                        Dense(128, use_bias=False),
                        BatchNormalization(), Activation('elu'), Dropout(drop*0.75),
                        Dense(64, use_bias=False),
                        BatchNormalization(), Activation('elu'), Dropout(drop*0.5),
                        Dense(4, activation='softmax')
                    ])
                    m.compile(optimizer=Adam(0.001), loss='categorical_crossentropy', metrics=['accuracy'])
                    return m

                split_results = []
                for sname, ts in SPLITS:
                    with st.spinner(f"Training {sname} split…"):
                        X_tr, X_te, y_tr, y_te = train_test_split(X_cls, y_cls, test_size=ts, random_state=SEED, stratify=y_cls)
                        sc = StandardScaler()
                        X_tr = sc.fit_transform(X_tr); X_te = sc.transform(X_te)
                        y_tr_cat = to_categorical(y_tr, 4)
                        model = build_cls(X_tr.shape[1], dropout)
                        cb = [EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True, verbose=0),
                              ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=0)]
                        hist = model.fit(X_tr, y_tr_cat, epochs=epochs, batch_size=32,
                                         validation_split=0.15, callbacks=cb, verbose=0)
                        yp = np.argmax(model.predict(X_te, verbose=0), axis=1)
                        acc = accuracy_score(y_te, yp)
                        split_results.append({'Split': sname, 'Accuracy': f"{acc*100:.2f}%", 'Epochs run': len(hist.history['accuracy'])})

                st.dataframe(pd.DataFrame(split_results), use_container_width=True, hide_index=True)
                st.subheader("Training Curves (last split)")
                fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                axes[0].plot(hist.history['accuracy'], color='#3B8BD4', label='Train')
                axes[0].plot(hist.history['val_accuracy'], color='#1D9E75', label='Val')
                axes[0].set_title('Accuracy'); axes[0].set_xlabel('Epoch'); axes[0].legend()
                axes[1].plot(hist.history['loss'], color='#3B8BD4', label='Train')
                axes[1].plot(hist.history['val_loss'], color='#E24B4A', label='Val')
                axes[1].set_title('Loss'); axes[1].set_xlabel('Epoch'); axes[1].legend()
                show_fig(fig)
                plot_confusion(confusion_matrix(y_te, yp), CLASS_NAMES, "Confusion Matrix (last split)")

    with tab_reg:
        if not tf_available:
            st.warning("TensorFlow not installed. Showing pre-computed results from the report.")
            reg_report_df = pd.DataFrame([
                {"Split":"80:20","RMSE (cm)":"17.77","R²":"0.802"},
                {"Split":"70:30","RMSE (cm)":"17.77","R²":"0.802"},
                {"Split":"50:50","RMSE (cm)":"17.79","R²":"0.802"},
            ])
            st.dataframe(reg_report_df, use_container_width=True, hide_index=True)
        else:
            if st.button("▶ Train NN Regression", key="run_nn_reg"):
                SEED = 42; np.random.seed(SEED); tf.random.set_seed(SEED)
                dropout_reg = st.session_state.get("nn_dropout", 0.2)

                def build_reg(input_dim, drop):
                    m = Sequential([
                        Dense(256, input_dim=input_dim, use_bias=False),
                        BatchNormalization(), Activation('elu'), Dropout(drop),
                        Dense(128, use_bias=False),
                        BatchNormalization(), Activation('elu'), Dropout(drop*0.75),
                        Dense(64, use_bias=False),
                        BatchNormalization(), Activation('elu'), Dropout(drop*0.5),
                        Dense(1, activation='linear')
                    ])
                    m.compile(optimizer=Adam(0.001), loss='mse', metrics=['mae'])
                    return m

                reg_results = []
                for sname, ts in SPLITS:
                    with st.spinner(f"Training regression {sname}…"):
                        X_tr, X_te, y_tr, y_te = train_test_split(X_reg, y_reg, test_size=ts, random_state=SEED)
                        sc = StandardScaler()
                        X_tr = sc.fit_transform(X_tr); X_te = sc.transform(X_te)
                        y_sc = StandardScaler()
                        y_tr_s = y_sc.fit_transform(y_tr.reshape(-1,1)).ravel()
                        model = build_reg(X_tr.shape[1], dropout_reg)
                        cb = [EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=0)]
                        model.fit(X_tr, y_tr_s, epochs=100, batch_size=32, validation_split=0.15, callbacks=cb, verbose=0)
                        yp = y_sc.inverse_transform(model.predict(X_te, verbose=0)).ravel()
                        reg_results.append({'Split': sname, 'R²': round(r2_score(y_te, yp), 4),
                                            'RMSE': round(np.sqrt(mean_squared_error(y_te, yp)), 3),
                                            'MAE': round(mean_absolute_error(y_te, yp), 3)})
                st.dataframe(pd.DataFrame(reg_results), use_container_width=True, hide_index=True)
                plot_compare_bars([r['Split'] for r in reg_results], [r['R²'] for r in reg_results],
                                  'R² Score', 'NN Regression R² Across Splits', '#3B8BD4')

# ─────────────────────────────────────────────────────────────────
# ══════════════════ PAGE: MODEL COMPARISON ══════════════════
# ─────────────────────────────────────────────────────────────────
elif "Comparison" in page:
    st.title("🏆 Model Comparison")

    st.markdown('<div class="axora-card">', unsafe_allow_html=True)
    st.subheader("Classification Performance Summary")
    cls_summary = pd.DataFrame([
        {"Model":"🤖 KNN (k=22)",            "Best Accuracy":"63.09%","Macro F1":"0.63","Best Split":"80:20","5-Fold CV":"61.9%"},
        {"Model":"🌳 Decision Tree (pruned)", "Best Accuracy":"72.15%","Macro F1":"0.71","Best Split":"80:20","5-Fold CV":"70.39%"},
        {"Model":"⚙️ SVM (RBF, C=10)",        "Best Accuracy":"71.27%","Macro F1":"0.71","Best Split":"80:20","5-Fold CV":"70.8%"},
        {"Model":"🧠 Neural Network (MLP)",   "Best Accuracy":"74.80%","Macro F1":"~0.74","Best Split":"70:30","5-Fold CV":"73.98%"},
        {"Model":"📈 Linear Reg. (adapted)",  "Best Accuracy":"~50%",  "Macro F1":"~0.48","Best Split":"N/A",  "5-Fold CV":"N/A"},
    ])
    st.dataframe(cls_summary, use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="axora-card">', unsafe_allow_html=True)
    st.subheader("Regression Performance Summary")
    reg_summary = pd.DataFrame([
        {"Model":"🧠 Neural Network",   "Best R²":"0.802","Best RMSE (cm)":"17.77","Best Split":"80:20 / 70:30"},
        {"Model":"📈 Linear Regression","Best R²":"0.8033","Best RMSE (cm)":"17.89","Best Split":"70:30"},
        {"Model":"⚙️ SVM (SVR)",         "Best R²":"0.79", "Best RMSE (cm)":"18.4", "Best Split":"80:20"},
        {"Model":"🌳 Decision Tree",     "Best R²":"0.76", "Best RMSE (cm)":"20.1", "Best Split":"80:20"},
        {"Model":"🤖 KNN Regression",    "Best R²":"0.788","Best RMSE (cm)":"13.8 (MAE)","Best Split":"70:30"},
    ])
    st.dataframe(reg_summary, use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.subheader("Visual Comparison")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    models_cls  = ["KNN","Decision\nTree","SVM","Neural\nNetwork","Linear\nReg."]
    accs_cls    = [63.09, 72.15, 71.27, 74.80, 50.0]
    colors_cls  = ['#3B8BD4','#534AB7','#1D9E75','#E24B4A','#85B7EB']
    bars = axes[0].bar(models_cls, accs_cls, color=colors_cls, edgecolor='#0D1B2A', width=0.55)
    for b in bars:
        axes[0].text(b.get_x()+b.get_width()/2, b.get_height()+0.5, f'{b.get_height():.1f}%', ha='center', color='#C8D8EA', fontsize=9)
    axes[0].set_ylim(0, 90); axes[0].set_ylabel('Best Accuracy (%)')
    axes[0].set_title('Classification Accuracy')
    axes[0].axhline(72, color='#E24B4A', linestyle='--', lw=1, alpha=0.6, label='~72% ceiling')
    axes[0].legend()
    models_reg = ["Neural\nNetwork","Linear\nReg.","SVM\n(SVR)","Decision\nTree","KNN\nReg."]
    r2s        = [0.802, 0.803, 0.79, 0.76, 0.788]
    bars2 = axes[1].bar(models_reg, r2s, color=colors_cls, edgecolor='#0D1B2A', width=0.55)
    for b in bars2:
        axes[1].text(b.get_x()+b.get_width()/2, b.get_height()+0.003, f'{b.get_height():.3f}', ha='center', color='#C8D8EA', fontsize=9)
    axes[1].set_ylim(0, 1.0); axes[1].set_ylabel('R² Score')
    axes[1].set_title('Regression R² Score')
    show_fig(fig)

    st.subheader("Model Radar — Classification Metrics")
    cats = ['Accuracy','Precision','Recall','F1','CV Score']
    model_scores = {
        "KNN":      [63, 65, 63, 63, 62],
        "Dec.Tree": [72, 72, 71, 71, 70],
        "SVM":      [71, 71, 71, 71, 71],
        "MLP":      [75, 74, 75, 74, 74],
    }
    N = len(cats)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    fig2, ax2 = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    palette_radar = ['#3B8BD4','#534AB7','#1D9E75','#E24B4A']
    for (name, scores), col in zip(model_scores.items(), palette_radar):
        vals = scores + scores[:1]
        ax2.plot(angles, vals, color=col, linewidth=2, label=name)
        ax2.fill(angles, vals, color=col, alpha=0.12)
    ax2.set_thetagrids(np.degrees(angles[:-1]), cats, color='#85B7EB', fontsize=9)
    ax2.set_ylim(50, 85); ax2.set_yticks([55,65,75]); ax2.set_yticklabels(['55','65','75'], fontsize=7, color='#85B7EB')
    ax2.set_title("Model Radar (Classification %)", pad=20)
    ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    show_fig(fig2)

# ─────────────────────────────────────────────────────────────────
# ══════════════════ PAGE: LIVE PREDICTOR ══════════════════
# ─────────────────────────────────────────────────────────────────
elif "Predictor" in page:
    st.title("🔮 Live Predictor")
    st.markdown("Enter participant measurements to predict **performance class** and **broad jump distance**.")

    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            age     = st.number_input("Age (yr)",         18, 80, 30)
            gender  = st.selectbox("Gender",              ["M","F"])
            height  = st.number_input("Height (cm)",      140.0, 200.0, 170.0)
            weight  = st.number_input("Weight (kg)",      30.0, 150.0, 68.0)
        with c2:
            bf      = st.number_input("Body Fat (%)",     3.0, 60.0, 22.0)
            dia     = st.number_input("Diastolic BP",     50.0, 130.0, 79.0)
            sys_bp  = st.number_input("Systolic BP",      80.0, 180.0, 130.0)
        with c3:
            grip    = st.number_input("Grip Force (kg)",  5.0, 80.0, 37.0)
            flex    = st.number_input("Sit&Bend (cm)",   -20.0, 50.0, 15.0)
            situps  = st.number_input("Sit-ups",          0.0, 80.0, 40.0)
        model_choice = st.selectbox("Classifier", ["KNN (k=22)","Decision Tree","SVM (RBF)"])
        submitted = st.form_submit_button("🚀 Predict")

    if submitted:
        bmi = weight / ((height/100)**2)
        g   = 0 if gender == 'M' else 1
        x_cls = np.array([[age, g, height, weight, bf, dia, sys_bp, grip, flex, situps, df['broad jump_cm'].median(), bmi]])
        x_reg = np.array([[age, g, height, weight, bf, dia, sys_bp, grip, flex, situps, bmi]])

        sc_cls = StandardScaler(); X_cls_sc = sc_cls.fit_transform(X_cls)
        sc_reg = StandardScaler(); X_reg_sc = sc_reg.fit_transform(X_reg)

        if "KNN" in model_choice:
            clf = KNeighborsClassifier(n_neighbors=22, n_jobs=-1).fit(X_cls_sc, y_cls)
            rgr = KNeighborsRegressor(n_neighbors=35, n_jobs=-1).fit(X_reg_sc, y_reg)
            x_cls_in = sc_cls.transform(x_cls)
            x_reg_in = sc_reg.transform(x_reg)
        elif "Decision Tree" in model_choice:
            clf = DecisionTreeClassifier(random_state=42, ccp_alpha=0.000377).fit(X_cls, y_cls)
            rgr = DecisionTreeRegressor(max_depth=6, random_state=42).fit(X_reg, y_reg)
            x_cls_in = x_cls
            x_reg_in = x_reg
        else:
            clf = SVC(kernel='rbf', C=10, gamma=0.1, random_state=42).fit(X_cls_sc, y_cls)
            rgr = SVR(kernel='rbf', C=10).fit(X_reg_sc, y_reg)
            x_cls_in = sc_cls.transform(x_cls)
            x_reg_in = sc_reg.transform(x_reg)

        pred_cls   = clf.predict(x_cls_in)[0]
        class_name = CLASS_NAMES[pred_cls]
        pred_jump  = rgr.predict(x_reg_in)[0]

        class_info = {
            'A': ("🥇 Excellent", "#1D9E75",  "Outstanding fitness — maintain your regime."),
            'B': ("🥈 Good",      "#3B8BD4",  "Above average — targeted strength work recommended."),
            'C': ("🥉 Average",   "#534AB7",  "Moderate fitness — increase endurance training."),
            'D': ("⚠️ Below Avg", "#E24B4A",  "Below average — structured fitness programme advised."),
        }
        label, colour, advice = class_info[class_name]
        st.markdown(f"""
        <div class="axora-card" style="border-color:{colour}; text-align:center;">
            <div style="font-size:3rem">{label}</div>
            <div style="color:{colour}; font-size:1.4rem; font-weight:700; margin:8px 0">Performance Class: {class_name}</div>
            <div style="color:#C8D8EA">{advice}</div>
        </div>""", unsafe_allow_html=True)
        metric_cols({"Predicted Class": class_name, "Predicted Jump": f"{pred_jump:.1f} cm",
                     "BMI": f"{bmi:.1f}", "Model Used": model_choice.split()[0]})

# ─────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<hr style="border-color:#1D4570; margin-top:40px">
<div style="text-align:center; color:#4d7fa8; font-size:0.82rem; padding:12px">
    <strong style="color:#3B8BD4">AXORA</strong> — Body Performance Analytics &amp; Intelligent Classification System<br>
    Alaa Issawi · Amira Salama · Aya Abdel Maksoud · Aya El-Sabi · Aya Imam · Aya Khalil<br>
    <span style="color:#534AB7">Introduction to AI &amp; ML  ·  2024–2025</span>
</div>
""", unsafe_allow_html=True)
