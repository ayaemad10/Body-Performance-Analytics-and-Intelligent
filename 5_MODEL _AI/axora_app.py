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
import matplotlib.patches as mpatches
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
[data-testid="stSidebar"]          { background: #0F1E33; border-right: 1px solid #1D4570; }
[data-testid="stSidebar"] * { color: #C8D8EA !important; }
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
# DATA LOADING & PREPROCESSING
# ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data(uploaded=None):
    if uploaded is not None:
        df = pd.read_csv(uploaded)
    else:
        # Try common paths
        for p in ["bodyPerformance.csv", "bodyPerformance_-_Copy.csv",
                  "/mnt/user-data/uploads/bodyPerformance_-_Copy.csv"]:
            try:
                df = pd.read_csv(p); break
            except FileNotFoundError:
                continue
        else:
            # Generate synthetic demo data if no file found
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
    # Clean
    d = d.drop_duplicates()
    d = d[(d['systolic'] > 40) & (d['diastolic'] > 40)]
    d['sit and bend forward_cm'] = d['sit and bend forward_cm'].clip(-20, 50)
    # BMI feature
    d['BMI'] = d['weight_kg'] / ((d['height_cm'] / 100) ** 2)
    # Encode
    d['gender'] = d['gender'].map({'M': 0, 'F': 1})
    le = LabelEncoder()
    d['class_enc'] = le.fit_transform(d['class'])
    return d, le

# ─────────────────────────────────────────────────────────────────
# HELPER: matplotlib figure → streamlit
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

def plot_confusion(cm, classes, title, cmap='Blues'):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap,
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

# ─────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(LOGO_SVG, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### 📂 Dataset")
    uploaded_file = st.file_uploader("Upload bodyPerformance.csv", type=["csv"])
    st.markdown("---")
    st.markdown("### 🧭 Navigation")
    page = st.radio("", [
        "🏠  Overview",
        "📊  EDA",
        "🤖  KNN",
        "📈  Linear Regression",
        "🌳  Decision Tree",
        "⚙️   SVM",
        "🧠  Neural Network",
        "🏆  Model Comparison",
        "🔮  Live Predictor",
    ], label_visibility="collapsed")
    st.markdown("---")
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
# SHARED SPLIT HELPERS  (defined before all page blocks)
# ─────────────────────────────────────────────────────────────────
def run_splits_cls(estimator_fn, X, y, splits, scale=True):
    results = []
    for name, ts in splits:
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=ts,
                                                    random_state=42, stratify=y)
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
                <span class="badge-green">5 Models</span>
                <span class="badge-blue">Classification + Regression</span>
                <span class="badge-purple">Cross Validation</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    for col, lbl, val in zip(
        [c1,c2,c3,c4],
        ["Records","Features","Classes","Models"],
        [f"{len(df):,}", len(FEATURES_CLS), 4, 5]
    ):
        col.markdown(f"""<div class="metric-box">
            <div class="metric-val">{val}</div>
            <div class="metric-lbl">{lbl}</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns([1,1])
    with c1:
        st.markdown('<div class="axora-card">', unsafe_allow_html=True)
        st.subheader("📋 Dataset Sample")
        st.dataframe(raw_df.head(8), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="axora-card">', unsafe_allow_html=True)
        st.subheader("📐 Feature Descriptions")
        feat_desc = pd.DataFrame({
            "Feature": ["age","gender","height_cm","weight_kg","body fat_%",
                        "diastolic","systolic","gripForce",
                        "sit & bend fwd","sit-ups","broad jump","class"],
            "Type": ["Numeric","Categorical","Numeric","Numeric","Numeric",
                     "Numeric","Numeric","Numeric","Numeric","Numeric","Numeric","Target"],
            "Description": ["Age (yr)","M / F","Height (cm)","Weight (kg)","Body fat %",
                             "Diastolic BP","Systolic BP","Grip strength",
                             "Flexibility","Endurance","Explosive power","A/B/C/D"]
        })
        st.dataframe(feat_desc, use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="axora-card">', unsafe_allow_html=True)
    st.subheader("📊 Class Distribution")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    vc = raw_df['class'].value_counts().sort_index()
    axes[0].bar(vc.index, vc.values, color=['#3B8BD4','#534AB7','#1D9E75','#E24B4A'], edgecolor='#0D1B2A')
    axes[0].set_title("Class Frequency"); axes[0].set_xlabel("Class"); axes[0].set_ylabel("Count")
    axes[1].pie(vc.values, labels=vc.index, autopct='%1.1f%%',
                colors=['#3B8BD4','#534AB7','#1D9E75','#E24B4A'],
                textprops={'color':'#C8D8EA'})
    axes[1].set_title("Class Balance")
    show_fig(fig)
    st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# ══════════════════ PAGE: EDA ══════════════════
# ─────────────────────────────────────────────────────────────────
elif "EDA" in page:
    st.title("📊 Exploratory Data Analysis")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Distributions", "📦 Boxplots", "🔗 Correlations", "🔍 Pairplot"
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
        scaled = pd.DataFrame(scaler_viz.fit_transform(raw_df[num_cols].dropna()),
                               columns=num_cols)
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
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
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
        fig3 = plt.figure(figsize=(10, 8))
        enc2 = samp.copy()
        enc2['class_n'] = LabelEncoder().fit_transform(enc2['class'])
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
# ══════════════════ PAGE: KNN ══════════════════
# ─────────────────────────────────────────────────────────────────
elif "KNN" in page:
    st.title("🤖 K-Nearest Neighbors (KNN)")
    st.markdown('<div class="axora-card">', unsafe_allow_html=True)
    st.markdown("""
    KNN classifies each instance by majority vote among its **k nearest neighbors** in
    scaled feature space. `StandardScaler` was applied. Values of k from **1 to 30**
    were evaluated across three splits and 5-fold CV.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    tab_cls, tab_reg = st.tabs(["🎯 Classification", "📈 Regression"])

    # ── Classification ──────────────────────────────────────────
    with tab_cls:
        st.subheader("KNN Classification — Hyperparameter Tuning")

        col1, col2 = st.columns([1, 2])
        with col1:
            k_max   = st.slider("Max k to evaluate", 5, 50, 25, key="knn_kmax")
            split_s = st.selectbox("Split for tuning", ["80:20","70:30","50:50"], key="knn_split")
        with col2:
            ts_map = {"80:20":0.20,"70:30":0.30,"50:50":0.50}
            ts = ts_map[split_s]

        if st.button("▶ Run KNN Classification", key="run_knn_cls"):
            with st.spinner("Evaluating k values…"):
                X_tr, X_te, y_tr, y_te = train_test_split(
                    X_cls, y_cls, test_size=ts, random_state=42, stratify=y_cls)
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

            metric_cols({
                "Best k": best_k,
                "Best Accuracy": f"{best_acc*100:.2f}%",
                "Split": split_s
            })

            # k vs accuracy plot
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(list(k_range), [a*100 for a in accs], color='#3B8BD4', linewidth=2)
            ax.axvline(best_k, color='#1D9E75', linestyle='--', linewidth=1.5,
                       label=f'Best k={best_k} ({best_acc*100:.1f}%)')
            ax.set_xlabel('k'); ax.set_ylabel('Test Accuracy (%)')
            ax.set_title('k vs Test Accuracy'); ax.legend()
            show_fig(fig)

            # Best model detailed report
            best_model = KNeighborsClassifier(n_neighbors=best_k, n_jobs=-1)
            best_model.fit(X_tr, y_tr)
            yp = best_model.predict(X_te)

            c1, c2 = st.columns(2)
            with c1:
                plot_confusion(confusion_matrix(y_te, yp), CLASS_NAMES,
                               f"Confusion Matrix (k={best_k})")
            with c2:
                rpt = classification_report(y_te, yp, target_names=CLASS_NAMES, output_dict=True)
                rpt_df = pd.DataFrame(rpt).T.round(3)
                st.dataframe(rpt_df, use_container_width=True)

            # Cross splits
            st.subheader("Cross-Split Comparison")
            results = run_splits_cls(
                lambda: KNeighborsClassifier(n_neighbors=best_k, n_jobs=-1),
                X_cls, y_cls, SPLITS)
            res_df = pd.DataFrame(results)
            st.dataframe(res_df, use_container_width=True, hide_index=True)
            plot_compare_bars(res_df['Split'], res_df['Accuracy'],
                              'Accuracy (%)', 'KNN Accuracy Across Splits', '#3B8BD4')

            # 5-fold CV
            st.subheader("5-Fold Cross-Validation")
            sc_all = StandardScaler()
            X_sc = sc_all.fit_transform(X_cls)
            cv_sc = cross_val_score(KNeighborsClassifier(n_neighbors=best_k, n_jobs=-1),
                                    X_sc, y_cls, cv=5, scoring='accuracy')
            metric_cols({
                "CV Mean": f"{cv_sc.mean()*100:.2f}%",
                "CV Std":  f"±{cv_sc.std()*100:.2f}%",
                "Min Fold": f"{cv_sc.min()*100:.2f}%",
                "Max Fold": f"{cv_sc.max()*100:.2f}%"
            })
            fig2, ax2 = plt.subplots(figsize=(7, 3.5))
            ax2.bar([f'Fold {i+1}' for i in range(5)], cv_sc*100,
                    color='#534AB7', edgecolor='#0D1B2A')
            ax2.axhline(cv_sc.mean()*100, color='#1D9E75', linestyle='--',
                        label=f'Mean {cv_sc.mean()*100:.1f}%')
            ax2.set_ylabel('Accuracy (%)'); ax2.set_title('5-Fold CV Accuracy')
            ax2.legend()
            show_fig(fig2)

    # ── Regression ──────────────────────────────────────────────
    with tab_reg:
        st.subheader("KNN Regression — Predicting broad_jump_cm")

        k_reg = st.slider("k for regression", 1, 50, 37, key="knn_k_reg")
        if st.button("▶ Run KNN Regression", key="run_knn_reg"):
            with st.spinner("Running…"):
                results = run_splits_reg(
                    lambda: KNeighborsRegressor(n_neighbors=k_reg, n_jobs=-1),
                    X_reg, y_reg, SPLITS)
            res_df = pd.DataFrame(results)
            st.dataframe(res_df, use_container_width=True, hide_index=True)
            plot_compare_bars(res_df['Split'], res_df['R²'],
                              'R² Score', 'KNN Regression R² Across Splits', '#1D9E75')

            # Scatter: actual vs predicted
            sc = StandardScaler()
            X_tr, X_te, y_tr, y_te = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
            X_tr = sc.fit_transform(X_tr); X_te = sc.transform(X_te)
            m = KNeighborsRegressor(n_neighbors=k_reg, n_jobs=-1)
            m.fit(X_tr, y_tr); yp = m.predict(X_te)
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.scatter(y_te, yp, alpha=0.4, color='#3B8BD4', s=10)
            mn,mx = y_te.min(), y_te.max()
            ax.plot([mn,mx],[mn,mx],'r--',lw=2, label='Perfect')
            ax.set_xlabel('Actual'); ax.set_ylabel('Predicted')
            ax.set_title('Actual vs Predicted  (80:20)'); ax.legend()
            show_fig(fig)

# ─────────────────────────────────────────────────────────────────
# ══════════════════ PAGE: LINEAR REGRESSION ══════════════════
# ─────────────────────────────────────────────────────────────────
elif "Linear Regression" in page:
    st.title("📈 Linear Regression")
    st.markdown('<div class="axora-card">', unsafe_allow_html=True)
    st.markdown("""
    Linear Regression predicts **broad_jump_cm** as the primary numerical target.
    It is also adapted for classification by binning continuous outputs.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["📈 Regression (Primary)", "🎯 Classification (Adapted)"])

    with tab1:
        if st.button("▶ Run Linear Regression", key="run_lr"):
            with st.spinner("Training…"):
                results = run_splits_reg(LinearRegression, X_reg, y_reg, SPLITS, scale=False)

            res_df = pd.DataFrame(results)
            st.dataframe(res_df, use_container_width=True, hide_index=True)

            c1, c2 = st.columns(2)
            with c1:
                plot_compare_bars(res_df['Split'], res_df['R²'],
                                  'R² Score', 'R² Across Splits', '#1D9E75')
            with c2:
                plot_compare_bars(res_df['Split'], res_df['RMSE'],
                                  'RMSE (cm)', 'RMSE Across Splits', '#E24B4A')

            # Actual vs predicted
            X_tr, X_te, y_tr, y_te = train_test_split(X_reg, y_reg,
                                                        test_size=0.2, random_state=42)
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

            # Feature coefficients
            feat_names = FEATURES_REG + []
            # need dummy encoding for gender
            df_reg = df[FEATURES_REG + [TARGET_REG]].copy()
            X_fe = df_reg.drop(TARGET_REG, axis=1)
            y_fe = df_reg[TARGET_REG]
            m2 = LinearRegression(); m2.fit(X_fe, y_fe)
            coef_df = pd.DataFrame({'Feature': FEATURES_REG,
                                    'Coefficient': m2.coef_}).sort_values('Coefficient', key=abs, ascending=False)
            fig2, ax2 = plt.subplots(figsize=(9, 4))
            colors2 = ['#1D9E75' if c>=0 else '#E24B4A' for c in coef_df['Coefficient'].values[:10]]
            ax2.barh(coef_df['Feature'][:10], coef_df['Coefficient'][:10], color=colors2)
            ax2.set_title('Top Feature Coefficients'); ax2.set_xlabel('Coefficient')
            show_fig(fig2)

            # 5-Fold CV
            st.subheader("5-Fold Cross-Validation")
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            cv_r2 = cross_val_score(LinearRegression(), X_reg, y_reg, cv=kf, scoring='r2')
            metric_cols({
                "CV Mean R²": f"{cv_r2.mean():.4f}",
                "CV Std":     f"±{cv_r2.std():.4f}",
                "Min":        f"{cv_r2.min():.4f}",
                "Max":        f"{cv_r2.max():.4f}"
            })

    with tab2:
        st.info("Linear Regression adapted: continuous output is binned into performance quartiles.")
        if st.button("▶ Run Adapted Classification", key="run_lr_cls"):
            with st.spinner("Running…"):
                results = []
                for name, ts in SPLITS:
                    X_tr, X_te, y_tr, y_te = train_test_split(
                        X_reg, y_reg, test_size=ts, random_state=42)
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
# ══════════════════ PAGE: DECISION TREE ══════════════════
# ─────────────────────────────────────────────────────────────────
elif "Decision Tree" in page:
    st.title("🌳 Decision Tree")
    st.markdown('<div class="axora-card">', unsafe_allow_html=True)
    st.markdown("""
    Decision trees recursively partition feature space. **Cost-complexity pruning (alpha)**
    was optimised via 5-fold CV. Optimal alpha ≈ **0.000377** → depth 14, 477 nodes.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    tab_cls, tab_reg = st.tabs(["🎯 Classification", "📈 Regression"])

    with tab_cls:
        alpha = st.slider("ccp_alpha", 0.0, 0.01, 0.000377, step=0.000050,
                          format="%.6f", key="dt_alpha")
        if st.button("▶ Run Decision Tree Classification", key="run_dt_cls"):
            with st.spinner("Training…"):
                le2 = LabelEncoder()
                df_dt = df.copy()
                X_dt = df_dt[FEATURES_CLS].values
                y_dt = df_dt[TARGET_CLS].values

                X_tr, X_te, y_tr, y_te = train_test_split(
                    X_dt, y_dt, test_size=0.2, random_state=42, stratify=y_dt)
                m = DecisionTreeClassifier(random_state=42, ccp_alpha=alpha)
                m.fit(X_tr, y_tr); yp = m.predict(X_te)

            metric_cols({
                "Test Accuracy": f"{accuracy_score(y_te,yp)*100:.2f}%",
                "Tree Depth": m.get_depth(),
                "Nodes": m.tree_.node_count,
                "Alpha": f"{alpha:.6f}"
            })

            c1, c2 = st.columns(2)
            with c1:
                plot_confusion(confusion_matrix(y_te, yp), CLASS_NAMES, "Confusion Matrix")
            with c2:
                rpt = classification_report(y_te, yp, target_names=CLASS_NAMES, output_dict=True)
                st.dataframe(pd.DataFrame(rpt).T.round(3), use_container_width=True)

            # Alpha sweep
            st.subheader("Alpha Pruning Sweep")
            with st.spinner("Sweeping alphas…"):
                alphas = np.linspace(0, 0.005, 40)
                tr_accs, te_accs = [], []
                for a in alphas:
                    mm = DecisionTreeClassifier(random_state=42, ccp_alpha=a)
                    mm.fit(X_tr, y_tr)
                    tr_accs.append(mm.score(X_tr, y_tr))
                    te_accs.append(mm.score(X_te, y_te))
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(alphas, [a*100 for a in tr_accs], label='Train', color='#3B8BD4')
            ax.plot(alphas, [a*100 for a in te_accs], label='Test',  color='#1D9E75')
            ax.axvline(alpha, color='#E24B4A', linestyle='--', label=f'Selected α={alpha:.4f}')
            ax.set_xlabel('ccp_alpha'); ax.set_ylabel('Accuracy (%)')
            ax.set_title('Train vs Test Accuracy Across Alpha'); ax.legend()
            show_fig(fig)

            # Multi-split
            st.subheader("Cross-Split Results")
            results = run_splits_cls(
                lambda: DecisionTreeClassifier(random_state=42, ccp_alpha=alpha),
                X_cls, y_cls, SPLITS, scale=False)
            st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)

            # CV
            cv_sc = cross_val_score(
                DecisionTreeClassifier(random_state=42, ccp_alpha=alpha),
                X_cls, y_cls, cv=5, scoring='accuracy')
            metric_cols({
                "CV Mean": f"{cv_sc.mean()*100:.2f}%",
                "CV Std":  f"±{cv_sc.std()*100:.2f}%"
            })

    with tab_reg:
        dt_reg_alpha = st.slider("Regression ccp_alpha", 0.0, 5.0, 0.0, key="dt_reg_alpha")
        if st.button("▶ Run Decision Tree Regression", key="run_dt_reg"):
            with st.spinner("Training…"):
                results = run_splits_reg(
                    lambda: DecisionTreeRegressor(random_state=42, ccp_alpha=dt_reg_alpha),
                    X_reg, y_reg, SPLITS, scale=False)
            res_df = pd.DataFrame(results)
            st.dataframe(res_df, use_container_width=True, hide_index=True)
            plot_compare_bars(res_df['Split'], res_df['R²'],
                              'R² Score', 'Decision Tree Regression R²', '#1D9E75')

# ─────────────────────────────────────────────────────────────────
# ══════════════════ PAGE: SVM ══════════════════
# ─────────────────────────────────────────────────────────────────
elif "SVM" in page:
    st.title("⚙️ Support Vector Machine (SVM)")
    st.markdown('<div class="axora-card">', unsafe_allow_html=True)
    st.markdown("""
    SVM maximises the margin between classes. **Linear** and **RBF** kernels compared.
    Best config: **RBF, C=10, γ=0.1** → 71.27% accuracy.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    tab_cls, tab_reg = st.tabs(["🎯 Classification", "📈 Regression (SVR)"])

    with tab_cls:
        col1, col2, col3 = st.columns(3)
        with col1:
            kernel = st.selectbox("Kernel", ["rbf","linear","poly"], key="svm_kernel")
        with col2:
            C = st.select_slider("C", [0.1,1,10,100], value=10, key="svm_C")
        with col3:
            gamma_opt = st.selectbox("Gamma", ["scale","auto","0.01","0.1","1"], key="svm_gamma")

        gamma_val = gamma_opt if gamma_opt in ["scale","auto"] else float(gamma_opt)

        if st.button("▶ Run SVM Classification", key="run_svm_cls"):
            with st.spinner("Training SVM (this may take ~30s)…"):
                X_tr, X_te, y_tr, y_te = train_test_split(
                    X_cls, y_cls, test_size=0.2, random_state=42, stratify=y_cls)
                sc = StandardScaler()
                X_tr = sc.fit_transform(X_tr); X_te = sc.transform(X_te)
                m = SVC(kernel=kernel, C=C, gamma=gamma_val, random_state=42, probability=True)
                m.fit(X_tr, y_tr); yp = m.predict(X_te)

            metric_cols({
                "Accuracy":  f"{accuracy_score(y_te,yp)*100:.2f}%",
                "Macro F1":  f"{f1_score(y_te,yp,average='macro'):.3f}",
                "Kernel":    kernel.upper(),
                "C":         C
            })

            c1, c2 = st.columns(2)
            with c1:
                plot_confusion(confusion_matrix(y_te,yp), CLASS_NAMES, f"Confusion Matrix ({kernel})")
            with c2:
                rpt = classification_report(y_te, yp, target_names=CLASS_NAMES, output_dict=True)
                st.dataframe(pd.DataFrame(rpt).T.round(3), use_container_width=True)

            st.subheader("Cross-Split Results")
            results = run_splits_cls(
                lambda: SVC(kernel=kernel, C=C, gamma=gamma_val, random_state=42),
                X_cls, y_cls, SPLITS)
            res_df = pd.DataFrame(results)
            st.dataframe(res_df, use_container_width=True, hide_index=True)
            plot_compare_bars(res_df['Split'], res_df['Accuracy'],
                              'Accuracy (%)', 'SVM Accuracy Across Splits', '#534AB7')

    with tab_reg:
        svr_C = st.select_slider("SVR C", [0.1,1,10,100], value=10, key="svr_C")
        svr_eps = st.slider("Epsilon", 0.01, 1.0, 0.1, key="svr_eps")
        if st.button("▶ Run SVR Regression", key="run_svr"):
            with st.spinner("Training SVR…"):
                results = run_splits_reg(
                    lambda: SVR(kernel='rbf', C=svr_C, epsilon=svr_eps),
                    X_reg, y_reg, SPLITS)
            res_df = pd.DataFrame(results)
            st.dataframe(res_df, use_container_width=True, hide_index=True)
            plot_compare_bars(res_df['Split'], res_df['R²'],
                              'R² Score', 'SVR Regression R²', '#534AB7')

# ─────────────────────────────────────────────────────────────────
# ══════════════════ PAGE: NEURAL NETWORK ══════════════════
# ─────────────────────────────────────────────────────────────────
elif "Neural Network" in page:
    st.title("🧠 Neural Network (MLP)")
    st.markdown('<div class="axora-card">', unsafe_allow_html=True)
    st.markdown("""
    MLP implemented with **TensorFlow/Keras** — `BatchNormalization`, `Dropout`,
    `ReduceLROnPlateau`, and `EarlyStopping (patience=15)`.
    Corrected accuracy after leakage fix: **74.76 – 74.80%**.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    try:
        import tensorflow as tf
        from tensorflow.keras.utils import to_categorical
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Activation
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        TF_OK = True
    except ImportError:
        TF_OK = False
        st.error("TensorFlow not installed. Run: `pip install tensorflow`")

    if TF_OK:
        tab_cls, tab_reg = st.tabs(["🎯 Classification", "📈 Regression"])

        with tab_cls:
            col1, col2, col3 = st.columns(3)
            with col1:
                epochs = st.slider("Max Epochs", 30, 200, 100, key="nn_epochs")
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
                    m.compile(optimizer=Adam(0.001),
                              loss='categorical_crossentropy', metrics=['accuracy'])
                    return m

                split_results = []
                for sname, ts in SPLITS:
                    with st.spinner(f"Training {sname} split…"):
                        X_tr, X_te, y_tr, y_te = train_test_split(
                            X_cls, y_cls, test_size=ts, random_state=SEED, stratify=y_cls)
                        sc = StandardScaler()
                        X_tr = sc.fit_transform(X_tr); X_te = sc.transform(X_te)
                        y_tr_cat = to_categorical(y_tr, 4)

                        model = build_cls(X_tr.shape[1], dropout)
                        cb = [
                            EarlyStopping(monitor='val_loss', patience=patience,
                                          restore_best_weights=True, verbose=0),
                            ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                             patience=5, min_lr=1e-6, verbose=0)
                        ]
                        hist = model.fit(X_tr, y_tr_cat, epochs=epochs, batch_size=32,
                                         validation_split=0.15, callbacks=cb, verbose=0)
                        yp = np.argmax(model.predict(X_te, verbose=0), axis=1)
                        acc = accuracy_score(y_te, yp)
                        split_results.append({
                            'Split': sname,
                            'Accuracy': f"{acc*100:.2f}%",
                            'Epochs run': len(hist.history['accuracy'])
                        })

                st.dataframe(pd.DataFrame(split_results), use_container_width=True, hide_index=True)

                # Training curves for last model
                st.subheader("Training Curves (last split)")
                fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                axes[0].plot(hist.history['accuracy'],     color='#3B8BD4', label='Train')
                axes[0].plot(hist.history['val_accuracy'], color='#1D9E75', label='Val')
                axes[0].set_title('Accuracy'); axes[0].set_xlabel('Epoch'); axes[0].legend()
                axes[1].plot(hist.history['loss'],     color='#3B8BD4', label='Train')
                axes[1].plot(hist.history['val_loss'], color='#E24B4A', label='Val')
                axes[1].set_title('Loss'); axes[1].set_xlabel('Epoch'); axes[1].legend()
                show_fig(fig)

                # Final confusion matrix
                plot_confusion(confusion_matrix(y_te, yp), CLASS_NAMES,
                               "Confusion Matrix (last split)")

        with tab_reg:
            if st.button("▶ Train NN Regression", key="run_nn_reg"):
                SEED = 42; np.random.seed(SEED); tf.random.set_seed(SEED)

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
                        X_tr, X_te, y_tr, y_te = train_test_split(
                            X_reg, y_reg, test_size=ts, random_state=SEED)
                        sc = StandardScaler()
                        X_tr = sc.fit_transform(X_tr); X_te = sc.transform(X_te)
                        y_sc = StandardScaler()
                        y_tr_s = y_sc.fit_transform(y_tr.reshape(-1,1)).ravel()

                        model = build_reg(X_tr.shape[1], dropout)
                        cb = [EarlyStopping(monitor='val_loss', patience=15,
                                            restore_best_weights=True, verbose=0)]
                        model.fit(X_tr, y_tr_s, epochs=100, batch_size=32,
                                  validation_split=0.15, callbacks=cb, verbose=0)
                        yp = y_sc.inverse_transform(
                            model.predict(X_te, verbose=0)).ravel()
                        reg_results.append({
                            'Split': sname,
                            'R²':   round(r2_score(y_te, yp), 4),
                            'RMSE': round(np.sqrt(mean_squared_error(y_te, yp)), 3),
                            'MAE':  round(mean_absolute_error(y_te, yp), 3)
                        })

                st.dataframe(pd.DataFrame(reg_results), use_container_width=True, hide_index=True)
                plot_compare_bars(
                    [r['Split'] for r in reg_results],
                    [r['R²'] for r in reg_results],
                    'R² Score', 'NN Regression R² Across Splits', '#3B8BD4')

# ─────────────────────────────────────────────────────────────────
# ══════════════════ PAGE: MODEL COMPARISON ══════════════════
# ─────────────────────────────────────────────────────────────────
elif "Comparison" in page:
    st.title("🏆 Model Comparison")

    st.markdown('<div class="axora-card">', unsafe_allow_html=True)
    st.subheader("Classification Performance Summary")
    cls_summary = pd.DataFrame([
        {"Model":"KNN (k=22)",           "Best Accuracy":"63.09%","Macro F1":"0.63","Best Split":"80:20","5-Fold CV":"61.9%"},
        {"Model":"Decision Tree (pruned)","Best Accuracy":"72.15%","Macro F1":"0.71","Best Split":"80:20","5-Fold CV":"70.39%"},
        {"Model":"SVM (RBF, C=10)",       "Best Accuracy":"71.27%","Macro F1":"0.71","Best Split":"80:20","5-Fold CV":"70.8%"},
        {"Model":"Neural Network (MLP)",  "Best Accuracy":"74.80%","Macro F1":"~0.74","Best Split":"70:30","5-Fold CV":"73.98%"},
        {"Model":"Linear Reg. (adapted)", "Best Accuracy":"~50%",  "Macro F1":"~0.48","Best Split":"N/A",  "5-Fold CV":"N/A"},
    ])
    st.dataframe(cls_summary, use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="axora-card">', unsafe_allow_html=True)
    st.subheader("Regression Performance Summary")
    reg_summary = pd.DataFrame([
        {"Model":"Neural Network",  "Best R²":"0.802","Best RMSE (cm)":"17.77","Best Split":"80:20 / 70:30"},
        {"Model":"Linear Regression","Best R²":"0.8033","Best RMSE (cm)":"17.89","Best Split":"70:30"},
        {"Model":"SVM (SVR)",        "Best R²":"0.79", "Best RMSE (cm)":"18.4", "Best Split":"80:20"},
        {"Model":"Decision Tree",    "Best R²":"0.76", "Best RMSE (cm)":"20.1", "Best Split":"80:20"},
        {"Model":"KNN Regression",   "Best R²":"0.788","Best RMSE (cm)":"13.8 (MAE)","Best Split":"70:30"},
    ])
    st.dataframe(reg_summary, use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Visual comparison
    st.subheader("Visual Comparison")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    models_cls  = ["KNN","Decision\nTree","SVM","Neural\nNetwork","Linear\nReg."]
    accs_cls    = [63.09, 72.15, 71.27, 74.80, 50.0]
    colors_cls  = ['#3B8BD4','#534AB7','#1D9E75','#E24B4A','#85B7EB']
    bars = axes[0].bar(models_cls, accs_cls, color=colors_cls, edgecolor='#0D1B2A', width=0.55)
    for b in bars:
        axes[0].text(b.get_x()+b.get_width()/2, b.get_height()+0.5,
                     f'{b.get_height():.1f}%', ha='center', color='#C8D8EA', fontsize=9)
    axes[0].set_ylim(0, 90); axes[0].set_ylabel('Best Accuracy (%)')
    axes[0].set_title('Classification Accuracy')
    axes[0].axhline(72, color='#E24B4A', linestyle='--', lw=1, alpha=0.6, label='~72% ceiling')
    axes[0].legend()

    models_reg = ["Neural\nNetwork","Linear\nReg.","SVM\n(SVR)","Decision\nTree","KNN\nReg."]
    r2s        = [0.802, 0.803, 0.79, 0.76, 0.788]
    bars2 = axes[1].bar(models_reg, r2s, color=colors_cls, edgecolor='#0D1B2A', width=0.55)
    for b in bars2:
        axes[1].text(b.get_x()+b.get_width()/2, b.get_height()+0.003,
                     f'{b.get_height():.3f}', ha='center', color='#C8D8EA', fontsize=9)
    axes[1].set_ylim(0, 1.0); axes[1].set_ylabel('R² Score')
    axes[1].set_title('Regression R² Score')

    show_fig(fig)

    # Radar chart
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
            age     = st.number_input("Age (yr)",        18, 80, 30)
            gender  = st.selectbox("Gender",             ["M","F"])
            height  = st.number_input("Height (cm)",     140.0, 200.0, 170.0)
            weight  = st.number_input("Weight (kg)",     30.0, 150.0, 68.0)
        with c2:
            bf      = st.number_input("Body Fat (%)",    3.0, 60.0, 22.0)
            dia     = st.number_input("Diastolic BP",    50.0, 130.0, 79.0)
            sys     = st.number_input("Systolic BP",     80.0, 180.0, 130.0)
        with c3:
            grip    = st.number_input("Grip Force (kg)", 5.0, 80.0, 37.0)
            flex    = st.number_input("Sit&Bend (cm)",  -20.0, 50.0, 15.0)
            situps  = st.number_input("Sit-ups",         0.0, 80.0, 40.0)

        model_choice = st.selectbox("Classifier",
            ["KNN (k=22)","Decision Tree","SVM (RBF)"])
        submitted = st.form_submit_button("🚀 Predict")

    if submitted:
        bmi = weight / ((height/100)**2)
        g   = 0 if gender == 'M' else 1

        x_cls = np.array([[age, g, height, weight, bf, dia, sys,
                           grip, flex, situps,
                           # broad jump placeholder = median
                           df['broad jump_cm'].median(), bmi]])
        x_reg = np.array([[age, g, height, weight, bf, dia, sys,
                           grip, flex, situps, bmi]])

        # Fit models on full data
        sc_cls = StandardScaler(); X_cls_sc = sc_cls.fit_transform(X_cls)
        sc_reg = StandardScaler(); X_reg_sc = sc_reg.fit_transform(X_reg)
        y_reg_sc = StandardScaler(); y_reg_s = y_reg_sc.fit_transform(y_reg.reshape(-1,1)).ravel()

        if "KNN" in model_choice:
            clf = KNeighborsClassifier(n_neighbors=22, n_jobs=-1).fit(X_cls_sc, y_cls)
            rgr = KNeighborsRegressor(n_neighbors=35, n_jobs=-1).fit(X_reg_sc, y_reg)
        elif "Decision Tree" in model_choice:
            clf = DecisionTreeClassifier(random_state=42, ccp_alpha=0.000377).fit(X_cls, y_cls)
            rgr = DecisionTreeRegressor(random_state=42).fit(X_reg, y_reg)
        else:
            clf = SVC(kernel='rbf', C=10, gamma=0.1, random_state=42).fit(X_cls_sc, y_cls)
            rgr = SVR(kernel='rbf', C=10).fit(X_reg_sc, y_reg)

        needs_scale = "Linear Reg" not in model_choice and "Decision" not in model_choice
        if needs_scale:
            x_cls_in = sc_cls.transform(x_cls)
            x_reg_in = sc_reg.transform(x_reg)
        else:
            x_cls_in = x_cls
            x_reg_in = x_reg

        pred_cls   = clf.predict(x_cls_in)[0]
        class_name = CLASS_NAMES[pred_cls]
        if needs_scale:
            pred_jump = rgr.predict(x_reg_in)[0]
        else:
            pred_jump = rgr.predict(x_reg_in)[0]

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
            <div style="color:{colour}; font-size:1.4rem; font-weight:700; margin:8px 0">
                Performance Class: {class_name}
            </div>
            <div style="color:#C8D8EA">{advice}</div>
        </div>""", unsafe_allow_html=True)

        metric_cols({
            "Predicted Class":      class_name,
            "Predicted Jump":       f"{pred_jump:.1f} cm",
            "BMI":                  f"{bmi:.1f}",
            "Model Used":           model_choice.split()[0]
        })

        # Feature bar
        st.subheader("Your Input Profile")
        feat_vals = pd.DataFrame({
            'Feature': ['Age','Height','Weight','Body Fat %','Diastolic','Systolic',
                        'Grip Force','Sit&Bend','Sit-ups','BMI'],
            'Your Value': [age, height, weight, bf, dia, sys, grip, flex, situps, round(bmi,1)],
            'Dataset Mean': [
                df['age'].mean(), df['height_cm'].mean(), df['weight_kg'].mean(),
                df['body fat_%'].mean(), df['diastolic'].mean(), df['systolic'].mean(),
                df['gripForce'].mean(), df['sit and bend forward_cm'].mean(),
                df['sit-ups counts'].mean(), df['BMI'].mean()
            ]
        }).round(1)
        fig, ax = plt.subplots(figsize=(10, 4))
        x = np.arange(len(feat_vals))
        ax.bar(x - 0.2, feat_vals['Your Value'],   width=0.38, label='You',         color='#3B8BD4')
        ax.bar(x + 0.2, feat_vals['Dataset Mean'], width=0.38, label='Dataset Mean',color='#534AB7', alpha=0.7)
        ax.set_xticks(x); ax.set_xticklabels(feat_vals['Feature'], rotation=40, ha='right', fontsize=8)
        ax.set_title("Your Values vs Dataset Averages"); ax.legend()
        show_fig(fig)

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
