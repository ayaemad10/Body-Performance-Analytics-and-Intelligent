"""
╔══════════════════════════════════════════════════════════════════╗
║   AXORA — Body Performance Analytics & Intelligent System        ║
║   Streamlit Application  |  5 Models  |  Team Axora  |  2026    ║
╚══════════════════════════════════════════════════════════════════╝
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import base64, os

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
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
# LOGO — load from file if present, otherwise inline SVG fallback
# ─────────────────────────────────────────────────────────────────
LOGO_PATHS = [
    "axora_team_logo_2026.svg",
    "/mount/src/body-performance-analytics-and-intelligent/axora_team_logo_2026.svg",
]

def _load_logo():
    for p in LOGO_PATHS:
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                return f.read()
    # Minimal inline fallback (matches brand colours)
    return """<svg viewBox="0 0 680 420" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="bg2" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#3B8BD4"/><stop offset="100%" stop-color="#534AB7"/>
    </linearGradient>
    <linearGradient id="ac2" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0%" stop-color="#1D9E75"/><stop offset="100%" stop-color="#3B8BD4"/>
    </linearGradient>
  </defs>
  <rect x="40" y="20" width="600" height="380" rx="20" fill="#0D1B2A" stroke="#534AB7" stroke-width="1" opacity="0.97"/>
  <polygon points="340,70 374,90 374,130 340,150 306,130 306,90" fill="none" stroke="url(#bg2)" stroke-width="2"/>
  <polygon points="340,80 366,96 366,124 340,140 314,124 314,96" fill="#1a2a45" opacity="0.9"/>
  <line x1="340" y1="92" x2="326" y2="130" stroke="url(#bg2)" stroke-width="2.5" stroke-linecap="round"/>
  <line x1="340" y1="92" x2="354" y2="130" stroke="url(#bg2)" stroke-width="2.5" stroke-linecap="round"/>
  <line x1="330" y1="116" x2="350" y2="116" stroke="url(#ac2)" stroke-width="2" stroke-linecap="round"/>
  <circle cx="340" cy="60" r="5" fill="#378ADD" opacity="0.9"/>
  <circle cx="386" cy="82" r="4" fill="#534AB7" opacity="0.9"/>
  <circle cx="390" cy="140" r="4" fill="#1D9E75" opacity="0.9"/>
  <circle cx="340" cy="162" r="5" fill="#378ADD" opacity="0.9"/>
  <circle cx="290" cy="140" r="4" fill="#534AB7" opacity="0.9"/>
  <circle cx="294" cy="82" r="4" fill="#1D9E75" opacity="0.9"/>
  <circle cx="340" cy="110" r="68" fill="none" stroke="#3B8BD4" stroke-width="0.8" stroke-dasharray="4 6" opacity="0.3"/>
  <text x="340" y="214" text-anchor="middle" font-family="Georgia,serif" font-size="52" font-weight="700" letter-spacing="12" fill="url(#bg2)">AXORA</text>
  <rect x="180" y="222" width="320" height="2.5" rx="1.5" fill="url(#ac2)" opacity="0.9"/>
  <text x="340" y="252" text-anchor="middle" font-family="Courier New,monospace" font-size="12" letter-spacing="4" fill="#85B7EB" opacity="0.85">INTELLIGENCE · ILLUMINATED</text>
  <rect x="190" y="272" width="300" height="34" fill="#0F1E33" stroke="#378ADD" stroke-width="1" opacity="0.9"/>
  <text x="340" y="293" text-anchor="middle" font-family="Courier New,monospace" font-size="11" letter-spacing="2" fill="#5DCAA5">DATA ANALYSIS  ·  ARTIFICIAL INTELLIGENCE</text>
  <text x="340" y="336" text-anchor="middle" font-family="Georgia,serif" font-size="10.5" letter-spacing="1" fill="#4d7fa8" opacity="0.9">Aya Emad · Aya Samir · Aya Shaaban · Aya Ashraf · Alaa Esawy · Amira Salama</text>
  <line x1="100" y1="356" x2="580" y2="356" stroke="#534AB7" stroke-width="0.5" opacity="0.4"/>
  <text x="340" y="378" text-anchor="middle" font-family="Courier New,monospace" font-size="9" letter-spacing="3" fill="#3C3489" opacity="0.7">EST. 2026 · TEAM PROJECT</text>
</svg>"""

LOGO_SVG = _load_logo()

# ─────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background:#0D1B2A; }
[data-testid="stSidebar"]          { background:#0A1628; border-right:1px solid #1D4570; }
[data-testid="stSidebar"] *        { color:#C8D8EA !important; }
h1,h2,h3,h4,h5,h6 { color:#3B8BD4 !important; }
p, li, label, .stMarkdown { color:#C8D8EA !important; }

.axora-card {
    background:#0F1E33; border:1px solid #1D4570;
    border-radius:14px; padding:22px 26px; margin-bottom:18px;
}
.metric-box {
    background:linear-gradient(135deg,#112640,#0F1E33);
    border:1px solid #1D9E75; border-radius:10px;
    padding:16px; text-align:center; margin:6px;
}
.metric-val { font-size:2rem; font-weight:700; color:#3B8BD4; }
.metric-lbl { font-size:0.78rem; color:#85B7EB; text-transform:uppercase; letter-spacing:1px; }

.badge-blue   { background:#1D4570; color:#85B7EB; border-radius:8px; padding:3px 10px; font-size:0.78rem; display:inline-block; margin:2px; }
.badge-green  { background:#0D3D2A; color:#1D9E75; border-radius:8px; padding:3px 10px; font-size:0.78rem; display:inline-block; margin:2px; }
.badge-purple { background:#1E1A40; color:#AFA9EC; border-radius:8px; padding:3px 10px; font-size:0.78rem; display:inline-block; margin:2px; }

.stButton>button {
    background:linear-gradient(90deg,#1F5C9E,#534AB7);
    color:white; border:none; border-radius:8px;
    padding:10px 28px; font-weight:600; transition:all .2s;
}
.stButton>button:hover { opacity:.85; transform:translateY(-1px); }
.stSelectbox label, .stSlider label, .stNumberInput label { color:#85B7EB !important; }

/* Welcome page */
.welcome-wrap {
    display:flex; flex-direction:column;
    align-items:center; justify-content:center;
    min-height:75vh; text-align:center;
}
.welcome-title {
    font-size:2.6rem; font-weight:900;
    background:linear-gradient(90deg,#3B8BD4,#534AB7);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    letter-spacing:2px; margin-top:10px;
}
.welcome-sub {
    color:#85B7EB; font-size:1.1rem; margin:6px 0 22px 0;
    font-family:'Courier New',monospace; letter-spacing:3px;
}
.welcome-desc {
    color:#C8D8EA; max-width:620px; line-height:1.75;
    font-size:1.0rem; margin-bottom:28px;
}

/* Prediction result cards */
.pred-card {
    border-radius:12px; padding:18px 20px; margin:6px 0;
    border:1px solid; text-align:center;
}
.pred-model  { font-size:0.72rem; color:#85B7EB; text-transform:uppercase; letter-spacing:1px; margin-bottom:4px; }
.pred-class  { font-size:2.2rem; font-weight:800; }
.pred-jump   { font-size:1.0rem; color:#C8D8EA; margin-top:4px; }
.pred-badge  { display:inline-block; border-radius:6px; padding:2px 9px; font-size:0.75rem; font-weight:600; margin-top:6px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# MODEL METADATA
# ─────────────────────────────────────────────────────────────────
MODEL_META = {
    "🤖  KNN":               {"icon":"🤖","short":"KNN",         "cls_accuracy":"63.09%","reg_r2":"0.786","best_split":"80:20","tags":["k=22","StandardScaler"],"color":"#3B8BD4","cls_f1":"0.63","cls_cv":"61.9%","reg_rmse":"13.8 cm (MAE)"},
    "📈  Linear Regression": {"icon":"📈","short":"Lin. Reg.",   "cls_accuracy":"~50%",  "reg_r2":"0.803","best_split":"70:30","tags":["OLS","No scaling"],     "color":"#85B7EB","cls_f1":"~0.48","cls_cv":"N/A","reg_rmse":"17.89 cm"},
    "🌳  Decision Tree":     {"icon":"🌳","short":"Dec. Tree",   "cls_accuracy":"72.15%","reg_r2":"0.760","best_split":"80:20","tags":["α=0.000377","Depth 14"],"color":"#534AB7","cls_f1":"0.71","cls_cv":"70.39%","reg_rmse":"20.1 cm"},
    "⚙️   SVM":              {"icon":"⚙️","short":"SVM",         "cls_accuracy":"71.27%","reg_r2":"0.790","best_split":"80:20","tags":["RBF","C=10","γ=0.1"],  "color":"#1D9E75","cls_f1":"0.71","cls_cv":"70.8%","reg_rmse":"18.4 cm"},
    "🧠  Neural Network":    {"icon":"🧠","short":"MLP",         "cls_accuracy":"74.80%","reg_r2":"0.802","best_split":"70:30","tags":["256→128→64","Adam"],    "color":"#E24B4A","cls_f1":"~0.74","cls_cv":"73.98%","reg_rmse":"17.77 cm"},
}

CLASS_INFO = {
    'A': ("🥇 Excellent", "#1D9E75",  "Outstanding fitness — maintain your regime."),
    'B': ("🥈 Good",      "#3B8BD4",  "Above average — targeted strength work recommended."),
    'C': ("🥉 Average",   "#534AB7",  "Moderate fitness — increase endurance training."),
    'D': ("⚠️ Below Avg", "#E24B4A",  "Below average — structured fitness programme advised."),
}

# ─────────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data(uploaded=None):
    if uploaded is not None:
        return pd.read_csv(uploaded)
    for p in ["bodyPerformance.csv","bodyPerformance_-_Copy.csv",
              "/mount/src/body-performance-analytics-and-intelligent/bodyPerformance.csv"]:
        try:
            return pd.read_csv(p)
        except FileNotFoundError:
            continue
    np.random.seed(42); n = 2000
    return pd.DataFrame({
        'age':np.random.uniform(21,64,n),'gender':np.random.choice(['M','F'],n),
        'height_cm':np.random.normal(168,8,n),'weight_kg':np.random.normal(67,12,n),
        'body fat_%':np.random.normal(23,7,n).clip(3,45),
        'diastolic':np.random.normal(79,11,n).clip(50,120),
        'systolic':np.random.normal(130,15,n).clip(80,180),
        'gripForce':np.random.normal(37,11,n).clip(5,70),
        'sit and bend forward_cm':np.random.normal(15,8,n).clip(-20,50),
        'sit-ups counts':np.random.normal(40,14,n).clip(0,80),
        'broad jump_cm':np.random.normal(190,40,n).clip(50,310),
        'class':np.random.choice(['A','B','C','D'],n)
    })

@st.cache_data
def preprocess(df):
    d = df.copy().drop_duplicates()
    d = d[(d['systolic']>40)&(d['diastolic']>40)]
    d['sit and bend forward_cm'] = d['sit and bend forward_cm'].clip(-20,50)
    d['BMI'] = d['weight_kg']/((d['height_cm']/100)**2)
    d['gender'] = d['gender'].map({'M':0,'F':1})
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
        for spine in ax.spines.values(): spine.set_edgecolor('#1D4570')
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

def metric_cols(metrics: dict):
    cols = st.columns(len(metrics))
    for col,(lbl,val) in zip(cols,metrics.items()):
        col.markdown(f'<div class="metric-box"><div class="metric-val">{val}</div><div class="metric-lbl">{lbl}</div></div>', unsafe_allow_html=True)

def plot_confusion(cm, classes, title):
    fig,ax = plt.subplots(figsize=(5,4))
    sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',xticklabels=classes,yticklabels=classes,ax=ax,linewidths=.5)
    ax.set_title(title,pad=12); ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    show_fig(fig)

def plot_compare_bars(labels,values,ylabel,title,color='#3B8BD4'):
    fig,ax = plt.subplots(figsize=(6,3.5))
    bars = ax.bar(labels,values,color=color,edgecolor='#1D4570',width=0.5)
    for b in bars:
        ax.text(b.get_x()+b.get_width()/2,b.get_height()+0.005,f'{b.get_height():.3f}',
                ha='center',va='bottom',color='#C8D8EA',fontsize=9)
    ax.set_ylabel(ylabel); ax.set_title(title); ax.set_ylim(0,max(values)*1.2+0.05)
    show_fig(fig)

def run_splits_cls(fn, X, y, splits, scale=True):
    out=[]
    for name,ts in splits:
        Xr,Xe,yr,ye = train_test_split(X,y,test_size=ts,random_state=42,stratify=y)
        if scale:
            sc=StandardScaler(); Xr=sc.fit_transform(Xr); Xe=sc.transform(Xe)
        m=fn(); m.fit(Xr,yr); yp=m.predict(Xe)
        out.append({'Split':name,'Accuracy':round(accuracy_score(ye,yp)*100,2),
                    'Precision':round(precision_score(ye,yp,average='macro',zero_division=0)*100,2),
                    'Recall':round(recall_score(ye,yp,average='macro',zero_division=0)*100,2),
                    'F1':round(f1_score(ye,yp,average='macro',zero_division=0)*100,2)})
    return out

def run_splits_reg(fn, X, y, splits, scale=True):
    out=[]
    for name,ts in splits:
        Xr,Xe,yr,ye = train_test_split(X,y,test_size=ts,random_state=42)
        if scale:
            sc=StandardScaler(); Xr=sc.fit_transform(Xr); Xe=sc.transform(Xe)
        m=fn(); m.fit(Xr,yr); yp=m.predict(Xe)
        out.append({'Split':name,'R²':round(r2_score(ye,yp),4),
                    'RMSE':round(np.sqrt(mean_squared_error(ye,yp)),3),
                    'MAE':round(mean_absolute_error(ye,yp),3)})
    return out

def model_summary_card(meta, page_key):
    c = meta['color']
    tags = "".join([f'<span style="display:inline-block;background:#1a2a45;border:1px solid #2a4a70;border-radius:4px;padding:1px 6px;font-size:0.65rem;color:#AFA9EC;margin:2px">{t}</span>' for t in meta['tags']])
    st.markdown(f"""
    <div class="axora-card" style="border-color:{c};">
      <div style="display:flex;gap:28px;flex-wrap:wrap;align-items:center;">
        <div style="flex:1;min-width:200px;">
          <div style="font-size:0.72rem;color:#85B7EB;text-transform:uppercase;letter-spacing:1px;">Classification</div>
          <div style="font-size:0.9rem;color:#C8D8EA;margin-bottom:8px;">Multi-class Performance Class (A–D)</div>
          <div style="font-size:0.72rem;color:#85B7EB;text-transform:uppercase;letter-spacing:1px;">Regression</div>
          <div style="font-size:0.9rem;color:#C8D8EA;">{("Primary task" if "Reg" in meta['short'] else "Predict broad_jump_cm")}</div>
          <div style="margin-top:8px;">{tags}</div>
        </div>
        <div style="display:flex;gap:14px;flex-wrap:wrap;">
          <div style="text-align:center;background:#112640;border:1px solid #1D4570;border-radius:10px;padding:12px 18px;">
            <div style="font-size:1.7rem;font-weight:800;color:#3B8BD4;">{meta['cls_accuracy']}</div>
            <div style="font-size:0.65rem;color:#85B7EB;text-transform:uppercase;">Best Cls Accuracy</div>
          </div>
          <div style="text-align:center;background:#112640;border:1px solid #1D9E75;border-radius:10px;padding:12px 18px;">
            <div style="font-size:1.7rem;font-weight:800;color:#1D9E75;">{meta['reg_r2']}</div>
            <div style="font-size:0.65rem;color:#85B7EB;text-transform:uppercase;">Regression R²</div>
          </div>
          <div style="text-align:center;background:#112640;border:1px solid #534AB7;border-radius:10px;padding:12px 18px;">
            <div style="font-size:1.7rem;font-weight:800;color:#AFA9EC;">{meta['cls_f1']}</div>
            <div style="font-size:0.65rem;color:#85B7EB;text-transform:uppercase;">Macro F1</div>
          </div>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

SPLITS = [("80:20",0.20),("70:30",0.30),("50:50",0.50)]
CLASS_NAMES = ['A','B','C','D']

# ─────────────────────────────────────────────────────────────────
# SIDEBAR  ── logo appears ONCE here only
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(LOGO_SVG, unsafe_allow_html=True)
    st.markdown('<hr style="border-color:#1D4570;margin:8px 0 12px 0">', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("📂 Upload bodyPerformance.csv", type=["csv"], label_visibility="collapsed")

    st.markdown('<hr style="border-color:#1D4570;margin:8px 0 12px 0">', unsafe_allow_html=True)
    st.markdown("**🧭 Navigation**")

    nav_page = st.radio("nav", [
        "🏠  Welcome",
        "📊  EDA",
        "🏆  Model Comparison",
        "🔮  Live Predictor",
    ], label_visibility="collapsed")

    st.markdown('<hr style="border-color:#1D4570;margin:8px 0 12px 0">', unsafe_allow_html=True)
    st.markdown("**🤖 ML Models**")

    if "sel_model" not in st.session_state:
        st.session_state.sel_model = None

    for mkey, meta in MODEL_META.items():
        is_active = st.session_state.sel_model == mkey
        bc = meta['color'] if is_active else "#1D4570"
        bg = "#0D2820" if is_active else "#0F1E33"
        tags_h = "".join([f'<span style="display:inline-block;background:#1a2a45;border:1px solid #2a4a70;border-radius:3px;padding:1px 5px;font-size:0.62rem;color:#AFA9EC;margin:1px">{t}</span>' for t in meta['tags']])
        st.markdown(f"""
        <div style="background:{bg};border:1px solid {bc};border-radius:10px;padding:10px 12px;margin-bottom:6px;">
          <div style="font-size:0.86rem;font-weight:700;color:{meta['color']};margin-bottom:5px;">{meta['icon']} {meta['short']}</div>
          <div style="display:flex;justify-content:space-between;margin-bottom:2px;">
            <span style="font-size:0.65rem;color:#85B7EB;text-transform:uppercase">Cls Acc.</span>
            <span style="font-size:0.83rem;font-weight:700;color:#3B8BD4">{meta['cls_accuracy']}</span>
          </div>
          <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
            <span style="font-size:0.65rem;color:#85B7EB;text-transform:uppercase">Reg R²</span>
            <span style="font-size:0.83rem;font-weight:700;color:#1D9E75">{meta['reg_r2']}</span>
          </div>
          <div style="margin-top:2px;">{tags_h}</div>
        </div>""", unsafe_allow_html=True)
        lbl = "✓ Selected" if is_active else f"Open {meta['short']}"
        if st.button(lbl, key=f"btn_{mkey}", use_container_width=True):
            st.session_state.sel_model = None if is_active else mkey
            st.rerun()

    st.markdown('<hr style="border-color:#1D4570;margin:8px 0 8px 0">', unsafe_allow_html=True)
    st.markdown('<span class="badge-green">Team Axora</span><span class="badge-blue">AI & ML 2024–25</span>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────
raw_df = load_data(uploaded_file)
df, le = preprocess(raw_df)

FEATURES_CLS = ['age','gender','height_cm','weight_kg','body fat_%','diastolic','systolic',
                 'gripForce','sit and bend forward_cm','sit-ups counts','broad jump_cm','BMI']
FEATURES_REG = ['age','gender','height_cm','weight_kg','body fat_%','diastolic','systolic',
                 'gripForce','sit and bend forward_cm','sit-ups counts','BMI']
X_cls = df[FEATURES_CLS].values
y_cls = df['class_enc'].values
X_reg = df[FEATURES_REG].values
y_reg = df['broad jump_cm'].values

# ─────────────────────────────────────────────────────────────────
# PAGE ROUTING
# ─────────────────────────────────────────────────────────────────
sel = st.session_state.get("sel_model", None)
page = sel if sel else nav_page

# ══════════════════════════════════════════════════════
# 🏠 WELCOME — logo + title only, clean landing page
# ══════════════════════════════════════════════════════
if "Welcome" in page:
    st.markdown("""
    <div class="welcome-wrap">
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align:center; padding: 60px 20px 0 20px;">
        <div class="welcome-title">Body Performance Analytics</div>
        <div class="welcome-sub">INTELLIGENT CLASSIFICATION SYSTEM</div>
        <div class="welcome-desc">
            A comparative study of <strong style="color:#3B8BD4">five machine learning models</strong>
            applied to the Body Performance Dataset — 13,393 records, 12 physiological features,
            four fitness performance classes (A–D).
        </div>
        <div style="margin-bottom:30px;">
            <span class="badge-green">13,393 Records</span>
            <span class="badge-blue">12 Features</span>
            <span class="badge-purple">5 ML Models</span>
            <span class="badge-blue">KNN · LR · DT · SVM · MLP</span>
        </div>
        <div style="color:#4d7fa8;font-family:'Courier New',monospace;font-size:0.82rem;letter-spacing:2px;">
            ← Use the sidebar to navigate or open a model
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # Quick stats row
    st.markdown('<br>', unsafe_allow_html=True)
    cols = st.columns(5)
    stats = [
        ("🤖 KNN","63.09%","0.786"),
        ("📈 Lin. Reg.","~50%*","0.803"),
        ("🌳 Dec. Tree","72.15%","0.760"),
        ("⚙️ SVM","71.27%","0.790"),
        ("🧠 MLP","74.80%","0.802"),
    ]
    for col,(name,acc,r2) in zip(cols,stats):
        col.markdown(f"""
        <div style="background:#0F1E33;border:1px solid #1D4570;border-radius:10px;
                    padding:14px 10px;text-align:center;margin:4px 2px;">
          <div style="font-size:0.85rem;font-weight:700;color:#3B8BD4;margin-bottom:6px;">{name}</div>
          <div style="font-size:1.3rem;font-weight:800;color:#3B8BD4;">{acc}</div>
          <div style="font-size:0.65rem;color:#85B7EB;letter-spacing:1px">CLS ACC</div>
          <div style="font-size:1.1rem;font-weight:700;color:#1D9E75;margin-top:4px;">R² {r2}</div>
          <div style="font-size:0.65rem;color:#85B7EB;letter-spacing:1px">REGRESSION</div>
        </div>""", unsafe_allow_html=True)
    st.markdown('<p style="text-align:center;font-size:0.75rem;color:#4d7fa8;margin-top:4px;">* Linear Regression adapted for classification — task mismatch</p>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
# 📊 EDA
# ══════════════════════════════════════════════════════
elif "EDA" in page:
    st.title("📊 Exploratory Data Analysis")
    tab1,tab2,tab3,tab4 = st.tabs(["📈 Distributions","📦 Boxplots","🔥 Correlations","🔵 Pairplot"])
    num_cols = ['age','height_cm','weight_kg','body fat_%','diastolic',
                'systolic','gripForce','sit and bend forward_cm','sit-ups counts','broad jump_cm']

    with tab1:
        fig,axes = plt.subplots(2,5,figsize=(18,7)); axes=axes.flatten()
        for i,c in enumerate(num_cols):
            axes[i].hist(raw_df[c].dropna(),bins=30,color='#3B8BD4',alpha=0.8,edgecolor='#0D1B2A')
            axes[i].set_title(c,fontsize=9)
        plt.tight_layout(); show_fig(fig)

    with tab2:
        scv=StandardScaler(); scaled=pd.DataFrame(scv.fit_transform(raw_df[num_cols].dropna()),columns=num_cols)
        fig,ax=plt.subplots(figsize=(14,5))
        scaled.boxplot(ax=ax,notch=False,patch_artist=True,
                       boxprops=dict(facecolor='#1D4570',color='#3B8BD4'),
                       medianprops=dict(color='#1D9E75',linewidth=2),
                       whiskerprops=dict(color='#85B7EB'),capprops=dict(color='#85B7EB'),
                       flierprops=dict(marker='o',color='#E24B4A',alpha=0.4,markersize=3))
        ax.set_title("Standardised Feature Boxplots"); plt.xticks(rotation=40,ha='right',fontsize=8)
        show_fig(fig)

    with tab3:
        enc=raw_df.copy(); enc['gender']=enc['gender'].map({'M':0,'F':1})
        enc['class_n']=LabelEncoder().fit_transform(enc['class'])
        corr=enc[num_cols+['class_n']].corr()
        fig,ax=plt.subplots(figsize=(12,9))
        sns.heatmap(corr,annot=True,fmt='.2f',cmap='coolwarm',center=0,linewidths=.4,ax=ax,annot_kws={'size':7})
        ax.set_title("Feature Correlation Matrix"); show_fig(fig)
        class_corr=corr['class_n'].drop('class_n').sort_values()
        fig2,ax2=plt.subplots(figsize=(9,4))
        colors=['#E24B4A' if v<0 else '#1D9E75' for v in class_corr.values]
        ax2.barh(class_corr.index,class_corr.values,color=colors,edgecolor='#0D1B2A')
        ax2.axvline(0,color='#85B7EB',linewidth=1); ax2.set_title("Pearson r with Performance Class"); show_fig(fig2)

    with tab4:
        samp=raw_df.sample(min(400,len(raw_df)),random_state=42)
        palette={'A':'#3B8BD4','B':'#534AB7','C':'#1D9E75','D':'#E24B4A'}
        g=sns.pairplot(samp[['gripForce','sit-ups counts','broad jump_cm','body fat_%','class']],
                       hue='class',palette=palette,plot_kws={'alpha':0.4,'s':14},diag_kind='kde')
        g.fig.patch.set_facecolor('#0D1B2A')
        for ax_ in g.axes.flatten():
            if ax_: ax_.set_facecolor('#0F1E33')
        st.pyplot(g.fig,use_container_width=True); plt.close()

# ══════════════════════════════════════════════════════
# KNN
# ══════════════════════════════════════════════════════
elif "KNN" in page:
    meta = MODEL_META["🤖  KNN"]
    st.title("🤖 K-Nearest Neighbors (KNN)")
    model_summary_card(meta, page)
    tab_cls,tab_reg = st.tabs(["🎯 Classification","📈 Regression"])

    with tab_cls:
        col1,col2 = st.columns([1,2])
        with col1:
            k_max   = st.slider("Max k to evaluate",5,50,25,key="knn_kmax")
            split_s = st.selectbox("Split",["80:20","70:30","50:50"],key="knn_split")
        ts_map={"80:20":0.20,"70:30":0.30,"50:50":0.50}
        if st.button("▶ Run KNN Classification",key="run_knn_cls"):
            with st.spinner("Evaluating k values…"):
                Xr,Xe,yr,ye=train_test_split(X_cls,y_cls,test_size=ts_map[split_s],random_state=42,stratify=y_cls)
                sc=StandardScaler(); Xr=sc.fit_transform(Xr); Xe=sc.transform(Xe)
                accs=[accuracy_score(ye,KNeighborsClassifier(n_neighbors=k,n_jobs=-1).fit(Xr,yr).predict(Xe)) for k in range(1,k_max+1)]
                bk=int(np.argmax(accs))+1; ba=max(accs)
            metric_cols({"Best k":bk,"Best Accuracy":f"{ba*100:.2f}%","Split":split_s})
            fig,ax=plt.subplots(figsize=(10,4))
            ax.plot(range(1,k_max+1),[a*100 for a in accs],color='#3B8BD4',linewidth=2)
            ax.axvline(bk,color='#1D9E75',linestyle='--',linewidth=1.5,label=f'Best k={bk} ({ba*100:.1f}%)')
            ax.set_xlabel('k'); ax.set_ylabel('Test Accuracy (%)'); ax.set_title('k vs Accuracy'); ax.legend()
            show_fig(fig)
            bm=KNeighborsClassifier(n_neighbors=bk,n_jobs=-1).fit(Xr,yr); yp=bm.predict(Xe)
            c1,c2=st.columns(2)
            with c1: plot_confusion(confusion_matrix(ye,yp),CLASS_NAMES,f"Confusion Matrix (k={bk})")
            with c2: st.dataframe(pd.DataFrame(classification_report(ye,yp,target_names=CLASS_NAMES,output_dict=True)).T.round(3),use_container_width=True)
            st.subheader("Cross-Split Comparison")
            res=run_splits_cls(lambda:KNeighborsClassifier(n_neighbors=bk,n_jobs=-1),X_cls,y_cls,SPLITS)
            st.dataframe(pd.DataFrame(res),use_container_width=True,hide_index=True)
            st.subheader("5-Fold CV")
            cv=cross_val_score(KNeighborsClassifier(n_neighbors=bk,n_jobs=-1),StandardScaler().fit_transform(X_cls),y_cls,cv=5)
            metric_cols({"CV Mean":f"{cv.mean()*100:.2f}%","CV Std":f"±{cv.std()*100:.2f}%","Min":f"{cv.min()*100:.2f}%","Max":f"{cv.max()*100:.2f}%"})

    with tab_reg:
        k_reg=st.slider("k for regression",1,50,37,key="knn_k_reg")
        if st.button("▶ Run KNN Regression",key="run_knn_reg"):
            res=run_splits_reg(lambda:KNeighborsRegressor(n_neighbors=k_reg,n_jobs=-1),X_reg,y_reg,SPLITS)
            st.dataframe(pd.DataFrame(res),use_container_width=True,hide_index=True)
            plot_compare_bars([r['Split'] for r in res],[r['R²'] for r in res],'R² Score','KNN Regression R²','#1D9E75')

# ══════════════════════════════════════════════════════
# LINEAR REGRESSION
# ══════════════════════════════════════════════════════
elif "Linear Regression" in page:
    meta = MODEL_META["📈  Linear Regression"]
    st.title("📈 Linear Regression")
    model_summary_card(meta, page)
    tab1,tab2=st.tabs(["📈 Regression (Primary)","🎯 Classification (Adapted)"])

    with tab1:
        if st.button("▶ Run Linear Regression",key="run_lr"):
            res=run_splits_reg(LinearRegression,X_reg,y_reg,SPLITS,scale=False)
            res_df=pd.DataFrame(res)
            st.dataframe(res_df,use_container_width=True,hide_index=True)
            c1,c2=st.columns(2)
            with c1: plot_compare_bars(res_df['Split'],res_df['R²'],'R² Score','R² Across Splits','#1D9E75')
            with c2: plot_compare_bars(res_df['Split'],res_df['RMSE'],'RMSE (cm)','RMSE Across Splits','#E24B4A')
            Xr,Xe,yr,ye=train_test_split(X_reg,y_reg,test_size=0.2,random_state=42)
            m=LinearRegression(); m.fit(Xr,yr); yp=m.predict(Xe); res2=ye-yp
            fig,axes=plt.subplots(1,3,figsize=(16,4.5))
            axes[0].scatter(ye,yp,alpha=0.4,color='#3B8BD4',s=10); axes[0].plot([ye.min(),ye.max()],[ye.min(),ye.max()],'r--',lw=2); axes[0].set_title('Actual vs Predicted')
            axes[1].scatter(yp,res2,alpha=0.4,color='#534AB7',s=10); axes[1].axhline(0,color='r',lw=2); axes[1].set_title('Residuals')
            axes[2].hist(res2,bins=35,color='#1D9E75',edgecolor='#0D1B2A',alpha=0.85); axes[2].set_title('Residual Dist.')
            show_fig(fig)
            st.subheader("5-Fold CV")
            cv=cross_val_score(LinearRegression(),X_reg,y_reg,cv=KFold(5,shuffle=True,random_state=42),scoring='r2')
            metric_cols({"CV Mean R²":f"{cv.mean():.4f}","CV Std":f"±{cv.std():.4f}","Min":f"{cv.min():.4f}","Max":f"{cv.max():.4f}"})

    with tab2:
        st.info("Linear Regression adapted: continuous output is binned into performance quartiles.")
        if st.button("▶ Run Adapted Classification",key="run_lr_cls"):
            results=[]
            for name,ts in SPLITS:
                Xr,Xe,yr,ye=train_test_split(X_reg,y_reg,test_size=ts,random_state=42)
                m=LinearRegression(); m.fit(Xr,yr); yp=m.predict(Xe)
                q=np.percentile(yr,[25,50,75])
                def _b(v): return 3 if v>=q[2] else (2 if v>=q[1] else (1 if v>=q[0] else 0))
                results.append({'Split':name,'Adapted Accuracy':f'{accuracy_score([_b(v) for v in ye],[_b(v) for v in yp])*100:.1f}%'})
            st.dataframe(pd.DataFrame(results),use_container_width=True,hide_index=True)
            st.warning("⚠️ ~50% confirms fundamental task mismatch — Linear Regression is not designed for classification.")

# ══════════════════════════════════════════════════════
# DECISION TREE
# ══════════════════════════════════════════════════════
elif "Decision Tree" in page:
    meta = MODEL_META["🌳  Decision Tree"]
    st.title("🌳 Decision Tree")
    model_summary_card(meta, page)
    tab_cls,tab_reg=st.tabs(["🎯 Classification","📈 Regression"])

    with tab_cls:
        alpha=st.slider("ccp_alpha",0.0,0.01,0.000377,step=0.000050,format="%.6f",key="dt_alpha")
        if st.button("▶ Run Decision Tree Classification",key="run_dt_cls"):
            res=run_splits_cls(lambda:DecisionTreeClassifier(random_state=42,ccp_alpha=alpha),X_cls,y_cls,SPLITS,scale=False)
            res_df=pd.DataFrame(res)
            st.dataframe(res_df,use_container_width=True,hide_index=True)
            plot_compare_bars(res_df['Split'],res_df['Accuracy'],'Accuracy (%)','DT Accuracy Across Splits','#534AB7')
            Xr,Xe,yr,ye=train_test_split(X_cls,y_cls,test_size=0.2,random_state=42,stratify=y_cls)
            bm=DecisionTreeClassifier(random_state=42,ccp_alpha=alpha).fit(Xr,yr); yp=bm.predict(Xe)
            c1,c2=st.columns(2)
            with c1: plot_confusion(confusion_matrix(ye,yp),CLASS_NAMES,"Confusion Matrix (80:20)")
            with c2: st.dataframe(pd.DataFrame(classification_report(ye,yp,target_names=CLASS_NAMES,output_dict=True)).T.round(3),use_container_width=True)
            metric_cols({"Depth":bm.get_depth(),"Leaves":bm.get_n_leaves(),"Train Acc":f"{accuracy_score(yr,bm.predict(Xr))*100:.1f}%","Test Acc":f"{accuracy_score(ye,yp)*100:.2f}%"})
            cv=cross_val_score(DecisionTreeClassifier(random_state=42,ccp_alpha=alpha),X_cls,y_cls,cv=5)
            metric_cols({"CV Mean":f"{cv.mean()*100:.2f}%","CV Std":f"±{cv.std()*100:.2f}%","Min":f"{cv.min()*100:.2f}%","Max":f"{cv.max()*100:.2f}%"})

    with tab_reg:
        depth_r=st.slider("Tree Depth",2,14,6,key="dt_depth_reg")
        if st.button("▶ Run DT Regression",key="run_dt_reg"):
            res=run_splits_reg(lambda:DecisionTreeRegressor(max_depth=depth_r,random_state=42),X_reg,y_reg,SPLITS,scale=False)
            st.dataframe(pd.DataFrame(res),use_container_width=True,hide_index=True)
            plot_compare_bars([r['Split'] for r in res],[r['R²'] for r in res],'R² Score','DT Regression R²','#1D9E75')

# ══════════════════════════════════════════════════════
# SVM
# ══════════════════════════════════════════════════════
elif "SVM" in page:
    meta = MODEL_META["⚙️   SVM"]
    st.title("⚙️ Support Vector Machine (SVM)")
    model_summary_card(meta, page)
    tab_cls,tab_reg=st.tabs(["🎯 Classification","📈 SVR Regression"])

    with tab_cls:
        c1,c2,c3=st.columns(3)
        with c1: kernel=st.selectbox("Kernel",["rbf","linear"],key="svm_kernel")
        with c2: C_val=st.selectbox("C",[0.1,1,10,100],index=2,key="svm_C")
        with c3: gamma_val=st.selectbox("γ (RBF)",[0.001,0.01,0.1,1],index=2,key="svm_gamma")
        if st.button("▶ Run SVM Classification",key="run_svm_cls"):
            guse=gamma_val if kernel=='rbf' else 'scale'
            with st.spinner("Training SVM…"):
                res=run_splits_cls(lambda:SVC(kernel=kernel,C=C_val,gamma=guse,random_state=42),X_cls,y_cls,SPLITS)
            res_df=pd.DataFrame(res)
            st.dataframe(res_df,use_container_width=True,hide_index=True)
            plot_compare_bars(res_df['Split'],res_df['Accuracy'],'Accuracy (%)','SVM Accuracy Across Splits','#1D9E75')
            sc=StandardScaler()
            Xr,Xe,yr,ye=train_test_split(X_cls,y_cls,test_size=0.2,random_state=42,stratify=y_cls)
            Xr=sc.fit_transform(Xr); Xe=sc.transform(Xe)
            bm=SVC(kernel=kernel,C=C_val,gamma=guse,random_state=42).fit(Xr,yr); yp=bm.predict(Xe)
            cc1,cc2=st.columns(2)
            with cc1: plot_confusion(confusion_matrix(ye,yp),CLASS_NAMES,"Confusion Matrix (80:20)")
            with cc2: st.dataframe(pd.DataFrame(classification_report(ye,yp,target_names=CLASS_NAMES,output_dict=True)).T.round(3),use_container_width=True)

    with tab_reg:
        c1,c2=st.columns(2)
        with c1: svr_C=st.selectbox("C",[0.1,1,10,100],index=2,key="svr_C")
        with c2: svr_eps=st.selectbox("Epsilon",[0.01,0.1,0.5,1.0],index=1,key="svr_eps")
        if st.button("▶ Run SVR Regression",key="run_svr_reg"):
            with st.spinner("Training SVR…"):
                res=run_splits_reg(lambda:SVR(kernel='rbf',C=svr_C,epsilon=svr_eps),X_reg,y_reg,SPLITS)
            st.dataframe(pd.DataFrame(res),use_container_width=True,hide_index=True)
            plot_compare_bars([r['Split'] for r in res],[r['R²'] for r in res],'R² Score','SVR Regression R²','#1D9E75')

# ══════════════════════════════════════════════════════
# NEURAL NETWORK
# ══════════════════════════════════════════════════════
elif "Neural Network" in page:
    meta = MODEL_META["🧠  Neural Network"]
    st.title("🧠 Neural Network (MLP)")
    model_summary_card(meta, page)

    from sklearn.neural_network import MLPClassifier, MLPRegressor
    st.info("ℹ️ Uses scikit-learn MLPClassifier/Regressor (equivalent MLP architecture). Pre-computed Keras results shown in tables.")

    tab_cls,tab_reg=st.tabs(["🎯 Classification","📈 Regression"])

    with tab_cls:
        st.markdown("#### Pre-computed Results (TensorFlow/Keras — from report)")
        st.dataframe(pd.DataFrame([
            {"Split":"80:20","Train Acc":"~82%","Val Acc":"~74%","Test Acc":"74.76%"},
            {"Split":"70:30","Train Acc":"~80%","Val Acc":"~74%","Test Acc":"74.80%"},
            {"Split":"50:50","Train Acc":"~77%","Val Acc":"~73%","Test Acc":"73.33%"},
            {"Split":"5-Fold CV","Train Acc":"—","Val Acc":"—","Test Acc":"73.98% ±0.47%"},
        ]),use_container_width=True,hide_index=True)
        st.markdown("#### Live sklearn MLP")
        c1,c2,c3=st.columns(3)
        with c1: h1=st.selectbox("Layer 1",[64,128,256],index=1,key="mlp_h1")
        with c2: h2=st.selectbox("Layer 2",[32,64,128],index=1,key="mlp_h2")
        with c3: al=st.select_slider("L2 alpha",[0.0001,0.001,0.01,0.1],value=0.001,key="mlp_al")
        if st.button("▶ Run sklearn MLP Classification",key="run_mlp_cls"):
            with st.spinner("Training…"):
                res=run_splits_cls(lambda:MLPClassifier(hidden_layer_sizes=(h1,h2),activation='relu',solver='adam',
                                   alpha=al,batch_size=64,max_iter=300,early_stopping=True,
                                   validation_fraction=0.15,n_iter_no_change=15,random_state=42),X_cls,y_cls,SPLITS)
            res_df=pd.DataFrame(res)
            st.dataframe(res_df,use_container_width=True,hide_index=True)
            plot_compare_bars(res_df['Split'],res_df['Accuracy'],'Accuracy (%)','MLP Accuracy Across Splits','#E24B4A')
            sc=StandardScaler(); Xr,Xe,yr,ye=train_test_split(X_cls,y_cls,test_size=0.2,random_state=42,stratify=y_cls)
            Xr=sc.fit_transform(Xr); Xe=sc.transform(Xe)
            bm=MLPClassifier(hidden_layer_sizes=(h1,h2),activation='relu',solver='adam',alpha=al,
                             batch_size=64,max_iter=300,early_stopping=True,validation_fraction=0.15,
                             n_iter_no_change=15,random_state=42).fit(Xr,yr)
            yp=bm.predict(Xe)
            cc1,cc2=st.columns(2)
            with cc1: plot_confusion(confusion_matrix(ye,yp),CLASS_NAMES,"Confusion Matrix (80:20)")
            with cc2: st.dataframe(pd.DataFrame(classification_report(ye,yp,target_names=CLASS_NAMES,output_dict=True)).T.round(3),use_container_width=True)
            cv=cross_val_score(MLPClassifier(hidden_layer_sizes=(h1,h2),activation='relu',solver='adam',alpha=al,
                               batch_size=64,max_iter=300,early_stopping=True,validation_fraction=0.15,
                               n_iter_no_change=15,random_state=42),StandardScaler().fit_transform(X_cls),y_cls,cv=5)
            metric_cols({"CV Mean":f"{cv.mean()*100:.2f}%","CV Std":f"±{cv.std()*100:.2f}%","Min":f"{cv.min()*100:.2f}%","Max":f"{cv.max()*100:.2f}%"})

    with tab_reg:
        st.dataframe(pd.DataFrame([
            {"Split":"80:20","RMSE (cm)":"17.77","R²":"0.802"},
            {"Split":"70:30","RMSE (cm)":"17.77","R²":"0.802"},
            {"Split":"50:50","RMSE (cm)":"17.79","R²":"0.802"},
        ]),use_container_width=True,hide_index=True)
        if st.button("▶ Run sklearn MLP Regression",key="run_mlp_reg"):
            with st.spinner("Training MLP Regressor…"):
                res=run_splits_reg(lambda:MLPRegressor(hidden_layer_sizes=(256,128,64),activation='relu',solver='adam',
                                   alpha=0.001,batch_size=64,max_iter=300,early_stopping=True,
                                   validation_fraction=0.15,n_iter_no_change=15,random_state=42),X_reg,y_reg,SPLITS)
            st.dataframe(pd.DataFrame(res),use_container_width=True,hide_index=True)
            plot_compare_bars([r['Split'] for r in res],[r['R²'] for r in res],'R² Score','MLP Regression R²','#E24B4A')

# ══════════════════════════════════════════════════════
# 🏆 MODEL COMPARISON
# ══════════════════════════════════════════════════════
elif "Comparison" in page:
    st.title("🏆 Model Comparison")
    st.markdown('<div class="axora-card">', unsafe_allow_html=True)
    st.subheader("Classification Performance")
    st.dataframe(pd.DataFrame([
        {"Model":"🤖 KNN (k=22)",            "Best Accuracy":"63.09%","Macro F1":"0.63","Best Split":"80:20","5-Fold CV":"61.9%"},
        {"Model":"🌳 Decision Tree (pruned)", "Best Accuracy":"72.15%","Macro F1":"0.71","Best Split":"80:20","5-Fold CV":"70.39%"},
        {"Model":"⚙️ SVM (RBF, C=10)",        "Best Accuracy":"71.27%","Macro F1":"0.71","Best Split":"80:20","5-Fold CV":"70.8%"},
        {"Model":"🧠 Neural Network (MLP)",   "Best Accuracy":"74.80%","Macro F1":"~0.74","Best Split":"70:30","5-Fold CV":"73.98%"},
        {"Model":"📈 Linear Reg. (adapted)",  "Best Accuracy":"~50%",  "Macro F1":"~0.48","Best Split":"N/A","5-Fold CV":"N/A"},
    ]),use_container_width=True,hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="axora-card">', unsafe_allow_html=True)
    st.subheader("Regression Performance")
    st.dataframe(pd.DataFrame([
        {"Model":"🧠 Neural Network",   "Best R²":"0.802","Best RMSE (cm)":"17.77","Best Split":"80:20"},
        {"Model":"📈 Linear Regression","Best R²":"0.8033","Best RMSE (cm)":"17.89","Best Split":"70:30"},
        {"Model":"⚙️ SVM (SVR)",         "Best R²":"0.79","Best RMSE (cm)":"18.4","Best Split":"80:20"},
        {"Model":"🌳 Decision Tree",     "Best R²":"0.76","Best RMSE (cm)":"20.1","Best Split":"80:20"},
        {"Model":"🤖 KNN Regression",    "Best R²":"0.788","Best RMSE (cm)":"13.8 (MAE)","Best Split":"70:30"},
    ]),use_container_width=True,hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

    fig,axes=plt.subplots(1,2,figsize=(14,5))
    mnames=["KNN","Decision\nTree","SVM","Neural\nNetwork","Linear\nReg."]
    accs=[63.09,72.15,71.27,74.80,50.0]
    clrs=['#3B8BD4','#534AB7','#1D9E75','#E24B4A','#85B7EB']
    bars=axes[0].bar(mnames,accs,color=clrs,edgecolor='#0D1B2A',width=0.55)
    for b in bars: axes[0].text(b.get_x()+b.get_width()/2,b.get_height()+0.5,f'{b.get_height():.1f}%',ha='center',color='#C8D8EA',fontsize=9)
    axes[0].set_ylim(0,90); axes[0].set_title('Classification Accuracy'); axes[0].axhline(72,color='#E24B4A',linestyle='--',lw=1,alpha=0.6,label='~72% ceiling'); axes[0].legend()
    r2s=[0.802,0.803,0.79,0.76,0.788]
    bars2=axes[1].bar(["Neural\nNetwork","Linear\nReg.","SVM\n(SVR)","Decision\nTree","KNN\nReg."],r2s,color=clrs,edgecolor='#0D1B2A',width=0.55)
    for b in bars2: axes[1].text(b.get_x()+b.get_width()/2,b.get_height()+0.003,f'{b.get_height():.3f}',ha='center',color='#C8D8EA',fontsize=9)
    axes[1].set_ylim(0,1.0); axes[1].set_title('Regression R² Score')
    show_fig(fig)

# ══════════════════════════════════════════════════════
# 🔮 LIVE PREDICTOR — ALL 5 MODELS
# ══════════════════════════════════════════════════════
elif "Predictor" in page:
    st.title("🔮 Live Predictor — All 5 Models")
    st.markdown("""
    <div class="axora-card">
    Enter participant measurements below. All <strong style="color:#3B8BD4">five models</strong>
    will predict the <strong>performance class (A–D)</strong> and
    <strong>broad jump distance (cm)</strong> simultaneously, so you can compare results side-by-side.
    </div>""", unsafe_allow_html=True)

    with st.form("predict_form"):
        c1,c2,c3 = st.columns(3)
        with c1:
            age    = st.number_input("Age (yr)",        18, 80,  30)
            gender = st.selectbox("Gender",             ["M","F"])
            height = st.number_input("Height (cm)",     140.0,200.0,170.0)
            weight = st.number_input("Weight (kg)",      30.0,150.0, 68.0)
        with c2:
            bf     = st.number_input("Body Fat (%)",      3.0, 60.0, 22.0)
            dia    = st.number_input("Diastolic BP",      50.0,130.0, 79.0)
            sys_bp = st.number_input("Systolic BP",       80.0,180.0,130.0)
        with c3:
            grip   = st.number_input("Grip Force (kg)",   5.0, 80.0, 37.0)
            flex   = st.number_input("Sit & Bend (cm)",  -20.0,50.0, 15.0)
            situps = st.number_input("Sit-ups",            0.0, 80.0, 40.0)
        submitted = st.form_submit_button("🚀 Predict with All 5 Models")

    if submitted:
        bmi = weight / ((height/100)**2)
        g   = 0 if gender == 'M' else 1

        # Feature vectors
        x_cls_raw = np.array([[age,g,height,weight,bf,dia,sys_bp,grip,flex,situps,df['broad jump_cm'].median(),bmi]])
        x_reg_raw = np.array([[age,g,height,weight,bf,dia,sys_bp,grip,flex,situps,bmi]])

        # Fit scalers on full training data
        sc_cls = StandardScaler().fit(X_cls)
        sc_reg = StandardScaler().fit(X_reg)
        x_cls_sc = sc_cls.transform(x_cls_raw)
        x_reg_sc = sc_reg.transform(x_reg_raw)

        # ── Train all 5 classifiers + 5 regressors on full data ──
        with st.spinner("Training all 5 models and predicting…"):

            models_cls = {
                "KNN":
                    KNeighborsClassifier(n_neighbors=22,n_jobs=-1).fit(sc_cls.transform(X_cls),y_cls),
                "Lin. Reg.":
                    None,  # handled separately via binning
                "Dec. Tree":
                    DecisionTreeClassifier(random_state=42,ccp_alpha=0.000377).fit(X_cls,y_cls),
                "SVM":
                    SVC(kernel='rbf',C=10,gamma=0.1,random_state=42).fit(sc_cls.transform(X_cls),y_cls),
                "MLP":
                    MLPClassifier(hidden_layer_sizes=(256,128,64),activation='relu',solver='adam',
                                  alpha=0.001,batch_size=64,max_iter=300,early_stopping=True,
                                  validation_fraction=0.15,n_iter_no_change=15,
                                  random_state=42).fit(sc_cls.transform(X_cls),y_cls),
            }

            models_reg = {
                "KNN":
                    KNeighborsRegressor(n_neighbors=35,n_jobs=-1).fit(sc_reg.transform(X_reg),y_reg),
                "Lin. Reg.":
                    LinearRegression().fit(X_reg,y_reg),
                "Dec. Tree":
                    DecisionTreeRegressor(max_depth=6,random_state=42).fit(X_reg,y_reg),
                "SVM":
                    SVR(kernel='rbf',C=10,epsilon=0.1).fit(sc_reg.transform(X_reg),y_reg),
                "MLP":
                    MLPRegressor(hidden_layer_sizes=(256,128,64),activation='relu',solver='adam',
                                 alpha=0.001,batch_size=64,max_iter=300,early_stopping=True,
                                 validation_fraction=0.15,n_iter_no_change=15,
                                 random_state=42).fit(sc_reg.transform(X_reg),y_reg),
            }

            # Linear Regression adapted classifier (binning)
            lr_cls_model = LinearRegression().fit(X_reg,y_reg)
            lr_pred_val  = lr_cls_model.predict(x_reg_raw)[0]
            q_lr         = np.percentile(y_reg,[25,50,75])
            lr_cls_idx   = 3 if lr_pred_val>=q_lr[2] else (2 if lr_pred_val>=q_lr[1] else (1 if lr_pred_val>=q_lr[0] else 0))

        # ── Collect predictions ──
        results = {}
        model_keys = ["KNN","Lin. Reg.","Dec. Tree","SVM","MLP"]
        model_icons= {"KNN":"🤖","Lin. Reg.":"📈","Dec. Tree":"🌳","SVM":"⚙️","MLP":"🧠"}
        model_colors={"KNN":"#3B8BD4","Lin. Reg.":"#85B7EB","Dec. Tree":"#534AB7","SVM":"#1D9E75","MLP":"#E24B4A"}
        needs_scale = {"KNN":True,"Lin. Reg.":False,"Dec. Tree":False,"SVM":True,"MLP":True}

        for name in model_keys:
            if name == "Lin. Reg.":
                cls_idx  = lr_cls_idx
                jump     = lr_cls_model.predict(x_reg_raw)[0]
            else:
                sc_x_cls = x_cls_sc if needs_scale[name] else x_cls_raw
                sc_x_reg = x_reg_sc if needs_scale[name] else x_reg_raw
                cls_idx  = models_cls[name].predict(sc_x_cls)[0]
                jump     = models_reg[name].predict(sc_x_reg)[0]
            cls_letter = CLASS_NAMES[cls_idx]
            results[name] = {"cls":cls_letter,"jump":jump}

        # ── Display results ──
        st.markdown("### 📊 Prediction Results — All 5 Models")

        cols = st.columns(5)
        for col, name in zip(cols, model_keys):
            r   = results[name]
            cls = r['cls']
            jmp = r['jump']
            label, colour, advice = CLASS_INFO[cls]
            icon = model_icons[name]
            mc   = model_colors[name]
            col.markdown(f"""
            <div style="background:#0F1E33;border:2px solid {colour};border-radius:12px;
                        padding:16px 10px;text-align:center;margin:2px;">
              <div style="font-size:0.7rem;color:{mc};text-transform:uppercase;
                          letter-spacing:1px;font-weight:700;margin-bottom:6px;">
                {icon} {name}
              </div>
              <div style="font-size:3rem;line-height:1;">{label.split()[0]}</div>
              <div style="font-size:2rem;font-weight:900;color:{colour};margin:4px 0;">
                Class {cls}
              </div>
              <div style="font-size:0.82rem;color:#1D9E75;font-weight:700;margin-bottom:4px;">
                {jmp:.1f} cm jump
              </div>
              <div style="font-size:0.68rem;color:#85B7EB;line-height:1.4;">{advice}</div>
            </div>""", unsafe_allow_html=True)

        # ── Agreement summary ──
        cls_votes = [results[n]['cls'] for n in model_keys]
        jump_preds = [results[n]['jump'] for n in model_keys]
        from collections import Counter
        majority = Counter(cls_votes).most_common(1)[0]
        avg_jump = np.mean(jump_preds)
        agreement = majority[1]

        st.markdown('<br>', unsafe_allow_html=True)
        ml, mr = st.columns(2)
        with ml:
            lbl, colour, advice = CLASS_INFO[majority[0]]
            st.markdown(f"""
            <div class="axora-card" style="border-color:{colour};text-align:center;">
              <div style="font-size:0.8rem;color:#85B7EB;letter-spacing:2px;text-transform:uppercase;margin-bottom:8px;">
                🗳️ Model Majority Vote
              </div>
              <div style="font-size:3.5rem;font-weight:900;color:{colour};">Class {majority[0]}</div>
              <div style="font-size:1rem;color:#C8D8EA;margin:6px 0;">{lbl} — {agreement}/5 models agree</div>
              <div style="font-size:0.85rem;color:#85B7EB;">{advice}</div>
            </div>""", unsafe_allow_html=True)

        with mr:
            st.markdown(f"""
            <div class="axora-card" style="border-color:#1D9E75;text-align:center;">
              <div style="font-size:0.8rem;color:#85B7EB;letter-spacing:2px;text-transform:uppercase;margin-bottom:8px;">
                📏 Jump Prediction Summary
              </div>
              <div style="font-size:3rem;font-weight:900;color:#1D9E75;">{avg_jump:.1f} cm</div>
              <div style="font-size:0.9rem;color:#C8D8EA;margin-top:6px;">Average across all 5 models</div>
              <div style="font-size:0.82rem;color:#85B7EB;margin-top:8px;">
                Range: {min(jump_preds):.1f} – {max(jump_preds):.1f} cm
              </div>
            </div>""", unsafe_allow_html=True)

        # ── Per-model jump comparison chart ──
        st.markdown("### 📈 Jump Distance — Model Comparison")
        fig, ax = plt.subplots(figsize=(10, 4))
        clrs = [model_colors[n] for n in model_keys]
        bars = ax.bar(model_keys, jump_preds, color=clrs, edgecolor='#0D1B2A', width=0.5)
        for b,v in zip(bars,jump_preds):
            ax.text(b.get_x()+b.get_width()/2,b.get_height()+0.5,f'{v:.1f}cm',ha='center',color='#C8D8EA',fontsize=10,fontweight='bold')
        ax.axhline(avg_jump,color='#1D9E75',linestyle='--',lw=2,label=f'Mean {avg_jump:.1f} cm')
        ax.set_ylabel('Predicted Broad Jump (cm)'); ax.set_title('Broad Jump Prediction by Model'); ax.legend()
        ax.set_ylim(0, max(jump_preds)*1.25)
        show_fig(fig)

        # ── Input profile ──
        st.markdown("### 📋 Your Input Profile vs Dataset Average")
        feat_vals = pd.DataFrame({
            'Feature':['Age','Height','Weight','Body Fat %','Diastolic','Systolic','Grip Force','Sit&Bend','Sit-ups','BMI'],
            'Your Value':[age,height,weight,bf,dia,sys_bp,grip,flex,situps,round(bmi,1)],
            'Dataset Mean':np.round([df['age'].mean(),df['height_cm'].mean(),df['weight_kg'].mean(),
                                      df['body fat_%'].mean(),df['diastolic'].mean(),df['systolic'].mean(),
                                      df['gripForce'].mean(),df['sit and bend forward_cm'].mean(),
                                      df['sit-ups counts'].mean(),df['BMI'].mean()],1)
        })
        fig2,ax2=plt.subplots(figsize=(12,4))
        x=np.arange(len(feat_vals))
        ax2.bar(x-0.2,feat_vals['Your Value'],width=0.38,label='You',color='#3B8BD4')
        ax2.bar(x+0.2,feat_vals['Dataset Mean'],width=0.38,label='Dataset Mean',color='#534AB7',alpha=0.7)
        ax2.set_xticks(x); ax2.set_xticklabels(feat_vals['Feature'],rotation=35,ha='right',fontsize=9)
        ax2.set_title("Your Values vs Dataset Averages"); ax2.legend()
        show_fig(fig2)

# ─────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<hr style="border-color:#1D4570;margin-top:40px">
<div style="text-align:center;color:#4d7fa8;font-size:0.82rem;padding:12px">
    <strong style="color:#3B8BD4">AXORA</strong> — Body Performance Analytics &amp; Intelligent Classification System<br>
    Alaa Issawi · Amira Salama · Aya Abdel Maksoud · Aya El-Sabi · Aya Imam · Aya Khalil<br>
    <span style="color:#534AB7">Introduction to AI &amp; ML · 2024–2025</span>
</div>
""", unsafe_allow_html=True)
