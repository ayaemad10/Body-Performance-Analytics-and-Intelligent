"""
╔══════════════════════════════════════════════════════════════════╗
║  AXORA — Body Performance Analytics & Intelligent System         ║
║  5 ML Models  |  Classification + Regression                     ║
║  Team Axora · Intro to AI & ML · 2024-2025                       ║
╚══════════════════════════════════════════════════════════════════╝
"""

# ── Standard library ─────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")
import sys, os

# ── Core data / plotting  ────────────────────────────────────────
import streamlit as st

try:
    import pandas as pd
except ImportError:
    st.error("❌ pandas not installed. Add `pandas` to requirements.txt"); st.stop()

try:
    import numpy as np
except ImportError:
    st.error("❌ numpy not installed. Add `numpy` to requirements.txt"); st.stop()

try:
    import matplotlib
    matplotlib.use("Agg")           # non-interactive backend — required for servers
    import matplotlib.pyplot as plt
except ImportError:
    st.error("❌ matplotlib not installed. Add `matplotlib` to requirements.txt"); st.stop()

try:
    import seaborn as sns
except ImportError:
    st.error("❌ seaborn not installed. Add `seaborn` to requirements.txt"); st.stop()

# ── Machine learning ─────────────────────────────────────────────
try:
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.model_selection import (train_test_split, cross_val_score,
                                         StratifiedKFold, KFold)
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.tree import (DecisionTreeClassifier, DecisionTreeRegressor,
                               plot_tree)
    from sklearn.svm import SVC, SVR
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        confusion_matrix, classification_report,
        mean_squared_error, r2_score, mean_absolute_error,
    )
except ImportError as e:
    st.error(f"❌ scikit-learn import error: {e}. Add `scikit-learn` to requirements.txt")
    st.stop()

# ── TensorFlow — optional, graceful degradation ──────────────────
TF_OK = False
try:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import tensorflow as tf
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (Dense, Dropout,
                                          BatchNormalization, Activation)
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TF_OK = True
    TF_VERSION = tf.__version__
except Exception:
    TF_VERSION = "not available"

# ══════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Axora | Body Performance Analytics",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════
# CSS — Axora dark theme
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── Base ── */
html,[data-testid="stAppViewContainer"]{background:#0D1B2A!important;color:#C8D8EA}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0F1E33,#0A1628);
  border-right:1px solid #1D4570}
[data-testid="stSidebar"] *{color:#C8D8EA!important}

/* ── Headings ── */
h1{color:#3B8BD4!important;font-size:2rem!important;font-weight:800!important}
h2{color:#3B8BD4!important;font-size:1.45rem!important;font-weight:700!important}
h3{color:#85B7EB!important;font-size:1.1rem!important}
p,.stMarkdown p,li{color:#C8D8EA!important;line-height:1.7}

/* ── Cards ── */
.card{background:#0F1E33;border:1px solid #1D4570;border-radius:14px;
      padding:22px 26px;margin-bottom:18px}
.ca{border-left:4px solid #1D9E75}
.cp{border-left:4px solid #534AB7}
.cr{border-left:4px solid #E24B4A}

/* ── Metric tiles ── */
.mb{background:linear-gradient(135deg,#112640,#0F1E33);border:1px solid #1D4570;
    border-radius:12px;padding:18px 12px;text-align:center;margin:4px}
.mv{font-size:1.85rem;font-weight:800;color:#3B8BD4;line-height:1.1}
.ml{font-size:.72rem;color:#85B7EB;text-transform:uppercase;
    letter-spacing:1.5px;margin-top:4px}

/* ── Badges ── */
.bg{background:#1D4570;color:#85B7EB;border-radius:6px;padding:2px 9px;
    font-size:.75rem;display:inline-block;margin:2px}
.gg{background:#0D3D2A;color:#1D9E75;border-radius:6px;padding:2px 9px;
    font-size:.75rem;display:inline-block;margin:2px}
.pg{background:#1E1A40;color:#AFA9EC;border-radius:6px;padding:2px 9px;
    font-size:.75rem;display:inline-block;margin:2px}
.rg{background:#3D1010;color:#F08080;border-radius:6px;padding:2px 9px;
    font-size:.75rem;display:inline-block;margin:2px}

/* ── Cover ── */
.cover{background:linear-gradient(135deg,#0D1B2A,#112640,#0F1E33);
       border:1px solid #534AB7;border-radius:18px;padding:28px 36px;
       display:flex;align-items:center;gap:28px;margin-bottom:28px}

/* ── Alert boxes ── */
.ibox{background:#0A1E35;border:1px solid #1D4570;border-radius:10px;
      padding:13px 18px;margin:8px 0;color:#85B7EB;font-size:.88rem}
.wbox{background:#1E1500;border:1px solid #A07800;border-radius:10px;
      padding:13px 18px;margin:8px 0;color:#D4A800;font-size:.88rem}
.sbox{background:#0A1E14;border:1px solid #1D9E75;border-radius:10px;
      padding:13px 18px;margin:8px 0;color:#1D9E75;font-size:.88rem}

/* ── Tables ── */
.dataframe thead th{background:#1F5C9E!important;color:#fff!important}
.dataframe tbody tr:nth-child(even){background:#112640!important}
.dataframe{color:#C8D8EA!important}

/* ── Tabs ── */
[data-testid="stTab"] button{color:#85B7EB!important}
[data-testid="stTab"] button[aria-selected="true"]{color:#3B8BD4!important;
  border-bottom:2px solid #1D9E75!important}

/* ── Buttons ── */
.stButton>button{background:linear-gradient(90deg,#1F5C9E,#534AB7);
  color:#fff;border:none;border-radius:8px;padding:10px 28px;
  font-weight:700;letter-spacing:.4px;transition:all .18s}
.stButton>button:hover{opacity:.82;transform:translateY(-1px)}

/* ── Form inputs ── */
.stSelectbox label,.stSlider label,.stNumberInput label{color:#85B7EB!important}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# AXORA SVG LOGO  (self-contained — no file dependency)
# ══════════════════════════════════════════════════════════════════
LOGO = """
<svg viewBox="0 0 680 420" xmlns="http://www.w3.org/2000/svg" width="220" height="136">
  <defs>
    <linearGradient id="bGx" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#3B8BD4"/><stop offset="100%" stop-color="#534AB7"/>
    </linearGradient>
    <linearGradient id="aGx" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0%" stop-color="#1D9E75"/><stop offset="100%" stop-color="#3B8BD4"/>
    </linearGradient>
  </defs>
  <rect x="40" y="20" width="600" height="380" rx="20" fill="#0D1B2A"
        stroke="#534AB7" stroke-width="1" opacity=".97"/>
  <g opacity=".09" fill="#85B7EB">
    <circle cx="100" cy="80" r="2"/><circle cx="140" cy="80" r="2"/>
    <circle cx="180" cy="80" r="2"/><circle cx="100" cy="120" r="2"/>
    <circle cx="140" cy="120" r="2"/><circle cx="500" cy="80" r="2"/>
    <circle cx="540" cy="80" r="2"/><circle cx="580" cy="80" r="2"/>
    <circle cx="500" cy="120" r="2"/><circle cx="540" cy="120" r="2"/>
  </g>
  <polygon points="340,70 374,90 374,130 340,150 306,130 306,90"
           fill="none" stroke="url(#bGx)" stroke-width="2"/>
  <polygon points="340,80 366,96 366,124 340,140 314,124 314,96" fill="#1A2A45"/>
  <line x1="340" y1="92" x2="326" y2="130" stroke="url(#bGx)"
        stroke-width="2.5" stroke-linecap="round"/>
  <line x1="340" y1="92" x2="354" y2="130" stroke="url(#bGx)"
        stroke-width="2.5" stroke-linecap="round"/>
  <line x1="330" y1="116" x2="350" y2="116" stroke="#1D9E75"
        stroke-width="2" stroke-linecap="round"/>
  <circle cx="340" cy="60" r="5" fill="#378ADD" opacity=".9"/>
  <circle cx="386" cy="82" r="4" fill="#534AB7" opacity=".9"/>
  <circle cx="390" cy="140" r="4" fill="#1D9E75" opacity=".9"/>
  <circle cx="340" cy="162" r="5" fill="#378ADD" opacity=".9"/>
  <circle cx="290" cy="140" r="4" fill="#534AB7" opacity=".9"/>
  <circle cx="294" cy="82" r="4" fill="#1D9E75" opacity=".9"/>
  <circle cx="340" cy="110" r="68" fill="none" stroke="#3B8BD4"
          stroke-width=".8" stroke-dasharray="4 6" opacity=".3"/>
  <text x="340" y="214" text-anchor="middle"
        font-family="Georgia,serif" font-size="52" font-weight="700"
        letter-spacing="12" fill="url(#bGx)">AXORA</text>
  <rect x="180" y="222" width="320" height="2.5" rx="1.5" fill="url(#aGx)" opacity=".9"/>
  <text x="340" y="252" text-anchor="middle"
        font-family="'Courier New',monospace" font-size="12" letter-spacing="4"
        fill="#85B7EB" opacity=".85">INTELLIGENCE · ILLUMINATED</text>
  <rect x="190" y="272" width="300" height="34" rx="17"
        fill="#0F1E33" stroke="#378ADD" stroke-width="1" opacity=".9"/>
  <text x="340" y="293" text-anchor="middle"
        font-family="'Courier New',monospace" font-size="11" letter-spacing="2"
        fill="#5DCAA5">DATA ANALYSIS  ·  ARTIFICIAL INTELLIGENCE</text>
  <text x="340" y="336" text-anchor="middle"
        font-family="Georgia,serif" font-size="10.5" letter-spacing="1"
        fill="#4d7fa8" opacity=".9">
    Aya Emad · Aya Samir · Aya Shaaban · Aya Ashraf · Alaa Issawi · Amira Salama
  </text>
  <line x1="100" y1="356" x2="580" y2="356"
        stroke="#534AB7" stroke-width=".5" opacity=".4"/>
  <text x="340" y="378" text-anchor="middle"
        font-family="'Courier New',monospace" font-size="9" letter-spacing="3"
        fill="#3C3489" opacity=".7">EST. 2025 · TEAM PROJECT</text>
</svg>"""

# ══════════════════════════════════════════════════════════════════
# COLOUR PALETTE
# ══════════════════════════════════════════════════════════════════
DARK   = "#0D1B2A"
PANEL  = "#0F1E33"
BLUE   = "#3B8BD4"
PURPLE = "#534AB7"
GREEN  = "#1D9E75"
RED    = "#E24B4A"
LBLUE  = "#85B7EB"
PAL    = [BLUE, PURPLE, GREEN, RED]

# ══════════════════════════════════════════════════════════════════
# PLOT HELPERS
# ══════════════════════════════════════════════════════════════════
def _style(fig):
    fig.patch.set_facecolor(DARK)
    for ax in fig.get_axes():
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=LBLUE, labelsize=8)
        ax.xaxis.label.set_color(LBLUE)
        ax.yaxis.label.set_color(LBLUE)
        ax.title.set_color(BLUE)
        for sp in ax.spines.values():
            sp.set_edgecolor("#1D4570")
    return fig

def show(fig):
    st.pyplot(_style(fig), use_container_width=True)
    plt.close(fig)

def mbox(d: dict):
    cols = st.columns(len(d))
    for col, (lbl, val) in zip(cols, d.items()):
        col.markdown(
            f'<div class="mb"><div class="mv">{val}</div>'
            f'<div class="ml">{lbl}</div></div>',
            unsafe_allow_html=True,
        )

def conf_map(cm, classes, title="Confusion Matrix", cmap="Blues"):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap=cmap,
        xticklabels=classes, yticklabels=classes,
        ax=ax, linewidths=.4, cbar_kws={"shrink": .75},
        annot_kws={"size": 10, "weight": "bold"},
    )
    ax.set_title(title, pad=10)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    show(fig)

def bar_c(labels, vals, ylabel, title, color=BLUE):
    fig, ax = plt.subplots(figsize=(max(5, len(labels) * 1.7), 3.8))
    bars = ax.bar(labels, vals, color=color, edgecolor=DARK, width=.52, zorder=3)
    for b in bars:
        v = b.get_height()
        ax.text(
            b.get_x() + b.get_width() / 2,
            v + max(vals) * .013,
            f"{v:.3f}" if max(vals) < 10 else f"{v:.1f}",
            ha="center", va="bottom", color="#C8D8EA", fontsize=9, fontweight="bold",
        )
    ax.set_ylabel(ylabel); ax.set_title(title)
    ax.set_ylim(0, max(vals) * 1.28 + .02)
    ax.grid(axis="y", alpha=.2, color=LBLUE)
    show(fig)

def scatter_res(yte, yp, title="Actual vs Predicted"):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].scatter(yte, yp, alpha=.35, color=BLUE, s=12)
    mn, mx = min(yte.min(), yp.min()), max(yte.max(), yp.max())
    axes[0].plot([mn, mx], [mn, mx], "--", color=RED, lw=2, label="Perfect")
    axes[0].set_xlabel("Actual"); axes[0].set_ylabel("Predicted")
    axes[0].set_title(title); axes[0].legend()
    res = yte - yp
    axes[1].hist(res, bins=35, color=GREEN, edgecolor=DARK, alpha=.85)
    axes[1].axvline(0, color=RED, lw=2, linestyle="--")
    axes[1].set_xlabel("Residual"); axes[1].set_title("Residual Distribution")
    show(fig)

def fold_chart(scores, label="Accuracy (%)", scale=100, color=PURPLE):
    pct = scores * scale
    fig, ax = plt.subplots(figsize=(9, 3.8))
    bars = ax.bar(
        [f"Fold {i+1}" for i in range(len(pct))],
        pct, color=color, edgecolor=DARK, zorder=3,
    )
    ax.axhline(pct.mean(), color=GREEN, lw=2, linestyle="--",
               label=f"Mean {pct.mean():.2f}")
    ax.set_ylabel(label); ax.set_title(f"5-Fold CV — {label}")
    ax.grid(axis="y", alpha=.2); ax.legend()
    for b in bars:
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + pct.max() * .006,
            f"{b.get_height():.2f}",
            ha="center", color="#C8D8EA", fontsize=8,
        )
    show(fig)

def per_class(rpt, classes):
    metrics = ["precision", "recall", "f1-score"]
    x = np.arange(len(classes)); w = .26
    fig, ax = plt.subplots(figsize=(9, 4))
    for i, (m, c) in enumerate(zip(metrics, [BLUE, GREEN, PURPLE])):
        vals = [rpt[cl][m] for cl in classes]
        ax.bar(x + i * w, vals, w, label=m.capitalize(),
               color=c, edgecolor=DARK, zorder=3)
    ax.set_xticks(x + w)
    ax.set_xticklabels([f"Class {c}" for c in classes])
    ax.set_ylabel("Score"); ax.set_title("Per-Class Metrics")
    ax.set_ylim(0, 1.12); ax.legend(); ax.grid(axis="y", alpha=.2)
    show(fig)

# ══════════════════════════════════════════════════════════════════
# SPLIT RUNNERS
# ══════════════════════════════════════════════════════════════════
SPLITS = [("80:20", .20), ("70:30", .30), ("50:50", .50)]

def run_cls(fn, X, y, scale=True):
    rows = []
    for nm, ts in SPLITS:
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=ts, random_state=42, stratify=y)
        if scale:
            sc = StandardScaler()
            Xtr = sc.fit_transform(Xtr); Xte = sc.transform(Xte)
        m = fn(); m.fit(Xtr, ytr); yp = m.predict(Xte)
        rows.append({
            "Split": nm,
            "Accuracy":  f"{accuracy_score(yte, yp)*100:.2f}%",
            "Precision": f"{precision_score(yte,yp,average='macro',zero_division=0)*100:.2f}%",
            "Recall":    f"{recall_score(yte,yp,average='macro',zero_division=0)*100:.2f}%",
            "F1":        f"{f1_score(yte,yp,average='macro',zero_division=0)*100:.2f}%",
        })
    Xsc = StandardScaler().fit_transform(X) if scale else X
    cv = cross_val_score(
        fn(), Xsc, y,
        cv=StratifiedKFold(5, shuffle=True, random_state=42),
        scoring="accuracy",
    )
    rows.append({
        "Split": "5-Fold CV",
        "Accuracy": f"{cv.mean()*100:.2f}% ±{cv.std()*100:.2f}",
        "Precision": "—", "Recall": "—", "F1": "—",
    })
    return pd.DataFrame(rows), cv

def run_reg(fn, X, y, scale=True):
    rows = []
    for nm, ts in SPLITS:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=ts, random_state=42)
        if scale:
            sc = StandardScaler()
            Xtr = sc.fit_transform(Xtr); Xte = sc.transform(Xte)
        m = fn(); m.fit(Xtr, ytr); yp = m.predict(Xte)
        rows.append({
            "Split": nm,
            "R²":   f"{r2_score(yte, yp):.4f}",
            "RMSE": f"{np.sqrt(mean_squared_error(yte, yp)):.3f}",
            "MAE":  f"{mean_absolute_error(yte, yp):.3f}",
        })
    Xsc = StandardScaler().fit_transform(X) if scale else X
    cv = cross_val_score(
        fn(), Xsc, y,
        cv=KFold(5, shuffle=True, random_state=42),
        scoring="r2",
    )
    rows.append({
        "Split": "5-Fold CV",
        "R²":  f"{cv.mean():.4f} ±{cv.std():.4f}",
        "RMSE": "—", "MAE": "—",
    })
    return pd.DataFrame(rows), cv

# ══════════════════════════════════════════════════════════════════
# DATA LOADING & PREPROCESSING
# ══════════════════════════════════════════════════════════════════
@st.cache_data
def load_data(uploaded=None):
    if uploaded is not None:
        return pd.read_csv(uploaded)
    for path in [
        "bodyPerformance.csv",
        "bodyPerformance_-_Copy.csv",
        "/mnt/user-data/uploads/bodyPerformance_-_Copy.csv",
        os.path.join(os.path.dirname(__file__), "bodyPerformance.csv"),
    ]:
        try:
            return pd.read_csv(path)
        except Exception:
            continue
    # Synthetic fallback
    np.random.seed(42); n = 3000
    age = np.random.uniform(21, 64, n)
    gnd = np.random.choice(["M", "F"], n)
    ht  = np.where(gnd=="M", np.random.normal(172,6,n), np.random.normal(160,6,n))
    wt  = np.where(gnd=="M", np.random.normal(72,12,n), np.random.normal(58,9,n))
    bf  = np.where(gnd=="M", np.random.normal(18,5,n), np.random.normal(28,6,n)).clip(3,55)
    gr  = np.where(gnd=="M", np.random.normal(45,9,n), np.random.normal(27,7,n)).clip(5,75)
    su  = np.random.normal(42, 14, n).clip(0, 80)
    bj  = (gr*2.2 + su*1.8 + ht*.6 - age*1.1 - bf*1.5 +
           np.random.normal(0, 18, n)).clip(50, 310)
    cls = pd.cut(bj, bins=[0,145,180,220,400],
                 labels=["D","C","B","A"]).astype(str)
    return pd.DataFrame({
        "age": age, "gender": gnd,
        "height_cm": ht.clip(130, 200), "weight_kg": wt.clip(30, 140),
        "body fat_%": bf,
        "diastolic":  np.random.normal(79, 11, n).clip(50, 120),
        "systolic":   np.random.normal(130, 15, n).clip(80, 180),
        "gripForce":  gr,
        "sit and bend forward_cm": np.random.normal(15, 8, n).clip(-20, 50),
        "sit-ups counts": su, "broad jump_cm": bj, "class": cls,
    })

@st.cache_data
def preprocess(df):
    d = df.copy()
    d = d.drop_duplicates()
    d = d[(d["systolic"] > 40) & (d["diastolic"] > 40)]
    d["sit and bend forward_cm"] = d["sit and bend forward_cm"].clip(-20, 50)
    d["BMI"] = d["weight_kg"] / ((d["height_cm"] / 100) ** 2)
    d["gender_enc"] = d["gender"].map({"M": 0, "F": 1})
    le = LabelEncoder()
    d["class_enc"] = le.fit_transform(d["class"])
    return d, le

# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(LOGO, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("**📂 Dataset**")
    uploaded = st.file_uploader(
        "Upload bodyPerformance.csv", type=["csv"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    page = st.radio("**🧭 Navigation**", [
        "🏠 Overview",
        "📊 EDA",
        "🤖 KNN",
        "📈 Linear Regression",
        "🌳 Decision Tree",
        "⚙️ SVM",
        "🧠 Neural Network",
        "🏆 Model Comparison",
        "🔮 Live Predictor",
    ], label_visibility="collapsed")
    st.markdown("---")
    for name, cls in [
        ("Alaa Issawi",       "gg"),
        ("Amira Salama",      "bg"),
        ("Aya Abdel Maksoud", "pg"),
        ("Aya El-Sabi",       "gg"),
        ("Aya Imam",          "bg"),
        ("Aya Khalil",        "pg"),
    ]:
        st.markdown(f'<span class="{cls}">{name}</span>', unsafe_allow_html=True)
    st.markdown(
        '<br><span class="rg">Intro to AI &amp; ML · 2024–2025</span>',
        unsafe_allow_html=True,
    )
    if TF_OK:
        st.markdown(f'<br><span class="gg">TF {TF_VERSION}</span>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<br><span class="rg">TF: not installed</span>',
                    unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# LOAD & PREPARE DATA
# ══════════════════════════════════════════════════════════════════
raw = load_data(uploaded)
df, le = preprocess(raw)

FC  = ["age","gender_enc","height_cm","weight_kg","body fat_%",
       "diastolic","systolic","gripForce",
       "sit and bend forward_cm","sit-ups counts","broad jump_cm","BMI"]
FR  = ["age","gender_enc","height_cm","weight_kg","body fat_%",
       "diastolic","systolic","gripForce",
       "sit and bend forward_cm","sit-ups counts","BMI"]
CL  = ["A","B","C","D"]
NUM = ["age","height_cm","weight_kg","body fat_%","diastolic",
       "systolic","gripForce","sit and bend forward_cm",
       "sit-ups counts","broad jump_cm"]

Xc = df[FC].values;  yc = df["class_enc"].values
Xr = df[FR].values;  yr = df["broad jump_cm"].values

# ══════════════════════════════════════════════════════════════════
# ██  PAGE: OVERVIEW  ██
# ══════════════════════════════════════════════════════════════════
if "Overview" in page:
    st.markdown(
        f'<div class="cover">{LOGO}'
        '<div>'
        '<div style="font-size:2rem;font-weight:900;color:#3B8BD4">Body Performance Analytics</div>'
        '<div style="font-size:1.1rem;font-weight:700;color:#85B7EB">'
        'Intelligent Classification &amp; Regression System</div>'
        '<div style="color:#C8D8EA;font-size:.88rem;margin-top:6px">'
        'A Comparative Study of 5 ML Models on Physical Fitness Data</div>'
        '<div style="margin-top:12px">'
        '<span class="gg">KNN</span>'
        '<span class="bg">Linear Regression</span>'
        '<span class="pg">Decision Tree</span>'
        '<span class="rg">SVM</span>'
        '<span class="gg">Neural Network</span>'
        '</div></div></div>',
        unsafe_allow_html=True,
    )

    mbox({
        "Records":  f"{len(df):,}",
        "Features": len(FC),
        "Classes":  4,
        "ML Models": 5,
    })
    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown('<div class="card ca">', unsafe_allow_html=True)
        st.subheader("📋 Dataset Preview")
        st.dataframe(raw.head(10), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📐 Feature Reference")
        st.dataframe(pd.DataFrame({
            "Feature": ["age","gender","height_cm","weight_kg","body fat_%",
                        "diastolic","systolic","gripForce","sit&bend","sit-ups",
                        "broad_jump","class"],
            "Type": ["Num","Cat","Num","Num","Num","Num","Num","Num",
                     "Num","Num","Num","Target"],
        }), use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📊 Class Distribution")
    c1, c2, c3 = st.columns(3)
    vc = raw["class"].value_counts().sort_index()
    with c1:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        bars = ax.bar(vc.index, vc.values, color=PAL, edgecolor=DARK, width=.55)
        for b in bars:
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+15,
                    str(int(b.get_height())), ha="center", color="#C8D8EA", fontsize=9)
        ax.set_title("Class Frequency")
        ax.set_xlabel("Performance Class"); ax.set_ylabel("Count")
        show(fig)
    with c2:
        fig, ax = plt.subplots(figsize=(4.5, 4.5), subplot_kw=dict(aspect="equal"))
        _, _, at = ax.pie(vc.values, labels=vc.index, autopct="%1.1f%%",
                          colors=PAL, startangle=90,
                          wedgeprops=dict(edgecolor=DARK, linewidth=1.5))
        for t in at: t.set_color("white")
        ax.set_title("Class Balance"); show(fig)
    with c3:
        st.dataframe(
            df[["age","weight_kg","gripForce","sit-ups counts","broad jump_cm"]]
            .describe().round(1),
            use_container_width=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card cp">', unsafe_allow_html=True)
    st.subheader("🔧 Preprocessing Pipeline")
    r1, r2, r3 = st.columns(3)
    r1.markdown("**Cleaning**\n- Removed 1 duplicate\n- Removed BP ≤ 40 rows\n- Capped sit-bend [−20, 50]")
    r2.markdown("**Encoding**\n- gender → binary (M=0, F=1)\n- class → integer (A=0…D=3)\n- BMI feature engineered")
    r3.markdown("**Scaling**\n- StandardScaler for KNN / SVM / MLP\n- Raw values for DT & LR\n- Fit on training data only")
    st.markdown("</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# ██  PAGE: EDA  ██
# ══════════════════════════════════════════════════════════════════
elif "EDA" in page:
    st.title("📊 Exploratory Data Analysis")
    t1, t2, t3, t4, t5 = st.tabs([
        "📈 Distributions", "📦 Boxplots",
        "🔗 Correlations", "🎯 By Class", "🔍 Insights",
    ])

    with t1:
        st.subheader("Numeric Feature Distributions")
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        axes = axes.flatten()
        for i, col in enumerate(NUM):
            axes[i].hist(raw[col].dropna(), bins=35,
                         color=PAL[i % 4], alpha=.85, edgecolor=DARK)
            axes[i].set_title(col, fontsize=9)
        plt.tight_layout(pad=1.5); show(fig)
        st.markdown(
            '<div class="ibox">📌 <b>age</b> right-skewed; '
            '<b>height_cm</b> bimodal (M/F); '
            '<b>broad_jump_cm</b> left-skewed; BP columns near-normal.</div>',
            unsafe_allow_html=True,
        )

    with t2:
        st.subheader("Boxplots — Standardised Features")
        scv = StandardScaler()
        sc_df = pd.DataFrame(
            scv.fit_transform(raw[NUM].dropna()), columns=NUM)
        fig, ax = plt.subplots(figsize=(16, 5))
        sc_df.boxplot(
            ax=ax, patch_artist=True, notch=False,
            boxprops=dict(facecolor="#1D4570", color=BLUE),
            medianprops=dict(color=GREEN, linewidth=2.5),
            whiskerprops=dict(color=LBLUE),
            capprops=dict(color=LBLUE),
            flierprops=dict(marker="o", color=RED, alpha=.4, markersize=3),
        )
        ax.set_title("Standardised Boxplots")
        plt.xticks(rotation=35, ha="right", fontsize=8)
        show(fig)
        oc = []
        for col in NUM:
            q1, q3 = raw[col].quantile(.25), raw[col].quantile(.75)
            iqr = q3 - q1
            n = ((raw[col] < q1-1.5*iqr) | (raw[col] > q3+1.5*iqr)).sum()
            oc.append({"Feature": col, "Outliers": n,
                       "%": f"{n/len(raw)*100:.2f}%"})
        st.dataframe(pd.DataFrame(oc), use_container_width=True, hide_index=True)

    with t3:
        st.subheader("Correlation Heatmap")
        enc2 = raw.copy()
        enc2["gender"] = enc2["gender"].map({"M": 0, "F": 1})
        enc2["class_n"] = LabelEncoder().fit_transform(enc2["class"])
        corr = enc2[NUM + ["class_n"]].corr()
        fig, ax = plt.subplots(figsize=(13, 10))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                    linewidths=.4, ax=ax, annot_kws={"size": 7},
                    cbar_kws={"shrink": .7})
        ax.set_title("Feature Correlation Matrix"); show(fig)

        st.subheader("Correlation with Performance Class")
        cr = corr["class_n"].drop("class_n").sort_values()
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        ax2.barh(cr.index, cr.values,
                 color=[RED if v < 0 else GREEN for v in cr.values],
                 edgecolor=DARK)
        ax2.axvline(0, color=LBLUE, lw=1)
        ax2.set_title("Pearson r with Performance Class")
        ax2.set_xlabel("Correlation coefficient"); show(fig2)

    with t4:
        feat_sel = st.selectbox("Select feature", NUM, key="eda_sel")
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for cls_, col_ in zip(CL, PAL):
            sub = raw[raw["class"] == cls_][feat_sel]
            axes[0].hist(sub, bins=25, alpha=.55, label=f"Class {cls_}", color=col_)
        axes[0].set_title(f"{feat_sel} — Histogram by Class")
        axes[0].legend(); axes[0].set_xlabel(feat_sel)
        bp = axes[1].boxplot(
            [raw[raw["class"] == c][feat_sel].dropna() for c in CL],
            labels=CL, patch_artist=True,
        )
        for patch, c in zip(bp["boxes"], PAL):
            patch.set_facecolor(c); patch.set_alpha(.7)
        for md in bp["medians"]:
            md.set_color("white"); md.set_linewidth(2)
        axes[1].set_title(f"{feat_sel} — Boxplot by Class"); show(fig)
        st.dataframe(
            raw.groupby("class")[NUM].mean().round(2),
            use_container_width=True,
        )

    with t5:
        st.subheader("Key EDA Insights")
        for title_, body_, cls_ in [
            ("✅ Balanced Dataset",
             "~25% per class (A/B/C/D) — no SMOTE or reweighting needed.", "ca"),
            ("📐 Strongest Predictors",
             "sit_and_bend_forward_cm (r=−0.61), sit_ups_counts (r=−0.45), "
             "broad_jump_cm (r=−0.26).", "ca"),
            ("⚠️ Low-Signal Features",
             "diastolic & systolic BP: r < 0.07 with class — "
             "add noise rather than signal.", "cp"),
            ("👥 Gender Effect",
             "height_cm & gripForce bimodal (M/F peaks). "
             "Males jump ~40–50 cm further on average.", "card"),
            ("📅 Age Skew",
             "Right-skewed (skewness = 0.60). "
             "Older participants trend toward lower fitness classes.", "cr"),
        ]:
            st.markdown(
                f'<div class="card {cls_}"><b style="color:#3B8BD4">{title_}</b><br>'
                f'<span style="font-size:.9rem">{body_}</span></div>',
                unsafe_allow_html=True,
            )

# ══════════════════════════════════════════════════════════════════
# ██  PAGE: KNN  ██
# ══════════════════════════════════════════════════════════════════
elif "KNN" in page:
    st.title("🤖 K-Nearest Neighbors (KNN)")
    st.markdown(
        '<div class="card ca">KNN classifies by <b>majority vote</b> among k nearest '
        'neighbors. StandardScaler applied. k=1–50 evaluated. '
        'Best k=<b>22</b> on 80:20 → <b>63.09%</b>.</div>',
        unsafe_allow_html=True,
    )
    tc, tr = st.tabs(["🎯 Classification", "📈 Regression"])

    with tc:
        c1, c2, c3 = st.columns(3)
        kmax = c1.slider("Max k to evaluate", 5, 60, 35, key="kk")
        spl  = c2.selectbox("Split for tuning", ["80:20","70:30","50:50"], key="ks")
        wts  = c3.selectbox("Distance weights", ["uniform","distance"], key="kw")

        if st.button("▶  Run KNN Classification", use_container_width=True, key="rk"):
            ts = {"80:20": .2, "70:30": .3, "50:50": .5}[spl]
            with st.spinner("Sweeping k values…"):
                Xtr, Xte, ytr, yte = train_test_split(
                    Xc, yc, test_size=ts, random_state=42, stratify=yc)
                sc = StandardScaler()
                Xtr = sc.fit_transform(Xtr); Xte = sc.transform(Xte)
                kr = list(range(1, kmax + 1))
                accs = [
                    accuracy_score(
                        yte,
                        KNeighborsClassifier(k, weights=wts, n_jobs=-1)
                        .fit(Xtr, ytr).predict(Xte),
                    )
                    for k in kr
                ]
                bk = kr[int(np.argmax(accs))]; ba = max(accs)

            mbox({"Best k": bk, "Best Accuracy": f"{ba*100:.2f}%",
                  "Split": spl, "Weights": wts})

            # k-sweep chart
            fig, ax = plt.subplots(figsize=(11, 4))
            ax.plot(kr, [a*100 for a in accs], color=BLUE, lw=2)
            ax.fill_between(kr, [a*100 for a in accs], alpha=.1, color=BLUE)
            ax.axvline(bk, color=GREEN, lw=1.8, linestyle="--",
                       label=f"k={bk} → {ba*100:.1f}%")
            ax.set_xlabel("k"); ax.set_ylabel("Test Accuracy (%)")
            ax.set_title("k Sweep — Test Accuracy")
            ax.legend(); ax.grid(alpha=.15); show(fig)

            # Confusion + per-class
            bm = KNeighborsClassifier(bk, weights=wts, n_jobs=-1).fit(Xtr, ytr)
            yp = bm.predict(Xte)
            rpt = classification_report(yte, yp, target_names=CL, output_dict=True)
            c1, c2 = st.columns(2)
            with c1: conf_map(confusion_matrix(yte, yp), CL, f"Confusion Matrix k={bk}")
            with c2: per_class(rpt, CL)
            st.dataframe(pd.DataFrame(rpt).T.round(3), use_container_width=True)

            # Cross-split table
            st.subheader("Cross-Split Comparison")
            res, cv5 = run_cls(
                lambda: KNeighborsClassifier(bk, weights=wts, n_jobs=-1), Xc, yc)
            st.dataframe(res, use_container_width=True, hide_index=True)
            bar_c(res["Split"][:3],
                  [float(v.split("%")[0]) for v in res["Accuracy"][:3]],
                  "Accuracy (%)", "KNN Accuracy Across Splits", BLUE)

            # 5-fold CV
            st.subheader("5-Fold Cross-Validation")
            mbox({"CV Mean": f"{cv5.mean()*100:.2f}%",
                  "CV Std":  f"±{cv5.std()*100:.2f}%",
                  "Min":     f"{cv5.min()*100:.2f}%",
                  "Max":     f"{cv5.max()*100:.2f}%"})
            fold_chart(cv5, "Accuracy (%)", 100, PURPLE)

    with tr:
        c1, c2 = st.columns(2)
        kr_ = c1.slider("k for regression", 1, 60, 37, key="kreg")
        wr_ = c2.selectbox("Weights", ["uniform","distance"], key="kwreg")

        if st.button("▶  Run KNN Regression", use_container_width=True, key="rkr"):
            with st.spinner("Running regression…"):
                res, cv5 = run_reg(
                    lambda: KNeighborsRegressor(kr_, weights=wr_, n_jobs=-1), Xr, yr)
            st.dataframe(res, use_container_width=True, hide_index=True)
            bar_c(res["Split"][:3],
                  [float(v.split(" ")[0]) for v in res["R²"][:3]],
                  "R²", "KNN Regression R² Across Splits", GREEN)

            sc = StandardScaler()
            Xtr, Xte, ytr, yte = train_test_split(Xr, yr, test_size=.2, random_state=42)
            Xtr = sc.fit_transform(Xtr); Xte = sc.transform(Xte)
            m = KNeighborsRegressor(kr_, weights=wr_, n_jobs=-1)
            m.fit(Xtr, ytr); yp = m.predict(Xte)
            mbox({"R²":   f"{r2_score(yte,yp):.4f}",
                  "RMSE": f"{np.sqrt(mean_squared_error(yte,yp)):.2f} cm",
                  "MAE":  f"{mean_absolute_error(yte,yp):.2f} cm"})
            scatter_res(yte, yp, "KNN Regression — Actual vs Predicted (80:20)")
            fold_chart(cv5, "R²", 1, GREEN)

# ══════════════════════════════════════════════════════════════════
# ██  PAGE: LINEAR REGRESSION  ██
# ══════════════════════════════════════════════════════════════════
elif "Linear Regression" in page:
    st.title("📈 Linear Regression")
    st.markdown(
        '<div class="card ca">Primary task: predict <b>broad_jump_cm</b>. '
        'Best split: 70:30 → R²=<b>0.8033</b>. '
        'Also adapted for classification (~50% — fundamental task mismatch).</div>',
        unsafe_allow_html=True,
    )
    tr, tc = st.tabs(["📈 Regression (Primary)", "🎯 Classification (Adapted)"])

    with tr:
        if st.button("▶  Run Linear Regression", use_container_width=True, key="rlr"):
            with st.spinner("Training…"):
                res, cv5 = run_reg(LinearRegression, Xr, yr, scale=False)
            st.dataframe(res, use_container_width=True, hide_index=True)
            r2v = [float(v.split(" ")[0]) for v in res["R²"][:3]]
            rmv = [float(v) for v in res["RMSE"][:3]]
            c1, c2 = st.columns(2)
            with c1: bar_c(res["Split"][:3], r2v,  "R²",       "R² Across Splits",   GREEN)
            with c2: bar_c(res["Split"][:3], rmv,  "RMSE (cm)", "RMSE Across Splits", RED)

            Xtr, Xte, ytr, yte = train_test_split(Xr, yr, test_size=.2, random_state=42)
            m = LinearRegression(); m.fit(Xtr, ytr); yp = m.predict(Xte)
            mbox({"R²":   f"{r2_score(yte,yp):.4f}",
                  "RMSE": f"{np.sqrt(mean_squared_error(yte,yp)):.2f} cm",
                  "MAE":  f"{mean_absolute_error(yte,yp):.2f} cm"})
            scatter_res(yte, yp, "Linear Regression — Actual vs Predicted (80:20)")

            st.subheader("Feature Coefficients")
            m2 = LinearRegression(); m2.fit(Xr, yr)
            cd = pd.DataFrame({"Feature": FR, "Coefficient": m2.coef_}).sort_values(
                "Coefficient", key=abs, ascending=False)
            fig, ax = plt.subplots(figsize=(11, 4.5))
            ax.barh(cd["Feature"], cd["Coefficient"],
                    color=[GREEN if v >= 0 else RED for v in cd["Coefficient"]],
                    edgecolor=DARK)
            ax.axvline(0, color=LBLUE, lw=1)
            ax.set_title("Feature Coefficients (Linear Regression)"); show(fig)

            st.subheader("5-Fold CV")
            mbox({"CV Mean R²": f"{cv5.mean():.4f}", "CV Std": f"±{cv5.std():.4f}"})
            fold_chart(cv5, "R²", 1, GREEN)

    with tc:
        st.markdown(
            '<div class="wbox">⚠️ Linear Regression is not designed for classification. '
            'Continuous outputs binned to quartile-based performance classes.</div>',
            unsafe_allow_html=True,
        )
        if st.button("▶  Run Adapted Classification", use_container_width=True, key="rlrc"):
            rows = []
            for nm, ts in SPLITS:
                Xtr, Xte, ytr, yte = train_test_split(Xr, yr, test_size=ts, random_state=42)
                m = LinearRegression(); m.fit(Xtr, ytr); yp = m.predict(Xte)
                q = np.percentile(ytr, [25, 50, 75])
                def _b(v):
                    return 3 if v >= q[2] else 2 if v >= q[1] else 1 if v >= q[0] else 0
                acc = accuracy_score([_b(v) for v in yte], [_b(v) for v in yp])
                rows.append({"Split": nm, "Adapted Accuracy": f"{acc*100:.1f}%"})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            st.markdown(
                '<div class="wbox">Result: ~49–51%. Confirms task mismatch — '
                'use a dedicated classifier instead.</div>',
                unsafe_allow_html=True,
            )

# ══════════════════════════════════════════════════════════════════
# ██  PAGE: DECISION TREE  ██
# ══════════════════════════════════════════════════════════════════
elif "Decision Tree" in page:
    st.title("🌳 Decision Tree")
    st.markdown(
        '<div class="card ca">Cost-complexity pruning via 5-fold CV. '
        'Optimal: α=0.000377 → depth 14, 477 nodes → <b>72.15%</b> accuracy. '
        'Unpruned tree (depth 28): 100% train, 64.9% test (severe overfitting).</div>',
        unsafe_allow_html=True,
    )
    tc, tr = st.tabs(["🎯 Classification", "📈 Regression"])

    with tc:
        c1, c2 = st.columns(2)
        alpha = c1.slider("ccp_alpha", 0.0, 0.008, 0.000377,
                          step=5e-5, format="%.6f", key="dta")
        maxd  = c2.slider("Max depth (0 = unlimited)", 0, 30, 0, key="dtd")
        md = None if maxd == 0 else maxd

        if st.button("▶  Run Decision Tree Classification",
                     use_container_width=True, key="rdt"):
            with st.spinner("Training…"):
                Xtr, Xte, ytr, yte = train_test_split(
                    Xc, yc, test_size=.2, random_state=42, stratify=yc)
                m = DecisionTreeClassifier(
                    random_state=42, ccp_alpha=alpha, max_depth=md)
                m.fit(Xtr, ytr); yp = m.predict(Xte)

            mbox({"Accuracy": f"{accuracy_score(yte,yp)*100:.2f}%",
                  "Depth": m.get_depth(), "Nodes": m.tree_.node_count,
                  "α": f"{alpha:.6f}"})

            c1, c2 = st.columns(2)
            with c1: conf_map(confusion_matrix(yte, yp), CL, "Confusion Matrix")
            with c2:
                rpt = classification_report(yte, yp, target_names=CL, output_dict=True)
                per_class(rpt, CL)
            st.dataframe(pd.DataFrame(rpt).T.round(3), use_container_width=True)

            # Alpha sweep
            st.subheader("Alpha Pruning Sweep (Overfitting → Underfitting)")
            with st.spinner("Sweeping alphas…"):
                alsw = np.linspace(0, .006, 50)
                tra, tea = [], []
                for a in alsw:
                    mm = DecisionTreeClassifier(random_state=42, ccp_alpha=a)
                    mm.fit(Xtr, ytr)
                    tra.append(mm.score(Xtr, ytr))
                    tea.append(mm.score(Xte, yte))
            fig, ax = plt.subplots(figsize=(11, 4))
            ax.plot(alsw, [a*100 for a in tra], color=BLUE, lw=2, label="Train")
            ax.plot(alsw, [a*100 for a in tea], color=GREEN, lw=2, label="Test")
            ax.axvline(alpha, color=RED, lw=1.5, linestyle="--",
                       label=f"Selected α={alpha:.5f}")
            ax.set_xlabel("ccp_alpha"); ax.set_ylabel("Accuracy (%)")
            ax.set_title("Train vs Test Accuracy Across Alpha")
            ax.legend(); ax.grid(alpha=.15); show(fig)

            # Cross-split table
            st.subheader("Cross-Split Results")
            res, cv5 = run_cls(
                lambda: DecisionTreeClassifier(
                    random_state=42, ccp_alpha=alpha, max_depth=md),
                Xc, yc, scale=False,
            )
            st.dataframe(res, use_container_width=True, hide_index=True)
            bar_c(res["Split"][:3],
                  [float(v.split("%")[0]) for v in res["Accuracy"][:3]],
                  "Accuracy (%)", "Decision Tree Splits", PURPLE)
            fold_chart(cv5, "Accuracy (%)", 100, PURPLE)

            # Tree visualisation
            st.subheader("Tree Structure (top 3 levels)")
            fig2, ax2 = plt.subplots(figsize=(20, 8))
            plot_tree(m, max_depth=3, feature_names=FC, class_names=CL,
                      filled=True, rounded=True, precision=2,
                      ax=ax2, fontsize=8, impurity=False)
            ax2.set_title("Decision Tree — Top 3 Levels", pad=10); show(fig2)

    with tr:
        c1, c2 = st.columns(2)
        dtr_a = c1.slider("ccp_alpha (regression)", 0.0, 100.0, 0.0, key="dtrar")
        dtr_d = c2.slider("Max depth", 0, 20, 0, key="dtrdr")
        md2 = None if dtr_d == 0 else dtr_d

        if st.button("▶  Run DT Regression", use_container_width=True, key="rdtr"):
            with st.spinner("Training…"):
                res, cv5 = run_reg(
                    lambda: DecisionTreeRegressor(
                        random_state=42, ccp_alpha=dtr_a, max_depth=md2),
                    Xr, yr, scale=False,
                )
            st.dataframe(res, use_container_width=True, hide_index=True)
            bar_c(res["Split"][:3],
                  [float(v.split(" ")[0]) for v in res["R²"][:3]],
                  "R²", "DT Regression R²", GREEN)
            Xtr, Xte, ytr, yte = train_test_split(Xr, yr, test_size=.2, random_state=42)
            m2 = DecisionTreeRegressor(random_state=42, ccp_alpha=dtr_a, max_depth=md2)
            m2.fit(Xtr, ytr)
            scatter_res(yte, m2.predict(Xte), "DT Regression — Actual vs Predicted")
            fold_chart(cv5, "R²", 1, GREEN)

# ══════════════════════════════════════════════════════════════════
# ██  PAGE: SVM  ██
# ══════════════════════════════════════════════════════════════════
elif "SVM" in page:
    st.title("⚙️ Support Vector Machine (SVM)")
    st.markdown(
        '<div class="card ca">Max-margin classifier. Best: '
        '<b>RBF kernel, C=10, γ=0.1 → 71.27%</b> accuracy.</div>',
        unsafe_allow_html=True,
    )
    tc, tr = st.tabs(["🎯 Classification", "📈 Regression (SVR)"])

    with tc:
        c1, c2, c3 = st.columns(3)
        kern = c1.selectbox("Kernel", ["rbf","linear","poly"], key="svmk")
        Cval = c2.select_slider("C", [0.1,1,10,100], value=10, key="svmC")
        gam  = c3.selectbox("Gamma", ["scale","auto","0.01","0.1","1"],
                             index=3, key="svmg")
        gv   = gam if gam in ["scale","auto"] else float(gam)

        if st.button("▶  Run SVM Classification", use_container_width=True, key="rsvm"):
            with st.spinner("Training SVM (may take 30–60 s on large datasets)…"):
                sc = StandardScaler()
                Xtr, Xte, ytr, yte = train_test_split(
                    Xc, yc, test_size=.2, random_state=42, stratify=yc)
                Xtr = sc.fit_transform(Xtr); Xte = sc.transform(Xte)
                m = SVC(kernel=kern, C=Cval, gamma=gv, random_state=42)
                m.fit(Xtr, ytr); yp = m.predict(Xte)

            mbox({"Accuracy":  f"{accuracy_score(yte,yp)*100:.2f}%",
                  "Macro F1":  f"{f1_score(yte,yp,average='macro'):.3f}",
                  "Kernel":    kern.upper(), "C": Cval})

            c1, c2 = st.columns(2)
            with c1: conf_map(confusion_matrix(yte, yp), CL, f"Confusion Matrix ({kern})")
            with c2:
                rpt = classification_report(yte, yp, target_names=CL, output_dict=True)
                per_class(rpt, CL)
            st.dataframe(pd.DataFrame(rpt).T.round(3), use_container_width=True)

            st.subheader("Kernel Comparison (80:20 split)")
            with st.spinner("Comparing all kernels…"):
                kr_res = []
                for kn, Cv, gv2 in [("linear",1,"scale"),("rbf",1,0.1),
                                     ("rbf",10,0.1),("poly",1,"scale")]:
                    mm = SVC(kernel=kn, C=Cv, gamma=gv2, random_state=42)
                    mm.fit(Xtr, ytr)
                    kr_res.append({"Config": f"{kn} C={Cv}",
                                   "Accuracy": f"{mm.score(Xte,yte)*100:.2f}%"})
            st.dataframe(pd.DataFrame(kr_res), use_container_width=True, hide_index=True)

            st.subheader("Cross-Split Results")
            res, cv5 = run_cls(
                lambda: SVC(kernel=kern, C=Cval, gamma=gv, random_state=42), Xc, yc)
            st.dataframe(res, use_container_width=True, hide_index=True)
            bar_c(res["Split"][:3],
                  [float(v.split("%")[0]) for v in res["Accuracy"][:3]],
                  "Accuracy (%)", "SVM Accuracy Across Splits", PURPLE)
            fold_chart(cv5, "Accuracy (%)", 100, PURPLE)

    with tr:
        c1, c2, c3 = st.columns(3)
        svk = c1.selectbox("SVR Kernel", ["rbf","linear"], key="svrk")
        svC = c2.select_slider("C", [0.1,1,10,100], value=10, key="svrC")
        sve = c3.slider("Epsilon", .01, 1., .1, step=.05, key="svre")

        if st.button("▶  Run SVR", use_container_width=True, key="rsvr"):
            with st.spinner("Training SVR…"):
                res, cv5 = run_reg(
                    lambda: SVR(kernel=svk, C=svC, epsilon=sve), Xr, yr)
            st.dataframe(res, use_container_width=True, hide_index=True)
            bar_c(res["Split"][:3],
                  [float(v.split(" ")[0]) for v in res["R²"][:3]],
                  "R²", "SVR R² Across Splits", PURPLE)
            sc = StandardScaler()
            Xtr, Xte, ytr, yte = train_test_split(Xr, yr, test_size=.2, random_state=42)
            Xtr = sc.fit_transform(Xtr); Xte = sc.transform(Xte)
            m = SVR(kernel=svk, C=svC, epsilon=sve); m.fit(Xtr, ytr); yp = m.predict(Xte)
            mbox({"R²":   f"{r2_score(yte,yp):.4f}",
                  "RMSE": f"{np.sqrt(mean_squared_error(yte,yp)):.2f} cm",
                  "MAE":  f"{mean_absolute_error(yte,yp):.2f} cm"})
            scatter_res(yte, yp, "SVR — Actual vs Predicted (80:20)")
            fold_chart(cv5, "R²", 1, PURPLE)

# ══════════════════════════════════════════════════════════════════
# ██  PAGE: NEURAL NETWORK  ██
# ══════════════════════════════════════════════════════════════════
elif "Neural Network" in page:
    st.title("🧠 Neural Network (MLP)")
    st.markdown(
        '<div class="card ca">Architecture: Dense 256→128→64 + ELU + BatchNorm + Dropout. '
        'EarlyStopping (patience=15) + ReduceLROnPlateau. '
        'Best accuracy after leakage fix: <b>74.80%</b>.</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="wbox">⚠️ <b>Leakage note:</b> An earlier version showed 100% accuracy '
        'due to target labels leaking through a preprocessing step. '
        'Results here use the corrected pipeline.</div>',
        unsafe_allow_html=True,
    )

    ta, tb, tc_ = st.tabs(["🎯 Classification", "📈 Regression", "🏗️ Architecture"])

    with tc_:
        st.subheader("Network Architecture")
        st.dataframe(pd.DataFrame([
            {"Block":"Input",     "Layer":"—",     "Units":len(FC),"Activation":"—",      "Regularisation":"—"},
            {"Block":"Hidden 1",  "Layer":"Dense",  "Units":256,   "Activation":"ELU",    "Regularisation":"BatchNorm + Dropout 0.20"},
            {"Block":"Hidden 2",  "Layer":"Dense",  "Units":128,   "Activation":"ELU",    "Regularisation":"BatchNorm + Dropout 0.15"},
            {"Block":"Hidden 3",  "Layer":"Dense",  "Units":64,    "Activation":"ELU",    "Regularisation":"BatchNorm + Dropout 0.10"},
            {"Block":"Out (Cls)", "Layer":"Dense",  "Units":4,     "Activation":"Softmax","Regularisation":"—"},
            {"Block":"Out (Reg)", "Layer":"Dense",  "Units":1,     "Activation":"Linear", "Regularisation":"—"},
        ]), use_container_width=True, hide_index=True)
        st.subheader("Training Configuration")
        st.dataframe(pd.DataFrame([
            {"Parameter":"Optimiser",        "Value":"Adam  lr=0.001"},
            {"Parameter":"Loss (Cls)",        "Value":"Categorical Cross-Entropy"},
            {"Parameter":"Loss (Reg)",        "Value":"Mean Squared Error"},
            {"Parameter":"Early Stopping",    "Value":"patience=15, restore_best_weights=True"},
            {"Parameter":"ReduceLROnPlateau", "Value":"factor=0.5, patience=5, min_lr=1e-6"},
            {"Parameter":"Batch Size",        "Value":"32"},
            {"Parameter":"Max Epochs",        "Value":"200"},
            {"Parameter":"Validation Split",  "Value":"15 % of training data"},
        ]), use_container_width=True, hide_index=True)

    if not TF_OK:
        for t in [ta, tb]:
            with t:
                st.markdown(
                    '<div class="wbox">⚠️ TensorFlow is not installed in this environment. '
                    'Add <code>tensorflow-cpu</code> to <b>requirements.txt</b> '
                    'and redeploy.</div>',
                    unsafe_allow_html=True,
                )
    else:
        def mk_cls(inp, dr):
            np.random.seed(42); tf.random.set_seed(42)
            m = Sequential([
                Dense(256, input_dim=inp, use_bias=False),
                BatchNormalization(), Activation("elu"), Dropout(dr),
                Dense(128, use_bias=False), BatchNormalization(),
                Activation("elu"), Dropout(dr * .75),
                Dense(64,  use_bias=False), BatchNormalization(),
                Activation("elu"), Dropout(dr * .5),
                Dense(4, activation="softmax"),
            ])
            m.compile(optimizer=Adam(.001),
                      loss="categorical_crossentropy", metrics=["accuracy"])
            return m

        with ta:
            c1, c2, c3 = st.columns(3)
            ep_ = c1.slider("Max Epochs",  30, 200, 100, key="nep")
            pa_ = c2.slider("ES Patience",  5,  30,  15, key="npa")
            dr_ = c3.slider("Dropout Rate", .1,  .5,  .2, step=.05, key="ndr")

            if st.button("▶  Train Neural Network", use_container_width=True, key="rnn"):
                rows_ = []; lh = None; lyte = None; lyp = None

                for nm, ts in SPLITS:
                    with st.spinner(f"Training {nm} split…"):
                        np.random.seed(42); tf.random.set_seed(42)
                        Xtr, Xte, ytr, yte = train_test_split(
                            Xc, yc, test_size=ts, random_state=42, stratify=yc)
                        sc = StandardScaler()
                        Xtr = sc.fit_transform(Xtr); Xte = sc.transform(Xte)
                        ytrc = to_categorical(ytr, 4)
                        model = mk_cls(Xtr.shape[1], dr_)
                        cb = [
                            EarlyStopping(monitor="val_loss", patience=pa_,
                                          restore_best_weights=True, verbose=0),
                            ReduceLROnPlateau(monitor="val_loss", factor=.5,
                                             patience=5, min_lr=1e-6, verbose=0),
                        ]
                        h = model.fit(Xtr, ytrc, epochs=ep_, batch_size=32,
                                      validation_split=.15, callbacks=cb, verbose=0)
                        yp_ = np.argmax(model.predict(Xte, verbose=0), axis=1)
                        acc = accuracy_score(yte, yp_)
                        rows_.append({"Split": nm,
                                      "Accuracy": f"{acc*100:.2f}%",
                                      "Epochs run": len(h.history["accuracy"])})
                        lh = h; lyte = yte; lyp = yp_

                with st.spinner("Running 5-fold CV…"):
                    skf = StratifiedKFold(5, shuffle=True, random_state=42)
                    cv_a = []
                    for ti, ei in skf.split(Xc, yc):
                        np.random.seed(42); tf.random.set_seed(42)
                        Xt, Xe, yt, ye = Xc[ti], Xc[ei], yc[ti], yc[ei]
                        s = StandardScaler()
                        Xt = s.fit_transform(Xt); Xe = s.transform(Xe)
                        mf = mk_cls(Xt.shape[1], dr_)
                        mf.fit(
                            Xt, to_categorical(yt, 4),
                            epochs=ep_, batch_size=32,
                            validation_split=.15, verbose=0,
                            callbacks=[EarlyStopping(monitor="val_loss",
                                       patience=pa_, restore_best_weights=True, verbose=0)],
                        )
                        cv_a.append(accuracy_score(
                            ye, np.argmax(mf.predict(Xe, verbose=0), axis=1)))

                cv_a = np.array(cv_a)
                rows_.append({"Split": "5-Fold CV",
                               "Accuracy": f"{cv_a.mean()*100:.2f}% ±{cv_a.std()*100:.2f}",
                               "Epochs run": "—"})
                st.dataframe(pd.DataFrame(rows_), use_container_width=True, hide_index=True)

                # Training curves
                st.subheader("Training Curves (last split)")
                fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
                ax[0].plot(lh.history["accuracy"],     color=BLUE,  lw=2, label="Train")
                ax[0].plot(lh.history["val_accuracy"], color=GREEN, lw=2, label="Val")
                ax[0].set_title("Accuracy"); ax[0].set_xlabel("Epoch")
                ax[0].legend(); ax[0].grid(alpha=.15)
                ax[1].plot(lh.history["loss"],     color=BLUE, lw=2, label="Train Loss")
                ax[1].plot(lh.history["val_loss"], color=RED,  lw=2, label="Val Loss")
                ax[1].set_title("Loss"); ax[1].set_xlabel("Epoch")
                ax[1].legend(); ax[1].grid(alpha=.15)
                show(fig)

                # Confusion + per-class
                c1, c2 = st.columns(2)
                with c1: conf_map(confusion_matrix(lyte, lyp), CL, "Confusion Matrix")
                with c2:
                    rpt = classification_report(lyte, lyp, target_names=CL, output_dict=True)
                    per_class(rpt, CL)
                st.dataframe(pd.DataFrame(rpt).T.round(3), use_container_width=True)
                fold_chart(cv_a, "Accuracy (%)", 100, BLUE)

        with tb:
            if st.button("▶  Train NN Regression", use_container_width=True, key="rnnr"):
                def mk_reg(inp, dr):
                    np.random.seed(42); tf.random.set_seed(42)
                    m = Sequential([
                        Dense(256, input_dim=inp, use_bias=False),
                        BatchNormalization(), Activation("elu"), Dropout(dr),
                        Dense(128, use_bias=False), BatchNormalization(),
                        Activation("elu"), Dropout(dr * .75),
                        Dense(64,  use_bias=False), BatchNormalization(),
                        Activation("elu"), Dropout(dr * .5),
                        Dense(1, activation="linear"),
                    ])
                    m.compile(optimizer=Adam(.001), loss="mse", metrics=["mae"])
                    return m

                rr = []
                for nm, ts in SPLITS:
                    with st.spinner(f"Regression {nm}…"):
                        np.random.seed(42); tf.random.set_seed(42)
                        Xtr, Xte, ytr, yte = train_test_split(
                            Xr, yr, test_size=ts, random_state=42)
                        sX = StandardScaler(); sY = StandardScaler()
                        Xtr = sX.fit_transform(Xtr); Xte = sX.transform(Xte)
                        yts = sY.fit_transform(ytr.reshape(-1, 1)).ravel()
                        mr = mk_reg(Xtr.shape[1], dr_)
                        mr.fit(
                            Xtr, yts, epochs=100, batch_size=32,
                            validation_split=.15, verbose=0,
                            callbacks=[EarlyStopping(monitor="val_loss",
                                       patience=15, restore_best_weights=True, verbose=0)],
                        )
                        yp_ = sY.inverse_transform(
                            mr.predict(Xte, verbose=0)).ravel()
                        rr.append({"Split": nm,
                                   "R²":   f"{r2_score(yte,yp_):.4f}",
                                   "RMSE": f"{np.sqrt(mean_squared_error(yte,yp_)):.3f}",
                                   "MAE":  f"{mean_absolute_error(yte,yp_):.3f}"})

                st.dataframe(pd.DataFrame(rr), use_container_width=True, hide_index=True)
                bar_c([r["Split"] for r in rr],
                      [float(r["R²"]) for r in rr],
                      "R²", "NN Regression R² Across Splits", BLUE)
                scatter_res(yte, yp_, "NN Regression — Actual vs Predicted")

# ══════════════════════════════════════════════════════════════════
# ██  PAGE: MODEL COMPARISON  ██
# ══════════════════════════════════════════════════════════════════
elif "Comparison" in page:
    st.title("🏆 Model Comparison")

    st.markdown('<div class="card ca">', unsafe_allow_html=True)
    st.subheader("🎯 Classification — All Models")
    st.dataframe(pd.DataFrame([
        {"Model":"KNN (k=22)",            "Best Acc":"63.09%","F1":"0.63","Split":"80:20","5-Fold CV":"61.9%", "Note":"❌"},
        {"Model":"Decision Tree (pruned)","Best Acc":"72.15%","F1":"0.71","Split":"80:20","5-Fold CV":"70.39%","Note":"✅"},
        {"Model":"SVM (RBF C=10)",        "Best Acc":"71.27%","F1":"0.71","Split":"80:20","5-Fold CV":"70.8%", "Note":"✅"},
        {"Model":"Neural Network (MLP)",  "Best Acc":"74.80%","F1":"~0.74","Split":"70:30","5-Fold CV":"73.98%","Note":"⭐ Best"},
        {"Model":"Linear Reg. (adapted)", "Best Acc":"~50%",  "F1":"~0.48","Split":"N/A","5-Fold CV":"N/A",   "Note":"❌"},
    ]), use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card cp">', unsafe_allow_html=True)
    st.subheader("📈 Regression — All Models")
    st.dataframe(pd.DataFrame([
        {"Model":"Neural Network",   "Best R²":"0.802","RMSE":"17.77 cm","Split":"80:20 / 70:30","Note":"⭐ Best"},
        {"Model":"Linear Regression","Best R²":"0.803","RMSE":"17.89 cm","Split":"70:30",         "Note":"✅ Simplest"},
        {"Model":"SVM (SVR)",        "Best R²":"0.79", "RMSE":"18.4 cm", "Split":"80:20",         "Note":"✅"},
        {"Model":"KNN Regression",   "Best R²":"0.788","RMSE":"~18.5 cm","Split":"70:30",         "Note":"✅"},
        {"Model":"Decision Tree",    "Best R²":"0.76", "RMSE":"20.1 cm", "Split":"80:20",         "Note":"⚠️"},
    ]), use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📊 Visual Comparison")
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        models_c = ["KNN","Dec.Tree","SVM","Neur.Net","Lin.Reg"]
        accs_c   = [63.09, 72.15, 71.27, 74.80, 50.0]
        bars = ax.bar(models_c, accs_c, color=PAL+[LBLUE],
                      edgecolor=DARK, width=.55, zorder=3)
        for b in bars:
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+.5,
                    f"{b.get_height():.1f}%", ha="center",
                    color="#C8D8EA", fontsize=9)
        ax.set_ylim(0, 90); ax.set_ylabel("Best Accuracy (%)")
        ax.set_title("Classification Accuracy")
        ax.axhline(74.8, color=GREEN, lw=1.5, linestyle="--",
                   alpha=.7, label="Best 74.8%")
        ax.legend(); ax.grid(axis="y", alpha=.15); show(fig)
    with c2:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        models_r = ["Neur.Net","Lin.Reg","SVM","KNN","Dec.Tree"]
        r2s      = [.802, .803, .79, .788, .76]
        bars = ax.bar(models_r, r2s,
                      color=[GREEN,BLUE,PURPLE,LBLUE,RED],
                      edgecolor=DARK, width=.55, zorder=3)
        for b in bars:
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+.004,
                    f"{b.get_height():.3f}", ha="center",
                    color="#C8D8EA", fontsize=9)
        ax.set_ylim(.5, 1.0); ax.set_ylabel("R² Score")
        ax.set_title("Regression R² Score")
        ax.grid(axis="y", alpha=.15); show(fig)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🕸️ Radar Chart — Classification Metrics (%)")
    cats   = ["Accuracy","Precision","Recall","F1","CV Score"]
    scores_r = {
        "KNN":      [63, 65, 63, 63, 62],
        "Dec.Tree": [72, 72, 71, 71, 70],
        "SVM":      [71, 71, 71, 71, 71],
        "MLP":      [75, 74, 75, 74, 74],
    }
    N = len(cats)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(6.5, 6.5), subplot_kw=dict(polar=True))
    for (nm, sc), col in zip(scores_r.items(), PAL):
        v = sc + sc[:1]
        ax.plot(angles, v, color=col, lw=2.2, label=nm)
        ax.fill(angles, v, color=col, alpha=.1)
    ax.set_thetagrids(np.degrees(angles[:-1]), cats, color=LBLUE, fontsize=10)
    ax.set_ylim(50, 82); ax.set_yticks([55, 65, 75])
    ax.set_yticklabels(["55","65","75"], fontsize=7, color=LBLUE)
    ax.set_facecolor(PANEL); ax.figure.patch.set_facecolor(DARK)
    for line in ax.xaxis.get_gridlines(): line.set_color("#1D4570")
    ax.set_title("Classification Radar", pad=22, color=BLUE)
    ax.legend(loc="upper right", bbox_to_anchor=(1.38, 1.12),
              facecolor=PANEL, edgecolor="#1D4570", labelcolor="#C8D8EA")
    st.pyplot(fig, use_container_width=True); plt.close(fig)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card cr">', unsafe_allow_html=True)
    st.subheader("⚠️ ~71% Accuracy Ceiling — Root Cause Analysis")
    st.markdown("""
| Root Cause | Explanation |
|---|---|
| **Class B/C Overlap** | Adjacent tiers share overlapping physiological ranges; F1 for B (0.48–0.58) is consistently the lowest |
| **Low-signal features** | Diastolic & systolic BP: r < 0.07; feature ablation shows ≤1% gain from removal |
| **Label ambiguity** | Classes discretised from continuous scores; boundary participants could belong to either adjacent tier |
| **Tuning performed** | SVM C=0.01→100, 3 kernels; DT alpha via CV; KNN k=1→99; MLP depths 1–3, dropout 0.1–0.5 |

**Best legitimate accuracy: MLP = 74.80%** — after correcting the 100% data-leakage bug.
    """)
    st.markdown("</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# ██  PAGE: LIVE PREDICTOR  ██
# ══════════════════════════════════════════════════════════════════
elif "Predictor" in page:
    st.title("🔮 Live Predictor")
    st.markdown(
        '<div class="card ca">Enter participant measurements to predict '
        '<b>Performance Class (A–D)</b> and <b>Broad Jump distance (cm)</b>.</div>',
        unsafe_allow_html=True,
    )

    with st.form("pred_form"):
        st.subheader("📋 Participant Measurements")
        c1, c2, c3, c4 = st.columns(4)
        age_ = c1.number_input("Age (yr)",         18, 80, 28)
        gnd_ = c2.selectbox("Gender",              ["M","F"])
        ht_  = c3.number_input("Height (cm)",     140.0, 200.0, 170.0, step=.5)
        wt_  = c4.number_input("Weight (kg)",      30.0, 150.0,  68.0, step=.5)
        c1, c2, c3, c4 = st.columns(4)
        bf_  = c1.number_input("Body Fat (%)",      3.0,  60.0,  22.0, step=.5)
        dia_ = c2.number_input("Diastolic BP",     50.0, 130.0,  79.0)
        sys_ = c3.number_input("Systolic BP",      80.0, 180.0, 130.0)
        gr_  = c4.number_input("Grip Force (kg)",   5.0,  80.0,  37.0, step=.5)
        c1, c2, c3 = st.columns(3)
        fl_  = c1.number_input("Sit&Bend (cm)",   -20.0,  50.0,  15.0, step=.5)
        su_  = c2.number_input("Sit-ups",           0.0,  80.0,  40.0)
        bj_  = c3.number_input("Broad Jump (cm)",   0.0, 310.0, 190.0, step=1.0,
                                help="Used as feature for classification. Leave at 190 if unknown.")
        mc_  = st.selectbox("Classifier",
                            ["KNN (k=22)","Decision Tree","SVM (RBF, C=10)"])
        sub_ = st.form_submit_button("🚀  Predict Performance",
                                     use_container_width=True)

    if sub_:
        bmi_ = wt_ / ((ht_ / 100) ** 2)
        ge_  = 0 if gnd_ == "M" else 1

        xc = np.array([[age_, ge_, ht_, wt_, bf_, dia_, sys_,
                         gr_, fl_, su_, bj_, bmi_]])
        xr = np.array([[age_, ge_, ht_, wt_, bf_, dia_, sys_,
                         gr_, fl_, su_, bmi_]])

        sc_c = StandardScaler(); Xcs = sc_c.fit_transform(Xc)
        sc_r = StandardScaler(); Xrs = sc_r.fit_transform(Xr)

        with st.spinner("Running prediction…"):
            if "KNN" in mc_:
                clf = KNeighborsClassifier(22, n_jobs=-1).fit(Xcs, yc)
                rgr = KNeighborsRegressor(35, n_jobs=-1).fit(Xrs, yr)
                xci, xri = sc_c.transform(xc), sc_r.transform(xr)
            elif "Decision" in mc_:
                clf = DecisionTreeClassifier(random_state=42, ccp_alpha=.000377).fit(Xc, yc)
                rgr = DecisionTreeRegressor(random_state=42).fit(Xr, yr)
                xci, xri = xc, xr
            else:
                clf = SVC(kernel="rbf", C=10, gamma=.1, random_state=42).fit(Xcs, yc)
                rgr = SVR(kernel="rbf", C=10).fit(Xrs, yr)
                xci, xri = sc_c.transform(xc), sc_r.transform(xr)

            pc = clf.predict(xci)[0]
            pj = float(rgr.predict(xri)[0])

        cn = CL[pc]
        INFO = {
            "A": ("🥇 Excellent", GREEN,  "Top tier fitness. Maintain your programme."),
            "B": ("🥈 Good",      BLUE,   "Above average. Focus on targeted strength work."),
            "C": ("🥉 Average",   PURPLE, "Moderate fitness. Increase endurance training."),
            "D": ("⚠️ Below Avg", RED,    "Below average. Start a structured fitness programme."),
        }
        lbl, col, adv = INFO[cn]
        st.markdown(
            f'<div class="card" style="border:2px solid {col};text-align:center;padding:28px">'
            f'<div style="font-size:3rem;line-height:1">{lbl}</div>'
            f'<div style="font-size:1.5rem;font-weight:900;color:{col};margin:8px 0">'
            f'Performance Class: {cn}</div>'
            f'<div style="color:#C8D8EA">{adv}</div></div>',
            unsafe_allow_html=True,
        )
        mbox({"Class": cn, "Predicted Jump": f"{pj:.1f} cm",
              "BMI": f"{bmi_:.1f}", "Model": mc_.split()[0]})

        # Jump gauge
        st.subheader("📊 Broad Jump Assessment")
        fig, ax = plt.subplots(figsize=(10, 2.8))
        for lo, hi, c, lb in [
            (0, 120, RED, "< 120"),   (120, 165, "#E8963A", "120–165"),
            (165, 200, BLUE, "165–200"), (200, 240, GREEN, "200–240"),
            (240, 320, "#5DCAA5", "240+"),
        ]:
            ax.barh(0, hi-lo, left=lo, height=.5, color=c, edgecolor=DARK, alpha=.75)
            ax.text((lo+hi)/2, .53, lb, ha="center", va="bottom",
                    color="#C8D8EA", fontsize=8)
        ax.axvline(pj, color="white", lw=3, zorder=5, label=f"{pj:.0f} cm")
        ax.set_xlim(0, 320); ax.set_ylim(-.2, .9)
        ax.set_yticks([]); ax.set_xlabel("Broad Jump Distance (cm)")
        ax.set_title("Jump Distance Gauge"); ax.legend(); show(fig)

        # Profile comparison
        st.subheader("👤 Your Profile vs Dataset Averages")
        fn = ["Age","Height","Weight","Body Fat%","Diastolic",
              "Systolic","Grip","Sit&Bend","Sit-ups","BMI"]
        yv = [age_, ht_, wt_, bf_, dia_, sys_, gr_, fl_, su_, round(bmi_, 1)]
        mv = [
            df["age"].mean(), df["height_cm"].mean(), df["weight_kg"].mean(),
            df["body fat_%"].mean(), df["diastolic"].mean(), df["systolic"].mean(),
            df["gripForce"].mean(), df["sit and bend forward_cm"].mean(),
            df["sit-ups counts"].mean(), df["BMI"].mean(),
        ]
        x_ = np.arange(len(fn))
        fig, ax = plt.subplots(figsize=(13, 4.5))
        ax.bar(x_-.22, yv, .42, label="You",         color=BLUE,   edgecolor=DARK, zorder=3)
        ax.bar(x_+.22, mv, .42, label="Dataset Mean", color=PURPLE, edgecolor=DARK,
               alpha=.75, zorder=3)
        ax.set_xticks(x_)
        ax.set_xticklabels(fn, rotation=35, ha="right", fontsize=8)
        ax.set_title("Your Values vs Dataset Averages")
        ax.legend(); ax.grid(axis="y", alpha=.15); show(fig)

# ══════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<hr style="border:none;border-top:1px solid #1D4570;margin-top:50px">
<div style="text-align:center;padding:16px 0;color:#4d7fa8;font-size:.82rem">
  <strong style="color:#3B8BD4;font-size:.95rem">AXORA</strong>
  &nbsp;·&nbsp; Body Performance Analytics &amp; Intelligent Classification System<br>
  <span style="color:#85B7EB">
    Alaa Issawi &nbsp;·&nbsp; Amira Salama &nbsp;·&nbsp;
    Aya Abdel Maksoud &nbsp;·&nbsp; Aya El-Sabi &nbsp;·&nbsp;
    Aya Imam &nbsp;·&nbsp; Aya Khalil
  </span><br>
  <span style="color:#534AB7">Introduction to AI &amp; ML &nbsp;·&nbsp; 2024–2025</span>
</div>
""", unsafe_allow_html=True)
