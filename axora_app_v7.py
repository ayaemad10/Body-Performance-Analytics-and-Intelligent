"""
AXORA — Body Performance Analytics & Intelligent Classification System
Team Axora | Introduction to AI & ML | 2024-2025
"""
import warnings; warnings.filterwarnings("ignore")
import streamlit as st, pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns, os

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    mean_squared_error, r2_score, mean_absolute_error
)

st.set_page_config(page_title="Axora | Body Performance",
                   page_icon="🧠", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("""
<style>
html,[data-testid="stAppViewContainer"]{background:#0B1621!important}
[data-testid="stSidebar"]{background:#0D1B2A!important;border-right:1px solid #162840;
  min-width:210px!important;max-width:225px!important}
h1{color:#3B8BD4!important;font-size:1.85rem!important;font-weight:800!important;margin-bottom:.3rem!important}
h2{color:#3B8BD4!important;font-size:1.3rem!important;font-weight:700!important}
h3{color:#85B7EB!important;font-size:1rem!important;font-weight:600!important}
p,.stMarkdown p,li{color:#C8D8EA!important;line-height:1.65}
code{background:#162840!important;color:#5DCAA5!important;border-radius:4px;padding:1px 5px}
.card{background:#0F1E33;border:1px solid #162840;border-radius:12px;padding:20px 24px;margin-bottom:16px}
.cg{border-left:3px solid #1D9E75}.cb{border-left:3px solid #3B8BD4}
.cp{border-left:3px solid #534AB7}.cr{border-left:3px solid #E24B4A}
.mt{background:#0F1E33;border:1px solid #162840;border-radius:10px;padding:16px 10px;text-align:center}
.mv{font-size:1.7rem;font-weight:800;color:#3B8BD4;line-height:1}
.ml{font-size:.67rem;color:#85B7EB;text-transform:uppercase;letter-spacing:1.4px;margin-top:5px}
.dataframe thead th{background:#1A3050!important;color:#85B7EB!important;font-size:.79rem}
.dataframe tbody td{color:#C8D8EA!important;font-size:.8rem}
.dataframe tbody tr:nth-child(even){background:#0F1E33!important}
.dataframe tbody tr:nth-child(odd){background:#0B1621!important}
[data-testid="stTab"] button{color:#85B7EB!important;font-size:.83rem}
[data-testid="stTab"] button[aria-selected="true"]{color:#3B8BD4!important;border-bottom:2px solid #1D9E75!important}
.stButton>button{background:linear-gradient(90deg,#1A3A6C,#3B2E8A);color:#fff;border:none;
  border-radius:8px;padding:9px 24px;font-weight:700;transition:all .15s}
.stButton>button:hover{opacity:.85;transform:translateY(-1px)}
.stSelectbox label,.stSlider label,.stNumberInput label{color:#85B7EB!important;font-size:.81rem}
.ibox{background:#0A1E35;border:1px solid #1D4570;border-radius:8px;padding:11px 15px;margin:7px 0;color:#85B7EB;font-size:.84rem}
.wbox{background:#1A1200;border:1px solid #9A7000;border-radius:8px;padding:11px 15px;margin:7px 0;color:#C8A800;font-size:.84rem}
</style>
""", unsafe_allow_html=True)

# ── Logo ──────────────────────────────────────────────
LOGO_INLINE = """<svg viewBox="0 0 340 170" xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">
  <defs>
    <linearGradient id="bGa" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#3B8BD4"/><stop offset="100%" stop-color="#534AB7"/>
    </linearGradient>
    <linearGradient id="aGa" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0%" stop-color="#1D9E75"/><stop offset="100%" stop-color="#3B8BD4"/>
    </linearGradient>
  </defs>
  <rect width="340" height="170" rx="14" fill="#0D1B2A" stroke="#534AB7" stroke-width=".8"/>
  <polygon points="170,18 191,30 191,54 170,66 149,54 149,30" fill="none" stroke="url(#bGa)" stroke-width="1.5"/>
  <polygon points="170,24 185,33 185,51 170,60 155,51 155,33" fill="#1a2a45"/>
  <line x1="170" y1="30" x2="162" y2="54" stroke="url(#bGa)" stroke-width="2" stroke-linecap="round"/>
  <line x1="170" y1="30" x2="178" y2="54" stroke="url(#bGa)" stroke-width="2" stroke-linecap="round"/>
  <line x1="164" y1="46" x2="176" y2="46" stroke="#1D9E75" stroke-width="1.5" stroke-linecap="round"/>
  <circle cx="170" cy="12" r="3.5" fill="#378ADD"/><circle cx="196" cy="25" r="2.8" fill="#534AB7"/>
  <circle cx="196" cy="55" r="2.8" fill="#1D9E75"/><circle cx="170" cy="73" r="3.5" fill="#378ADD"/>
  <circle cx="144" cy="55" r="2.8" fill="#534AB7"/><circle cx="144" cy="25" r="2.8" fill="#1D9E75"/>
  <circle cx="170" cy="42" r="50" fill="none" stroke="#3B8BD4" stroke-width=".6" stroke-dasharray="3 5" opacity=".3"/>
  <text x="170" y="104" text-anchor="middle" font-family="Georgia,serif" font-size="28"
        font-weight="700" letter-spacing="7" fill="url(#bGa)">AXORA</text>
  <rect x="72" y="110" width="196" height="1.5" rx="1" fill="url(#aGa)" opacity=".9"/>
  <text x="170" y="124" text-anchor="middle" font-family="Courier New,monospace"
        font-size="7" letter-spacing="2.5" fill="#85B7EB" opacity=".85">INTELLIGENCE · ILLUMINATED</text>
  <rect x="62" y="134" width="216" height="22" rx="11" fill="#0F1E33" stroke="#378ADD" stroke-width=".7"/>
  <text x="170" y="149" text-anchor="middle" font-family="Courier New,monospace"
        font-size="6.5" letter-spacing="1.2" fill="#5DCAA5">DATA ANALYSIS · ARTIFICIAL INTELLIGENCE</text>
</svg>"""

def _logo(w=200,h=124):
    paths=["axora_team_logo.svg",
           "/mnt/user-data/uploads/axora_team_logo.svg",
           "/mount/src/body-performance-analytics-and-intelligent/axora_team_logo.svg"]
    for p in paths:
        if os.path.exists(p):
            with open(p,"r",encoding="utf-8") as f: svg=f.read()
            svg=svg.replace('width="100%"',f'width="{w}"')
            return svg
    return LOGO_INLINE.format(w=w,h=h)

# ── Colours ───────────────────────────────────────────
DARK="#0B1621";PANEL="#0F1E33";BLUE="#3B8BD4"
PURPLE="#534AB7";GREEN="#1D9E75";RED="#E24B4A";LBLUE="#85B7EB"
PAL=[BLUE,PURPLE,GREEN,RED]

def _style(fig):
    fig.patch.set_facecolor(DARK)
    for ax in fig.get_axes():
        ax.set_facecolor(PANEL); ax.tick_params(colors=LBLUE,labelsize=8)
        ax.xaxis.label.set_color(LBLUE); ax.yaxis.label.set_color(LBLUE)
        ax.title.set_color(BLUE)
        for sp in ax.spines.values(): sp.set_edgecolor("#162840")
    return fig

def show(fig): st.pyplot(_style(fig),use_container_width=True); plt.close(fig)

def mbox(d):
    cols=st.columns(len(d))
    for col,(lbl,val) in zip(cols,d.items()):
        col.markdown(f'<div class="mt"><div class="mv">{val}</div>'
                     f'<div class="ml">{lbl}</div></div>',unsafe_allow_html=True)

def conf_map(cm,classes,title="Confusion Matrix",cmap="Blues"):
    fig,ax=plt.subplots(figsize=(5,4))
    sns.heatmap(cm,annot=True,fmt="d",cmap=cmap,xticklabels=classes,
                yticklabels=classes,ax=ax,linewidths=.4,cbar_kws={"shrink":.75},
                annot_kws={"size":11,"weight":"bold"})
    ax.set_title(title,pad=10); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); show(fig)

def bar_chart(labels,vals,ylabel,title,color=BLUE,hline=None):
    if isinstance(color,list): cols=color
    else: cols=[color]*len(vals)
    fig,ax=plt.subplots(figsize=(max(5,len(labels)*1.8),4))
    bars=ax.bar(labels,vals,color=cols,edgecolor=DARK,width=.52,zorder=3)
    for b in bars:
        v=b.get_height()
        ax.text(b.get_x()+b.get_width()/2,v+max(vals)*.015,
                f"{v:.3f}" if max(vals)<5 else f"{v:.1f}",
                ha="center",va="bottom",color="#C8D8EA",fontsize=9,fontweight="bold")
    if hline:
        ax.axhline(hline[0],color=hline[1],lw=1.5,linestyle="--",label=hline[2])
        ax.legend(labelcolor="#C8D8EA")
    ax.set_ylabel(ylabel); ax.set_title(title)
    ax.set_ylim(0,max(vals)*1.3); ax.grid(axis="y",alpha=.15,color=LBLUE); show(fig)

def scatter_res(yte,yp,title):
    fig,axes=plt.subplots(1,2,figsize=(13,4.5))
    axes[0].scatter(yte,yp,alpha=.35,color=BLUE,s=12)
    mn,mx=min(yte.min(),yp.min()),max(yte.max(),yp.max())
    axes[0].plot([mn,mx],[mn,mx],"--",color=RED,lw=2,label="Perfect fit")
    axes[0].set_xlabel("Actual"); axes[0].set_ylabel("Predicted")
    axes[0].set_title(title); axes[0].legend(labelcolor="#C8D8EA")
    res=yte-yp
    axes[1].hist(res,bins=35,color=GREEN,edgecolor=DARK,alpha=.85)
    axes[1].axvline(0,color=RED,lw=2,linestyle="--")
    axes[1].set_xlabel("Residual"); axes[1].set_title("Residual Distribution"); show(fig)

def fold_bars(scores,label,scale=100,color=PURPLE):
    pct=scores*scale; fig,ax=plt.subplots(figsize=(9,3.8))
    bars=ax.bar([f"Fold {i+1}" for i in range(len(pct))],pct,color=color,edgecolor=DARK,zorder=3)
    ax.axhline(pct.mean(),color=GREEN,lw=2,linestyle="--",label=f"Mean {pct.mean():.2f}")
    ax.set_ylabel(label); ax.set_title(f"5-Fold CV — {label}")
    ax.grid(axis="y",alpha=.2); ax.legend(labelcolor="#C8D8EA")
    for b in bars:
        ax.text(b.get_x()+b.get_width()/2,b.get_height()+pct.max()*.006,
                f"{b.get_height():.2f}",ha="center",color="#C8D8EA",fontsize=8)
    show(fig)

def per_class_bars(rpt,classes):
    x=np.arange(len(classes)); w=.26; fig,ax=plt.subplots(figsize=(9,4))
    for i,(m,c) in enumerate(zip(["precision","recall","f1-score"],[BLUE,GREEN,PURPLE])):
        ax.bar(x+i*w,[rpt[cl][m] for cl in classes],w,label=m.capitalize(),color=c,edgecolor=DARK,zorder=3)
    ax.set_xticks(x+w); ax.set_xticklabels([f"Class {c}" for c in classes])
    ax.set_ylim(0,1.12); ax.set_ylabel("Score"); ax.set_title("Per-Class Metrics")
    ax.legend(labelcolor="#C8D8EA"); ax.grid(axis="y",alpha=.2); show(fig)

SPLITS=[("80:20",.20),("70:30",.30),("50:50",.50)]

def run_cls(fn,X,y,scale=True):
    rows=[]
    for nm,ts in SPLITS:
        Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=ts,random_state=42,stratify=y)
        if scale:
            sc=StandardScaler(); Xtr=sc.fit_transform(Xtr); Xte=sc.transform(Xte)
        m=fn(); m.fit(Xtr,ytr); yp=m.predict(Xte)
        rows.append({"Split":nm,
            "Accuracy":f"{accuracy_score(yte,yp)*100:.2f}%",
            "Precision":f"{precision_score(yte,yp,average='macro',zero_division=0)*100:.2f}%",
            "Recall":f"{recall_score(yte,yp,average='macro',zero_division=0)*100:.2f}%",
            "F1":f"{f1_score(yte,yp,average='macro',zero_division=0)*100:.2f}%"})
    Xsc=StandardScaler().fit_transform(X) if scale else X
    cv=cross_val_score(fn(),Xsc,y,cv=StratifiedKFold(5,shuffle=True,random_state=42),scoring="accuracy")
    rows.append({"Split":"5-Fold CV","Accuracy":f"{cv.mean()*100:.2f}% ±{cv.std()*100:.2f}",
                 "Precision":"—","Recall":"—","F1":"—"})
    return pd.DataFrame(rows),cv

def run_reg(fn,X,y,scale=True):
    rows=[]
    for nm,ts in SPLITS:
        Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=ts,random_state=42)
        if scale:
            sc=StandardScaler(); Xtr=sc.fit_transform(Xtr); Xte=sc.transform(Xte)
        m=fn(); m.fit(Xtr,ytr); yp=m.predict(Xte)
        rows.append({"Split":nm,"R²":f"{r2_score(yte,yp):.4f}",
            "RMSE":f"{np.sqrt(mean_squared_error(yte,yp)):.3f}",
            "MAE":f"{mean_absolute_error(yte,yp):.3f}"})
    Xsc=StandardScaler().fit_transform(X) if scale else X
    cv=cross_val_score(fn(),Xsc,y,cv=KFold(5,shuffle=True,random_state=42),scoring="r2")
    rows.append({"Split":"5-Fold CV","R²":f"{cv.mean():.4f} ±{cv.std():.4f}","RMSE":"—","MAE":"—"})
    return pd.DataFrame(rows),cv

@st.cache_data
def load(up=None):
    if up: return pd.read_csv(up)
    for p in ["bodyPerformance.csv","bodyPerformance_-_Copy.csv",
              "/mount/src/body-performance-analytics-and-intelligent/bodyPerformance.csv"]:
        try: return pd.read_csv(p)
        except: pass
    np.random.seed(42); n=3000
    gnd=np.random.choice(["M","F"],n)
    ht=np.where(gnd=="M",np.random.normal(172,6,n),np.random.normal(160,6,n))
    wt=np.where(gnd=="M",np.random.normal(72,12,n),np.random.normal(58,9,n))
    bf=np.where(gnd=="M",np.random.normal(18,5,n),np.random.normal(28,6,n)).clip(3,55)
    gr=np.where(gnd=="M",np.random.normal(45,9,n),np.random.normal(27,7,n)).clip(5,75)
    su=np.random.normal(42,14,n).clip(0,80); age=np.random.uniform(21,64,n)
    bj=(gr*2.2+su*1.8+ht*.6-age*1.1-bf*1.5+np.random.normal(0,18,n)).clip(50,310)
    cls=pd.cut(bj,bins=[0,145,180,220,400],labels=["D","C","B","A"]).astype(str)
    return pd.DataFrame({"age":age,"gender":gnd,"height_cm":ht.clip(130,200),
        "weight_kg":wt.clip(30,140),"body fat_%":bf,
        "diastolic":np.random.normal(79,11,n).clip(50,120),
        "systolic":np.random.normal(130,15,n).clip(80,180),"gripForce":gr,
        "sit and bend forward_cm":np.random.normal(15,8,n).clip(-20,50),
        "sit-ups counts":su,"broad jump_cm":bj,"class":cls})

@st.cache_data
def prep(df):
    d=df.copy(); d=d.drop_duplicates()
    d=d[(d["systolic"]>40)&(d["diastolic"]>40)]
    d["sit and bend forward_cm"]=d["sit and bend forward_cm"].clip(-20,50)
    d["BMI"]=d["weight_kg"]/((d["height_cm"]/100)**2)
    d["gender_enc"]=d["gender"].map({"M":0,"F":1})
    le=LabelEncoder(); d["class_enc"]=le.fit_transform(d["class"]); return d,le

# ── Sidebar ───────────────────────────────────────────
with st.sidebar:
    st.markdown(_logo(200,124),unsafe_allow_html=True)
    st.markdown("<br>",unsafe_allow_html=True)
    st.markdown('<div style="color:#3B8BD4;font-size:.72rem;font-weight:700;'
                'letter-spacing:1.5px;text-transform:uppercase;margin-bottom:4px">'
                'Navigation</div>',unsafe_allow_html=True)
    page=st.radio("",["🏠 Overview","📊 EDA","🤖 KNN","📈 Linear Regression",
                       "🌳 Decision Tree","⚙️ SVM","🧠 Neural Network",
                       "🏆 Model Comparison","🔮 Live Predictor"],
                  label_visibility="collapsed")
    st.markdown('<hr style="border:none;border-top:1px solid #162840;margin:14px 0">',
                unsafe_allow_html=True)
    up=st.file_uploader("Upload CSV",type=["csv"],label_visibility="collapsed")
    st.markdown('<div style="color:#2E4A6A;font-size:.68rem;line-height:1.8;margin-top:10px">'
                'Alaa Issawi<br>Amira Salama<br>Aya Abdel Maksoud<br>'
                'Aya El-Sabi<br>Aya Imam<br>Aya Khalil</div>',unsafe_allow_html=True)

raw=load(up); df,le=prep(raw)
CL=["A","B","C","D"]
NUM=["age","height_cm","weight_kg","body fat_%","diastolic","systolic",
     "gripForce","sit and bend forward_cm","sit-ups counts","broad jump_cm"]
FC=["age","gender_enc","height_cm","weight_kg","body fat_%","diastolic","systolic",
    "gripForce","sit and bend forward_cm","sit-ups counts","broad jump_cm","BMI"]
FR=["age","gender_enc","height_cm","weight_kg","body fat_%","diastolic","systolic",
    "gripForce","sit and bend forward_cm","sit-ups counts","BMI"]
Xc=df[FC].values; yc=df["class_enc"].values
Xr=df[FR].values; yr=df["broad jump_cm"].values

# ══════════════════════════════════════════════════════
if "Overview" in page:
    c1,c2,c3=st.columns([1,2,1])
    with c2:
        st.markdown(_logo(380,235),unsafe_allow_html=True)
        st.markdown("""<div style="text-align:center;margin-top:16px">
        <div style="font-size:1.65rem;font-weight:900;color:#3B8BD4">Body Performance Analytics</div>
        <div style="font-size:1rem;color:#85B7EB;margin-top:4px">Intelligent Classification &amp; Regression System</div>
        <div style="color:#4d7fa8;font-size:.8rem;margin-top:6px">5 Machine Learning Models · Team Axora · 2024–2025</div>
        </div>""",unsafe_allow_html=True)
    st.markdown("<br>",unsafe_allow_html=True)
    mbox({"Records":f"{len(df):,}","Features":len(FC),"Classes":"A · B · C · D","Models":"5"})
    st.markdown("<br>",unsafe_allow_html=True)
    c1,c2=st.columns([3,2])
    with c1:
        st.markdown('<div class="card cb">',unsafe_allow_html=True)
        st.subheader("📋 Dataset Sample"); st.dataframe(raw.head(8),use_container_width=True)
        st.markdown("</div>",unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card">',unsafe_allow_html=True)
        st.subheader("🔧 Preprocessing")
        st.markdown("- Removed 1 duplicate → 13,392 rows\n- Removed BP ≤ 40 mmHg\n"
                    "- Capped sit-bend [−20, 50] cm\n- Engineered BMI feature\n"
                    "- Encoded gender & class\n- StandardScaler for KNN/SVM/MLP")
        st.markdown("</div>",unsafe_allow_html=True)
    st.markdown('<div class="card cg">',unsafe_allow_html=True)
    st.subheader("📊 Class Distribution")
    vc=raw["class"].value_counts().sort_index()
    c1,c2,c3=st.columns(3)
    with c1:
        fig,ax=plt.subplots(figsize=(5,3.5))
        bars=ax.bar(vc.index,vc.values,color=PAL,edgecolor=DARK,width=.55)
        for b in bars:
            ax.text(b.get_x()+b.get_width()/2,b.get_height()+20,
                    str(int(b.get_height())),ha="center",color="#C8D8EA",fontsize=9)
        ax.set_title("Class Frequency"); show(fig)
    with c2:
        fig,ax=plt.subplots(figsize=(4.5,4.5),subplot_kw=dict(aspect="equal"))
        _,_,at=ax.pie(vc.values,labels=vc.index,autopct="%1.1f%%",colors=PAL,
                      startangle=90,wedgeprops=dict(edgecolor=DARK,lw=1.5))
        for t in at: t.set_color("white")
        ax.set_title("Class Balance"); show(fig)
    with c3:
        st.dataframe(df[["age","weight_kg","gripForce","sit-ups counts",
                          "broad jump_cm"]].describe().round(1),use_container_width=True)
    st.markdown("</div>",unsafe_allow_html=True)

elif "EDA" in page:
    st.title("📊 Exploratory Data Analysis")
    t1,t2,t3,t4,t5=st.tabs(["📈 Distributions","📦 Boxplots","🔗 Correlations","🎯 By Class","💡 Insights"])
    with t1:
        fig,axes=plt.subplots(2,5,figsize=(20,8)); axes=axes.flatten()
        for i,col in enumerate(NUM):
            axes[i].hist(raw[col].dropna(),bins=35,color=PAL[i%4],alpha=.85,edgecolor=DARK)
            axes[i].set_title(col,fontsize=8)
        plt.tight_layout(pad=1.5); show(fig)
    with t2:
        scv=StandardScaler()
        sc_df=pd.DataFrame(scv.fit_transform(raw[NUM].dropna()),columns=NUM)
        fig,ax=plt.subplots(figsize=(16,5))
        sc_df.boxplot(ax=ax,patch_artist=True,boxprops=dict(facecolor="#1D4570",color=BLUE),
            medianprops=dict(color=GREEN,lw=2.5),whiskerprops=dict(color=LBLUE),
            capprops=dict(color=LBLUE),flierprops=dict(marker="o",color=RED,alpha=.4,markersize=3))
        ax.set_title("Standardised Boxplots"); plt.xticks(rotation=35,ha="right",fontsize=8); show(fig)
    with t3:
        enc2=raw.copy(); enc2["gender"]=enc2["gender"].map({"M":0,"F":1})
        enc2["class_n"]=LabelEncoder().fit_transform(enc2["class"])
        corr=enc2[NUM+["class_n"]].corr()
        fig,ax=plt.subplots(figsize=(13,10))
        sns.heatmap(corr,annot=True,fmt=".2f",cmap="coolwarm",center=0,
                    linewidths=.3,ax=ax,annot_kws={"size":7},cbar_kws={"shrink":.7})
        ax.set_title("Correlation Matrix"); show(fig)
        cr=corr["class_n"].drop("class_n").sort_values()
        fig2,ax2=plt.subplots(figsize=(10,4))
        ax2.barh(cr.index,cr.values,color=[RED if v<0 else GREEN for v in cr.values],edgecolor=DARK)
        ax2.axvline(0,color=LBLUE,lw=1); ax2.set_title("Correlation with Performance Class"); show(fig2)
    with t4:
        sel=st.selectbox("Feature",NUM,key="eda_s")
        fig,ax=plt.subplots(1,2,figsize=(14,5))
        for cls_,col_ in zip(CL,PAL):
            ax[0].hist(raw[raw["class"]==cls_][sel],bins=25,alpha=.55,label=f"Class {cls_}",color=col_)
        ax[0].set_title(f"{sel} by Class"); ax[0].legend(labelcolor="#C8D8EA")
        bp=ax[1].boxplot([raw[raw["class"]==c][sel].dropna() for c in CL],labels=CL,patch_artist=True)
        for patch,c in zip(bp["boxes"],PAL): patch.set_facecolor(c); patch.set_alpha(.7)
        for md in bp["medians"]: md.set_color("white"); md.set_lw(2)
        ax[1].set_title(f"{sel} — Boxplot by Class"); show(fig)
        st.dataframe(raw.groupby("class")[NUM].mean().round(2),use_container_width=True)
    with t5:
        for ttl,body,cls in [
            ("✅ Balanced Dataset","~25% per class — no oversampling needed.","cg"),
            ("📐 Strongest Predictors","sit_and_bend (r=−0.61), sit_ups (r=−0.45), broad_jump (r=−0.26).","cg"),
            ("⚠️ Weak Predictors","Diastolic & systolic BP: r<0.07 — minimal discriminative power.","cp"),
            ("👥 Gender Effect","Height & grip force bimodal (M/F). Males jump ~40–50 cm further.","cb"),
            ("📅 Age","Right-skewed (0.60). Older → lower fitness class.","cr")]:
            st.markdown(f'<div class="card {cls}"><b style="color:#3B8BD4">{ttl}</b>'
                        f'<br><span style="font-size:.87rem">{body}</span></div>',unsafe_allow_html=True)

elif "KNN" in page:
    st.title("🤖 K-Nearest Neighbors")
    st.markdown('<div class="card cg">Classifies by majority vote among <b>k nearest neighbors</b> '
                'in scaled space. Best k=22 (classification) → <b>63.09%</b>. '
                'Best k=37 (regression) → <b>R²=0.7858</b>.</div>',unsafe_allow_html=True)
    st.markdown('<div class="card cb">',unsafe_allow_html=True)
    st.subheader("📒 Notebook Results (Verified)")
    mbox({"Best k (Cls)":"22","Accuracy (80:20)":"63.09%","Best CV k":"24 (61.83%)",
          "Best k (Reg)":"37","R² (80:20)":"0.7858"})
    c1,c2=st.columns(2)
    with c1:
        st.markdown("**Classification**")
        st.dataframe(pd.DataFrame([
            {"Split":"80:20","K":22,"Accuracy":"63.09%","F1":"0.632"},
            {"Split":"70:30","K":15,"Accuracy":"62.40%","F1":"0.625"},
            {"Split":"50:50","K":23,"Accuracy":"61.78%","F1":"0.619"},
            {"Split":"5-Fold CV","K":24,"Accuracy":"61.83% ±0.35%","F1":"—"},
        ]),use_container_width=True,hide_index=True)
    with c2:
        st.markdown("**Regression**")
        st.dataframe(pd.DataFrame([
            {"Split":"80:20","K":37,"R²":"0.7858","MAE":"13.79 cm"},
            {"Split":"70:30","K":36,"R²":"0.7879","MAE":"13.84 cm"},
            {"Split":"50:50","K":28,"R²":"0.7849","MAE":"13.95 cm"},
            {"Split":"5-Fold CV","K":35,"R²":"0.782 ±0.014","MAE":"—"},
        ]),use_container_width=True,hide_index=True)
    st.markdown("</div>",unsafe_allow_html=True)
    tc,tr=st.tabs(["🎯 Run Classification","📈 Run Regression"])
    with tc:
        c1,c2=st.columns(2)
        kmax=c1.slider("Max k",5,60,35,key="kk"); spl=c2.selectbox("Split",["80:20","70:30","50:50"],key="ks")
        if st.button("▶ Run KNN Classification",use_container_width=True,key="rk"):
            ts={"80:20":.2,"70:30":.3,"50:50":.5}[spl]
            with st.spinner("Sweeping k…"):
                Xtr,Xte,ytr,yte=train_test_split(Xc,yc,test_size=ts,random_state=42,stratify=yc)
                sc=StandardScaler(); Xtr=sc.fit_transform(Xtr); Xte=sc.transform(Xte)
                kr=list(range(1,kmax+1))
                accs=[accuracy_score(yte,KNeighborsClassifier(k,n_jobs=-1).fit(Xtr,ytr).predict(Xte)) for k in kr]
                bk=kr[int(np.argmax(accs))]; ba=max(accs)
            mbox({"Best k":bk,"Accuracy":f"{ba*100:.2f}%"})
            fig,ax=plt.subplots(figsize=(11,4))
            ax.plot(kr,[a*100 for a in accs],color=BLUE,lw=2)
            ax.fill_between(kr,[a*100 for a in accs],alpha=.1,color=BLUE)
            ax.axvline(bk,color=GREEN,lw=1.8,linestyle="--",label=f"k={bk} → {ba*100:.1f}%")
            ax.set_xlabel("k"); ax.set_ylabel("Accuracy (%)"); ax.set_title("k Sweep")
            ax.legend(labelcolor="#C8D8EA"); ax.grid(alpha=.15); show(fig)
            bm=KNeighborsClassifier(bk,n_jobs=-1).fit(Xtr,ytr); yp=bm.predict(Xte)
            rpt=classification_report(yte,yp,target_names=CL,output_dict=True)
            c1,c2=st.columns(2)
            with c1: conf_map(confusion_matrix(yte,yp),CL,f"Confusion Matrix k={bk}")
            with c2: per_class_bars(rpt,CL)
            st.dataframe(pd.DataFrame(rpt).T.round(3),use_container_width=True)
            Xsc=StandardScaler().fit_transform(Xc)
            cv5=cross_val_score(KNeighborsClassifier(bk,n_jobs=-1),Xsc,yc,
                cv=StratifiedKFold(5,shuffle=True,random_state=42),scoring="accuracy")
            mbox({"CV Mean":f"{cv5.mean()*100:.2f}%","CV Std":f"±{cv5.std()*100:.2f}%"})
            fold_bars(cv5,"Accuracy (%)",100,PURPLE)
    with tr:
        kr_=st.slider("k (regression)",1,60,37,key="kreg")
        if st.button("▶ Run KNN Regression",use_container_width=True,key="rkr"):
            res,cv5=run_reg(lambda:KNeighborsRegressor(kr_,n_jobs=-1),Xr,yr)
            st.dataframe(res,use_container_width=True,hide_index=True)
            bar_chart(res["Split"][:3],[float(v.split(" ")[0]) for v in res["R²"][:3]],"R²","KNN R²",GREEN)
            sc=StandardScaler(); Xtr,Xte,ytr,yte=train_test_split(Xr,yr,test_size=.2,random_state=42)
            Xtr=sc.fit_transform(Xtr); Xte=sc.transform(Xte)
            m=KNeighborsRegressor(kr_,n_jobs=-1); m.fit(Xtr,ytr); yp=m.predict(Xte)
            mbox({"R²":f"{r2_score(yte,yp):.4f}","RMSE":f"{np.sqrt(mean_squared_error(yte,yp)):.2f} cm",
                  "MAE":f"{mean_absolute_error(yte,yp):.2f} cm"})
            scatter_res(yte,yp,"KNN Regression — 80:20"); fold_bars(cv5,"R²",1,GREEN)

elif "Linear Regression" in page:
    st.title("📈 Linear Regression")
    st.markdown('<div class="card cg">Predicts <b>broad_jump_cm</b> (regression). '
                'Best: 70:30 split → <b>R²=0.8033, RMSE=17.89 cm</b>. '
                '5-Fold CV R²=<b>0.787</b>. Adapted classification ~50% (task mismatch).</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="card cb">',unsafe_allow_html=True)
    st.subheader("📒 Notebook Results (Verified)")
    mbox({"Best R² (70:30)":"0.8033","RMSE":"17.89 cm","CV R²":"0.787","CV RMSE":"18.37 cm"})
    st.dataframe(pd.DataFrame([
        {"Split":"80:20","R²":"0.7999","RMSE":"18.03 cm","MSE":"324.95"},
        {"Split":"70:30","R²":"0.8033","RMSE":"17.89 cm","MSE":"319.87"},
        {"Split":"50:50","R²":"0.7992","RMSE":"17.89 cm","MSE":"319.90"},
        {"Split":"5-Fold CV","R²":"0.787 ±0.016","RMSE":"18.37 cm","MSE":"—"},
    ]),use_container_width=True,hide_index=True)
    st.markdown("</div>",unsafe_allow_html=True)
    if st.button("▶ Run Linear Regression",use_container_width=True,key="rlr"):
        res,cv5=run_reg(LinearRegression,Xr,yr,scale=False)
        st.dataframe(res,use_container_width=True,hide_index=True)
        r2v=[float(v.split(" ")[0]) for v in res["R²"][:3]]; rmv=[float(v) for v in res["RMSE"][:3]]
        c1,c2=st.columns(2)
        with c1: bar_chart(res["Split"][:3],r2v,"R²","R² Across Splits",GREEN)
        with c2: bar_chart(res["Split"][:3],rmv,"RMSE (cm)","RMSE Across Splits",RED)
        Xtr,Xte,ytr,yte=train_test_split(Xr,yr,test_size=.2,random_state=42)
        m=LinearRegression(); m.fit(Xtr,ytr); yp=m.predict(Xte)
        mbox({"R²":f"{r2_score(yte,yp):.4f}","RMSE":f"{np.sqrt(mean_squared_error(yte,yp)):.2f} cm",
              "MAE":f"{mean_absolute_error(yte,yp):.2f} cm"})
        scatter_res(yte,yp,"Linear Regression — 80:20")
        m2=LinearRegression(); m2.fit(Xr,yr)
        cd=pd.DataFrame({"Feature":FR,"Coefficient":m2.coef_}).sort_values("Coefficient",key=abs,ascending=False)
        fig,ax=plt.subplots(figsize=(11,4.5))
        ax.barh(cd["Feature"],cd["Coefficient"],
                color=[GREEN if v>=0 else RED for v in cd["Coefficient"]],edgecolor=DARK)
        ax.axvline(0,color=LBLUE,lw=1); ax.set_title("Feature Coefficients"); show(fig)
        mbox({"CV Mean R²":f"{cv5.mean():.4f}","CV Std":f"±{cv5.std():.4f}"})
        fold_bars(cv5,"R²",1,GREEN)

elif "Decision Tree" in page:
    st.title("🌳 Decision Tree")
    st.markdown('<div class="card cg">Cost-complexity pruning (α) prevents overfitting. '
                '<b>Classification:</b> α=0.000377 → depth 14, 477 nodes → <b>71.07%</b>. '
                '<b>Regression:</b> α=16.21 → depth 3 → <b>R²=0.677</b>.</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="card cb">',unsafe_allow_html=True)
    st.subheader("📒 Notebook Results (Verified)")
    mbox({"Best Cls Acc":"71.07%","CV Cls Acc":"70.39%","Best Reg R²":"0.677","CV Reg R²":"0.695"})
    st.markdown("**Combined Classification & Regression Results**")
    st.dataframe(pd.DataFrame([
        {"Split":"80:20","Cls Acc":"71.07%","Cls Train Acc":"78.04%","Precision":"0.71","Recall":"0.71","F1":"0.71","Reg R²":"0.6770","RMSE":"22.82 cm"},
        {"Split":"70:30","Cls Acc":"70.58%","Cls Train Acc":"79.08%","Precision":"—","Recall":"—","F1":"—","Reg R²":"0.6699","RMSE":"23.09 cm"},
        {"Split":"50:50","Cls Acc":"67.06%","Cls Train Acc":"85.96%","Precision":"—","Recall":"—","F1":"—","Reg R²":"0.6852","RMSE":"22.47 cm"},
        {"Split":"5-Fold CV","Cls Acc":"70.39% ±1.39%","Cls Train Acc":"—","Precision":"—","Recall":"—","F1":"—","Reg R²":"0.6947 ±0.015","RMSE":"—"},
    ]),use_container_width=True,hide_index=True)
    st.markdown("</div>",unsafe_allow_html=True)
    tc,tr=st.tabs(["🎯 Run Classification","📈 Run Regression"])
    with tc:
        c1,c2=st.columns(2)
        alpha=c1.slider("ccp_alpha",0.0,0.008,0.000377,step=5e-5,format="%.6f",key="dta")
        md=c2.slider("Max depth (0=auto)",0,30,0,key="dtd"); mxd=None if md==0 else md
        if st.button("▶ Run Decision Tree Classification",use_container_width=True,key="rdt"):
            Xtr,Xte,ytr,yte=train_test_split(Xc,yc,test_size=.2,random_state=42,stratify=yc)
            m=DecisionTreeClassifier(random_state=42,ccp_alpha=alpha,max_depth=mxd)
            m.fit(Xtr,ytr); yp=m.predict(Xte)
            mbox({"Accuracy":f"{accuracy_score(yte,yp)*100:.2f}%","Depth":m.get_depth(),"Nodes":m.tree_.node_count})
            alsw=np.linspace(0,.006,50); tra,tea=[],[]
            for a in alsw:
                mm=DecisionTreeClassifier(random_state=42,ccp_alpha=a)
                mm.fit(Xtr,ytr); tra.append(mm.score(Xtr,ytr)); tea.append(mm.score(Xte,yte))
            fig,ax=plt.subplots(figsize=(11,4))
            ax.plot(alsw,[a*100 for a in tra],color=BLUE,lw=2,label="Train")
            ax.plot(alsw,[a*100 for a in tea],color=GREEN,lw=2,label="Test")
            ax.axvline(alpha,color=RED,lw=1.5,linestyle="--",label=f"α={alpha:.5f}")
            ax.set_xlabel("ccp_alpha"); ax.set_ylabel("Accuracy (%)"); ax.set_title("Alpha Sweep")
            ax.legend(labelcolor="#C8D8EA"); ax.grid(alpha=.15); show(fig)
            rpt=classification_report(yte,yp,target_names=CL,output_dict=True)
            c1,c2=st.columns(2)
            with c1: conf_map(confusion_matrix(yte,yp),CL,"Confusion Matrix")
            with c2: per_class_bars(rpt,CL)
            st.dataframe(pd.DataFrame(rpt).T.round(3),use_container_width=True)
            res,cv5=run_cls(lambda:DecisionTreeClassifier(random_state=42,ccp_alpha=alpha,max_depth=mxd),
                            Xc,yc,scale=False)
            st.dataframe(res,use_container_width=True,hide_index=True)
            fold_bars(cv5,"Accuracy (%)",100,PURPLE)
            st.subheader("Tree Structure (top 3 levels)")
            fig2,ax2=plt.subplots(figsize=(20,8))
            plot_tree(m,max_depth=3,feature_names=FC,class_names=CL,
                      filled=True,rounded=True,precision=2,ax=ax2,fontsize=8,impurity=False); show(fig2)
    with tr:
        c1,c2=st.columns(2)
        dta=c1.slider("ccp_alpha (reg)",0.0,50.0,0.0,key="dtrar")
        dtd=c2.slider("Max depth",0,20,0,key="dtrdr"); md2=None if dtd==0 else dtd
        if st.button("▶ Run DT Regression",use_container_width=True,key="rdtr"):
            res,cv5=run_reg(lambda:DecisionTreeRegressor(random_state=42,ccp_alpha=dta,max_depth=md2),Xr,yr,scale=False)
            st.dataframe(res,use_container_width=True,hide_index=True)
            bar_chart(res["Split"][:3],[float(v.split(" ")[0]) for v in res["R²"][:3]],"R²","DT Regression R²",GREEN)
            Xtr,Xte,ytr,yte=train_test_split(Xr,yr,test_size=.2,random_state=42)
            m2=DecisionTreeRegressor(random_state=42,ccp_alpha=dta,max_depth=md2)
            m2.fit(Xtr,ytr); scatter_res(yte,m2.predict(Xte),"DT Regression — 80:20"); fold_bars(cv5,"R²",1,GREEN)

elif "SVM" in page:
    st.title("⚙️ Support Vector Machine")
    st.markdown('<div class="card cg">Maximum-margin classifier. '
                'RBF kernel: C=10, γ=0.1 → <b>71.27%</b> on full dataset. '
                'SVR regression: R²≈<b>0.79</b>.</div>',unsafe_allow_html=True)
    st.markdown('<div class="card cb">',unsafe_allow_html=True)
    st.subheader("📒 Notebook Results (Verified)")
    mbox({"Linear (full)":"63.13%","RBF Tuned (full)":"71.31%","Best Params":"C=100, γ=0.01","Kernel":"RBF"})
    st.markdown("**Combined Classification & Regression Results (RBF Kernel)**")
    st.dataframe(pd.DataFrame([
        {"Split":"80:20","Cls Acc (RBF)":"71.31%","Cls Acc (Linear)":"63.13%","Reg R² (RBF)":"0.7967","Reg RMSE":"17.99 cm","Reg MAE":"13.38 cm"},
        {"Split":"70:30","Cls Acc (RBF)":"71.07%","Cls Acc (Linear)":"63.02%","Reg R² (RBF)":"0.7980","Reg RMSE":"17.97 cm","Reg MAE":"—"},
        {"Split":"50:50","Cls Acc (RBF)":"70.33%","Cls Acc (Linear)":"62.72%","Reg R² (RBF)":"0.7980","Reg RMSE":"17.96 cm","Reg MAE":"—"},
        {"Split":"5-Fold CV","Cls Acc (RBF)":"67.19%","Cls Acc (Linear)":"62.24%","Reg R² (RBF)":"0.7905 ±0.015","Reg RMSE":"18.23 cm","Reg MAE":"—"},
    ]),use_container_width=True,hide_index=True)
    st.markdown('<div class="wbox">📌 Note: CV classification is 67.19% on full dataset with grid search best params (C=100, γ=0.01). '
                'SVR RBF achieves R²≈0.797–0.798 across all splits.</div>',unsafe_allow_html=True)
    st.markdown("</div>",unsafe_allow_html=True)
    tc,tr=st.tabs(["🎯 Run Classification","📈 Run SVR"])
    with tc:
        c1,c2,c3=st.columns(3)
        kern=c1.selectbox("Kernel",["rbf","linear","poly"],key="svmk")
        Cv=c2.select_slider("C",[0.1,1,10,100],value=10,key="svmC")
        gam=c3.selectbox("Gamma",["scale","auto","0.01","0.1","1"],index=3,key="svmg")
        gv=gam if gam in ["scale","auto"] else float(gam)
        if st.button("▶ Run SVM (30–60s on full data)",use_container_width=True,key="rsvm"):
            with st.spinner("Training SVM…"):
                sc=StandardScaler()
                Xtr,Xte,ytr,yte=train_test_split(Xc,yc,test_size=.2,random_state=42,stratify=yc)
                Xtr=sc.fit_transform(Xtr); Xte=sc.transform(Xte)
                m=SVC(kernel=kern,C=Cv,gamma=gv,random_state=42); m.fit(Xtr,ytr); yp=m.predict(Xte)
            mbox({"Accuracy":f"{accuracy_score(yte,yp)*100:.2f}%",
                  "F1":f"{f1_score(yte,yp,average='macro'):.3f}","Kernel":kern.upper(),"C":Cv})
            rpt=classification_report(yte,yp,target_names=CL,output_dict=True)
            c1,c2=st.columns(2)
            with c1: conf_map(confusion_matrix(yte,yp),CL,f"Confusion Matrix ({kern})")
            with c2: per_class_bars(rpt,CL)
            st.dataframe(pd.DataFrame(rpt).T.round(3),use_container_width=True)
            res,cv5=run_cls(lambda:SVC(kernel=kern,C=Cv,gamma=gv,random_state=42),Xc,yc)
            st.dataframe(res,use_container_width=True,hide_index=True)
            bar_chart(res["Split"][:3],[float(v.split("%")[0]) for v in res["Accuracy"][:3]],
                      "Accuracy (%)","SVM Splits",PURPLE)
            fold_bars(cv5,"Accuracy (%)",100,PURPLE)
    with tr:
        c1,c2,c3=st.columns(3)
        svk=c1.selectbox("Kernel",["rbf","linear"],key="svrk")
        svC=c2.select_slider("C",[0.1,1,10,100],value=10,key="svrC")
        sve=c3.slider("Epsilon",.01,1.,.1,step=.05,key="svre")
        if st.button("▶ Run SVR",use_container_width=True,key="rsvr"):
            res,cv5=run_reg(lambda:SVR(kernel=svk,C=svC,epsilon=sve),Xr,yr)
            st.dataframe(res,use_container_width=True,hide_index=True)
            bar_chart(res["Split"][:3],[float(v.split(" ")[0]) for v in res["R²"][:3]],"R²","SVR R²",PURPLE)
            sc=StandardScaler(); Xtr,Xte,ytr,yte=train_test_split(Xr,yr,test_size=.2,random_state=42)
            Xtr=sc.fit_transform(Xtr); Xte=sc.transform(Xte)
            m=SVR(kernel=svk,C=svC,epsilon=sve); m.fit(Xtr,ytr); yp=m.predict(Xte)
            mbox({"R²":f"{r2_score(yte,yp):.4f}","RMSE":f"{np.sqrt(mean_squared_error(yte,yp)):.2f} cm",
                  "MAE":f"{mean_absolute_error(yte,yp):.2f} cm"})
            scatter_res(yte,yp,"SVR — 80:20"); fold_bars(cv5,"R²",1,PURPLE)

elif "Neural Network" in page:
    st.title("🧠 Neural Network (MLP)")
    st.markdown('<div class="card cg">Multi-Layer Perceptron — 256→128→64 units, ReLU, '
                'L2 regularisation, Early Stopping.<br>'
                '<b>Note:</b> Original notebook showed 100% (data leakage bug). '
                'Corrected sklearn MLP gives realistic <b>~74.76–74.80%</b>.</div>',unsafe_allow_html=True)
    st.markdown('<div class="card cb">',unsafe_allow_html=True)
    st.subheader("📒 Architecture")
    c1,c2=st.columns(2)
    with c1:
        st.dataframe(pd.DataFrame([
            {"Layer":"Input","Units":len(FC),"Activation":"—","Reg":"—"},
            {"Layer":"Hidden 1","Units":256,"Activation":"ReLU","Reg":"L2 α=0.001"},
            {"Layer":"Hidden 2","Units":128,"Activation":"ReLU","Reg":"L2 α=0.001"},
            {"Layer":"Hidden 3","Units":64,"Activation":"ReLU","Reg":"L2 α=0.001"},
            {"Layer":"Output (Cls)","Units":4,"Activation":"Softmax","Reg":"—"},
            {"Layer":"Output (Reg)","Units":1,"Activation":"Linear","Reg":"—"},
        ]),use_container_width=True,hide_index=True)
    with c2:
        st.dataframe(pd.DataFrame([
            {"Parameter":"Solver","Value":"Adam"},
            {"Parameter":"Learning Rate","Value":"Adaptive"},
            {"Parameter":"Early Stopping","Value":"patience=15"},
            {"Parameter":"Validation Split","Value":"10%"},
            {"Parameter":"Max Iterations","Value":"500"},
            {"Parameter":"L2 Alpha","Value":"0.001"},
        ]),use_container_width=True,hide_index=True)
    st.markdown('<div class="wbox">⚠️ Original notebook: 100% accuracy from data leakage '
                '(target labels in features). Corrected below.</div>',unsafe_allow_html=True)
    st.markdown("**Combined Classification & Regression Results**")
    st.dataframe(pd.DataFrame([
        {"Split":"80:20","Cls Accuracy":"74.76%","Cls F1":"~0.74","Cls CV":"73.98% ±0.47%","Reg R²":"0.8001","Reg RMSE":"17.84 cm","Reg MAE":"13.38 cm"},
        {"Split":"70:30","Cls Accuracy":"74.80%","Cls F1":"~0.74","Cls CV":"—","Reg R²":"0.8025","Reg RMSE":"17.77 cm","Reg MAE":"13.41 cm"},
        {"Split":"50:50","Cls Accuracy":"73.33%","Cls F1":"~0.73","Cls CV":"—","Reg R²":"0.8010","Reg RMSE":"17.83 cm","Reg MAE":"13.44 cm"},
        {"Split":"5-Fold CV","Cls Accuracy":"73.98% ±0.47%","Cls F1":"—","Cls CV":"—","Reg R²":"0.7932 ±0.014","Reg RMSE":"18.11 cm","Reg MAE":"13.62 cm"},
    ]),use_container_width=True,hide_index=True)
    st.markdown("</div>",unsafe_allow_html=True)

    def mk_cls(al,it): return MLPClassifier(hidden_layer_sizes=(256,128,64),activation="relu",
        solver="adam",alpha=al,learning_rate="adaptive",max_iter=it,early_stopping=True,
        validation_fraction=.10,n_iter_no_change=15,random_state=42,verbose=False)
    def mk_reg(al,it): return MLPRegressor(hidden_layer_sizes=(256,128,64),activation="relu",
        solver="adam",alpha=al,learning_rate="adaptive",max_iter=it,early_stopping=True,
        validation_fraction=.10,n_iter_no_change=15,random_state=42,verbose=False)

    ta,tb=st.tabs(["🎯 Classification","📈 Regression"])
    with ta:
        c1,c2=st.columns(2)
        mi=c1.slider("Max Iterations",100,800,400,step=50,key="nn_i")
        al=c2.select_slider("L2 Alpha",[0.0001,0.001,0.01,0.1],value=0.001,key="nn_a")
        if st.button("▶ Train MLP Classifier",use_container_width=True,key="rnn"):
            with st.spinner("Training…"):
                res,cv5=run_cls(lambda:mk_cls(al,mi),Xc,yc)
                Xtr,Xte,ytr,yte=train_test_split(Xc,yc,test_size=.2,random_state=42,stratify=yc)
                sc=StandardScaler(); Xtr=sc.fit_transform(Xtr); Xte=sc.transform(Xte)
                bm=mk_cls(al,mi); bm.fit(Xtr,ytr); yp=bm.predict(Xte)
            st.dataframe(res,use_container_width=True,hide_index=True)
            mbox({"80:20 Acc":f"{accuracy_score(yte,yp)*100:.2f}%",
                  "CV Mean":f"{cv5.mean()*100:.2f}%","Iterations":bm.n_iter_})
            fig,ax=plt.subplots(figsize=(11,4))
            ax.plot(bm.loss_curve_,color=BLUE,lw=2,label="Train Loss")
            ax.set_xlabel("Iteration"); ax.set_ylabel("Loss"); ax.set_title("MLP Training Loss")
            ax.legend(labelcolor="#C8D8EA"); ax.grid(alpha=.15); show(fig)
            rpt=classification_report(yte,yp,target_names=CL,output_dict=True)
            c1,c2=st.columns(2)
            with c1: conf_map(confusion_matrix(yte,yp),CL,"Confusion Matrix (80:20)")
            with c2: per_class_bars(rpt,CL)
            st.dataframe(pd.DataFrame(rpt).T.round(3),use_container_width=True)
            fold_bars(cv5,"Accuracy (%)",100,BLUE)
    with tb:
        c1,c2=st.columns(2)
        mi_r=c1.slider("Max Iterations",100,800,400,step=50,key="nn_ir")
        al_r=c2.select_slider("L2 Alpha",[0.0001,0.001,0.01,0.1],value=0.001,key="nn_ar")
        if st.button("▶ Train MLP Regressor",use_container_width=True,key="rnnr"):
            res,cv5=run_reg(lambda:mk_reg(al_r,mi_r),Xr,yr)
            st.dataframe(res,use_container_width=True,hide_index=True)
            bar_chart(res["Split"][:3],[float(v.split(" ")[0]) for v in res["R²"][:3]],"R²","MLP R²",BLUE)
            sc=StandardScaler(); Xtr,Xte,ytr,yte=train_test_split(Xr,yr,test_size=.2,random_state=42)
            Xtr=sc.fit_transform(Xtr); Xte=sc.transform(Xte)
            mr=mk_reg(al_r,mi_r); mr.fit(Xtr,ytr); yp=mr.predict(Xte)
            mbox({"R²":f"{r2_score(yte,yp):.4f}","RMSE":f"{np.sqrt(mean_squared_error(yte,yp)):.2f} cm",
                  "MAE":f"{mean_absolute_error(yte,yp):.2f} cm"})
            scatter_res(yte,yp,"MLP Regression — 80:20"); fold_bars(cv5,"R²",1,BLUE)

elif "Comparison" in page:
    st.title("🏆 Model Comparison")
    st.markdown('<div class="card cg">',unsafe_allow_html=True)
    st.subheader("🎯 Classification — Verified Results")
    st.dataframe(pd.DataFrame([
        {"Model":"KNN (k=22)","Best Acc":"63.09%","F1":"0.632","Split":"80:20","CV":"61.83%","Rank":"5th"},
        {"Model":"Decision Tree (α=0.000377)","Best Acc":"71.07%","F1":"0.71","Split":"80:20","CV":"70.39%","Rank":"3rd ✅"},
        {"Model":"SVM (RBF, C=100, γ=0.01)","Best Acc":"71.31%","F1":"0.713","Split":"80:20","CV":"67.19%","Rank":"2nd ✅"},
        {"Model":"Neural Network (MLP)","Best Acc":"74.80%","F1":"~0.74","Split":"70:30","CV":"73.98%","Rank":"1st ⭐"},
        {"Model":"Linear Reg. (adapted)","Best Acc":"~50%","F1":"~0.48","Split":"N/A","CV":"N/A","Rank":"❌"},
    ]),use_container_width=True,hide_index=True); st.markdown("</div>",unsafe_allow_html=True)

    st.markdown('<div class="card cb">',unsafe_allow_html=True)
    st.subheader("📈 Regression — Verified Results")
    st.dataframe(pd.DataFrame([
        {"Model":"Linear Regression","Best R²":"0.8033","RMSE":"17.89 cm","CV R²":"0.787","Rank":"1st ⭐"},
        {"Model":"Neural Network (MLP)","Best R²":"0.8025","RMSE":"17.77 cm","CV R²":"0.793","Rank":"2nd ✅"},
        {"Model":"SVM (SVR, RBF)","Best R²":"0.798","RMSE":"~17.97 cm","CV R²":"~0.791","Rank":"3rd ✅"},
        {"Model":"KNN (k=37)","Best R²":"0.786","RMSE":"~18.5 cm","CV R²":"0.782","Rank":"4th"},
        {"Model":"Decision Tree","Best R²":"0.677","RMSE":"22.82 cm","CV R²":"0.695","Rank":"5th"},
    ]),use_container_width=True,hide_index=True); st.markdown("</div>",unsafe_allow_html=True)

    c1,c2=st.columns(2)
    with c1:
        bar_chart(["KNN","Dec.Tree","SVM","MLP","Lin.Reg"],
                  [63.1,72.15,71.31,74.80,50.0],"Best Accuracy (%)","Classification Accuracy",
                  color=PAL+[LBLUE],hline=(70,"#1D9E75","70% line"))
    with c2:
        bar_chart(["Lin.Reg","MLP","SVR","KNN","Dec.Tree"],
                  [.8033,.8025,.798,.786,.677],"R² Score","Regression R²",
                  color=[GREEN,BLUE,PURPLE,LBLUE,RED])

    st.markdown('<div class="card">',unsafe_allow_html=True)
    st.subheader("🕸️ Radar — Classification Metrics (%)")
    cats=["Accuracy","Precision","Recall","F1","CV Score"]
    scores_r={"KNN":[63,65,63,63,62],"Dec.Tree":[72,72,71,71,70],
              "SVM":[71,71,71,71,67],"MLP":[75,74,75,74,74]}
    N=len(cats); angles=np.linspace(0,2*np.pi,N,endpoint=False).tolist(); angles+=angles[:1]
    fig,ax=plt.subplots(figsize=(6,6),subplot_kw=dict(polar=True))
    for (nm,sc_),col in zip(scores_r.items(),PAL):
        v=sc_+sc_[:1]; ax.plot(angles,v,color=col,lw=2.2,label=nm); ax.fill(angles,v,color=col,alpha=.1)
    ax.set_thetagrids(np.degrees(angles[:-1]),cats,color=LBLUE,fontsize=10)
    ax.set_ylim(50,82); ax.set_yticks([55,65,75]); ax.set_yticklabels(["55","65","75"],fontsize=7,color=LBLUE)
    ax.set_facecolor(PANEL); ax.figure.patch.set_facecolor(DARK)
    for l in ax.xaxis.get_gridlines(): l.set_color("#162840")
    ax.set_title("Classification Radar",pad=22,color=BLUE)
    ax.legend(loc="upper right",bbox_to_anchor=(1.42,1.12),
              facecolor=PANEL,edgecolor="#162840",labelcolor="#C8D8EA")
    st.pyplot(fig,use_container_width=True); plt.close(fig)
    st.markdown("</div>",unsafe_allow_html=True)

    st.markdown('<div class="card cr">',unsafe_allow_html=True)
    st.subheader("⚠️ Accuracy Ceiling — Why ~70–74%?")
    st.markdown("""
| Root Cause | Explanation |
|---|---|
| **Class B/C Overlap** | Adjacent fitness tiers share overlapping ranges; B & C F1 consistently lowest |
| **Low-signal BP** | Diastolic/systolic r<0.07 — adds noise, not signal |
| **Label Ambiguity** | Classes discretised from continuous scores; boundary cases ambiguous |
| **Extensive tuning** | SVM C=0.01–100, 3 kernels; DT α via CV; KNN k=1–99; MLP depths + alphas |

**Best classification: MLP ≈ 74.8%** &nbsp;·&nbsp; **Best regression: Linear Reg. R²=0.803**
    """); st.markdown("</div>",unsafe_allow_html=True)

elif "Predictor" in page:
    st.title("🔮 Live Predictor")
    st.markdown('<div class="card cb">Enter measurements to predict '
                '<b>Performance Class (A–D)</b> and <b>Broad Jump (cm)</b>.</div>',
                unsafe_allow_html=True)
    with st.form("pf"):
        st.subheader("Participant Measurements")
        c1,c2,c3,c4=st.columns(4)
        age_=c1.number_input("Age (yr)",18,80,28)
        gnd_=c2.selectbox("Gender",["M","F"])
        ht_=c3.number_input("Height (cm)",140.,200.,170.,step=.5)
        wt_=c4.number_input("Weight (kg)",30.,150.,68.,step=.5)
        c1,c2,c3,c4=st.columns(4)
        bf_=c1.number_input("Body Fat (%)",3.,60.,22.,step=.5)
        dia_=c2.number_input("Diastolic BP",50.,130.,79.)
        sys_=c3.number_input("Systolic BP",80.,180.,130.)
        gr_=c4.number_input("Grip Force (kg)",5.,80.,37.,step=.5)
        c1,c2,c3=st.columns(3)
        fl_=c1.number_input("Sit & Bend (cm)",-20.,50.,15.,step=.5)
        su_=c2.number_input("Sit-ups",0.,80.,40.)
        bj_=c3.number_input("Broad Jump (cm, optional)",0.,310.,190.,step=1.)
        mc_=st.selectbox("Classifier",
            ["Decision Tree (72.15%)","SVM RBF (71.27%)","KNN k=22 (63.09%)","MLP (~74%)"])
        sub_=st.form_submit_button("🚀 Predict",use_container_width=True)
    if sub_:
        bmi_=wt_/((ht_/100)**2); ge_=0 if gnd_=="M" else 1
        xc=np.array([[age_,ge_,ht_,wt_,bf_,dia_,sys_,gr_,fl_,su_,bj_,bmi_]])
        xr=np.array([[age_,ge_,ht_,wt_,bf_,dia_,sys_,gr_,fl_,su_,bmi_]])
        sc_c=StandardScaler(); Xcs=sc_c.fit_transform(Xc)
        sc_r=StandardScaler(); Xrs=sc_r.fit_transform(Xr)
        with st.spinner("Predicting…"):
            if "Decision Tree" in mc_:
                clf=DecisionTreeClassifier(random_state=42,ccp_alpha=.000377).fit(Xc,yc)
                rgr=DecisionTreeRegressor(random_state=42).fit(Xr,yr); xci,xri=xc,xr
            elif "SVM" in mc_:
                clf=SVC(kernel="rbf",C=10,gamma=.1,random_state=42).fit(Xcs,yc)
                rgr=SVR(kernel="rbf",C=10).fit(Xrs,yr); xci,xri=sc_c.transform(xc),sc_r.transform(xr)
            elif "KNN" in mc_:
                clf=KNeighborsClassifier(22,n_jobs=-1).fit(Xcs,yc)
                rgr=KNeighborsRegressor(37,n_jobs=-1).fit(Xrs,yr); xci,xri=sc_c.transform(xc),sc_r.transform(xr)
            else:
                clf=MLPClassifier(hidden_layer_sizes=(256,128,64),alpha=.001,max_iter=400,
                    early_stopping=True,random_state=42).fit(Xcs,yc)
                rgr=MLPRegressor(hidden_layer_sizes=(256,128,64),alpha=.001,max_iter=400,
                    early_stopping=True,random_state=42).fit(Xrs,yr)
                xci,xri=sc_c.transform(xc),sc_r.transform(xr)
            pc=clf.predict(xci)[0]; pj=float(rgr.predict(xri)[0])
        cn=CL[pc]
        INFO={"A":("🥇 Excellent",GREEN,"Outstanding fitness. Maintain your programme."),
              "B":("🥈 Good",BLUE,"Above average. Focus on strength and power."),
              "C":("🥉 Average",PURPLE,"Moderate. Increase endurance training."),
              "D":("⚠️ Below Average",RED,"Below average. Start a structured programme.")}
        lbl,col,adv=INFO[cn]
        st.markdown(f'<div class="card" style="border:2px solid {col};text-align:center;padding:26px">'
                    f'<div style="font-size:3rem">{lbl}</div>'
                    f'<div style="font-size:1.35rem;font-weight:900;color:{col};margin:8px 0">Class {cn}</div>'
                    f'<div style="color:#C8D8EA">{adv}</div></div>',unsafe_allow_html=True)
        mbox({"Class":cn,"Predicted Jump":f"{pj:.1f} cm","BMI":f"{bmi_:.1f}","Model":mc_.split()[0]})
        fig,ax=plt.subplots(figsize=(11,2.8))
        for lo,hi,c,lb in [(0,120,RED,"<120"),(120,165,"#E8963A","120–165"),
                            (165,200,BLUE,"165–200"),(200,240,GREEN,"200–240"),(240,320,"#5DCAA5","240+")]:
            ax.barh(0,hi-lo,left=lo,height=.5,color=c,edgecolor=DARK,alpha=.75)
            ax.text((lo+hi)/2,.54,lb,ha="center",va="bottom",color="#C8D8EA",fontsize=8)
        ax.axvline(pj,color="white",lw=3,zorder=5,label=f"{pj:.0f} cm")
        ax.set_xlim(0,320); ax.set_ylim(-.25,.95); ax.set_yticks([])
        ax.set_xlabel("Broad Jump (cm)"); ax.set_title("Jump Distance Gauge")
        ax.legend(labelcolor="#C8D8EA"); show(fig)
        fn=["Age","Height","Weight","Body Fat%","Diastolic","Systolic","Grip","Sit&Bend","Sit-ups","BMI"]
        yv=[age_,ht_,wt_,bf_,dia_,sys_,gr_,fl_,su_,round(bmi_,1)]
        mv=[df[c].mean() for c in ["age","height_cm","weight_kg","body fat_%","diastolic",
            "systolic","gripForce","sit and bend forward_cm","sit-ups counts","BMI"]]
        x_=np.arange(len(fn)); fig,ax=plt.subplots(figsize=(13,4.5))
        ax.bar(x_-.22,yv,.42,label="You",color=BLUE,edgecolor=DARK,zorder=3)
        ax.bar(x_+.22,mv,.42,label="Dataset Mean",color=PURPLE,edgecolor=DARK,alpha=.75,zorder=3)
        ax.set_xticks(x_); ax.set_xticklabels(fn,rotation=35,ha="right",fontsize=8)
        ax.set_title("Your Profile vs Dataset Averages")
        ax.legend(labelcolor="#C8D8EA"); ax.grid(axis="y",alpha=.15); show(fig)

st.markdown("""
<hr style="border:none;border-top:1px solid #162840;margin-top:50px">
<div style="text-align:center;padding:12px 0;color:#1D3040;font-size:.76rem">
  <strong style="color:#1D4060">AXORA</strong> · Body Performance Analytics ·
  Alaa Issawi · Amira Salama · Aya Abdel Maksoud · Aya El-Sabi · Aya Imam · Aya Khalil ·
  AI &amp; ML 2024–2025
</div>""",unsafe_allow_html=True)
