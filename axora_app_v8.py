"""
AXORA — Body Performance Analytics & Intelligent Classification System
Version 8.0 | Team Axora
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

st.markdown("""
    <style>
    .stApp {
        background-color: #0B1621;
        color: #C8D8EA;
    }
    section[data-testid="stSidebar"] {
        background-color: #0D1B2A !important;
        border-right: 1px solid #162840;
    }
    .centered-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        padding-top: 5rem;
    }
    h1 { color: #3B8BD4 !important; }
    h2, h3 { color: #FFFFFF !important; }
    .stRadio [data-testid="stWidgetLabel"] { color: #3B8BD4; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

plt.style.use('dark_background')
AXORA_COLORS = ["#3B8BD4", "#9B51E0", "#4CC9F0", "#7209B7"]
sns.set_palette(sns.color_palette(AXORA_COLORS))

with st.sidebar:
    st.image("axora_team_logo.svg", width=150)
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

if menu == "🏠 Home":
    st.markdown('<div class="centered-container">', unsafe_allow_html=True)
    st.image("axora_team_logo.svg", width=450)
    st.markdown('<h1 style="font-size: 3.5rem; margin-bottom: 0;">AXORA</h1>', unsafe_allow_html=True)
    st.markdown('<p style="font-size: 1.2rem; opacity: 0.8;">Intelligent Body Performance Analytics System</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

elif menu == "📊 Data Overview":
    st.header("📋 Dataset Overview")
    st.write("Add dataset preview here")

elif menu == "📈 EDA":
    st.header("📊 Exploratory Data Analysis")
    st.write("Add EDA plots here")

elif menu == "🤖 Model Analysis":
    st.header(f"Model Performance: {model_choice}")
    st.write("Add model metrics here")

elif menu == "🎯 Prediction":
    st.header("🔮 Intelligent Prediction")
    st.write("Add prediction inputs here")
