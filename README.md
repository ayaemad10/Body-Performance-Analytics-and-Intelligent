📋 Project Overview

This project presents a comprehensive machine learning analysis of human body performance using the Body Performance Dataset. It focuses on classifying individuals into performance categories (A–D) and predicting physical performance metrics such as broad jump distance.

The system applies a full ML pipeline including preprocessing, feature engineering, model training, and evaluation, along with an interactive application for real-world usage.

🎯 Objective
Classify individuals into performance levels (A, B, C, D)
Predict physical performance (broad jump distance) using regression models
📊 Pipeline Steps
Step	Description
1. EDA	Analyze distributions, correlations, and feature importance
2. Data Cleaning	Remove duplicates, fix invalid values, handle outliers
3. Feature Engineering	Encoding (gender, class), scaling (StandardScaler)
4. Train/Test Split	Multiple splits (80/20, 70/30, 50/50)
5. Modeling	KNN, Decision Tree, SVM, Neural Network, Linear Regression
6. Evaluation	Accuracy, F1 Score, R², RMSE, Cross-validation
7. Prediction	Classification + regression predictions
🤖 Models
📌 Classification Results
Model	Accuracy	F1 Score
KNN	63.09%	0.63
Decision Tree	72.15%	0.71
SVM (RBF)	71.31%	0.71
Neural Network (MLP)	74.80%	0.75
Linear Regression (adapted)	~51%	~0.48

✅ Best Model: Neural Network (MLP)

📌 Regression Results (Broad Jump Prediction)
Model	R² Score	RMSE
Neural Network	0.803	17.77
Linear Regression	0.803	17.89
SVM (SVR)	0.798	17.96
KNN	0.788	—
Decision Tree	0.685	22.43
🚀 Getting Started
1. Clone the Repository
git clone https://github.com/ayaemad10/Body-Performance-Analytics-and-Intelligent.git
cd Body-Performance-Analytics-and-Intelligent
2. Install Dependencies
pip install -r requirements.txt
3. Run the Project
python main.py
🌐 Interactive App

The project includes a Streamlit web application that allows users to:

Input their own body data
Get instant performance classification
Explore model insights

Run it using:

streamlit run app.py
📁 Project Structure
Body-Performance-Analytics-and-Intelligent/
├── data/                # Dataset files
├── notebooks/           # Jupyter notebooks
├── src/                 # ML pipeline & models
├── app.py               # Streamlit app
├── requirements.txt     # Dependencies
├── README.md            # Documentation
└── .gitignore           # Ignore rules
🔑 Key Features
sit_and_bend_forward_cm → strongest predictor
sit_ups_counts → endurance indicator
broad_jump_cm → explosive power
age → negatively correlated with performance
gender → affects strength-related metrics
📚 Dataset
Source: Kaggle (Body Performance Dataset)
Records: 13,393
Features: 12 (10 numerical + 2 categorical)
Target: Performance class (A, B, C, D)
Class Balance: Balanced (~25% each)
⚠️ Key Insights
يوجد ceiling في الأداء ~75% لكل الموديلات
السبب:
تداخل بين Class B و C
Features ضعيفة (زي blood pressure)
عدم وضوح الحدود بين الكلاسات
🛠️ Tech Stack
Python
Pandas / NumPy
Scikit-learn
TensorFlow / Keras
Matplotlib / Seaborn
Streamlit
📝 License

MIT License — feel free to use and modify.
