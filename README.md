.

🏃‍♂️ Body Performance Analytics & Intelligent Prediction
An end-to-end Machine Learning pipeline designed to analyze human physical performance. This project classifies individuals into performance categories (A–D) and predicts explosive power (Broad Jump distance) using a variety of supervised learning algorithms.

🎯 Objectives
Classification: Categorize individuals into four performance levels (A, B, C, D) based on physical metrics.

Regression: Predict the Broad Jump distance (cm) as a measure of explosive physical power.

📊 The ML Pipeline
The project follows a rigorous data science workflow:
Stage,Description
1. EDA,"Distribution analysis, correlation matrices, and identifying feature importance."
2. Cleaning,"Duplicate removal, handling outliers, and validating data integrity."
3. Engineering,Categorical encoding (Gender/Class) and feature scaling via StandardScaler.
4. Modeling,"Comparative analysis of KNN, Decision Trees, SVM, and MLP Neural Networks."
5. Evaluation,"Assessment using Accuracy, F1-Score, R2, and RMSE."
🤖 Model Performance
🏆 Classification (Performance Grade)
The Neural Network (MLP) emerged as the champion, though many models hit a performance "ceiling" at ~75% due to feature overlap between Class B and C.
Model,Accuracy,F1 Score
Neural Network (MLP),74.80%,0.75
Decision Tree,72.15%,0.71
SVM (RBF),71.31%,0.71
KNN,63.09%,0.63
📏 Regression (Broad Jump Prediction)
Model,R2 Score,RMSE
Neural Network,0.803,17.77
Linear Regression,0.803,17.89
SVM (SVR),0.798,17.96
🔑 Key Insights & Challenges
Top Predictor: sit_and_bend_forward_cm proved to be the strongest indicator of overall performance.

The "75% Ceiling": Analysis shows significant data overlap between middle-tier classes (B and C).

Weak Features: Features like blood pressure showed minimal correlation with physical performance categories.

🚀 Interactive Application
Experience the model in real-time! The integrated Streamlit app allows you to input body metrics and receive an instant performance grade.
# To run the app locally:
streamlit run app.py
🛠️ Installation & Usage
1.Clone the repo
git clone https://github.com/ayaemad10/Body-Performance-Analytics-and-Intelligent.git
cd Body-Performance-Analytics-and-Intelligent
2.Install dependencies
pip install -r requirements.txt
3.Execute the pipeline
pip install -r requirements.txt
📁 Project Structure
├── data/                # Raw and processed datasets
├── notebooks/           # Experimental Jupyer Notebooks
├── src/                 # Modular ML pipeline code
├── app.py               # Streamlit web application
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
📄 License
Distributed under the MIT License. See LICENSE for more information.
Author: Aya Emad


