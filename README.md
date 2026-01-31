# ❤️ Heart Disease Prediction System

A professional-grade **Streamlit** web application that leverages machine learning to predict the likelihood of heart disease based on clinical parameters.

---

## 📊 Overview
This project uses a **Random Forest** classification model to analyze patient data—such as age, cholesterol, and chest pain type—to determine cardiovascular risk. The application provides real-time predictions, probability scores, and health recommendations through an interactive dashboard.

## 🚀 Key Features
* **Predictive Engine:** Utilizes a Random Forest model with an **85% accuracy** and **0.94 ROC-AUC** score.
* **Interactive Input:** Users can input 13 medical attributes through a clean, two-column interface.
* **Data Visualization:** Includes a dynamic Plotly gauge chart to visualize heart disease risk scores.
* **Automatic Scaling:** Features a pre-fitted `StandardScaler` to ensure numerical inputs are processed correctly.
* **Health Insights:** Generates specific recommendations based on "High Risk" or "Low Risk" classifications.

## 🛠️ Tech Stack
* **Frontend:** Streamlit
* **Data Analysis:** Pandas, NumPy
* **Machine Learning:** Scikit-learn (Random Forest & StandardScaler)
* **Visualization:** Plotly Graph Objects
* **Model Storage:** Pickle

## 🚀 Getting Started
* **1. Prerequisites:**
        Ensure you have Python installed, then clone this repository and navigate to the project folder.
* **2. Install Dependencies:**
        Create a virtual environment and install the required libraries:
      text->pip install streamlit pandas numpy plotly scikit-learn
* **3. Run the Application:**
      ->streamlit run app.py

## Medical Attributes Used
The model evaluates the following top predictors and clinical metrics:
* **Chest Pain Type:** Typical Angina, Atypical Angina, Non-anginal, or Asymptomatic.
* **Thalassemia:** Normal, Fixed Defect, or Reversible Defect.
* **Max Heart Rate:** Highest heart rate achieved during exercise.
* **Other Metrics:** Age, Sex, Resting BP, Cholesterol, Fasting Blood Sugar, Resting ECG, ST Depression, and more.

*⚠️ Disclaimer*
For Educational Purposes Only. This application is a machine learning demonstration and should not be used for actual medical diagnosis. Always consult a healthcare professional for medical advice.


## 📂 Project Structure
```text
├── data/
│   ├── heart_disease.csv            # Raw dataset
│   ├── X_train.csv / X_test.csv     # Training/Testing features
│   └── y_train.csv / y_test.csv     # Training/Testing labels
├── models/
│   ├── best_model_random_forest.pkl # The trained classifier
│   └── scaler.pkl                   # The fitted StandardScaler
├── notebooks/
│   ├── 01_data_exploration.ipynb    # Initial EDA
│   ├── 02_data_preprocessing.ipynb  # Cleaning and scaling
│   └── 03_models_training.ipynb     # Model training and tuning
├── app.py                           # The main Streamlit application code
├── download_data.py                 # Script to fetch/update dataset
├── requirements.txt                 # Python dependencies
└── README.md                        # Documentation







        
