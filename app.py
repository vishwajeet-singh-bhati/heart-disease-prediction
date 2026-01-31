import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide"
)

# Load the model and scaler
@st.cache_resource
def load_model():
    with open('models/best_model_random_forest.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_model()

# Title and description
st.title("❤️ Heart Disease Prediction System")
st.markdown("""
This application predicts the likelihood of heart disease based on medical attributes.
**Disclaimer:** This is a machine learning model for educational purposes only. 
Always consult healthcare professionals for medical advice.
""")

# Sidebar with information
st.sidebar.header("About")
st.sidebar.info("""
**Model Information:**
- Algorithm: Random Forest
- Accuracy: 85%
- ROC-AUC: 0.94

**Top Predictors:**
1. Chest Pain Type
2. Thalassemia
3. Max Heart Rate
""")

# Create two columns for input
col1, col2 = st.columns(2)

with col1:
    st.subheader("📋 Patient Information")
    
    age = st.number_input("Age", min_value=20, max_value=100, value=50, 
                          help="Patient's age in years")
    
    sex = st.selectbox("Sex", options=[1, 0], 
                       format_func=lambda x: "Male" if x == 1 else "Female",
                       help="Biological sex")
    
    cp = st.selectbox("Chest Pain Type", options=[0, 1, 2, 3],
                      format_func=lambda x: {
                          0: "Typical Angina",
                          1: "Atypical Angina", 
                          2: "Non-anginal Pain",
                          3: "Asymptomatic"
                      }[x],
                      help="Type of chest pain experienced")
    
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 
                                min_value=90, max_value=200, value=120,
                                help="Blood pressure at rest")
    
    chol = st.number_input("Cholesterol (mg/dl)", 
                           min_value=100, max_value=600, value=200,
                           help="Serum cholesterol level")
    
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1],
                       format_func=lambda x: "Yes" if x == 1 else "No",
                       help="Is fasting blood sugar greater than 120 mg/dl?")
    
    restecg = st.selectbox("Resting ECG Results", options=[0, 1, 2],
                           format_func=lambda x: {
                               0: "Normal",
                               1: "ST-T Wave Abnormality",
                               2: "Left Ventricular Hypertrophy"
                           }[x],
                           help="Resting electrocardiographic results")

with col2:
    st.subheader("🏃 Exercise & Heart Metrics")
    
    thalach = st.number_input("Maximum Heart Rate Achieved", 
                              min_value=60, max_value=220, value=150,
                              help="Highest heart rate during exercise test")
    
    exang = st.selectbox("Exercise Induced Angina", options=[0, 1],
                         format_func=lambda x: "Yes" if x == 1 else "No",
                         help="Chest pain during exercise")
    
    oldpeak = st.number_input("ST Depression (oldpeak)", 
                              min_value=0.0, max_value=7.0, value=1.0, step=0.1,
                              help="ST depression induced by exercise")
    
    slope = st.selectbox("Slope of Peak Exercise ST Segment", options=[0, 1, 2],
                         format_func=lambda x: {
                             0: "Upsloping",
                             1: "Flat",
                             2: "Downsloping"
                         }[x],
                         help="Slope of the peak exercise ST segment")
    
    ca = st.selectbox("Number of Major Vessels (0-3)", options=[0, 1, 2, 3],
                      help="Number of major vessels colored by fluoroscopy")
    
    thal = st.selectbox("Thalassemia", options=[1, 2, 3],
                        format_func=lambda x: {
                            1: "Normal",
                            2: "Fixed Defect",
                            3: "Reversible Defect"
                        }[x],
                        help="Blood disorder status")

# Prediction button
st.markdown("---")
if st.button("🔍 Predict Heart Disease Risk", type="primary", use_container_width=True):
    # Prepare input data
    input_data = pd.DataFrame({
        'age': [age],
        'sex': [sex],
        'cp': [cp],
        'trestbps': [trestbps],
        'chol': [chol],
        'fbs': [fbs],
        'restecg': [restecg],
        'thalach': [thalach],
        'exang': [exang],
        'oldpeak': [oldpeak],
        'slope': [slope],
        'ca': [ca],
        'thal': [thal]
    })
    
    # Scale numerical features
    numerical_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    input_data[numerical_features] = scaler.transform(input_data[numerical_features])
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]
    
    # Display results
    st.markdown("---")
    st.subheader("📊 Prediction Results")
    
    # Create three columns for results
    res_col1, res_col2, res_col3 = st.columns(3)
    
    with res_col1:
        if prediction == 1:
            st.error("⚠️ **High Risk**")
            st.markdown("Heart disease detected")
        else:
            st.success("✅ **Low Risk**")
            st.markdown("No heart disease detected")
    
    with res_col2:
        st.metric("Disease Probability", f"{probability[1]*100:.1f}%")
    
    with res_col3:
        st.metric("Healthy Probability", f"{probability[0]*100:.1f}%")
    
    # Probability gauge chart
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = probability[1] * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Heart Disease Risk Score"},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkred" if prediction == 1 else "darkgreen"},
            'steps': [
                {'range': [0, 33], 'color': "lightgreen"},
                {'range': [33, 66], 'color': "yellow"},
                {'range': [66, 100], 'color': "lightcoral"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)
    
    # Medical recommendation
    st.markdown("---")
    st.subheader("💡 Recommendations")
    
    if prediction == 1:
        st.warning("""
        **Important Notice:**
        - This model suggests a higher risk of heart disease
        - Please consult a cardiologist for proper diagnosis
        - Consider lifestyle modifications: diet, exercise, stress management
        - Regular health checkups are recommended
        """)
    else:
        st.info("""
        **Maintain Your Heart Health:**
        - Continue regular exercise and healthy diet
        - Monitor blood pressure and cholesterol levels
        - Annual health checkups recommended
        - Manage stress and get adequate sleep
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Built with ❤️ using Streamlit | Model Accuracy: 85% | For Educational Purposes Only</p>
</div>
""", unsafe_allow_html=True)