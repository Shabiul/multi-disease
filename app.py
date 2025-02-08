import os
import pickle
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from streamlit_option_menu import option_menu
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import tensorflow_hub as hub
from keras.models import load_model
from keras.utils import custom_object_scope



# Set page configuration
st.set_page_config(page_title="Health Assistant", layout="wide", page_icon="🧑‍⚕️")

# Getting the working directory
def get_working_dir():
    return os.path.dirname(os.path.abspath(__file__))

working_dir = get_working_dir()

# Load saved models
def load_pickle_model(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)

diabetes_model = load_pickle_model(f'{working_dir}/saved_models/diabetes_model.sav')
heart_disease_model = load_pickle_model(f'{working_dir}/saved_models/heart_disease_model.sav')
parkinsons_model = load_pickle_model(f'{working_dir}/saved_models/parkinsons_model.sav')
with custom_object_scope({'KerasLayer': hub.KerasLayer}):
    brain_tumor_model = load_model(f'{working_dir}/saved_models/brain.h5')

# Sidebar navigation
with st.sidebar:
    selected = option_menu('Multiple Disease Prediction System',
                           ['Diabetes Prediction', 'Heart Disease Prediction', 'Parkinsons Prediction', 'Brain Tumor Prediction'],
                           menu_icon='hospital-fill',
                           icons=['activity', 'heart', 'person', 'brain'],
                           default_index=0)

# Utility function to convert input to float
def process_inputs(inputs):
    try:
        return [float(x) for x in inputs]
    except ValueError:
        st.error("Please enter valid numerical values.")
        return None

# Diabetes Prediction Page
if selected == 'Diabetes Prediction':
    st.title('Diabetes Prediction using ML')
    user_input = [st.text_input(label) for label in [
        'Number of Pregnancies', 'Glucose Level', 'Blood Pressure value',
        'Skin Thickness value', 'Insulin Level', 'BMI value',
        'Diabetes Pedigree Function value', 'Age of the Person']]
    
    if st.button('Diabetes Test Result'):
        processed_input = process_inputs(user_input)
        if processed_input:
            prediction = diabetes_model.predict([processed_input])
            st.success('The person is diabetic' if prediction[0] == 1 else 'The person is not diabetic')

# Heart Disease Prediction Page
elif selected == 'Heart Disease Prediction':
    st.title('Heart Disease Prediction using ML')
    user_input = [st.text_input(label) for label in [
        'Age', 'Sex', 'Chest Pain types', 'Resting Blood Pressure', 'Serum Cholestoral in mg/dl',
        'Fasting Blood Sugar > 120 mg/dl', 'Resting Electrocardiographic results', 'Maximum Heart Rate achieved',
        'Exercise Induced Angina', 'ST depression induced by exercise', 'Slope of the peak exercise ST segment',
        'Major vessels colored by fluoroscopy', 'Thal (0 = normal; 1 = fixed defect; 2 = reversible defect)']]
    
    if st.button('Heart Disease Test Result'):
        processed_input = process_inputs(user_input)
        if processed_input:
            prediction = heart_disease_model.predict([processed_input])
            st.success('The person has heart disease' if prediction[0] == 1 else 'The person does not have heart disease')

# Parkinson's Prediction Page
elif selected == "Parkinsons Prediction":
    st.title("Parkinson's Disease Prediction using ML")
    user_input = [st.text_input(label) for label in [
        'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)', 'MDVP:RAP',
        'MDVP:PPQ', 'Jitter:DDP', 'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5',
        'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA', 'Spread1', 'Spread2', 'D2', 'PPE']]
    
    if st.button("Parkinson's Test Result"):
        processed_input = process_inputs(user_input)
        if processed_input:
            prediction = parkinsons_model.predict([processed_input])
            st.success("The person has Parkinson's disease" if prediction[0] == 1 else "The person does not have Parkinson's disease")

# Brain Tumor Prediction Page
elif selected == 'Brain Tumor Prediction':
    st.title("Brain Tumor Prediction")
    tumor_classes = ["Glioma", "Meningioma", "Pituitary", "No_Tumor"]

    def preprocess_image(img):
        img = img.resize((224, 224)).convert("RGB")  # Resize & ensure RGB format
        img = img_to_array(img) / 255.0  # Normalize
        return np.expand_dims(img, axis=0)  # Add batch dimension

    uploaded_file = st.file_uploader("Choose an MRI scan", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded MRI Scan", use_column_width=True)
        processed_image = preprocess_image(image)
        prediction = brain_tumor_model.predict(processed_image)
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction) * 100
        st.write(f"### Prediction: 🧬 **Tumor Type:** {tumor_classes[predicted_class]} ({confidence:.2f}%)")
