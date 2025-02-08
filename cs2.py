import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array  # Import img_to_array correctly
from PIL import Image
import tensorflow_hub as hub  # Import TensorFlow Hub

# Load model with custom KerasLayer
model = load_model(
    "C:/Users/Shabiul/Downloads/multiple-disease-prediction-streamlit-app-main/saved_models/brain.h5",
    custom_objects={'KerasLayer': hub.KerasLayer}
)

print("Model loaded successfully!")

# Define the tumor classes (Update based on your dataset)
tumor_classes = ["Glioma", "Meningioma", "Pituitary", "No_Tumor"]

# Define function to preprocess the image
def preprocess_image(img):
    img = img.resize((224, 224))  # Resize image to match model input size
    img = img.convert("RGB")  # Ensure image has 3 channels (RGB)
    img = img_to_array(img)  # Convert image to NumPy array
    img = img / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Streamlit UI
st.title("🧠 Brain Tumor Classification App")
st.write("Upload an MRI scan to classify the type of brain tumor.")

# Upload image
uploaded_file = st.file_uploader("Choose an MRI scan", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded MRI Scan", use_column_width=True)

    # Preprocess the image
    processed_image = preprocess_image(image)

    # Predict using the model
    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction)  # Get the class index
    confidence = np.max(prediction) * 100  # Get confidence score

    # Display result
    st.write("### Prediction:")
    st.write(f"🧬 **Tumor Type:** {tumor_classes[predicted_class]}")
    st.write(f"📊 **Confidence:** {confidence:.2f}%")

    # Highlight results
    if predicted_class == 3:
        st.success("🟢 No Tumor Detected! MRI is normal.")
    else:
        st.error(f"🔴 **{tumor_classes[predicted_class]} detected!** Consult a doctor.")
