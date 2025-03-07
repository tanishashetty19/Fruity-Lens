import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
import os
import uuid

def save_correction(image, corrected_class):
    correction_dir = "data/corrections"
    os.makedirs(correction_dir, exist_ok=True)
    image.save(os.path.join(correction_dir, f"{corrected_class}_{uuid.uuid4().hex}.jpg"))

# Load the trained model
@st.cache_resource
def load_trained_model():
    return load_model('models/food_scan_model.h5')

# Define the path to your dataset
data_dir = r"C:\Users\Lenovo\Desktop\Food_scan\data\images"
class_labels = sorted(os.listdir(data_dir))  # Get food class names from subfolder names

# Function to predict the uploaded image
def predict_image(image, model, class_labels):
    img = image.resize((128, 128))  # Resize image to match model input size
    img_array = img_to_array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    predictions = model.predict(img_array)
    predicted_class = class_labels[np.argmax(predictions)]  # Map index to class label
    confidence = np.max(predictions)

    return predicted_class, confidence

# Streamlit app interface

page_bg_color = """ 
<style>
[data-testid="stAppViewContainer"] {
    background-color: #e8e8e4;
}
</style>
"""
st.markdown(page_bg_color, unsafe_allow_html=True)

st.title("FruityLens")

uploaded_file = st.file_uploader("Upload a food image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image',use_container_width=True)
    # use_column_width
    model = load_trained_model()
    predicted_class, confidence = predict_image(image, model, class_labels)

    st.write(f"**Predicted Class**: {predicted_class}")
    st.write(f"**Confidence**: {confidence * 100:.2f}%")

#correction(feedback)
corrected_class = st.selectbox("If the prediction is wrong, select the correct class:", class_labels)
if st.button("Submit Correction"):
    save_correction(image, corrected_class)
    st.write("Thank you! Your feedback will help improve the model.")

