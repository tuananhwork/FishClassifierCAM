import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('fish_classifier_model.h5')

# Load fish species names
species_dict = {}
with open('fish_species.txt', 'r') as f:
    for line in f:
        idx, species = line.strip().split(',')
        species_dict[int(idx)] = species

# Function to make prediction
def predict(image):
    # Preprocess the image to match the model's input requirements
    img = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)  # Adjust size to your model input
    img = np.asarray(img)
    img = img / 255.0  # Normalize image
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Make prediction
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions, axis=1)
    confidence = np.max(predictions, axis=1) * 100  # Get the confidence percentage
    
    return predicted_class[0], confidence[0]

# Streamlit interface
st.set_page_config(page_title="AnhCBT | Fish Species Classifier", page_icon="🐟")
st.title("Fish Species Classifier")

st.image("avt.jpg", caption="Quả app này làm mệt vãi chưởng HAHAHAHA")


st.write("Upload an image of a fish, and the model will predict its species.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

st.warning("Don't have fish? Download Dataset to test")
st.link_button("Download now", 'https://drive.google.com/drive/folders/10faXPNaqy6fCDRKz95ZAVE_SGCqaOdVi?usp=sharing')

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    
    predicted_class, confidence = predict(image)
    predicted_species = species_dict[predicted_class]
    st.success(f"Prediction: {predicted_species} ({confidence:.2f}%)")

st.link_button("Contact me", 'https://www.facebook.com/chu.anh.11')

