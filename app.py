import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os

st.set_page_config(page_title="🌿 Plant Disease Recognition", layout="centered")
st.title("🌿 Plant Disease Recognition")

# ---- MODEL LOADING ----
model_path = "plant_disease_recong_model.keras"

if os.path.exists(model_path):
    st.write("✅ Model file found")
    model = tf.keras.models.load_model(model_path)
else:
    st.error("❌ Model file NOT found! Make sure it's in the project root.")
    st.stop()

# ---- DISEASE INFO LOADING ----
disease_info_path = "assets/disease_info.json"

if os.path.exists(disease_info_path):
    with open(disease_info_path, "r") as f:
        disease_info = json.load(f)
    st.write("✅ Disease info loaded")
else:
    st.error("❌ disease_info.json NOT found! Check assets folder.")
    st.stop()

# ---- IMAGE UPLOAD ----
uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    try:
        # Open and display the image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Preprocess image to match model input
        image = image.resize((160, 160))  # Correct input size for your model
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        st.write("✅ Image preprocessed:", image_array.shape)

        # Prediction
        st.write("Predicting...")
        prediction = model.predict(image_array)
        class_idx = int(np.argmax(prediction))
        st.write(f"Prediction array: {prediction}")

        # Display results
        info = disease_info.get(str(class_idx), None)
        if info:
            st.success(f"**Predicted Disease:** {info['name']}")
            st.info(f"**Cause:** {info['cause']}")
            st.info(f"**Cure:** {info['cure']}")
        else:
            st.warning("Prediction not found in disease_info.json")

    except Exception as e:
        st.error(f"❌ Error processing image: {e}")
