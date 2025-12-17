import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

IMG_SIZE = 256  # EfficientNetB3
MODEL_PATH = 'best_dr_model.h5'

@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        return model
    except Exception as e:
        st.error(f"Error loading model. Check if '{MODEL_PATH}' exists. Error: {e}")
        return None

model = load_model()
class_names = ['Mild DR', 'Moderate DR', 'Healthy', 'Proliferate_DR', 'Severe DR']

# Layout
st.set_page_config(page_title="DR Detector")
st.title("Diabetic Retinopathy Detection")
st.write("Upload a retinal scan to analyze severity.")

file = st.file_uploader("Choose a retina image", type=["jpg", "png", "jpeg"])

if file is not None and model is not None:
    image = Image.open(file).convert('RGB')
    st.image(image, caption='Uploaded Scan', use_container_width=True)
    # Resizing 
    image_resized = ImageOps.fit(image, (IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)
    img_array = np.array(image_resized)
    # Normalization
    img_array = img_array.astype(np.float32) / 255.0
    # Batch Dimension (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)

    if st.button("Analyze"):
        with st.spinner("Analyzing..."):
            prediction = model.predict(img_array)
            scores = prediction[0]
            predicted_index = np.argmax(scores)
            predicted_class = class_names[predicted_index]
            confidence = scores[predicted_index] * 100
            
        if predicted_class in ["Mild DR", "Moderate DR"]:
            st.warning("Diagnosis: Non-Proliferative DR (Early Stage)")
        elif predicted_class in ["Severe DR", "Proliferate DR"]:
            st.error("Diagnosis: Proliferative DR (Advanced Stage)")
        else:
            st.success("Diagnosis: Healthy")
        st.bar_chart({label: score for label, score in zip(class_names, scores)})
