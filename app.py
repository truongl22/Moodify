import streamlit as st
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Load trained best_model.h5 
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("best_model.h5")
    return model

model = load_model()

emotion_labels = ["Angry", "Happy", "Neutral", "Sad"]

def process_img_from_streamlit(uploaded_file):
    img = Image.open(uploaded_file).convert("L")  # grayscale
    img = img.resize((48, 48))
    x = np.array(img).astype("float32") / 255.0
    x = np.expand_dims(x, axis=(0, -1))
    return x

# streamlit Setup
st.set_page_config(page_title="Moodify Emotion Detector", layout="centered")

# two tabs: emotion detection and About. 
tab1, tab2 = st.tabs(["Emotion Detection", "About Project"])

with tab1:
    st.title("Emotion Detection Using CNN Model")
    st.write("Upload a face image for emotion recognition using the trained CNN classifier.")

    uploaded_file = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", width=250)

        img_processed = process_img_from_streamlit(uploaded_file)

        probs = model.predict(img_processed)[0]
        emotion_index = int(np.argmax(probs))
        emotion = emotion_labels[emotion_index]

        st.subheader(f"Predicted Emotion: **{emotion}**")

        st.write("Prediction Probabilities:")
        for label, p in zip(emotion_labels, probs):
            st.write(f"- **{label}**: {p:.4f}")

with tab2:
    st.title("About Moodify")
    st.write("""
    **Moodify Emotion Detector**  
    This app uses a Convolutional Neural Network (CNN) trained on grayscale 48×48 images 
    to classify facial emotion into the following categories: Angry, Happy, Neutral, Sad  

    **About**  
    - Built for DS 4420 (Machine Learning and Data Mining 2)  
    - Deployed using Streamlit Cloud  
    - Emotion classifier built in Python  
    - Recommender system built separately in R  

    **Technologies Used:**  
    - TensorFlow / Keras: Tensorflow 2.1.xx
    - Streamlit 
    - Python: python< 3.10.sxx
    - NumPy: 1.2.3
    - PIL: match up with tensorflow and numpy versions  
    """)
    st.write("---")
    st.write("Created by **Son Tran** and **Lam Truong** for DS 4420 Extra Credit.")
