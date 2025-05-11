import streamlit as st
import numpy as np
import os
import librosa
from tensorflow.keras.models import load_model

# Emotion labels (in correct training order!)
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'pleasant_surprise']

# Page title
st.title("🎧 Speech Emotion Recognition - TESS")

# Load trained model (only once)
@st.cache_resource
def load_trained_model():
    model_path = "speech_emotion_model.h5"
    if os.path.exists(model_path):
        return load_model(model_path)
    else:
        st.error("❌ Model file not found! Please train and save the model first.")
        return None

model = load_trained_model()

# Feature extraction function
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfcc, axis=1)

# File uploader
uploaded_file = st.file_uploader("Upload a .wav file", type=["wav"])

if uploaded_file is not None and model is not None:
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.read())

    # Extract features from uploaded audio
    features = extract_features("temp.wav")
    features = np.expand_dims(features, axis=0)  # Reshape for model input (1, 13)

    # Predict emotion
    prediction = model.predict(features)
    predicted_index = np.argmax(prediction)
    predicted_emotion = emotion_labels[predicted_index]

    # Show results
    st.success(f"🎤 Predicted Emotion: **{predicted_emotion.capitalize()}**")

    # Debug: Show raw prediction scores
    st.write("Raw prediction scores:", prediction)
    
    # Show predicted index (helps in understanding which class the model predicted)
    st.write("Predicted index:", predicted_index)
    st.write("Extracted features:", features)
    st.write("Raw prediction scores:", prediction)


