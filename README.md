# Speech_emotion_recognition project
Speech Emotion Recognition using TESS Dataset
This project is a Speech Emotion Recognition system that classifies emotions from audio files using the TESS dataset. The model is implemented using Keras for deep learning and Streamlit for deployment as a web application.
F\FEATURES:
Real-time emotion detection from speech audio files.
Classification of seven emotions:
Angry, Disgust, Fear, Happy, Neutral, Sad, and Pleasant Surprise.
Feature extraction using Mel Frequency Cepstral Coefficients (MFCC).
User-friendly interface for uploading and analyzing audio files.
TECHNOLOGY Used:
Python
TensorFlow / Keras
Streamlit
Librosa (for audio processing)
Numpy
DATASET: TESS (Toronto Emotional Speech Set)
Run the Streamlit app:
streamlit run app.py
MODEL TRAINING:
model is trained using MFCC features extracted from the audio files.
The trained model is saved as speech_emotion_model.h5 and is used for real-time emotion prediction.
 RESULT AND ACCURACY:
 The model achieved significant accuracy in classifying emotions from audio.
Detailed performance metrics and confusion matrix are available in the report.
