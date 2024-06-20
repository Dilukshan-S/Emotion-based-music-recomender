import os
import numpy as np
import cv2  
from flask import Flask, request, jsonify
from flask_cors import CORS
from keras.models import load_model
import pickle
from sklearn.base import BaseEstimator, TransformerMixin

# Define paths to the models
facial_emotion_model_path = '../models/facial_emotion_model.h5'
text_emotion_model_path = '../models/text_emotion_pipeline.pkl'

# Load pre-trained models
facial_emotion_model = load_model(facial_emotion_model_path)
with open(text_emotion_model_path, 'rb') as file:
    text_emotion_model = pickle.load(file)

# Emotion labels for facial emotion model
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})  # Enable CORS for all routes

# Function to predict emotion from text input
def predict_emotion_from_text(text):
    emotion = text_emotion_model.predict([text])[0]
    return emotion

# Custom transformer for face preprocessing
class FacePreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, target_size=(48, 48)):
        self.target_size = target_size
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        gray_face = cv2.cvtColor(X, cv2.COLOR_BGR2GRAY)
        resized_face = cv2.resize(gray_face, self.target_size)
        expanded_face = np.expand_dims(resized_face, axis=-1)
        return expanded_face

# Function to predict emotion from video
def predict_emotion_from_video():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return "Error: Could not open video capture."
    start_time = cv2.getTickCount()
    detected_emotions = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame, emotions = predict_emotion_with_bounding_box(frame)
        if emotions:
            detected_emotions.extend(emotions)
        if (cv2.waitKey(1) & 0xFF == ord('q')) or (cv2.getTickCount() - start_time >= 10 * cv2.getTickFrequency()):
            break
    cap.release()
    cv2.destroyAllWindows()
    if detected_emotions:
        video_emotion = max(detected_emotions, key=detected_emotions.count)
    else:
        video_emotion = "Neutral"
    return video_emotion

def predict_emotion_with_bounding_box(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
    emotions = []
    for (x, y, w, h) in faces:
        face_region = frame[y:y+h, x:x+w]
        preprocessor = FacePreprocessor()
        processed_face = preprocessor.transform(face_region)
        emotion_prediction = facial_emotion_model.predict(processed_face[np.newaxis, ...])[0]
        detected_emotion_label = emotion_labels[np.argmax(emotion_prediction)]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, detected_emotion_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        emotions.append(detected_emotion_label)
    return frame, emotions

@app.route('/predict_text', methods=['POST'])
def predict_text():
    text = request.form.get('text')
    if text:
        text_emotion = predict_emotion_from_text(text)
        return jsonify({'emotion': text_emotion})
    return jsonify({'error': 'No text provided'}), 400

@app.route('/predict_video', methods=['POST'])
def predict_video():
    video_emotion = predict_emotion_from_video()
    if "Error" in video_emotion:
        return jsonify({'error': video_emotion}), 500
    return jsonify({'emotion': video_emotion})

@app.route('/recommend_song', methods=['POST'])
def recommend_song():
    text_emotion = request.form.get('text_emotion')
    video_emotion = request.form.get('video_emotion')
    if text_emotion and video_emotion:
        song_recommendation = "Song recommendation logic to be implemented"  # Replace with actual logic
        return jsonify({'recommendation': song_recommendation})
    return jsonify({'error': 'Missing emotions for recommendation'}), 400

if __name__ == '__main__':
    app.run(debug=True)
