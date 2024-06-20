import os
import numpy as np
import cv2  
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
    # Initialize video capture from default camera (0)
    cap = cv2.VideoCapture(0)

    # Check if the camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open video capture.")
        return None

    # Variables for time management
    start_time = cv2.getTickCount()
    frame_count = 0
    detected_emotions = []

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Error: Frame capture failed.")
            break

        # Increment frame count
        frame_count += 1

        # Draw bounding box around faces and predict emotions
        frame, emotions = predict_emotion_with_bounding_box(frame)

        # Display the captured frame with bounding box and emotion label
        cv2.imshow('Video', frame)

        # Collect detected emotions
        if emotions:
            detected_emotions.extend(emotions)

        # Check if 1 minute has passed or if user pressed 'q'
        if (cv2.waitKey(1) & 0xFF == ord('q')) or (cv2.getTickCount() - start_time >= 60 * cv2.getTickFrequency()):
            break

    # Release the capture
    cap.release()
    cv2.destroyAllWindows()

    # Determine the most frequent emotion detected
    if detected_emotions:
        video_emotion = max(detected_emotions, key=detected_emotions.count)
    else:
        video_emotion = "Neutral"  # Default if no emotions detected

    return video_emotion

# Function to predict emotion from frames with bounding boxes
def predict_emotion_with_bounding_box(frame):
    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Load the pre-trained face cascade classifier for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    emotions = []

    # Draw bounding boxes around faces and predict emotions
    for (x, y, w, h) in faces:
        # Extract face region from the frame
        face_region = frame[y:y+h, x:x+w]

        # Preprocess the face region using the custom transformer
        preprocessor = FacePreprocessor()
        processed_face = preprocessor.transform(face_region)

        # Predict emotion using the facial emotion model
        emotion_prediction = facial_emotion_model.predict(processed_face[np.newaxis, ...])[0]
        detected_emotion_label = emotion_labels[np.argmax(emotion_prediction)]

        # Draw bounding box around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Display emotion label near the bounding box
        cv2.putText(frame, detected_emotion_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Collect detected emotions
        emotions.append(detected_emotion_label)

    return frame, emotions

# Function to recommend a song based on combined emotions (to be implemented)
def recommend_song(text_emotion, video_emotion):
    # Placeholder for song recommendation logic
    # Implement this based on combined emotions from text and video
    return "Song recommendation logic to be implemented"  # Replace with actual logic

# Main function to coordinate the process
def main():
    # Get text input from user
    sample_text = input("Enter your current emotion: ")

    # Predict emotion from text input
    text_emotion = predict_emotion_from_text(sample_text)

    # Predict emotion from video
    video_emotion = predict_emotion_from_video()

    # Recommend a song based on combined emotions
    song_recommendation = recommend_song(text_emotion, video_emotion)

    print(f"Text Emotion: {text_emotion}")
    print(f"Video Emotion: {video_emotion}")
    print(f"Song Recommendation: {song_recommendation}")

# Entry point of the script
if __name__ == "__main__":
    main()
