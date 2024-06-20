import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import pickle

# Ensure the models directory exists
os.makedirs('models', exist_ok=True)

# Load and preprocess data
text_data = pd.read_csv('../data/TextEmotion/tweet_emotions.csv')

X_text = text_data['content']  # Ensure this is the correct column name for your text data
y_text = text_data['sentiment']  # Ensure this is the correct column name for your labels

# Create a pipeline that includes the vectorizer and the model
pipeline = make_pipeline(
    TfidfVectorizer(),
    StandardScaler(with_mean=False),  # TfidfVectorizer output does not support mean centering
    LogisticRegression(max_iter=5000, solver='liblinear')  # Increased max_iter and changed solver
)

# Train-test split
X_text_train, X_text_test, y_text_train, y_text_test = train_test_split(X_text, y_text, test_size=0.2, random_state=42)

# Train the pipeline
pipeline.fit(X_text_train, y_text_train)

# Save the pipeline (includes both the vectorizer and the model)
with open('../models/text_emotion_pipeline.pkl', 'wb') as pipeline_file:
    pickle.dump(pipeline, pipeline_file)

print("Text emotion pipeline saved successfully!")
