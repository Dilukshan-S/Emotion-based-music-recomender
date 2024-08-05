# BestMusiQ
 
This project combines text and facial emotion recognition models to predict user emotions and recommend music accordingly.

**Project Structure:**

EMODETECT/
scripts/
app.py: Starts the text and facial emotion recognition models.

models/
facial_emotion_model.h5
text_emotion_recognition_model3.h5

bestmusiq/ React frontend application to interact with the emotion detection models.

# Setup and Execution

# Step 1: Run the Emotion Detection Models

1. Navigate to the EMODETECT/scripts directory:

cd EMODETECT/scripts

2. Start the text and facial emotion recognition models:

python app.py

# Step 2: Start the BestMusiq Frontend

1. Open a new terminal window.
2. Navigate to the bestmusiq directory:

cd bestmusiq

3. Install the required npm packages:

npm install

4. Start the React application:

npm start

Open your web browser and navigate to http://localhost:3000 to view the web page.

# Note:
Music prediction logic is not implemented yet.
