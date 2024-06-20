import React, { useState } from 'react';
import './App.css';

function EmotionDetection() {
  const [textInput, setTextInput] = useState('');
  const [textEmotion, setTextEmotion] = useState('');
  const [videoEmotion, setVideoEmotion] = useState('');
  const [recommendation, setRecommendation] = useState('');
  const [error, setError] = useState('');

  const handleTextInputChange = (e) => {
    setTextInput(e.target.value);
  };

  const handleTextSubmit = async () => {
    try {
      const formData = new FormData();
      formData.append('text', textInput);
      const response = await fetch('http://localhost:5000/predict_text', {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      if (response.ok) {
        setTextEmotion(data.emotion);
      } else {
        setError(data.error);
      }
    } catch (error) {
      setError(`Error: ${error.message}`);
    }
  };

  const handleVideoSubmit = async () => {
    try {
      const response = await fetch('http://localhost:5000/predict_video', {
        method: 'POST',
      });
      const data = await response.json();
      if (response.ok) {
        setVideoEmotion(data.emotion);
      } else {
        setError(data.error);
      }
    } catch (error) {
      setError(`Error: ${error.message}`);
    }
  };

  const handleRecommendation = async () => {
    try {
      const formData = new FormData();
      formData.append('text_emotion', textEmotion);
      formData.append('video_emotion', videoEmotion);
      const response = await fetch('http://localhost:5000/recommend_song', {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      if (response.ok) {
        setRecommendation(data.recommendation);
      } else {
        setError(data.error);
      }
    } catch (error) {
      setError(`Error: ${error.message}`);
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Emotion Based Music Recomender</h1>
        <textarea
          type="text"
          value={textInput}
          className='InputBox'
          onChange={handleTextInputChange}
          placeholder="          Enter text here"
        />
        <button onClick={handleTextSubmit}>Submit Text</button>
        <button onClick={handleVideoSubmit}>Capture Video</button>
        <button onClick={handleRecommendation}>Get Song Recommendation</button>
        <p>Text Emotion: {textEmotion}</p>
        <p>Video Emotion: {videoEmotion}</p>
        <p>Song Recommendation: {recommendation}</p>
        <p>{error}</p>
      </header>
    </div>
  );
}

export default EmotionDetection;
