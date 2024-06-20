import React from 'react';

const VideoCaptureComponent = ({ onNext }) => {
  const handleCaptureVideo = async () => {
    try {
      const response = await fetch('http://localhost:5000/predict_video', {
        method: 'POST',
      });
      const data = await response.json();
      if (response.ok) {
        onNext(data.emotion);
      } else {
        console.error('Error capturing video:', data.error);
      }
    } catch (error) {
      console.error('Error capturing video:', error);
    }
  };

  return (
    <div>
      <h2>Step 2: Capture Video</h2>
      <button onClick={handleCaptureVideo}>Capture Video</button>
    </div>
  );
};

export default VideoCaptureComponent;
