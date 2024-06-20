import React, { useState } from 'react';

const TextInputComponent = ({ onNext }) => {
  const [textInput, setTextInput] = useState('');

  const handleTextInputChange = (e) => {
    setTextInput(e.target.value);
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    onNext(textInput);
  };

  return (
    <div>
      <h2>Step 1: Enter Your Current Emotion</h2>
      <form onSubmit={handleSubmit}>
        <input
          type="text"
          value={textInput}
          onChange={handleTextInputChange}
          placeholder="Enter text"
        />
        <button type="submit">Next</button>
      </form>
    </div>
  );
};

export default TextInputComponent;
