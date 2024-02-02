// src/TensorflowComponent.js
import React, { useState, useEffect } from 'react';
import * as tf from '@tensorflow/tfjs';
import * as use from '@tensorflow-models/universal-sentence-encoder';
import './TensorflowComponent.css'; // Import your CSS file

function TensorflowComponent() {
    const [userInput, setUserInput] = useState('');
    const [embeddings, setEmbeddings] = useState();
    const [isLoading, setIsLoading] = useState(false);
    const [model, setModel] = useState(null);
  
    useEffect(() => {
      const loadModel = async () => {
        try {
          const loadedModel = await use.load();
          console.log('Model loaded successfully!');
          console.log('Loaded model details:', loadedModel);
          setModel(loadedModel);
        } catch (error) {
          console.error('Error loading model:', error);
        }
      };
  
      tf.setBackend('webgl');
      loadModel();
    }, []);
  
    const handleButtonClick = async () => {
      try {
        setIsLoading(true);
        const userInputArray = userInput.split('\n');
        const newEmbeddings = await model.embed(userInputArray);
        setEmbeddings(newEmbeddings.arraySync());
      } catch (error) {
        console.error('Error processing input:', error);
      } finally {
        setIsLoading(false);
      }
    };

  return (
    <div className="tensorflow-container">
      <div className="input-container">
        <h1>TensorFlow Model Demo</h1>
        <textarea
          rows="10"
          placeholder="Enter sentences separated by new lines..."
          value={userInput}
          onChange={(e) => setUserInput(e.target.value)}
        ></textarea>
        <br />
        <button onClick={handleButtonClick} disabled={isLoading}>
          {isLoading ? 'Generating Embeddings...' : 'Generate Embeddings'}
        </button>
      </div>

      <div className="output-container">
        <h2>Embeddings:</h2>
        <div className="output-content">
          {JSON.stringify(embeddings, null, 2)}
        </div>
      </div>
    </div>
  );
}

export default TensorflowComponent;


// @tensorflow-models/universal-sentence-encoder library in TensorFlow.js, the model is loaded from a CDN (Content Delivery Network) where pre-trained models are hosted

// It also cache models once they are loaded, so subsequent requests to load the same model will reuse the cached version.