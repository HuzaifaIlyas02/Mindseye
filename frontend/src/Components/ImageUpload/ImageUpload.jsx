import React, { useState } from 'react';
import './ImageUpload.css';
import { uploadImage } from '../../api.js';  // Import the uploadImage function from api
import CytoscapeChart from '../CytoscapeChart/CytoscapeChart';  // Import CytoscapeChart component

const ImageUpload = () => {
  const [elements, setElements] = useState([]);  // Holds the nodes and edges for Cytoscape chart
  const [answer, setAnswer] = useState('');      // Holds the answer
  const [error, setError] = useState(null);

  const handleImageChange = async (e) => {
    if (e.target.files && e.target.files[0]) {
      const imageFile = e.target.files[0];
      
      try {
        setError(null);
        const response = await uploadImage(imageFile);  // Call the uploadImage function
        console.log('Backend response:', response);
        setElements(response.flowchart);  // Assume the backend returns flowchart data
        setAnswer(response.answer);       // Set the answer from the response
      } catch (err) {
        console.error('Error uploading image:', err);
        setError('Failed to upload the image. Please try again.');
      }
    }
  };

  return (
    <div className="image-upload-container"> {/* Updated wrapper div */}
      {/* Render the CytoscapeChart if there are elements to display */}
      {elements.length > 0 && (
        <CytoscapeChart elements={elements} answer={answer} />
      )}

      <div className="upload-button-container"> {/* Container for button */}
        <button onClick={() => document.getElementById('image-upload').click()}>
          Upload Image
        </button>
        <input
          type="file"
          id="image-upload"
          accept="image/*"
          onChange={handleImageChange}
          style={{ display: 'none' }}
        />
      </div>

      {error && <p className="error-message">{error}</p>}
    </div>
  );
};

export default ImageUpload;
