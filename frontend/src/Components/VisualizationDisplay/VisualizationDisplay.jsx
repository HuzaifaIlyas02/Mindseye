import React from 'react';
import './VisualizationDisplay.css';

const VisualizationDisplay = ({ visualization }) => {
  return (
    <div className="visualization-display">
      {visualization ? (
        <img src={visualization} alt="AI-generated visualization" />
      ) : (
        <p>Submit a question or upload an image to see the visualization.</p>
      )}
    </div>
  );
};

export default VisualizationDisplay;