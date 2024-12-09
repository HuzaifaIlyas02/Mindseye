import React from 'react';
import './features.css';

const Features = () => {
  return (
    <section className="features">
      <h2>Key Features</h2>
      <div className="feature-grid">
        <div className="feature-item">
          <h3>Text & Image Input</h3>
          <p>Upload or type your physics questions.</p>
        </div>
        <div className="feature-item">
          <h3>AI-Powered Analysis</h3>
          <p>Get concise and understandable answers using NLP.</p>
        </div>
        <div className="feature-item">
          <h3>Visual Explanations</h3>
          <p>Receive flowcharts and diagrams to clarify concepts.</p>
        </div>
        <div className="feature-item">
          <h3>Comprehensive Coverage</h3>
          <p>From high school to college-level physics topics.</p>
        </div>
      </div>
    </section>
  );
};

export default Features;