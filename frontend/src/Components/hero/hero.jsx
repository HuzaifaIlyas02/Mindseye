import React from 'react';
import './hero.css';
import PhysicsQA from '../PhysicsQA/PhysicsQA';


const Hero = () => {
  return (
    <div className="hero">
      <h1>Visualize Physics Concepts with AI</h1>
      <p>Upload your physics questions and get expert-level insights and visual explanations in seconds.</p>
      <PhysicsQA />
    </div>
  );
};

export default Hero;

