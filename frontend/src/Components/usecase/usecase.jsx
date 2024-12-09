import React from 'react';
import './usecase.css';

const UseCases = () => {
  const useCases = [
    {
      title: "Problem Solving",
      description: "Get step-by-step solutions to complex physics problems with detailed explanations and visual aids.",
      icon: "🧠"
    },
    {
      title: "Concept Visualization",
      description: "Visualize abstract physics concepts through interactive diagrams and animations.",
      icon: "👁️"
    },
    {
      title: "Exam Preparation",
      description: "Practice with a variety of physics problems and receive instant feedback to improve your understanding.",
      icon: "📚"
    },
    {
      title: "Research Assistance",
      description: "Explore advanced physics topics and get help with interpreting research papers and experiments.",
      icon: "🔬"
    }
  ];

  return (
    <div className="use-cases">
      <h1>Highlights for Mind's Eye</h1>
      <div className="use-cases-grid">
        {useCases.map((useCase, index) => (
          <div key={index} className="use-case-item">
            <div className="use-case-icon">{useCase.icon}</div>
            <h2>{useCase.title}</h2>
            <p>{useCase.description}</p>
          </div>
        ))}
      </div>
    </div>
  );
};

export default UseCases;