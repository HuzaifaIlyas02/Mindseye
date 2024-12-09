import React, { useState } from 'react';
import './QuestionInput.css';
import { submitQuestion } from '../../api.js';
import CytoscapeChart from '../CytoscapeChart/CytoscapeChart';  // Import CytoscapeChart component

const QuestionInput = () => {
  const [question, setQuestion] = useState('');
  const [elements, setElements] = useState([]);  // Holds the nodes and edges for Cytoscape chart
  const [answer, setAnswer] = useState('');      // Holds the answer
  const [error, setError] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      setError(null);
      const response = await submitQuestion(question);  // Fetch data from the backend
      console.log('Backend response:', response);
      setElements(response.flowchart);  // Assume the backend returns flowchart data
      setAnswer(response.answer);       // Set the answer from the response
      setQuestion('');
    } catch (err) {
      console.error('Error submitting question:', err);
      setError('Failed to submit the question. Please try again.');
    }
  };

  return (
    <div className="question-input-container">
      <form onSubmit={handleSubmit}>
        <input
          type="text"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Enter your question"
        />
        <button type="submit">Submit Question</button>
      </form>

      {error && <p className="error-message">{error}</p>}

      {elements.length > 0 && (
        <CytoscapeChart elements={elements} answer={answer} />
      )}
    </div>
  );
};

export default QuestionInput;
