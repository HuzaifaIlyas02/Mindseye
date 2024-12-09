import React from 'react';
import './FAQ.css';

const FAQ = () => {
  const faqItems = [
    {
      question: "What is Mind's Eye?",
      answer: "Mind's Eye is an AI-powered platform designed to help students visualize and understand complex physics concepts through interactive problem-solving and visual aids."
    },
    {
      question: "How does Mind's Eye work?",
      answer: "Users can input physics problems via text or image. Our AI processes the input, generates detailed explanations, and creates visual flowcharts to aid understanding."
    },
    {
      question: "What types of physics problems can Mind's Eye handle?",
      answer: "Mind's Eye covers a wide range of physics topics from high school to college level, including mechanics, thermodynamics, electromagnetism, and more."
    },
    {
      question: "Can I download the explanations and flowcharts?",
      answer: "Yes, users can download generated explanations and flowcharts in PDF or image formats for offline study."
    }
  ];

  return (
    <div className="faq-container">
      <h1>Frequently Asked Questions</h1>
      {faqItems.map((item, index) => (
        <div key={index} className="faq-item">
          <h2>{item.question}</h2>
          <p>{item.answer}</p>
        </div>
      ))}
    </div>
  );
};

export default FAQ;