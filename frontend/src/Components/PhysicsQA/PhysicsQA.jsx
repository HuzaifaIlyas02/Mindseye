import React, { useState } from "react";
import { useUser } from "../../UserContext";
import QuestionInput from "../QuestionInput/QuestionInput";
import ImageUpload from "../ImageUpload/ImageUpload";
import "./PhysicsQA.css"; // Import your updated styles

const PhysicsQA = () => {
  const [visualization, setVisualization] = useState(null);
  const [error, setError] = useState(null);
  const { user } = useUser();
  const [attemptedAccess, setAttemptedAccess] = useState(false);
  const [answer, setAnswer] = useState(null);
  const [flowchart, setFlowchart] = useState(null);

  const handleQuestionSubmit = async (question) => {
    if (!user) {
      setError("Please log in to submit questions.");
      return;
    }
    try {
      setError(null);
      const result = await submitQuestion(question, user.id);
      setAnswer(result.answer);

      if (result.flowchart) {
        setFlowchart(result.flowchart.trim());
        console.log("Flowchart to render:", result.flowchart.trim());
      } else {
        setFlowchart(null);
      }
    } catch (err) {
      setError("Failed to process question. Please try again.");
    }
  };

  const handleImageUpload = async (image) => {
    if (!user) {
      setError("Please log in to upload images.");
      return;
    }
    try {
      setError(null);
      const result = await uploadImage(image, user.id);
      setVisualization(result.visualization);
    } catch (err) {
      setError("Failed to process image. Please try again.");
    }
  };

  const handleAccessAttempt = () => {
    if (!user) {
      setAttemptedAccess(true);
      return;
    }
    setAttemptedAccess(false);
  };

  return (
    <div className="physics-qa-container">
      <button onClick={handleAccessAttempt} className="try-minds-eye-btn">
        Try Mind&apos;s Eye
      </button>
      {user ? (
        <>
          <QuestionInput onSubmit={handleQuestionSubmit} />
          <ImageUpload onUpload={handleImageUpload} />
        </>
      ) : (
        attemptedAccess && <p>Please log in to use the Physics Q&A feature.</p>
      )}
    </div>
  );
};

export default PhysicsQA;
