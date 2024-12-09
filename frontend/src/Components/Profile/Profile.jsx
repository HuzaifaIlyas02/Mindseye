import { useState } from 'react';
import { useUser } from '../../UserContext';
import { uploadImage, submitQuestion, getHistory } from '../../api'; 
import QuestionInput from '../QuestionInput/QuestionInput';
import './Profile.css';

function Profile() {
  const { user } = useUser();
  const [image, setImage] = useState(null);
  const [uploadStatus, setUploadStatus] = useState('');
  const [questionStatus, setQuestionStatus] = useState('');
  const [history, setHistory] = useState([]);

  const handleImageUpload = async (e) => {
    e.preventDefault();
    if (!image) {
      setUploadStatus('Please select an image first.');
      return;
    }

    try {
      await uploadImage(image);  // No need to assign 'result' since we're not using it
      setUploadStatus('Image uploaded successfully!');
    } catch (err) {
      setUploadStatus('Failed to upload image. Please try again.');
    }
  };

  const handleQuestionSubmit = async (question) => {
    try {
      await submitQuestion(question);  // No need to assign 'response' since we're not using it
      setQuestionStatus('Question submitted successfully!');
    } catch (err) {
      setQuestionStatus('Failed to submit question. Please try again.');
    }
  };

  const fetchHistory = async () => {
    try {
      const historyData = await getHistory();  // Use a more descriptive name like 'historyData' for clarity
      setHistory(historyData);
    } catch (err) {
      console.error('Failed to fetch history:', err);
    }
  };

  return (
    <div className="profile-container">
      <h1>Profile Page</h1>
      <p>Welcome, {user?.email || 'User'}!</p>
      <div className="history-section">
        <h2>Previous Questions</h2>
        <button onClick={fetchHistory}>Load History</button>
        {history.map((entry, idx) => (
          <div key={idx}>
            <p>Q: {entry.question}</p>
            <p>A: {entry.answer}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

export default Profile;
