import React from 'react';
import './Community.css';

const Community = () => {
  return (
    <div className="community-container">
      <h1>Mind's Eye Community</h1>
      <p>Join our community of physics enthusiasts! Share your questions, insights, and experiences.</p>
      <div className="community-features">
        <div className="feature">
          <h3>Discussion Forums</h3>
          <p>Engage in discussions about various physics topics.</p>
        </div>
        <div className="feature">
          <h3>Study Groups</h3>
          <p>Form or join study groups to collaborate with peers.</p>
        </div>
        <div className="feature">
          <h3>Ask an Expert</h3>
          <p>Get answers from our community of physics experts.</p>
        </div>
      </div>
    </div>
  );
};

export default Community;