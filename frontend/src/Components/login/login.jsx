import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useUser } from '../../UserContext';
import { login, setAuthToken } from '../../api';
import './login.css';

const LoginForm = () => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const { user, setUser } = useUser();  // Get user from context
  const navigate = useNavigate();

  // Check if there's a saved token in localStorage on component mount
  useEffect(() => {
    const token = localStorage.getItem('token');
    const savedUser = localStorage.getItem('user');

    if (token && savedUser) {
      setAuthToken(token);  // Set the token for future API requests
      setUser(JSON.parse(savedUser));  // Automatically set the user
      navigate('/');  // Redirect to home page
    }
  }, [setUser, navigate]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setIsLoading(true);
    
    try {
      const data = await login(email, password);

      if (data.access_token) {
        setAuthToken(data.access_token);
        
        // Save the token and user in localStorage
        localStorage.setItem('token', data.access_token);
        localStorage.setItem('user', JSON.stringify({ email }));

        setUser({ email });  // Set user context with email
        navigate('/');  // Redirect to home
      } else {
        setError('Login failed: Invalid response from server');
      }
    } catch (err) {
      setError(err.message || 'An error occurred during login');
    } finally {
      setIsLoading(false);
    }
  };

  // Logout function to clear localStorage and reset the user context
  const handleLogout = () => {
    localStorage.removeItem('token');
    localStorage.removeItem('user');
    setUser(null);
    navigate('/login');
  };

  if (user) {
    return (
      <div>
        <p>Welcome, {user.email}</p>
        <button onClick={handleLogout}>Logout</button>
      </div>
    );
  }

  return (
    <div className="login-container">
      <form onSubmit={handleSubmit} className="login-form">
        <h2>Login</h2>
        <input
          type="email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          placeholder="Email"
          required
          className="login-input"
          disabled={isLoading}
        />
        <input
          type="password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          placeholder="Password"
          required
          className="login-input"
          disabled={isLoading}
        />
        {error && <div className="error">{error}</div>}
        <button type="submit" className="login-button" disabled={isLoading}>
          {isLoading ? 'Logging in...' : 'Login'}
        </button>
      </form>
    </div>
  );
};

export default LoginForm;
