import { useNavigate, Link } from 'react-router-dom';
import { useUser } from '../../UserContext';
import { setAuthToken } from '../../api';
import './navbar.css';

const Navbar = () => {
  const navigate = useNavigate();
  const { user, setUser } = useUser();

  const handleLogout = () => {
    localStorage.removeItem('token');
    setAuthToken(null);
    setUser(null);
    navigate('/');
  };

  return (
    <>
      <nav className="navbar">
        <div className="navbar-left">
          <Link to="/use-cases" className="use-cases-btn">Highlights▾</Link>
        </div>
        <div className="navbar-center">
          <Link to="/" className="brand-name">Mind&apos;s Eye</Link>
        </div>
        <div className="navbar-right">
          <button onClick={() => navigate('/faq')} className="nav-btn faq-btn">FAQ</button>
          <button onClick={() => navigate('/community')} className="nav-btn community-btn">Community</button>
          {user ? (
            <>
              <button onClick={() => navigate('/profile')} className="nav-btn profile-btn">Profile</button>
              <button onClick={handleLogout} className="nav-btn logout-btn">Logout</button>
            </>
          ) : (
            <>
              <Link to="/signup" className="nav-btn signup-btn">Sign Up</Link>
              <Link to="/login" className="nav-btn login-btn">Log In</Link>
            </>
          )}
        </div>
      </nav>
    </>
  );
};

export default Navbar;
