import React, { createContext, useState, useContext, useEffect } from 'react';

const UserContext = createContext();

export const UserProvider = ({ children }) => {
  const [user, setUser] = useState(null);

  useEffect(() => {
    // Check for user and token in localStorage on initial load
    const storedUser = localStorage.getItem('user');
    const token = localStorage.getItem('token');  // Assuming you are storing the auth token
    if (storedUser && token) {
      setUser(JSON.parse(storedUser));
    } else {
      // Clear any old data if token is missing or invalid
      localStorage.removeItem('user');
      localStorage.removeItem('token');
    }
  }, []);

  const login = (userData, token) => {
    setUser(userData);
    localStorage.setItem('user', JSON.stringify(userData));
    localStorage.setItem('token', token);  // Store token when the user logs in
  };

  const logout = () => {
    setUser(null);
    localStorage.removeItem('user');
    localStorage.removeItem('token');
  };

  return (
    <UserContext.Provider value={{ user, setUser, login, logout }}>
      {children}
    </UserContext.Provider>
  );
};

export const useUser = () => useContext(UserContext);