import axios from "axios";

const API_URL = "http://localhost:5000/api";  // Flask backend URL

const api = axios.create({
  baseURL: API_URL,
  headers: {
    "Content-Type": "application/json",
  },
});

export const setAuthToken = (token) => {
  if (token) {
    api.defaults.headers.common["Authorization"] = `Bearer ${token}`;
    localStorage.setItem("token", token);
  } else {
    delete api.defaults.headers.common["Authorization"];
    localStorage.removeItem("token");
  }
};

export const login = async (email, password) => {
  try {
    const response = await api.post("/login", { email, password });
    if (response.data.access_token) {
      setAuthToken(response.data.access_token);  // Store the token in localStorage
    }
    return response.data;  // Should return access token
  } catch (error) {
    console.error("Login error:", error);
    throw error;
  }
};

export const register = async (email, password) => {
  try {
    const response = await api.post("/register", { email, password });
    return response.data;  // Return registration success response
  } catch (error) {
    if (error.response) {
      console.error("Registration error:", error.response.data);
      throw new Error(error.response.data.error || "Registration failed");
    } else {
      console.error("Registration error:", error);
      throw new Error("Registration failed");
    }
  }
};

export const uploadImage = async (image) => {
  try {
    const formData = new FormData();
    formData.append("image", image);
    const response = await api.post("/upload-image", formData, {
      headers: { "Content-Type": "multipart/form-data" },
    });
    return response.data;  // Handle image upload response
  } catch (error) {
    console.error("Error uploading image:", error);
    throw error;
  }
};

export const submitQuestion = async (question) => {
  try {
    const response = await api.post("/question", { question });  // Endpoint for submitting questions
    return response.data;  // Return response from Flask
  } catch (error) {
    console.error("Error submitting question:", error);
    throw error;
  }
};

export const getHistory = async () => {
  try {
    const response = await api.get("/history");
    return response.data;  // Assuming Flask returns user history
  } catch (error) {
    console.error("Error fetching history:", error);
    throw error;
  }
};

// Initialize token from localStorage
const token = localStorage.getItem("token");
if (token) {
  setAuthToken(token);
}

export default api;
