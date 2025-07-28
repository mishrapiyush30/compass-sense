// API Configuration
const config = {
  development: {
    API_BASE_URL: 'http://localhost:8000'
  },
  production: {
    API_BASE_URL: process.env.REACT_APP_API_BASE_URL || 'https://your-backend-url.railway.app'
  }
};

const environment = process.env.NODE_ENV || 'development';

export const API_BASE_URL = config[environment].API_BASE_URL;
export default config; 