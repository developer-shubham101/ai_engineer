// JWT Authentication Utility Functions

/**
 * Decode JWT token to extract payload
 * @param {string} token - JWT token
 * @returns {object|null} Decoded payload or null if invalid
 */
export function decodeJWT(token) {
  try {
    if (!token) return null;
    const parts = token.split('.');
    if (parts.length !== 3) return null;
    
    const payload = parts[1];
    const decoded = JSON.parse(atob(payload));
    return decoded;
  } catch (error) {
    console.error('Error decoding JWT:', error);
    return null;
  }
}

/**
 * Check if token is expired
 * @param {string} token - JWT token
 * @returns {boolean} True if expired
 */
export function isTokenExpired(token) {
  const decoded = decodeJWT(token);
  if (!decoded || !decoded.exp) return true;
  
  const currentTime = Math.floor(Date.now() / 1000);
  return decoded.exp < currentTime;
}

/**
 * Get stored token from localStorage
 * @returns {string|null} Token or null
 */
export function getStoredToken() {
  return localStorage.getItem('access_token');
}

/**
 * Store token in localStorage
 * @param {string} token - JWT token
 */
export function setStoredToken(token) {
  localStorage.setItem('access_token', token);
}

/**
 * Clear token from localStorage
 */
export function clearStoredToken() {
  localStorage.removeItem('access_token');
}

/**
 * Get stored user info from localStorage
 * @returns {object|null} User object or null
 */
export function getStoredUser() {
  const userStr = localStorage.getItem('user_info');
  if (!userStr) return null;
  try {
    return JSON.parse(userStr);
  } catch (error) {
    console.error('Error parsing user info:', error);
    return null;
  }
}

/**
 * Store user info in localStorage
 * @param {object} user - User object
 */
export function setStoredUser(user) {
  localStorage.setItem('user_info', JSON.stringify(user));
}

/**
 * Clear user info from localStorage
 */
export function clearStoredUser() {
  localStorage.removeItem('user_info');
}

/**
 * Extract session ID from JWT token
 * @param {string} token - JWT token
 * @returns {string|null} Session ID or null
 */
export function getSessionIdFromToken(token) {
  const decoded = decodeJWT(token);
  return decoded?.session_id || null;
}

/**
 * Clear all authentication data
 */
export function clearAuth() {
  clearStoredToken();
  clearStoredUser();
  // Also clear old API key if it exists
  localStorage.removeItem('api_key');
  localStorage.removeItem('remember_api_key');
  localStorage.removeItem('x-session-id');
}
