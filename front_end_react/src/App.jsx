import React, { useState, useEffect } from 'react'
import RAGChat from './components/RAGChat.jsx'
import Login from './components/Login.jsx'
import { getStoredToken, getStoredUser, setStoredToken, setStoredUser } from './utility/auth.js'

export default function App() {
  const [isAuthenticated, setIsAuthenticated] = useState(false)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    // Check if user is already logged in
    const token = getStoredToken()
    const user = getStoredUser()

    if (token && user) {
      setIsAuthenticated(true)
    }
    setLoading(false)
  }, [])

  function handleLoginSuccess(token, user) {
    if (token) {
      setStoredToken(token)
      setStoredUser(user)
    } else {
      // Guest mode - don't store token
      setStoredUser(user)
    }
    setIsAuthenticated(true)
  }

  function handleLogout() {
    setIsAuthenticated(false)
  }

  if (loading) {
    return (
      <div className="container-fluid d-flex justify-content-center align-items-center" style={{ minHeight: '100vh' }}>
        <div className="spinner-border" role="status">
          <span className="visually-hidden">Loading...</span>
        </div>
      </div>
    )
  }

  return (
    <div className="container-fluid">
      {isAuthenticated ? (
        <RAGChat onLogout={handleLogout} />
      ) : (
        <Login onLoginSuccess={handleLoginSuccess} />
      )}
    </div>
  )
}
