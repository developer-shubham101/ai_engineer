import React, { useState, useEffect } from 'react'
import RAGChat from './components/RAGChat.jsx'
import Login from './components/Login.jsx'
import { getStoredToken, getStoredUser, setStoredToken, setStoredUser } from './utility/auth.js'

export function setQueryParams(mode, convId) {
  const p = new URLSearchParams()
  if (mode && mode !== 'rag') p.set('mode', mode)
  if (convId) p.set('conv', convId)
  const str = p.toString()
  window.history.replaceState(null, '', str ? `?${str}` : window.location.pathname)
}

export default function App() {
  const [isAuthenticated, setIsAuthenticated] = useState(false)
  const [loading, setLoading] = useState(true)
  const [initialMode] = useState(() => new URLSearchParams(window.location.search).get('mode') || 'rag')
  const [initialConvId] = useState(() => new URLSearchParams(window.location.search).get('conv') || null)

  useEffect(() => {
    const token = getStoredToken()
    const user = getStoredUser()
    if (token && user) setIsAuthenticated(true)
    setLoading(false)
  }, [])

  function handleLoginSuccess(token, user) {
    if (token) {
      setStoredToken(token)
      setStoredUser(user)
    } else {
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
        <RAGChat onLogout={handleLogout} initialMode={initialMode} initialConvId={initialConvId} />
      ) : (
        <Login onLoginSuccess={handleLoginSuccess} />
      )}
    </div>
  )
}
