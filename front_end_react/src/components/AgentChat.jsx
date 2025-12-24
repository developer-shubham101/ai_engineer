import React, { useState } from 'react'
import { BASE_API_URL } from '../utility/const.js'
import { getStoredToken } from '../utility/auth.js'

export default function AgentChat({ onLogout, onExit }) {
  const [orchestratorType, setOrchestratorType] = useState('custom') // 'custom' or 'autogen'
  const [query, setQuery] = useState('')
  const [messages, setMessages] = useState([])
  const [loading, setLoading] = useState(false)

  async function handleSend(e) {
    if (e) e.preventDefault()
    if (!query.trim()) return

    setLoading(true)
    const userMsg = { role: 'user', content: query, ts: new Date().toLocaleString() }
    setMessages(prev => [...prev, userMsg])
    setQuery('')

    try {
      const token = getStoredToken()
      const headers = { 'Content-Type': 'application/json' }
      if (token) headers['Authorization'] = `Bearer ${token}`

      const res = await fetch(`${BASE_API_URL}/api/agents/query`, {
        method: 'POST',
        headers,
        body: JSON.stringify({
          question: userMsg.content,
          orchestrator_type: orchestratorType,
          options: {
            max_steps: 10
          }
        })
      })

      if (!res.ok) {
        throw new Error(`Agent API failed: ${res.status}`)
      }

      const data = await res.json()

      const botMsg = {
        role: 'assistant',
        content: data.answer,
        steps: data.steps || [],
        ts: new Date().toLocaleString()
      }

      setMessages(prev => [...prev, botMsg])

    } catch (err) {
      setMessages(prev => [...prev, {
        role: 'system',
        content: `Error: ${err.message}`,
        isError: true,
        ts: new Date().toLocaleString()
      }])
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="h-100 d-flex flex-column">
      {/* Header */}
      <div className="p-3 border-bottom d-flex justify-content-between align-items-center bg-light-subtle">
        <div className="d-flex align-items-center gap-2">
          <button className="btn btn-sm btn-outline-secondary" onClick={onExit} title="Exit Agent Mode">
            <i className="bi bi-arrow-left"></i>
          </button>
          <h5 className="mb-0">
            <i className="bi bi-cpu me-2"></i>
            AI Agents Sandbox
          </h5>
        </div>
        <div className="d-flex align-items-center gap-2">
          <span className="small text-muted">Orchestrator:</span>
          <select
            className="form-select form-select-sm"
            style={{ width: 'auto' }}
            value={orchestratorType}
            onChange={e => setOrchestratorType(e.target.value)}
            disabled={loading}
          >
            <option value="custom">Custom (Simulate)</option>
            <option value="autogen">AutoGen Agent</option>
          </select>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-grow-1 overflow-auto p-4" style={{ backgroundColor: 'var(--bs-body-bg)' }}>
        {messages.length === 0 && (
          <div className="text-center text-muted mt-5">
            <i className="bi bi-robot display-1"></i>
            <p className="lead mt-3">Select an orchestrator and start chatting.</p>
          </div>
        )}

        {messages.map((msg, idx) => (
          <div key={idx} className={`mb-4 card ${msg.role === 'user' ? 'border-primary' : ''} ${msg.isError ? 'border-danger' : ''}`}>
            <div className={`card-header d-flex justify-content-between ${msg.role === 'user' ? 'bg-primary-subtle' : ''}`}>
              <strong>{msg.role === 'user' ? 'You' : 'Agent Result'}</strong>
              <small className="text-muted">{msg.ts}</small>
            </div>
            <div className="card-body">
              {msg.isError ? (
                <div className="text-danger">{msg.content}</div>
              ) : (
                <div>
                  {msg.role === 'assistant' && msg.steps && msg.steps.length > 0 && (
                    <details className="mb-3 border rounded p-2 bg-light">
                      <summary className="cursor-pointer font-monospace small">View Reasoning Steps ({msg.steps.length})</summary>
                      <div className="mt-2 small">
                        {msg.steps.map((step, i) => (
                          <div key={i} className="mb-2 border-bottom pb-2">
                            <div><strong>Tool:</strong> {step.tool}</div>
                            <div><strong>Thought:</strong> {step.thought}</div>
                            <div className="text-muted"><strong>Result:</strong> {typeof step.result === 'object' ? JSON.stringify(step.result) : step.result}</div>
                          </div>
                        ))}
                      </div>
                    </details>
                  )}
                  <div style={{ whiteSpace: 'pre-wrap' }}>{msg.content}</div>
                </div>
              )}
            </div>
          </div>
        ))}

        {loading && (
          <div className="text-center my-4">
            <div className="spinner-border text-primary" role="status">
              <span className="visually-hidden">Loading...</span>
            </div>
          </div>
        )}
      </div>

      {/* Input */}
      <div className="p-3 border-top bg-light-subtle">
        <form onSubmit={handleSend}>
          <div className="input-group">
            <input
              type="text"
              className="form-control"
              placeholder="Enter your request..."
              value={query}
              onChange={e => setQuery(e.target.value)}
              disabled={loading}
            />
            <button className="btn btn-primary" type="submit" disabled={loading || !query.trim()}>
              {loading ? 'Thinking...' : 'Send'}
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}
