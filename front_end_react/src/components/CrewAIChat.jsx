import React, { useState, useEffect } from 'react'
import { BASE_API_URL } from '../utility/const.js'
import { getStoredToken, getSessionIdFromToken } from '../utility/auth.js'
import ReactMarkdown from 'react-markdown'

export default function CrewAIChat({ onLogout }) {
  const [topic, setTopic] = useState('')
  const [workflowType, setWorkflowType] = useState('debate')
  const [maxIterations, setMaxIterations] = useState(3)
  const [temperature, setTemperature] = useState(0.7)
  const [messages, setMessages] = useState([]) // For local history of runs
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [availableWorkflows, setAvailableWorkflows] = useState([])
  const [status, setStatus] = useState('idle') // idle, running, error, success

  useEffect(() => {
    fetchWorkflows()
  }, [])

  async function fetchWorkflows() {
    try {
      const res = await fetch(`${BASE_API_URL}/api/crew/status`)
      if (res.ok) {
        const data = await res.json()
        setAvailableWorkflows(data.available_workflows || [])
      }
    } catch (e) {
      console.error("Failed to fetch crew status", e)
    }
  }

  async function handleRunCrew(e) {
    if (e) e.preventDefault()
    if (!topic.trim()) return

    setLoading(true)
    setError(null)
    setStatus('running')

    // Add optimistic user message
    const userMsg = {
      role: 'user',
      content: `Run ${workflowType} on: ${topic}`,
      ts: new Date().toLocaleString()
    }
    setMessages(prev => [...prev, userMsg])

    try {
      const token = getStoredToken()
      const headers = { 'Content-Type': 'application/json' }
      if (token) headers['Authorization'] = `Bearer ${token}`

      const payload = {
        topic,
        workflow_type: workflowType,
        max_iterations: Number(maxIterations),
        temperature: Number(temperature),
        provider: 'local' // Fixed for now, can be extracted if needed
      }

      const res = await fetch(`${BASE_API_URL}/api/crew/query`, {
        method: 'POST',
        headers,
        body: JSON.stringify(payload)
      })

      if (!res.ok) {
        const txt = await res.text()
        throw new Error(`CrewAI failed: ${res.status} - ${txt}`)
      }

      const data = await res.json()

      const botMsg = {
        role: 'assistant',
        content: data.result,
        agents_used: data.agents_used,
        execution_time: data.execution_time_ms,
        ts: new Date().toLocaleString()
      }

      setMessages(prev => [...prev, botMsg])
      setStatus('success')
      setTopic('') // Clear input on success

    } catch (err) {
      console.error("Crew execution error:", err)
      setError(err.message)
      setStatus('error')
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
        <h5 className="mb-0">
          <i className="bi bi-robot me-2"></i>
          CrewAI Agents
        </h5>
        <div>
          {loading && <span className="badge bg-warning text-dark me-2">Agents Working...</span>}
          <span className="badge bg-secondary">{workflowType.toUpperCase()}</span>
        </div>
      </div>

      {/* Main Content - Results Area */}
      <div className="flex-grow-1 overflow-auto p-4" style={{ backgroundColor: 'var(--bs-body-bg)' }}>
        {messages.length === 0 && (
          <div className="text-center text-muted mt-5">
            <i className="bi bi-diagram-3 display-1"></i>
            <p className="lead mt-3">Select a workflow and topic to start a multi-agent session.</p>
          </div>
        )}

        {messages.map((msg, idx) => (
          <div key={idx} className={`mb-4 card ${msg.role === 'user' ? 'border-primary' : ''} ${msg.isError ? 'border-danger' : ''}`}>
            <div className={`card-header d-flex justify-content-between ${msg.role === 'user' ? 'bg-primary-subtle' : ''}`}>
              <strong>{msg.role === 'user' ? 'You' : 'CrewAI Result'}</strong>
              <small className="text-muted">{msg.ts}</small>
            </div>
            <div className="card-body">
              {msg.role === 'assistant' ? (
                <>
                  <div className="markdown-body">
                    <pre style={{ whiteSpace: 'pre-wrap', fontFamily: 'inherit' }}>{msg.content}</pre>
                  </div>
                  {msg.agents_used && (
                    <div className="mt-3 pt-2 border-top">
                      <small className="text-muted">
                        <i className="bi bi-people me-1"></i> Agents: {msg.agents_used.join(', ')} |
                        <i className="bi bi-clock ms-2 me-1"></i> Time: {(msg.execution_time / 1000).toFixed(2)}s
                      </small>
                    </div>
                  )}
                </>
              ) : (
                <div>{msg.content}</div>
              )}
            </div>
          </div>
        ))}

        {loading && (
          <div className="text-center my-4">
            <div className="spinner-border text-primary" role="status">
              <span className="visually-hidden">Loading...</span>
            </div>
            <p className="text-muted mt-2">Agents are collaborating...</p>
          </div>
        )}
      </div>

      {/* Input Area */}
      <div className="p-3 border-top bg-light-subtle">
        <form onSubmit={handleRunCrew}>
          <div className="row g-2">
            <div className="col-md-3">
              <select
                className="form-select"
                value={workflowType}
                onChange={e => setWorkflowType(e.target.value)}
                disabled={loading}
              >
                <option value="debate">Debate Team</option>
                <option value="research">Research Team</option>
                {availableWorkflows
                  .filter(w => !['debate', 'research'].includes(w.name))
                  .map(w => (
                    <option key={w.name} value={w.name}>{w.name}</option>
                  ))
                }
              </select>
            </div>
            <div className="col-md-7">
              <input
                type="text"
                className="form-control"
                placeholder="Enter topic (e.g., 'Is remote work better than office work?')"
                value={topic}
                onChange={e => setTopic(e.target.value)}
                disabled={loading}
              />
            </div>
            <div className="col-md-2">
              <button type="submit" className="btn btn-primary w-100" disabled={loading || !topic.trim()}>
                {loading ? 'Running...' : 'Start Agents'}
              </button>
            </div>
          </div>
          <div className="row g-2 mt-2">
            <div className="col-auto d-flex align-items-center">
              <small className="me-2">Iterations:</small>
              <input
                type="number"
                className="form-control form-control-sm"
                style={{ width: '70px' }}
                value={maxIterations}
                onChange={e => setMaxIterations(e.target.value)}
                min="1" max="10"
              />
            </div>
            <div className="col-auto d-flex align-items-center ms-3">
              <small className="me-2">Temp:</small>
              <input
                type="range"
                className="form-range"
                style={{ width: '100px' }}
                min="0" max="1" step="0.1"
                value={temperature}
                onChange={e => setTemperature(e.target.value)}
              />
              <small className="ms-1">{temperature}</small>
            </div>
          </div>
        </form>
      </div>
    </div>
  )
}
