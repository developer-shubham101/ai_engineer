import React, { useState, useEffect, useRef } from 'react'
import ReactMarkdown from 'react-markdown'
import { BASE_API_URL } from '../utility/const.js'
import { getStoredToken } from '../utility/auth.js'

export default function AgentChat({ onLogout, onExit, selectedConversationId }) {
  const [orchestratorType, setOrchestratorType] = useState('custom')
  const [query, setQuery] = useState('')
  const [messages, setMessages] = useState([])
  const [loading, setLoading] = useState(false)
  const [loadingHistory, setLoadingHistory] = useState(false)
  const [conversationId, setConversationId] = useState(selectedConversationId || null)
  const [maxSteps, setMaxSteps] = useState(5)
  const [temperature, setTemperature] = useState(0.1)
  const [debug, setDebug] = useState(false)
  const [autogenMeta, setAutogenMeta] = useState({ workflows: [], tools: [] })
  const [selectedWorkflow, setSelectedWorkflow] = useState('')
  const [selectedTools, setSelectedTools] = useState([])
  const messagesRef = useRef(null)

  useEffect(() => {
    if (orchestratorType !== 'autogen') return
    fetch(`${BASE_API_URL}/api/agents/autogen/workflows`, {
      headers: getHeaders(),
    })
      .then((r) => r.json())
      .then((data) => {
        setAutogenMeta(data);
        setSelectedWorkflow(data.workflows?.[0] || "");
        setSelectedTools([]);
      })
      .catch(console.error);
  }, [orchestratorType])

  const getHeaders = () => {
    const token = getStoredToken()
    const headers = { 'Content-Type': 'application/json' }
    if (token) headers['Authorization'] = `Bearer ${token}`
    return headers
  }

  useEffect(() => {
    if (messagesRef.current) messagesRef.current.scrollTop = messagesRef.current.scrollHeight
  }, [messages])

  // Load messages when conversationId changes (sidebar selection or new session)
  useEffect(() => {
    if (conversationId) loadConversationMessages(conversationId)
  }, [conversationId])

  // Sync when parent passes a new selectedConversationId (sidebar click)
  useEffect(() => {
    if (selectedConversationId && selectedConversationId !== conversationId) {
      setConversationId(selectedConversationId)
    }
  }, [selectedConversationId])

  async function loadConversationMessages(convId) {
    setLoadingHistory(true)
    try {
      const res = await fetch(`${BASE_API_URL}/api/conversations/${convId}/messages?limit=100`, {
        headers: getHeaders()
      })
      if (!res.ok) throw new Error('Failed to load agent messages')
      const data = await res.json()
      // API returns { messages: [...] } or a flat array
      const msgs = Array.isArray(data) ? data : (data.messages || [])
      const transformed = msgs.map(msg => ({
        role: msg.speaker === 'user' ? 'user' : 'assistant',
        content: msg.content,
        steps: msg.steps || [],
        tools_used: msg.tools_used || [],
        debug_info: msg.processing_time_ms ? { processing_time_ms: msg.processing_time_ms } : null,
        ts: new Date(msg.created_at).toLocaleString()
      }))
      setMessages(transformed)
    } catch (err) {
      console.error('Failed to load agent conversation messages:', err)
    } finally {
      setLoadingHistory(false)
    }
  }

  async function handleSend(e) {
    if (e) e.preventDefault()
    if (!query.trim()) return

    setLoading(true)
    const currentQuery = query
    setMessages(prev => [...prev, { role: 'user', content: currentQuery, ts: new Date().toLocaleString() }])
    setQuery('')

    try {
      const payload = {
        question: currentQuery,
        tools: orchestratorType === 'autogen' ? selectedTools : [],
        max_steps: maxSteps,
        temperature,
        orchestrator_type: orchestratorType,
        debug,
        ...(orchestratorType === 'autogen' && selectedWorkflow && { workflow: selectedWorkflow }),
        ...(conversationId && { conversation_id: conversationId })
      }

      const res = await fetch(`${BASE_API_URL}/api/agents/query`, {
        method: 'POST',
        headers: getHeaders(),
        body: JSON.stringify(payload)
      })

      if (!res.ok) throw new Error(`Agent API failed: ${res.status}`)

      const data = await res.json()

      if (data.conversation_id) {
        setConversationId(data.conversation_id)
      }

      setMessages(prev => [...prev, {
        role: 'assistant',
        content: data.answer,
        steps: data.steps || [],
        tools_used: data.tools_used || [],
        debug_info: data.debug_info || null,
        ts: new Date().toLocaleString()
      }])
    } catch (err) {
      setMessages(prev => [...prev, {
        role: 'error',
        content: `Error: ${err.message}`,
        ts: new Date().toLocaleString()
      }])
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="h-100 d-flex flex-column">
      {/* Header */}
      <div className="p-3 border-bottom d-flex justify-content-between align-items-center">
        <div className="d-flex align-items-center gap-2">
          <button className="btn btn-sm btn-outline-secondary" onClick={onExit}>
            <i className="bi bi-arrow-left"></i>
          </button>
          <h5 className="mb-0">
            <i className="bi bi-cpu me-2"></i>AI Agent
          </h5>
          {conversationId && (
            <span className="badge bg-secondary small font-monospace" title={conversationId}>
              {conversationId.slice(0, 16)}…
            </span>
          )}
        </div>
        <div className="d-flex align-items-center gap-2">
          <select
            className="form-select form-select-sm"
            style={{ width: 'auto' }}
            value={orchestratorType}
            onChange={e => setOrchestratorType(e.target.value)}
            disabled={loading}
          >
            <option value="custom">Custom</option>
            <option value="autogen">AutoGen</option>
          </select>
          {orchestratorType === 'autogen' && (
            <>
              <select
                className="form-select form-select-sm"
                style={{ width: 'auto' }}
                value={selectedWorkflow}
                onChange={e => setSelectedWorkflow(e.target.value)}
                disabled={loading}
              >
                {autogenMeta.workflows.map(w => <option key={w} value={w}>{w}</option>)}
              </select>
              <div className="dropdown">
                <button
                  className="btn btn-sm btn-outline-secondary dropdown-toggle"
                  type="button"
                  data-bs-toggle="dropdown"
                  disabled={loading}
                >
                  Tools {selectedTools.length > 0 && `(${selectedTools.length})`}
                </button>
                <ul className="dropdown-menu p-2" style={{ minWidth: 180 }}>
                  {autogenMeta.tools.map(t => (
                    <li key={t}>
                      <label className="dropdown-item d-flex align-items-center gap-2" style={{ cursor: 'pointer' }}>
                        <input
                          type="checkbox"
                          checked={selectedTools.includes(t)}
                          onChange={e => setSelectedTools(prev =>
                            e.target.checked ? [...prev, t] : prev.filter(x => x !== t)
                          )}
                        />
                        {t}
                      </label>
                    </li>
                  ))}
                </ul>
              </div>
            </>
          )}
          <div className="input-group input-group-sm" style={{ width: 110 }}>
            <span className="input-group-text">Steps</span>
            <input
              type="number" className="form-control" min={1} max={20}
              value={maxSteps} onChange={e => setMaxSteps(Number(e.target.value))}
              disabled={loading}
            />
          </div>
          <div className="input-group input-group-sm" style={{ width: 110 }}>
            <span className="input-group-text">Temp</span>
            <input
              type="number" className="form-control" min={0} max={1} step={0.1}
              value={temperature} onChange={e => setTemperature(parseFloat(e.target.value))}
              disabled={loading}
            />
          </div>
          <div className="form-check form-switch mb-0 d-flex align-items-center gap-1">
            <input
              className="form-check-input" type="checkbox"
              checked={debug} onChange={e => setDebug(e.target.checked)}
              id="debugToggle"
            />
            <label className="form-check-label small" htmlFor="debugToggle">Debug</label>
          </div>
          <button className="btn btn-sm btn-outline-danger" onClick={onLogout}>Logout</button>
        </div>
      </div>

      {/* Messages */}
      <div ref={messagesRef} className="flex-grow-1 overflow-auto p-3">
        {loadingHistory && (
          <div className="text-center my-3">
            <div className="spinner-border spinner-border-sm text-secondary me-2"></div>
            <span className="text-muted small">Loading history...</span>
          </div>
        )}

        {!loadingHistory && messages.length === 0 && (
          <div className="text-center text-muted mt-5">
            <i className="bi bi-robot display-1"></i>
            <p className="lead mt-3">Ask the agent anything. It will use tools to answer.</p>
          </div>
        )}

        {messages.map((msg, idx) => (
          <div key={idx} className={`mb-3 card ${msg.role === 'user' ? 'border-primary' : msg.role === 'error' ? 'border-danger' : ''}`}>
            <div className={`card-header d-flex justify-content-between align-items-center py-1 ${msg.role === 'user' ? 'bg-primary-subtle' : ''}`}>
              <strong className="small">{msg.role === 'user' ? 'You' : msg.role === 'error' ? 'Error' : 'Agent'}</strong>
              <div className="d-flex align-items-center gap-2">
                {msg.tools_used?.length > 0 && (
                  <span className="badge bg-info text-dark small">{msg.tools_used.join(', ')}</span>
                )}
                {msg.debug_info && (
                  <span className="badge bg-secondary small">
                    {msg.debug_info.processing_time_ms}ms · {msg.debug_info.actual_steps} steps
                  </span>
                )}
                <small className="text-muted">{msg.ts}</small>
              </div>
            </div>
            <div className="card-body py-2">
              {msg.role === 'error' ? (
                <div className="text-danger small">{msg.content}</div>
              ) : (
                <>
                  {msg.steps?.length > 0 && (
                    <details className="mb-2">
                      <summary className="small text-muted" style={{ cursor: 'pointer' }}>
                        Reasoning steps ({msg.steps.length})
                      </summary>
                      <div className="mt-2">
                        {msg.steps.map((step, i) => (
                          <div key={i} className="small border rounded p-2 mb-1 font-monospace">
                            <span className="badge bg-secondary me-1">Step {step.step}</span>
                            <strong>{step.tool}</strong>
                            {step.input && <div className="text-muted">Input: {step.input}</div>}
                            <div>Result: {typeof step.result === 'object' ? JSON.stringify(step.result) : step.result}</div>
                          </div>
                        ))}
                      </div>
                    </details>
                  )}
                  {msg.role === 'assistant'
                    ? <ReactMarkdown>{msg.content}</ReactMarkdown>
                    : <div>{msg.content}</div>
                  }
                </>
              )}
            </div>
          </div>
        ))}

        {loading && (
          <div className="text-center my-3">
            <div className="spinner-border spinner-border-sm text-primary me-2"></div>
            <span className="text-muted small">Agent is thinking...</span>
          </div>
        )}
      </div>

      {/* Input */}
      <div className="p-3 border-top">
        <form onSubmit={handleSend}>
          <div className="input-group">
            <input
              type="text"
              className="form-control"
              placeholder="Ask the agent..."
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
