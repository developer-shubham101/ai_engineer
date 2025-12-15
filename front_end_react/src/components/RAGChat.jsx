import React, { useEffect, useRef, useState } from 'react'
import AddJsonForm from './AddJsonForm.jsx'
import UploadFileForm from './UploadFileForm.jsx'
import UpdateMetadataForm from './UpdateMetadataForm.jsx'
import DocumentVersionModal from './DocumentVersionModal.jsx'
import PersonalizedTestModal from './PersonalizedTestModal.jsx'
import ConversationSidebar from './ConversationSidebar.jsx'
import ConversationMessageDetail from './ConversationMessageDetail.jsx'
import ToastList from './ToastList.jsx'
import { BASE_API_URL } from '../utility/const.js'
import { getStoredToken, getStoredUser, getSessionIdFromToken, clearAuth } from '../utility/auth.js'


export default function RAGChat({ onLogout }) {
  const [token, setToken] = useState(getStoredToken())
  const [user, setUser] = useState(getStoredUser())
  const [sessionId, setSessionId] = useState(() => {
    const storedToken = getStoredToken();
    return storedToken ? getSessionIdFromToken(storedToken) : '';
  })

  // Conversation state
  const [conversations, setConversations] = useState([])
  const [activeConversationId, setActiveConversationId] = useState(null)
  const [sidebarOpen, setSidebarOpen] = useState(true)

  // Existing state
  const [useLlm, setUseLlm] = useState(false)
  const [useDocuments, setUseDocuments] = useState(true)
  const [topK, setTopK] = useState(3)
  const [temperature, setTemperature] = useState(() => parseFloat(localStorage.getItem('temperature')) || 0.1)
  const [composer, setComposer] = useState('')
  const [messages, setMessages] = useState([])
  const [inFlight, setInFlight] = useState(false)
  const [selectedFile, setSelectedFile] = useState(null)
  const [fileMeta, setFileMeta] = useState({ department: '', sensitivity: 'public_internal', tags: '' })
  const [toasts, setToasts] = useState([])
  const [modelProvider, setModelProvider] = useState(localStorage.getItem('model_provider') || 'local')
  const [localLlmModel, setLocalLlmModel] = useState(localStorage.getItem('local_llm_model') || 'llama32-1b')
  const [theme, setTheme] = useState(localStorage.getItem('theme') || 'system')
  const [showAdminPanel, setShowAdminPanel] = useState(false)
  const messagesRef = useRef(null)

  // Load conversations on mount
  useEffect(() => {
    if (token) {
      loadConversations()
    }
  }, [token])

  // Auto-scroll messages
  useEffect(() => {
    if (messagesRef.current) messagesRef.current.scrollTop = messagesRef.current.scrollHeight
  }, [messages])

  // Persist settings
  useEffect(() => { localStorage.setItem('model_provider', modelProvider) }, [modelProvider])
  useEffect(() => { localStorage.setItem('local_llm_model', localLlmModel) }, [localLlmModel])
  useEffect(() => { localStorage.setItem('temperature', temperature.toString()) }, [temperature])
  useEffect(() => {
    localStorage.setItem('theme', theme)
    if (theme === 'system') {
      const systemIsDark = window.matchMedia('(prefers-color-scheme: dark)').matches
      document.documentElement.setAttribute('data-bs-theme', systemIsDark ? 'dark' : 'light')
    } else {
      document.documentElement.setAttribute('data-bs-theme', theme)
    }
  }, [theme])

  useEffect(() => {
    if (theme === 'system') {
      const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)')
      const handleChange = () => {
        document.documentElement.setAttribute('data-bs-theme', mediaQuery.matches ? 'dark' : 'light')
      }
      mediaQuery.addEventListener('change', handleChange)
      return () => mediaQuery.removeEventListener('change', handleChange)
    }
  }, [theme])

  function addToast(text, kind = 'danger', title = '') {
    const id = Date.now();
    setToasts(t => [...t, { id, text, kind, title }]);
    setTimeout(() => setToasts(t => t.filter(x => x.id !== id)), 5000)
  }

  function handleLogout() {
    clearAuth();
    setMessages([]);
    setConversations([]);
    setActiveConversationId(null);
    if (onLogout) onLogout();
  }

  // ========== Conversation API Functions ==========

  async function loadConversations() {
    if (!token) return
    try {
      const res = await fetch(`${BASE_API_URL}/api/conversations?limit=50&offset=0`, {
        headers: { 'Authorization': `Bearer ${token}` }
      })
      if (!res.ok) throw new Error('Failed to load conversations')
      const data = await res.json()
      setConversations(data)

      // If no active conversation, create one or select the first
      if (!activeConversationId && data.length > 0) {
        setActiveConversationId(data[0].id)
        await loadConversationMessages(data[0].id)
      } else if (!activeConversationId && data.length === 0) {
        await createNewConversation()
      }
    } catch (err) {
      console.error('Load conversations error:', err)
      addToast(err.message || 'Failed to load conversations', 'danger')
    }
  }

  async function createNewConversation(title = 'New Conversation') {
    if (!token) return
    try {
      const res = await fetch(`${BASE_API_URL}/api/conversations`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({ title })
      })
      if (!res.ok) throw new Error('Failed to create conversation')
      const newConv = await res.json()
      setConversations(prev => [newConv, ...prev])
      setActiveConversationId(newConv.id)
      setMessages([])
      addToast('New conversation created', 'success')
    } catch (err) {
      console.error('Create conversation error:', err)
      addToast(err.message || 'Failed to create conversation', 'danger')
    }
  }

  async function loadConversationMessages(conversationId) {
    if (!token) return
    try {
      const res = await fetch(`${BASE_API_URL}/api/conversations/${conversationId}/messages?limit=100`, {
        headers: { 'Authorization': `Bearer ${token}` }
      })
      if (!res.ok) throw new Error('Failed to load messages')
      const data = await res.json()

      // Transform backend messages to frontend format
      const transformedMessages = data.map(msg => ({
        role: msg.speaker === 'user' ? 'user' : 'bot',
        text: msg.speaker === 'user' ? msg.content : undefined,
        answer: msg.speaker === 'assistant' ? msg.content : undefined,
        ts: new Date(msg.created_at).toLocaleString(),
        // Include RAG logging data
        ...msg
      }))

      setMessages(transformedMessages)
    } catch (err) {
      console.error('Load messages error:', err)
      addToast(err.message || 'Failed to load messages', 'danger')
    }
  }

  async function switchConversation(conversationId) {
    setActiveConversationId(conversationId)
    await loadConversationMessages(conversationId)
  }

  async function renameConversation(conversationId, newTitle) {
    if (!token) return
    try {
      const res = await fetch(`${BASE_API_URL}/api/conversations/${conversationId}`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({ title: newTitle })
      })
      if (!res.ok) throw new Error('Failed to rename conversation')

      setConversations(prev => prev.map(c =>
        c.id === conversationId ? { ...c, title: newTitle } : c
      ))
      addToast('Conversation renamed', 'success')
    } catch (err) {
      console.error('Rename conversation error:', err)
      addToast(err.message || 'Failed to rename conversation', 'danger')
    }
  }

  async function deleteConversation(conversationId) {
    if (!token) return
    try {
      const res = await fetch(`${BASE_API_URL}/api/conversations/${conversationId}`, {
        method: 'DELETE',
        headers: { 'Authorization': `Bearer ${token}` }
      })
      if (!res.ok) throw new Error('Failed to delete conversation')

      setConversations(prev => prev.filter(c => c.id !== conversationId))

      // If deleting active conversation, switch to another or create new
      if (conversationId === activeConversationId) {
        const remaining = conversations.filter(c => c.id !== conversationId)
        if (remaining.length > 0) {
          await switchConversation(remaining[0].id)
        } else {
          await createNewConversation()
        }
      }

      addToast('Conversation deleted', 'success')
    } catch (err) {
      console.error('Delete conversation error:', err)
      addToast(err.message || 'Failed to delete conversation', 'danger')
    }
  }

  // ========== Query and Document Functions ==========

  async function sendQuery() {
    if (inFlight) return
    if (!composer.trim()) return addToast('Please enter a question', 'warning', 'Validation')

    // Create conversation if none exists
    if (!activeConversationId) {
      await createNewConversation()
      // Wait a bit for conversation to be created
      await new Promise(resolve => setTimeout(resolve, 500))
    }

    setInFlight(true)
    const userMsg = { role: 'user', text: composer, ts: new Date().toLocaleString() }
    setMessages(m => [...m, userMsg])
    setComposer('')

    const payload = {
      question: composer,
      top_k: Number(topK || 3),
      use_llm: !!useLlm,
      use_documents: !!useDocuments,
      temperature: temperature,
      conversation_id: activeConversationId // Include conversation ID
    }
    if (modelProvider === 'local' && localLlmModel) {
      payload.local_llm_model = localLlmModel
    }

    try {
      const headers = { "Content-Type": "application/json" };
      if (token) {
        headers["Authorization"] = `Bearer ${token}`;
        if (sessionId) headers['X-Session-ID'] = sessionId;
      }
      const res = await fetch(BASE_API_URL + `/api/rag/${modelProvider}/query`, {
        method: "POST",
        headers: headers,
        body: JSON.stringify(payload),
      });
      if (!res.ok) {
        const txt = await res.text().catch(() => res.statusText);
        throw new Error(`Server error: ${res.status} - ${txt}`)
      }
      const data = await res.json()
      const botMsg = {
        role: 'bot',
        answer: data.answer || 'No answer returned',
        retrieved: data.retrieved || [],
        context: data.context || '',
        filtered_out_count: data.filtered_out_count || 0,
        public_summaries: data.public_summaries || [],
        filtered_details: data.filtered_details || null,
        ts: new Date().toLocaleString()
      }
      setMessages(m => [...m, botMsg])

      // Reload conversations to update message count
      await loadConversations()

      if (botMsg.filtered_out_count > 0) {
        addToast(`${botMsg.filtered_out_count} results filtered for your role`, 'warning', 'Filtered')
      }
    } catch (err) {
      addToast(err.message || 'Network error', 'danger', 'Error')
      setMessages(m => [...m, {
        role: 'bot',
        answer: 'Error: ' + (err.message || 'Network error'),
        ts: new Date().toLocaleString()
      }])
    } finally {
      setTimeout(() => setInFlight(false), 2000)
    }
  }

  async function postAddJson({ source_name, text, metadata }) {
    const headers = { 'Content-Type': 'application/json' };
    if (token) {
      headers['Authorization'] = `Bearer ${token}`;
      if (sessionId) headers['X-Session-ID'] = sessionId;
    }
    try {
      const res = await fetch(BASE_API_URL + `/api/rag/documents/add`, {
        method: 'POST',
        headers: headers,
        body: JSON.stringify({ source_name, text, metadata })
      })
      if (!res.ok) {
        const txt = await res.text().catch(() => res.statusText);
        throw new Error(txt || 'Server error')
      }
      const data = await res.json();
      addToast('Document added', 'success');
      return data
    } catch (err) {
      addToast(err.message || 'Add error', 'danger');
      throw err
    }
  }

  async function postUploadFile(formData) {
    const headers = {};
    if (token) {
      headers['Authorization'] = `Bearer ${token}`;
      if (sessionId) headers['X-Session-ID'] = sessionId;
    }
    try {
      const res = await fetch(BASE_API_URL + `/api/rag/documents/add-file`, {
        method: 'POST',
        headers: headers,
        body: formData
      })
      if (!res.ok) {
        const txt = await res.text().catch(() => res.statusText);
        throw new Error(txt || 'Upload error')
      }
      const data = await res.json();
      addToast('File uploaded', 'success');
      return data
    } catch (err) {
      addToast(err.message || 'Upload error', 'danger');
      throw err
    }
  }

  async function requestAccess({ document_id, source_name, reason }) {
    if (!token) return addToast('Please login first', 'warning', 'Auth')
    const headers = { 'Content-Type': 'application/json', 'Authorization': `Bearer ${token}` };
    if (sessionId) headers['X-Session-ID'] = sessionId;
    try {
      const res = await fetch(BASE_API_URL + '/request-access', {
        method: 'POST',
        headers: headers,
        body: JSON.stringify({ document_id, source_name, reason })
      })
      if (!res.ok) {
        const txt = await res.text().catch(() => res.statusText);
        throw new Error(txt || 'Request error')
      }
      const data = await res.json();
      addToast('Access request submitted', 'success');
      return data
    } catch (err) {
      addToast(err.message || 'Request error', 'danger');
      throw err
    }
  }

  async function fetchAccessRequests() {
    if (!token) return addToast('Please login first', 'warning', 'Auth')
    const headers = { 'Authorization': `Bearer ${token}` };
    if (sessionId) headers['X-Session-ID'] = sessionId;
    try {
      const res = await fetch(BASE_API_URL + '/access-requests', { headers: headers })
      if (!res.ok) throw new Error('Failed to fetch');
      const data = await res.json();
      return data
    } catch (err) {
      addToast(err.message || 'Fetch error', 'danger');
      return []
    }
  }

  async function listDocuments(filters = {}) {
    if (!token) return addToast('Please login first', 'warning', 'Auth')
    const headers = { 'Authorization': `Bearer ${token}` };
    if (sessionId) headers['X-Session-ID'] = sessionId;
    const params = new URLSearchParams(filters).toString();
    try {
      const res = await fetch(BASE_API_URL + `/api/rag/documents/list${params ? '?' + params : ''}`, {
        headers: headers
      })
      if (!res.ok) throw new Error('Failed to fetch documents');
      const data = await res.json();
      return data
    } catch (err) {
      addToast(err.message || 'List error', 'danger');
      return []
    }
  }

  async function getDocumentVersions(documentId) {
    if (!token) return addToast('Please login first', 'warning', 'Auth')
    const headers = { 'Authorization': `Bearer ${token}` };
    if (sessionId) headers['X-Session-ID'] = sessionId;
    try {
      const res = await fetch(BASE_API_URL + `/api/rag/documents/${documentId}/versions`, {
        headers: headers
      })
      if (!res.ok) throw new Error('Failed to fetch versions');
      const data = await res.json();
      return data
    } catch (err) {
      addToast(err.message || 'Versions error', 'danger');
      return []
    }
  }

  async function updateDocument({ document_id, text, version_notes, status = 'published' }) {
    if (!token) return addToast('Please login first', 'warning', 'Auth')
    const headers = { 'Content-Type': 'application/json', 'Authorization': `Bearer ${token}` };
    if (sessionId) headers['X-Session-ID'] = sessionId;
    try {
      const res = await fetch(BASE_API_URL + `/api/rag/documents/update`, {
        method: 'POST',
        headers: headers,
        body: JSON.stringify({ document_id, text, version_notes, status })
      })
      if (!res.ok) {
        const txt = await res.text().catch(() => res.statusText);
        throw new Error(txt || 'Update error')
      }
      const data = await res.json();
      addToast('Document updated', 'success');
      return data
    } catch (err) {
      addToast(err.message || 'Update error', 'danger');
      throw err
    }
  }

  async function seedDocuments() {
    if (!token) return addToast('Please login first', 'warning', 'Auth')
    const headers = { 'Authorization': `Bearer ${token}` };
    if (sessionId) headers['X-Session-ID'] = sessionId;
    try {
      const res = await fetch(BASE_API_URL + `/api/rag/documents/seed`, {
        method: 'POST',
        headers: headers
      })
      if (!res.ok) {
        const txt = await res.text().catch(() => res.statusText);
        throw new Error(txt || 'Seed error')
      }
      const data = await res.json();
      addToast('Documents seeded', 'success');
      return data
    } catch (err) {
      addToast(err.message || 'Seed error', 'danger');
      throw err
    }
  }

  async function compareDocumentVersions(documentId, version1, version2) {
    if (!token) return addToast('Please login first', 'warning', 'Auth')
    const headers = { 'Authorization': `Bearer ${token}` };
    if (sessionId) headers['X-Session-ID'] = sessionId;
    const params = new URLSearchParams({ version1, version2 }).toString();
    try {
      const res = await fetch(BASE_API_URL + `/api/rag/documents/${documentId}/compare?${params}`, {
        headers: headers
      })
      if (!res.ok) throw new Error('Failed to compare versions');
      const data = await res.json();
      return data
    } catch (err) {
      addToast(err.message || 'Compare error', 'danger');
      return null
    }
  }

  async function archiveDocumentVersion(documentId, version) {
    if (!token) return addToast('Please login first', 'warning', 'Auth')
    const headers = { 'Authorization': `Bearer ${token}` };
    if (sessionId) headers['X-Session-ID'] = sessionId;
    const params = new URLSearchParams({ version }).toString();
    try {
      const res = await fetch(BASE_API_URL + `/api/rag/documents/${documentId}/archive?${params}`, {
        method: 'POST',
        headers: headers
      })
      if (!res.ok) {
        const txt = await res.text().catch(() => res.statusText);
        throw new Error(txt || 'Archive error')
      }
      const data = await res.json();
      addToast('Version archived', 'success');
      return data
    } catch (err) {
      addToast(err.message || 'Archive error', 'danger');
      throw err
    }
  }

  async function testRBACAccess() {
    if (!token) return addToast('Please login first', 'warning', 'Auth')
    try {
      const testDoc = {
        source_name: 'rbac_test.md',
        text: 'This is a test document for RBAC validation',
        metadata: {
          department: 'Engineering',
          sensitivity: 'highly_confidential'
        }
      };
      const result = await postAddJson(testDoc);
      addToast('RBAC Test: Document created successfully', 'success');
      return result;
    } catch (err) {
      if (err.message.includes('403') || err.message.includes('forbidden')) {
        addToast('RBAC Test: Access denied (expected for non-admin users)', 'warning');
      } else {
        addToast('RBAC Test: ' + err.message, 'danger');
      }
      return null;
    }
  }

  function truncate(text, n = 200) {
    if (!text) return '';
    return text.length <= n ? text : text.slice(0, n) + '...'
  }

  // Get active conversation title
  const activeConversation = conversations.find(c => c.id === activeConversationId)
  const conversationTitle = activeConversation?.title || 'New Conversation'

  return (
    <div className="chat-container">
      {/* Conversation Sidebar */}
      <ConversationSidebar
        conversations={conversations}
        activeConversationId={activeConversationId}
        onSelectConversation={switchConversation}
        onCreateConversation={() => createNewConversation()}
        onRenameConversation={renameConversation}
        onDeleteConversation={deleteConversation}
        isOpen={sidebarOpen}
        onToggle={() => setSidebarOpen(!sidebarOpen)}
      />

      {/* Main Chat Area */}
      <div className="chat-main-content">
        <div className="row g-3 h-100">
          <div className="col-md-9">
            <div className="card h-100">
              <div className="card-body d-flex flex-column">
                <div className="d-flex justify-content-between align-items-center mb-2">
                  <div className="d-flex align-items-center gap-2">
                    <button
                      className="btn btn-sm btn-outline-secondary"
                      onClick={() => setSidebarOpen(!sidebarOpen)}
                      title="Toggle sidebar"
                    >
                      <i className="bi bi-list"></i>
                    </button>
                    <h5 className="mb-0">{conversationTitle}</h5>
                  </div>
                  <small className="text-muted">Session: {sessionId || 'none'}</small>
                </div>

                <div
                  ref={messagesRef}
                  style={{
                    height: "73vh",
                    overflowY: "auto",
                    padding: 12,
                    borderRadius: 8,
                  }}
                >
                  {messages.map((m, idx) => (
                    <div
                      key={idx}
                      className={`d-flex mb-3 ${m.role === "user"
                        ? "justify-content-end"
                        : "justify-content-start"
                        }`}
                    >
                      <div
                        className={`p-3 rounded ${m.role === "user" ? "text-white" : ""
                          }`}
                        style={{
                          background: m.role === "user" ? "#0d6efd" : "#2d702fff",
                          maxWidth: "75%",
                        }}
                      >
                        <div className="small text-muted mb-1">
                          {m.role === "user" ? "You" : "Assistant"} • {m.ts}
                        </div>
                        <div style={{ whiteSpace: "pre-wrap" }}>
                          {m.role === "user" ? m.text : m.answer || m.text}
                        </div>

                        {/* RAG Debug Info */}
                        <ConversationMessageDetail message={m} />

                        {m.role === "bot" && m.context && (
                          <div className="small text-muted mt-2">
                            Context: {truncate(m.context, 300)}
                          </div>
                        )}
                        {m.role === "bot" &&
                          Array.isArray(m.retrieved) &&
                          m.retrieved.length > 0 && (
                            <details className="mt-2">
                              <summary>Retrieved ({m.retrieved.length})</summary>
                              <div className="mt-2">
                                {m.retrieved.map((r, i) => (
                                  <div
                                    key={i}
                                    className="p-2 mb-2"
                                    style={{
                                      borderLeft: "3px solid #e9ecef",
                                      background: "#2d702fff",
                                    }}
                                  >
                                    <div className="small text-muted">
                                      Source:{" "}
                                      {r.source ||
                                        (r.metadata && r.metadata.department) ||
                                        "unknown"}
                                    </div>
                                    <div style={{ whiteSpace: "pre-wrap" }}>
                                      {truncate(r.text || r.preview || "", 600)}
                                    </div>
                                    <div className="small text-muted mt-1">
                                      Metadata: {JSON.stringify(r.metadata || {})}
                                    </div>
                                    {r.metadata &&
                                      (r.metadata.sensitivity === "restricted" ||
                                        r.metadata.sensitivity ===
                                        "confidential") && (
                                        <button
                                          className="btn btn-sm btn-outline-primary mt-2"
                                          onClick={() =>
                                            requestAccess({
                                              document_id: r.id,
                                              source_name: r.source,
                                              reason: "Need access via UI",
                                            })
                                          }
                                        >
                                          Request Access
                                        </button>
                                      )}
                                  </div>
                                ))}
                              </div>
                            </details>
                          )}
                        {m.role === "bot" && m.filtered_out_count > 0 && (
                          <div className="mt-2">
                            <button
                              className="btn btn-sm btn-warning"
                              onClick={() => {
                                if (
                                  Array.isArray(m.public_summaries) &&
                                  m.public_summaries.length
                                ) {
                                  addToast(
                                    "Public summaries logged to console",
                                    "info"
                                  );
                                  console.log(
                                    "Public summaries:",
                                    m.public_summaries
                                  );
                                } else addToast("No public summaries", "warning");
                              }}
                            >
                              {m.filtered_out_count} results were filtered — show
                              public summaries
                            </button>
                          </div>
                        )}
                      </div>
                    </div>
                  ))}
                </div>

                <div className="mt-3">
                  <div className="d-flex gap-2">
                    <textarea
                      className="form-control"
                      value={composer}
                      onChange={(e) => setComposer(e.target.value)}
                      placeholder="Type your question..."
                      rows={2}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter' && !e.shiftKey) {
                          e.preventDefault()
                          sendQuery()
                        }
                      }}
                    />
                    <div className="d-flex flex-column align-items-end">
                      <div className="mb-2">
                        <button
                          className="btn btn-primary"
                          disabled={inFlight}
                          onClick={sendQuery}
                        >
                          {inFlight ? "Sending..." : "Send"}
                        </button>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Controls Panel */}
          <div className="col-md-3" style={{ height: '100vh', overflowY: 'auto' }}>
            <div className="card mb-3">
              <div className="card-header d-flex justify-content-between">
                Controls
                <button
                  className="btn btn-sm btn-outline-secondary"
                  onClick={() => setShowAdminPanel(!showAdminPanel)}
                >
                  {showAdminPanel ? 'Hide' : 'Show'} Admin
                </button>
              </div>
              <div className="card-body">
                <div className="d-flex flex-column gap-2">
                  {user && (
                    <div className="badge bg-primary">
                      {user.username} ({user.role})
                    </div>
                  )}
                  <div className="form-check form-switch">
                    <input
                      className="form-check-input"
                      type="checkbox"
                      checked={useLlm}
                      onChange={(e) => setUseLlm(e.target.checked)}
                    />
                    <label className="form-check-label">Use LLM</label>
                  </div>
                  <div className="form-check form-switch">
                    <input
                      className="form-check-input"
                      type="checkbox"
                      checked={useDocuments}
                      onChange={(e) => setUseDocuments(e.target.checked)}
                    />
                    <label className="form-check-label">Use Docs</label>
                  </div>
                  <div className="input-group input-group-sm">
                    <span className="input-group-text">Top K</span>
                    <input
                      type="number"
                      className="form-control"
                      min={1}
                      max={20}
                      value={topK}
                      onChange={(e) => setTopK(Number(e.target.value))}
                    />
                  </div>
                  <div className="input-group input-group-sm" title="Temperature controls randomness. Low = accurate and predictable. High = creative and unpredictable.">
                    <span className="input-group-text">Temp ({temperature.toFixed(1)})</span>
                    <input
                      type="range"
                      className="form-range"
                      min="0.0"
                      max="1.0"
                      step="0.1"
                      value={temperature}
                      onChange={(e) => setTemperature(parseFloat(e.target.value))}
                    />
                  </div>
                  <select
                    className="form-select form-select-sm"
                    value={modelProvider}
                    onChange={(e) => setModelProvider(e.target.value)}
                  >
                    <option value="local">Local</option>
                    <option value="google">Google</option>
                    <option value="gpt">OpenAI GPT</option>
                    <option value="hf">Hugging Face</option>
                  </select>
                  {modelProvider === 'local' && (
                    <select
                      className="form-select form-select-sm"
                      value={localLlmModel}
                      onChange={(e) => setLocalLlmModel(e.target.value)}
                    >
                      <option value="llama32-1b">Llama 3.2 1B</option>
                      <option value="llama32-3b">Llama 3.2 3B</option>
                      <option value="llama31-8b">Llama 3.1 8B</option>
                      <option value="phi3-mini">Phi-3 Mini</option>
                      <option value="gemma2-2b">Gemma 2 2B</option>
                      <option value="mistral-7b">Mistral 7B</option>
                      <option value="distilgpt2-company-tuned">DistilGPT2 Company Tuned</option>
                    </select>
                  )}
                  <select
                    className="form-select form-select-sm"
                    value={theme}
                    onChange={(e) => setTheme(e.target.value)}
                  >
                    <option value="system">System</option>
                    <option value="light">Light</option>
                    <option value="dark">Dark</option>
                  </select>
                  <button
                    className="btn btn-sm btn-outline-danger"
                    onClick={handleLogout}
                  >
                    Logout
                  </button>
                </div>
              </div>
            </div>

            {showAdminPanel && (
              <div>
                <div className="card mb-3">
                  <div className="card-header">Admin Panel</div>
                  <div className="card-body">
                    <div className="d-grid gap-2">
                      <button
                        className="btn btn-outline-primary"
                        onClick={async () => {
                          const list = await listDocuments();
                          console.log("Documents:", list);
                          addToast("Documents listed (console)", "info");
                        }}
                      >
                        List Documents
                      </button>
                      <button
                        className="btn btn-outline-success"
                        onClick={async () => {
                          await seedDocuments();
                        }}
                      >
                        Seed Documents
                      </button>
                      <button
                        className="btn btn-outline-warning"
                        data-bs-toggle="modal"
                        data-bs-target="#updateMetadataModal"
                      >
                        Update Metadata
                      </button>
                      <button
                        className="btn btn-outline-danger"
                        onClick={testRBACAccess}
                      >
                        Test RBAC
                      </button>
                      <button
                        className="btn btn-outline-secondary"
                        data-bs-toggle="modal"
                        data-bs-target="#documentVersionModal"
                      >
                        Version Manager
                      </button>
                      <button
                        className="btn btn-outline-primary"
                        data-bs-toggle="modal"
                        data-bs-target="#personalizedTestModal"
                      >
                        Test Personalization
                      </button>
                    </div>
                    <hr />
                    <div className="small text-muted">
                      Messages: {messages.length}
                    </div>
                    <div className="small text-muted">
                      Conversations: {conversations.length}
                    </div>
                  </div>
                </div>

                <div className="card">
                  <div className="card-header">Add Document</div>
                  <div className="card-body">
                    <button
                      className="btn btn-success w-100 mb-2"
                      data-bs-toggle="modal"
                      data-bs-target="#addJsonModal"
                    >
                      Add JSON Doc
                    </button>
                    <button
                      className="btn btn-info w-100"
                      data-bs-toggle="modal"
                      data-bs-target="#uploadFileModal"
                    >
                      Upload File
                    </button>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>


      {/* Modals */}
      <div
        className="modal fade"
        id="addJsonModal"
        tabIndex={-1}
        aria-hidden="true"
      >
        <div className="modal-dialog modal-lg">
          <div className="modal-content">
            <div className="modal-header">
              <h5 className="modal-title">Add JSON Document</h5>
              <button
                type="button"
                className="btn-close"
                data-bs-dismiss="modal"
                aria-label="Close"
              ></button>
            </div>
            <AddJsonForm onSubmit={postAddJson} />
          </div>
        </div>
      </div>
      <div
        className="modal fade"
        id="uploadFileModal"
        tabIndex={-1}
        aria-hidden="true"
      >
        <div className="modal-dialog">
          <div className="modal-content">
            <div className="modal-header">
              <h5 className="modal-title">Upload File</h5>
              <button
                type="button"
                className="btn-close"
                data-bs-dismiss="modal"
                aria-label="Close"
              ></button>
            </div>
            <UploadFileForm onSubmit={postUploadFile} />
          </div>
        </div>
      </div>
      <div
        className="modal fade"
        id="updateMetadataModal"
        tabIndex={-1}
        aria-hidden="true"
      >
        <div className="modal-dialog modal-lg">
          <div className="modal-content">
            <div className="modal-header">
              <h5 className="modal-title">Update Metadata</h5>
              <button
                type="button"
                className="btn-close"
                data-bs-dismiss="modal"
                aria-label="Close"
              ></button>
            </div>
            <UpdateMetadataForm />
          </div>
        </div>
      </div>
      <div
        className="modal fade"
        id="documentVersionModal"
        tabIndex={-1}
        aria-hidden="true"
      >
        <div className="modal-dialog modal-xl">
          <div className="modal-content">
            <div className="modal-header">
              <h5 className="modal-title">Document Version Manager</h5>
              <button
                type="button"
                className="btn-close"
                data-bs-dismiss="modal"
                aria-label="Close"
              ></button>
            </div>
            <DocumentVersionModal
              onGetVersions={getDocumentVersions}
              onCompareVersions={compareDocumentVersions}
              onArchiveVersion={archiveDocumentVersion}
            />
            <div className="modal-footer">
              <button
                type="button"
                className="btn btn-secondary"
                data-bs-dismiss="modal"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      </div>
      <div
        className="modal fade"
        id="personalizedTestModal"
        tabIndex={-1}
        aria-hidden="true"
      >
        <div className="modal-dialog modal-lg">
          <div className="modal-content">
            <div className="modal-header">
              <h5 className="modal-title">Personalized AI Response Test</h5>
              <button
                type="button"
                className="btn-close"
                data-bs-dismiss="modal"
                aria-label="Close"
              ></button>
            </div>
            <PersonalizedTestModal
              onSendQuery={sendQuery}
              modelProvider={modelProvider}
            />
            <div className="modal-footer">
              <button
                type="button"
                className="btn btn-secondary"
                data-bs-dismiss="modal"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      </div>

      <ToastList toasts={toasts} />
    </div>
  );
}
