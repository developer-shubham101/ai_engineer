import React, { useState, useEffect } from 'react'
import { BASE_API_URL } from '../utility/const.js'
import { getStoredToken, getSessionIdFromToken } from '../utility/auth.js'

const EMPTY_FORM = { name: '', messages: [{ role: 'system', content: '' }, { role: 'user', content: '{user_question}' }], prompt_variables: 'user_question' }

export default function PromptTemplateManager({ onClose }) {
  const [templates, setTemplates] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [editingTemplate, setEditingTemplate] = useState(null)
  const [formData, setFormData] = useState(EMPTY_FORM)
  const [feedback, setFeedback] = useState('')

  const getAuthHeaders = () => {
    const token = getStoredToken();
    if (!token) return {};
    const headers = { 'Authorization': `Bearer ${token}`, 'Content-Type': 'application/json' };
    const sessionId = getSessionIdFromToken(token);
    if (sessionId) headers['X-Session-ID'] = sessionId;
    return headers;
  }

  useEffect(() => { fetchTemplates() }, [])

  const fetchTemplates = async () => {
    setLoading(true)
    setError('')
    try {
      const res = await fetch(`${BASE_API_URL}/api/templates`, { headers: getAuthHeaders() })
      if (!res.ok) throw new Error('Failed to fetch templates')
      setTemplates(await res.json())
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const handleEdit = (tpl) => {
    setEditingTemplate(tpl)
    setFormData({
      name: tpl.name,
      messages: tpl.messages || [{ role: 'system', content: '' }, { role: 'user', content: '{user_question}' }],
      prompt_variables: tpl.prompt_variables || 'user_question'
    })
    setFeedback('')
  }

  const handleCreateNew = () => {
    setEditingTemplate(null)
    setFormData(EMPTY_FORM)
    setFeedback('')
  }

  const handleDelete = async (name) => {
    if (!window.confirm(`Are you sure you want to delete template "${name}"?`)) return
    try {
      const res = await fetch(`${BASE_API_URL}/api/templates/${name}`, { method: 'DELETE', headers: getAuthHeaders() })
      if (!res.ok) throw new Error('Failed to delete')
      setTemplates(prev => prev.filter(t => t.name !== name))
      if (editingTemplate?.name === name) handleCreateNew()
    } catch (err) {
      alert(err.message)
    }
  }

  const updateMessage = (index, field, value) => {
    const updated = formData.messages.map((m, i) => i === index ? { ...m, [field]: value } : m)
    setFormData({ ...formData, messages: updated })
  }

  const addMessage = () => {
    setFormData({ ...formData, messages: [...formData.messages, { role: 'user', content: '' }] })
  }

  const removeMessage = (index) => {
    setFormData({ ...formData, messages: formData.messages.filter((_, i) => i !== index) })
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setFeedback('')
    try {
      const payload = {
        name: formData.name,
        messages: formData.messages,
        prompt_variables: formData.prompt_variables
      }
      let res;
      if (editingTemplate) {
        res = await fetch(`${BASE_API_URL}/api/templates/${editingTemplate.name}`, {
          method: 'PUT',
          headers: getAuthHeaders(),
          body: JSON.stringify({ messages: payload.messages, prompt_variables: payload.prompt_variables })
        })
      } else {
        res = await fetch(`${BASE_API_URL}/api/templates`, {
          method: 'POST',
          headers: getAuthHeaders(),
          body: JSON.stringify(payload)
        })
      }
      if (!res.ok) throw new Error(await res.text() || 'Operation failed')

      if (editingTemplate) {
        setTemplates(prev => prev.map(t => t.name === editingTemplate.name ? { ...t, ...payload } : t))
        setFeedback('Template updated successfully')
      } else {
        setTemplates(prev => [...prev, payload])
        setFeedback('Template created successfully')
        setEditingTemplate(payload)
      }
    } catch (err) {
      setFeedback('Error: ' + err.message)
    }
  }

  return (
    <div className="card h-100">
      <div className="card-header d-flex justify-content-between align-items-center">
        <h5 className="mb-0">Prompt Template Manager</h5>
        <button className="btn btn-sm btn-outline-secondary" onClick={onClose}>Close</button>
      </div>
      <div className="card-body d-flex flex-column h-100" style={{ maxHeight: '70vh' }}>
        <div className="row h-100">
          {/* List Side */}
          <div className="col-md-4 border-end overflow-auto" style={{ maxHeight: '100%' }}>
            <button className="btn btn-primary w-100 mb-3" onClick={handleCreateNew}>
              <i className="bi bi-plus-lg me-2"></i> New Template
            </button>
            {loading && <div className="text-center"><div className="spinner-border spinner-border-sm"></div></div>}
            {error && <div className="alert alert-danger small">{error}</div>}
            <div className="list-group">
              {templates.map(t => (
                <div
                  key={t.name}
                  className={`list-group-item list-group-item-action d-flex justify-content-between align-items-center ${editingTemplate?.name === t.name ? 'active' : ''}`}
                  onClick={() => handleEdit(t)}
                  style={{ cursor: 'pointer' }}
                >
                  <div className="text-truncate" style={{ maxWidth: '80%' }}>{t.name}</div>
                  <button
                    className="btn btn-link btn-sm text-danger p-0"
                    onClick={(e) => { e.stopPropagation(); handleDelete(t.name); }}
                    title="Delete"
                  >
                    <i className="bi bi-trash"></i>
                  </button>
                </div>
              ))}
              {templates.length === 0 && !loading && <div className="text-muted text-center small p-3">No templates found</div>}
            </div>
          </div>

          {/* Editor Side */}
          <div className="col-md-8 overflow-auto" style={{ maxHeight: '100%' }}>
            <form onSubmit={handleSubmit} className="d-flex flex-column gap-3">
              <div>
                <label className="form-label">Template Name</label>
                <input
                  type="text"
                  className="form-control"
                  value={formData.name}
                  onChange={e => setFormData({ ...formData, name: e.target.value })}
                  disabled={!!editingTemplate}
                  placeholder="e.g. pirate_template"
                  required
                />
                {editingTemplate && <small className="text-muted">Template names cannot be changed once created.</small>}
              </div>

              <div>
                <div className="d-flex justify-content-between align-items-center mb-2">
                  <label className="form-label mb-0">Messages</label>
                  <button type="button" className="btn btn-sm btn-outline-secondary" onClick={addMessage}>
                    <i className="bi bi-plus-lg me-1"></i> Add Message
                  </button>
                </div>
                <div className="d-flex flex-column gap-2">
                  {formData.messages.map((msg, i) => (
                    <div key={i} className="border rounded p-2">
                      <div className="d-flex justify-content-between align-items-center mb-2">
                        <select
                          className="form-select form-select-sm w-auto"
                          value={msg.role}
                          onChange={e => updateMessage(i, 'role', e.target.value)}
                        >
                          <option value="system">system</option>
                          <option value="user">user</option>
                          <option value="assistant">assistant</option>
                        </select>
                        {formData.messages.length > 1 && (
                          <button type="button" className="btn btn-sm btn-link text-danger p-0" onClick={() => removeMessage(i)}>
                            <i className="bi bi-trash"></i>
                          </button>
                        )}
                      </div>
                      <textarea
                        className="form-control font-monospace"
                        style={{ fontSize: '0.85rem', minHeight: '80px' }}
                        value={msg.content}
                        onChange={e => updateMessage(i, 'content', e.target.value)}
                        placeholder={msg.role === 'system' ? 'You are a helpful assistant...' : '{user_question}'}
                        required
                      />
                    </div>
                  ))}
                </div>
                <div className="form-text mt-1">
                  Available variables: <code>{'{user_question}'}</code>, <code>{'{source_docs}'}</code>, <code>{'{chat_history}'}</code>
                </div>
              </div>

              <div>
                <label className="form-label">Prompt Variables</label>
                <input
                  type="text"
                  className="form-control"
                  value={formData.prompt_variables}
                  onChange={e => setFormData({ ...formData, prompt_variables: e.target.value })}
                  placeholder="e.g. user_question"
                />
                <div className="form-text">Pipe-separated list of variables used in the template.</div>
              </div>

              <div className="d-flex justify-content-between align-items-center">
                <span className={`small text-${feedback.includes('Error') ? 'danger' : 'success'}`}>{feedback}</span>
                <div className="d-flex gap-2">
                  {editingTemplate && <button type="button" className="btn btn-outline-secondary" onClick={handleCreateNew}>Cancel</button>}
                  <button type="submit" className="btn btn-primary">
                    {editingTemplate ? 'Update Template' : 'Create Template'}
                  </button>
                </div>
              </div>
            </form>
          </div>
        </div>
      </div>
    </div>
  )
}
