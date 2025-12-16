import React, { useState, useEffect } from 'react'
import { BASE_API_URL } from '../utility/const.js'
import { getStoredToken, getSessionIdFromToken } from '../utility/auth.js'

export default function PromptTemplateManager({ onClose }) {
  const [templates, setTemplates] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [editingTemplate, setEditingTemplate] = useState(null)

  // Form state
  const [formData, setFormData] = useState({ name: '', content: '' })
  const [feedback, setFeedback] = useState('')

  const getAuthHeaders = () => {
    const token = getStoredToken();
    if (!token) return {};
    const headers = { 'Authorization': `Bearer ${token}`, 'Content-Type': 'application/json' };
    const sessionId = getSessionIdFromToken(token);
    if (sessionId) headers['X-Session-ID'] = sessionId;
    return headers;
  }

  // Fetch templates on load
  useEffect(() => {
    fetchTemplates()
  }, [])

  const fetchTemplates = async () => {
    setLoading(true)
    setError('')
    try {
      const res = await fetch(`${BASE_API_URL}/api/templates`, {
        headers: getAuthHeaders()
      })
      if (!res.ok) throw new Error('Failed to fetch templates')
      const data = await res.json()
      // API returns a list of templates
      setTemplates(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const handleEdit = (tpl) => {
    setEditingTemplate(tpl)
    setFormData({ name: tpl.name, content: tpl.content })
    setFeedback('')
  }

  const handleCreateNew = () => {
    setEditingTemplate(null)
    setFormData({ name: '', content: '' })
    setFeedback('')
  }

  const handleDelete = async (name) => {
    if (!window.confirm(`Are you sure you want to delete template "${name}"?`)) return

    try {
      const res = await fetch(`${BASE_API_URL}/api/templates/${name}`, {
        method: 'DELETE',
        headers: getAuthHeaders()
      })
      if (!res.ok) throw new Error('Failed to delete')

      setTemplates(prev => prev.filter(t => t.name !== name))
      // If we were editing this one, clear form
      if (editingTemplate && editingTemplate.name === name) {
        handleCreateNew()
      }
    } catch (err) {
      alert(err.message)
    }
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setFeedback('')

    try {
      let res;
      if (editingTemplate) {
        // Update
        res = await fetch(`${BASE_API_URL}/api/templates/${editingTemplate.name}`, {
          method: 'PUT',
          headers: getAuthHeaders(),
          body: JSON.stringify({ content: formData.content })
        })
      } else {
        // Create
        res = await fetch(`${BASE_API_URL}/api/templates`, {
          method: 'POST',
          headers: getAuthHeaders(),
          body: JSON.stringify(formData)
        })
      }

      if (!res.ok) {
        const txt = await res.text()
        throw new Error(txt || 'Operation failed')
      }

      const updatedOrNew = { name: formData.name, content: formData.content }

      if (editingTemplate) {
        setTemplates(prev => prev.map(t => t.name === editingTemplate.name ? { ...t, content: formData.content } : t))
        setFeedback('Template updated successfully')
      } else {
        setTemplates(prev => [...prev, updatedOrNew])
        setFeedback('Template created successfully')
        // Switch to edit mode for the new template so user can keep tweaking
        setEditingTemplate(updatedOrNew)
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
                  <div className="text-truncate" style={{ maxWidth: '80%' }}>
                    {t.name}
                  </div>
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
          <div className="col-md-8 d-flex flex-column h-100">
            <form onSubmit={handleSubmit} className="d-flex flex-column h-100">
              <div className="mb-3">
                <label className="form-label">Template Name</label>
                <input
                  type="text"
                  className="form-control"
                  value={formData.name}
                  onChange={e => setFormData({ ...formData, name: e.target.value })}
                  disabled={!!editingTemplate} // Cannot rename for now as name is ID
                  placeholder="e.g. creative_writing_assistant"
                  required
                />
                {editingTemplate && <small className="text-muted">Template names cannot be changed once created.</small>}
              </div>

              <div className="mb-3 flex-grow-1 d-flex flex-column">
                <label className="form-label">Template Content</label>
                <textarea
                  className="form-control flex-grow-1 font-monospace"
                  style={{ minHeight: '300px', fontSize: '0.85rem' }}
                  value={formData.content}
                  onChange={e => setFormData({ ...formData, content: e.target.value })}
                  placeholder="System: You are a helpful assistant..."
                  required
                ></textarea>
                <div className="form-text">
                  Available variables: <code>{'{source_docs}'}</code>, <code>{'{user_question}'}</code>, <code>{'{chat_history}'}</code>
                </div>
              </div>

              <div className="d-flex justify-content-between align-items-center">
                <div>
                  {feedback && <span className={`text-${feedback.includes('Error') ? 'danger' : 'success'}`}>{feedback}</span>}
                </div>
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
