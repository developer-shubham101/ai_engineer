import React, { useState } from 'react'

export default function ConversationSidebar({
  conversations,
  activeConversationId,
  onSelectConversation,
  onCreateConversation,
  onRenameConversation,
  onDeleteConversation,
  isOpen,
  onToggle,
  mode = 'rag' // 'rag' | 'agent' | 'crew'
}) {
  const [editingId, setEditingId] = useState(null)
  const [editTitle, setEditTitle] = useState('')
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(null)

  const handleStartEdit = (conv) => {
    setEditingId(conv.id)
    setEditTitle(conv.title)
  }

  const handleSaveEdit = async () => {
    if (editTitle.trim() && editingId) {
      await onRenameConversation(editingId, editTitle.trim())
      setEditingId(null)
      setEditTitle('')
    }
  }

  const handleCancelEdit = () => {
    setEditingId(null)
    setEditTitle('')
  }

  const handleDelete = async (convId) => {
    await onDeleteConversation(convId)
    setShowDeleteConfirm(null)
  }

  const formatDate = (dateStr) => {
    const date = new Date(dateStr)
    const now = new Date()
    const diffMs = now - date
    const diffMins = Math.floor(diffMs / 60000)
    const diffHours = Math.floor(diffMs / 3600000)
    const diffDays = Math.floor(diffMs / 86400000)

    if (diffMins < 1) return 'Just now'
    if (diffMins < 60) return `${diffMins}m ago`
    if (diffHours < 24) return `${diffHours}h ago`
    if (diffDays < 7) return `${diffDays}d ago`

    return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' })
  }

  return (
    <>
      {/* Mobile overlay */}
      {isOpen && (
        <div
          className="conversation-sidebar-overlay"
          onClick={onToggle}
        />
      )}

      <div className={`conversation-sidebar ${isOpen ? 'open' : 'closed'}`}>
        {/* Header */}
        <div className="conversation-sidebar-header">
          <div className="d-flex align-items-center gap-2 mb-2">
            <i className={`bi ${mode === 'agent' ? 'bi-cpu' : mode === 'crew' ? 'bi-robot' : 'bi-chat-dots'} text-muted`}></i>
            <small className="text-muted fw-semibold text-uppercase" style={{ letterSpacing: '0.05em' }}>
              {mode === 'agent' ? 'Agent History' : mode === 'crew' ? 'CrewAI History' : 'Conversations'}
            </small>
          </div>
          <button
              className="btn btn-primary w-100 mb-3"
              onClick={onCreateConversation}
              title={mode === 'agent' ? 'New agent session' : mode === 'crew' ? 'New crew session' : 'Start a new conversation'}
            >
              <i className="bi bi-plus-lg me-2"></i>
              {mode === 'agent' ? 'New Agent Session' : mode === 'crew' ? 'New Crew Session' : 'New Conversation'}
            </button>
        </div>

        {/* Conversation List */}
        <div className="conversation-list">
          {conversations.length === 0 ? (
            <div className="text-center text-muted p-3">
              <i className="bi bi-chat-dots fs-1 d-block mb-2"></i>
              <small>No conversations yet</small>
            </div>
          ) : (
            conversations.map((conv) => (
              <div
                key={conv.id}
                className={`conversation-item ${conv.id === activeConversationId ? 'active' : ''
                  }`}
              >
                {editingId === conv.id ? (
                  // Edit mode
                  <div className="conversation-edit">
                    <input
                      type="text"
                      className="form-control form-control-sm mb-2"
                      value={editTitle}
                      onChange={(e) => setEditTitle(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter') handleSaveEdit()
                        if (e.key === 'Escape') handleCancelEdit()
                      }}
                      autoFocus
                    />
                    <div className="d-flex gap-1">
                      <button
                        className="btn btn-sm btn-success flex-grow-1"
                        onClick={handleSaveEdit}
                      >
                        <i className="bi bi-check-lg"></i>
                      </button>
                      <button
                        className="btn btn-sm btn-secondary flex-grow-1"
                        onClick={handleCancelEdit}
                      >
                        <i className="bi bi-x-lg"></i>
                      </button>
                    </div>
                  </div>
                ) : showDeleteConfirm === conv.id ? (
                  // Delete confirmation
                  <div className="conversation-delete-confirm">
                    <p className="small mb-2">Delete this conversation?</p>
                    <div className="d-flex gap-1">
                      <button
                        className="btn btn-sm btn-danger flex-grow-1"
                        onClick={() => handleDelete(conv.id)}
                      >
                        Delete
                      </button>
                      <button
                        className="btn btn-sm btn-secondary flex-grow-1"
                        onClick={() => setShowDeleteConfirm(null)}
                      >
                        Cancel
                      </button>
                    </div>
                  </div>
                ) : (
                  // Normal view
                  <>
                    <div
                      className="conversation-content"
                      onClick={() => onSelectConversation(conv.id)}
                    >
                      <div className="conversation-title">
                        {conv.title || 'New Conversation'}
                      </div>
                      <div className="conversation-meta">
                        <small className="text-muted">
                          {formatDate(conv.updated_at || conv.created_at)}
                          {conv.message_count > 0 && (
                            <span className="ms-2">
                              <i className="bi bi-chat-dots me-1"></i>
                              {conv.message_count}
                            </span>
                          )}
                        </small>
                      </div>
                    </div>
                    <div className="conversation-actions">
                      <button
                        className="btn btn-sm btn-link text-muted p-0"
                        onClick={(e) => {
                          e.stopPropagation()
                          handleStartEdit(conv)
                        }}
                        title="Rename conversation"
                      >
                        <i className="bi bi-pencil"></i>
                      </button>
                      <button
                        className="btn btn-sm btn-link text-danger p-0"
                        onClick={(e) => {
                          e.stopPropagation()
                          setShowDeleteConfirm(conv.id)
                        }}
                        title="Delete conversation"
                      >
                        <i className="bi bi-trash"></i>
                      </button>
                    </div>
                  </>
                )}
              </div>
            ))
          )}
        </div>
      </div>
    </>
  )
}
