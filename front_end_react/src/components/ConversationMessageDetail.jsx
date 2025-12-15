import React, { useState } from 'react'

export default function ConversationMessageDetail({ message }) {
  const [expanded, setExpanded] = useState(false)

  // Only show debug info for assistant messages with RAG logging
  if (message.speaker !== 'assistant' || !message.user_query) {
    return null
  }

  const getSentimentBadge = (sentiment) => {
    const badges = {
      positive: 'success',
      negative: 'danger',
      neutral: 'secondary'
    }
    return badges[sentiment] || 'secondary'
  }

  const getToneBadge = (tone) => {
    const badges = {
      polite: 'info',
      neutral: 'secondary',
      urgent: 'warning',
      frustrated: 'danger'
    }
    return badges[tone] || 'secondary'
  }

  return (
    <div className="message-debug-section mt-2">
      <button
        className="btn btn-sm btn-outline-secondary w-100 d-flex align-items-center justify-content-between"
        onClick={() => setExpanded(!expanded)}
      >
        <span>
          <i className={`bi bi-${expanded ? 'chevron-up' : 'chevron-down'} me-2`}></i>
          RAG Pipeline Details
        </span>
        {message.processing_time_ms && (
          <span className="badge bg-info">
            {message.processing_time_ms}ms
          </span>
        )}
      </button>

      {expanded && (
        <div className="debug-content mt-2">
          {/* Sentiment and Tone */}
          {(message.sentiment || message.tone) && (
            <div className="mb-3">
              <strong className="d-block mb-1">User Message Analysis:</strong>
              <div className="d-flex gap-2">
                {message.sentiment && (
                  <span className={`badge bg-${getSentimentBadge(message.sentiment)}`}>
                    Sentiment: {message.sentiment}
                  </span>
                )}
                {message.tone && (
                  <span className={`badge bg-${getToneBadge(message.tone)}`}>
                    Tone: {message.tone}
                  </span>
                )}
              </div>
            </div>
          )}

          {/* Retrieved Documents */}
          {message.retrieved_context && message.retrieved_context.length > 0 && (
            <div className="mb-3">
              <strong className="d-block mb-1">
                Retrieved Documents ({message.retrieved_context.length}):
              </strong>
              <div className="retrieved-docs">
                {message.retrieved_context.map((doc, idx) => (
                  <div key={idx} className="card card-body bg-light mb-2 small">
                    <div className="d-flex justify-content-between align-items-start mb-1">
                      <span className="badge bg-primary">Doc {idx + 1}</span>
                      {doc.distance && (
                        <span className="badge bg-secondary">
                          Score: {doc.distance.toFixed(3)}
                        </span>
                      )}
                    </div>
                    {doc.metadata && (
                      <div className="mb-1">
                        <small className="text-muted">
                          {doc.metadata.department && (
                            <span className="me-2">
                              <i className="bi bi-building me-1"></i>
                              {doc.metadata.department}
                            </span>
                          )}
                          {doc.metadata.sensitivity && (
                            <span className="badge bg-warning text-dark">
                              {doc.metadata.sensitivity}
                            </span>
                          )}
                        </small>
                      </div>
                    )}
                    <div className="text-truncate" title={doc.text}>
                      {doc.text?.substring(0, 150)}...
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* LLM Information */}
          <div className="mb-3">
            <strong className="d-block mb-1">LLM Configuration:</strong>
            <div className="small">
              <div><strong>Provider:</strong> {message.llm_provider || 'N/A'}</div>
              <div><strong>Model:</strong> {message.llm_model || 'N/A'}</div>
              {message.llm_temperature !== null && (
                <div><strong>Temperature:</strong> {message.llm_temperature}</div>
              )}
              {message.llm_max_tokens && (
                <div><strong>Max Tokens:</strong> {message.llm_max_tokens}</div>
              )}
              {message.llm_tokens_used && (
                <div><strong>Tokens Used:</strong> {message.llm_tokens_used}</div>
              )}
            </div>
          </div>

          {/* Embeddings */}
          {message.embeddings_used && (
            <div className="mb-3">
              <strong className="d-block mb-1">Embeddings:</strong>
              <div className="small">
                <div><strong>Model:</strong> {message.embeddings_used.model}</div>
                <div><strong>Dimensions:</strong> {message.embeddings_used.dimensions}</div>
              </div>
            </div>
          )}

          {/* Retrieval Settings */}
          <div className="mb-3">
            <strong className="d-block mb-1">Retrieval Settings:</strong>
            <div className="small">
              <div><strong>Top K:</strong> {message.retrieval_top_k || 'N/A'}</div>
              <div><strong>Use Documents:</strong> {message.use_documents ? 'Yes' : 'No'}</div>
              <div><strong>Use LLM:</strong> {message.use_llm ? 'Yes' : 'No'}</div>
            </div>
          </div>

          {/* LLM Prompt */}
          {message.llm_prompt && (
            <div className="mb-3">
              <strong className="d-block mb-1">LLM Prompt:</strong>
              <pre className="bg-dark text-light p-2 rounded small" style={{ maxHeight: '200px', overflow: 'auto' }}>
                {message.llm_prompt}
              </pre>
            </div>
          )}

          {/* Raw LLM Response */}
          {message.llm_response_raw && (
            <div className="mb-3">
              <strong className="d-block mb-1">Raw LLM Response:</strong>
              <pre className="bg-dark text-light p-2 rounded small" style={{ maxHeight: '200px', overflow: 'auto' }}>
                {message.llm_response_raw}
              </pre>
            </div>
          )}

          {/* Error Message */}
          {message.error_message && (
            <div className="alert alert-danger mb-0">
              <strong>Error:</strong> {message.error_message}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
