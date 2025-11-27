import React, { useState } from 'react'

export default function DocumentVersionModal({ onGetVersions, onCompareVersions, onArchiveVersion }) {
  const [documentId, setDocumentId] = useState('')
  const [version1, setVersion1] = useState('')
  const [version2, setVersion2] = useState('')
  const [archiveVersion, setArchiveVersion] = useState('')
  const [response, setResponse] = useState(null)
  const [loading, setLoading] = useState(false)

  async function handleGetVersions(e) {
    e.preventDefault()
    if (!documentId.trim()) {
      setResponse({ error: 'Document ID is required' })
      return
    }
    setLoading(true)
    try {
      const result = await onGetVersions(documentId)
      setResponse({ success: result })
      console.log('Version history:', result)
    } catch (err) {
      setResponse({ error: err.message })
    } finally {
      setLoading(false)
    }
  }

  async function handleCompareVersions(e) {
    e.preventDefault()
    if (!documentId.trim() || !version1.trim() || !version2.trim()) {
      setResponse({ error: 'Document ID and both versions are required' })
      return
    }
    setLoading(true)
    try {
      const result = await onCompareVersions(documentId, version1, version2)
      setResponse({ success: result })
      console.log('Version comparison:', result)
    } catch (err) {
      setResponse({ error: err.message })
    } finally {
      setLoading(false)
    }
  }

  async function handleArchiveVersion(e) {
    e.preventDefault()
    if (!documentId.trim() || !archiveVersion.trim()) {
      setResponse({ error: 'Document ID and version are required' })
      return
    }
    setLoading(true)
    try {
      const result = await onArchiveVersion(documentId, archiveVersion)
      setResponse({ success: result })
    } catch (err) {
      setResponse({ error: err.message })
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="modal-body">
      <div className="mb-3">
        <label className="form-label">Document ID</label>
        <input
          className="form-control"
          value={documentId}
          onChange={e => setDocumentId(e.target.value)}
          placeholder="doc_abc123..."
          required
        />
      </div>

      <div className="row g-3">
        <div className="col-md-4">
          <h6>Get Version History</h6>
          <button
            className="btn btn-primary w-100"
            onClick={handleGetVersions}
            disabled={loading}
          >
            {loading ? 'Loading...' : 'Get Versions'}
          </button>
        </div>

        <div className="col-md-4">
          <h6>Compare Versions</h6>
          <div className="mb-2">
            <input
              className="form-control form-control-sm mb-1"
              placeholder="Version 1 (e.g., 1.0)"
              value={version1}
              onChange={e => setVersion1(e.target.value)}
            />
            <input
              className="form-control form-control-sm"
              placeholder="Version 2 (e.g., 2.0)"
              value={version2}
              onChange={e => setVersion2(e.target.value)}
            />
          </div>
          <button
            className="btn btn-info w-100"
            onClick={handleCompareVersions}
            disabled={loading}
          >
            Compare
          </button>
        </div>

        <div className="col-md-4">
          <h6>Archive Version</h6>
          <div className="mb-2">
            <input
              className="form-control form-control-sm"
              placeholder="Version (e.g., 1.0)"
              value={archiveVersion}
              onChange={e => setArchiveVersion(e.target.value)}
            />
          </div>
          <button
            className="btn btn-danger w-100"
            onClick={handleArchiveVersion}
            disabled={loading}
          >
            Archive
          </button>
        </div>
      </div>

      {response && response.success && (
        <div className="alert alert-success mt-3">
          <strong>Success:</strong>
          <pre className="mt-2 mb-0">{JSON.stringify(response.success, null, 2)}</pre>
        </div>
      )}
      {response && response.error && (
        <div className="alert alert-danger mt-3">
          <strong>Error:</strong> {response.error}
        </div>
      )}

      <div className="mt-3">
        <small className="text-muted">
          Results are logged to browser console. Check DevTools → Console for detailed output.
        </small>
      </div>
    </div>
  )
}