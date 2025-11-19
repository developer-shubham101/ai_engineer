import React, { useState } from 'react'

export default function AddJsonForm({ onSubmit }){
  const [source, setSource] = useState('')
  const [text, setText] = useState('')
  const [metadata, setMetadata] = useState('{}')
  const [response, setResponse] = useState(null)

  async function handleSubmit(e){
    e.preventDefault()
    let meta = {}
    try { meta = metadata ? JSON.parse(metadata) : {} } catch(err){ setResponse({ error: 'Metadata must be valid JSON' }); return }
    try {
      const r = await onSubmit({ source_name: source, text, metadata: meta })
      setResponse({ success: r })
    } catch(err){ setResponse({ error: err.message }) }
  }

  return (
    <form onSubmit={handleSubmit}>
      <div className="modal-body">
        <div className="mb-2"><label className="form-label">Source Name</label><input className="form-control" value={source} onChange={e=>setSource(e.target.value)} required /></div>
        <div className="mb-2"><label className="form-label">Text</label><textarea className="form-control" rows={6} value={text} onChange={e=>setText(e.target.value)} required /></div>
        <div className="mb-2"><label className="form-label">Metadata (JSON)</label><textarea className="form-control" rows={4} value={metadata} onChange={e=>setMetadata(e.target.value)} placeholder='{"department":"HR","sensitivity":"internal"}' /></div>
        {response && response.success && <div className="alert alert-success">Added: <pre>{JSON.stringify(response.success, null, 2)}</pre></div>}
        {response && response.error && <div className="alert alert-danger">{response.error}</div>}
      </div>
      <div className="modal-footer"><button className="btn btn-primary" type="submit">Add</button><button className="btn btn-secondary" data-bs-dismiss="modal" type="button">Close</button></div>
    </form>
  )
}
