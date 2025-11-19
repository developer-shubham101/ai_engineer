import React, { useState } from 'react'

export default function UploadFileForm({ onSubmit }){
  const [file, setFile] = useState(null)
  const [department, setDepartment] = useState('')
  const [sensitivity, setSensitivity] = useState('public')
  const [tags, setTags] = useState('')
  const [publicSummary, setPublicSummary] = useState('')
  const [ownerId, setOwnerId] = useState('')
  const [response, setResponse] = useState(null)

  async function handleSubmit(e){
    e.preventDefault()
    if (!file){ setResponse({ error: 'Select a file' }); return }
    const fd = new FormData()
    fd.append('file', file)
    fd.append('department', department)
    fd.append('sensitivity', sensitivity)
    fd.append('tags', tags)
    fd.append('public_summary', publicSummary)
    fd.append('owner_id', ownerId)
    try { const r = await onSubmit(fd); setResponse({ success: r }) } catch(err){ setResponse({ error: err.message }) }
  }

  return (
    <form onSubmit={handleSubmit}>
      <div className="modal-body">
        <div className="mb-2"><label className="form-label">Select File</label><input type="file" className="form-control" onChange={e=>setFile(e.target.files[0])} required /></div>
        <div className="mb-2"><label className="form-label">Department</label><input className="form-control" value={department} onChange={e=>setDepartment(e.target.value)} /></div>
        <div className="mb-2"><label className="form-label">Sensitivity</label><select className="form-select" value={sensitivity} onChange={e=>setSensitivity(e.target.value)}><option value="public">public</option><option value="internal">internal</option><option value="restricted">restricted</option><option value="confidential">confidential</option></select></div>
        <div className="mb-2"><label className="form-label">Tags</label><input className="form-control" value={tags} onChange={e=>setTags(e.target.value)} /></div>
        <div className="mb-2"><label className="form-label">Public Summary</label><textarea className="form-control" rows={3} value={publicSummary} onChange={e=>setPublicSummary(e.target.value)} /></div>
        <div className="mb-2"><label className="form-label">Owner ID</label><input className="form-control" value={ownerId} onChange={e=>setOwnerId(e.target.value)} /></div>
        {response && response.success && <div className="alert alert-success">Uploaded: <pre>{JSON.stringify(response.success, null, 2)}</pre></div>}
        {response && response.error && <div className="alert alert-danger">{response.error}</div>}
      </div>
      <div className="modal-footer"><button className="btn btn-primary" type="submit">Upload</button><button className="btn btn-secondary" data-bs-dismiss="modal" type="button">Close</button></div>
    </form>
  )
}
