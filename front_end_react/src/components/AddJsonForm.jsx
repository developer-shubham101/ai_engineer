import React, { useState } from 'react'

export default function AddJsonForm({ onSubmit }){
  const [source, setSource] = useState('')
  const [text, setText] = useState('')
  const [department, setDepartment] = useState('Engineering')
  const [sensitivity, setSensitivity] = useState('public_internal')
  const [allowedRoles, setAllowedRoles] = useState('')
  const [ownerId, setOwnerId] = useState('')
  const [publicSummary, setPublicSummary] = useState('')
  const [useAdvanced, setUseAdvanced] = useState(false)
  const [metadata, setMetadata] = useState('{}')
  const [response, setResponse] = useState(null)

  async function handleSubmit(e){
    e.preventDefault()
    let meta = {}
    
    if (useAdvanced) {
      try { meta = metadata ? JSON.parse(metadata) : {} } catch(err){ setResponse({ error: 'Metadata must be valid JSON' }); return }
    } else {
      meta = {
        department,
        sensitivity,
        ...(allowedRoles && { allowed_roles: allowedRoles.split(',').map(r => r.trim()) }),
        ...(ownerId && { owner_id: ownerId }),
        ...(publicSummary && { public_summary: publicSummary })
      }
    }
    
    try {
      const r = await onSubmit({ source_name: source, text, metadata: meta })
      setResponse({ success: r })
    } catch(err){ setResponse({ error: err.message }) }
  }

  return (
    <form onSubmit={handleSubmit}>
      <div className="modal-body">
        <div className="mb-2">
          <label className="form-label">Source Name</label>
          <input className="form-control" value={source} onChange={e=>setSource(e.target.value)} required />
        </div>
        <div className="mb-2">
          <label className="form-label">Text</label>
          <textarea className="form-control" rows={4} value={text} onChange={e=>setText(e.target.value)} required />
        </div>
        
        <div className="mb-3">
          <div className="form-check form-switch">
            <input className="form-check-input" type="checkbox" checked={useAdvanced} onChange={e=>setUseAdvanced(e.target.checked)} />
            <label className="form-check-label">Advanced JSON Mode</label>
          </div>
        </div>
        
        {!useAdvanced ? (
          <div>
            <div className="row g-2 mb-2">
              <div className="col-md-6">
                <label className="form-label">Department</label>
                <select className="form-select" value={department} onChange={e=>setDepartment(e.target.value)}>
                  <option value="Engineering">Engineering</option>
                  <option value="HR">HR</option>
                  <option value="Finance">Finance</option>
                  <option value="Legal">Legal</option>
                  <option value="IT">IT</option>
                  <option value="Executive">Executive</option>
                  <option value="Admin">Admin</option>
                  <option value="General">General</option>
                </select>
              </div>
              <div className="col-md-6">
                <label className="form-label">Sensitivity</label>
                <select className="form-select" value={sensitivity} onChange={e=>setSensitivity(e.target.value)}>
                  <option value="public_internal">Public Internal</option>
                  <option value="department_confidential">Department Confidential</option>
                  <option value="role_confidential">Role Confidential</option>
                  <option value="highly_confidential">Highly Confidential</option>
                  <option value="personal">Personal</option>
                </select>
              </div>
            </div>
            <div className="row g-2 mb-2">
              <div className="col-md-6">
                <label className="form-label">Allowed Roles (comma-separated)</label>
                <input className="form-control" value={allowedRoles} onChange={e=>setAllowedRoles(e.target.value)} placeholder="SuperAdmin, Manager" />
              </div>
              <div className="col-md-6">
                <label className="form-label">Owner ID (for personal docs)</label>
                <input className="form-control" value={ownerId} onChange={e=>setOwnerId(e.target.value)} placeholder="u_emp_1" />
              </div>
            </div>
            <div className="mb-2">
              <label className="form-label">Public Summary</label>
              <textarea className="form-control" rows={2} value={publicSummary} onChange={e=>setPublicSummary(e.target.value)} placeholder="Brief summary for restricted access" />
            </div>
          </div>
        ) : (
          <div className="mb-2">
            <label className="form-label">Metadata (JSON)</label>
            <textarea className="form-control" rows={4} value={metadata} onChange={e=>setMetadata(e.target.value)} placeholder='{"department":"HR","sensitivity":"department_confidential"}' />
          </div>
        )}
        
        {response && response.success && <div className="alert alert-success">Added: <pre>{JSON.stringify(response.success, null, 2)}</pre></div>}
        {response && response.error && <div className="alert alert-danger">{response.error}</div>}
      </div>
      <div className="modal-footer">
        <button className="btn btn-primary" type="submit">Add Document</button>
        <button className="btn btn-secondary" data-bs-dismiss="modal" type="button">Close</button>
      </div>
    </form>
  )
}
