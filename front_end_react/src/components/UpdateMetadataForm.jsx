import React, { useState } from 'react'
import { BASE_API_URL } from '../utility/const'

export default function UpdateMetadataForm(){
  const [idsRaw, setIdsRaw] = useState('[]')
  const [metaRaw, setMetaRaw] = useState('{}')
  const [response, setResponse] = useState(null)

  async function handleSubmit(e){
    e.preventDefault()
    let ids=[]; let metadata={}
    try{ ids = JSON.parse(idsRaw) } catch(err){ setResponse({ error: 'IDs must be a JSON array' }); return }
    try{ metadata = JSON.parse(metaRaw) } catch(err){ setResponse({ error: 'Metadata must be valid JSON' }); return }
    try {
      const res = await fetch(BASE_API_URL + '/api/local/update-metadata', { method:'POST', headers:{ 'Content-Type':'application/json' }, body: JSON.stringify({ ids, metadata }) })
      if (!res.ok) throw new Error('Update failed')
      const data = await res.json(); setResponse({ success: data })
    } catch(err){ setResponse({ error: err.message }) }
  }

  return (
    <form onSubmit={handleSubmit}>
      <div className="modal-body">
        <div className="mb-2"><label>Chunk IDs (JSON array)</label><textarea className="form-control" rows={3} value={idsRaw} onChange={e=>setIdsRaw(e.target.value)} /></div>
        <div className="mb-2"><label>Metadata (JSON)</label><textarea className="form-control" rows={4} value={metaRaw} onChange={e=>setMetaRaw(e.target.value)} /></div>
        {response && response.success && <div className="alert alert-success">Updated: <pre>{JSON.stringify(response.success,null,2)}</pre></div>}
        {response && response.error && <div className="alert alert-danger">{response.error}</div>}
      </div>
      <div className="modal-footer"><button className="btn btn-warning" type="submit">Update</button><button className="btn btn-secondary" data-bs-dismiss="modal" type="button">Close</button></div>
    </form>
  )
}
