import React, { useEffect, useRef, useState } from 'react'
import AddJsonForm from './AddJsonForm.jsx'
import UploadFileForm from './UploadFileForm.jsx'
import UpdateMetadataForm from './UpdateMetadataForm.jsx'
import ToastList from './ToastList.jsx'
import { BASE_API_URL } from '../utility/const.js'


export default function RAGChat(){
  const [apiKey, setApiKey] = useState(localStorage.getItem('api_key') || '')
  const [rememberKey, setRememberKey] = useState(localStorage.getItem('remember_api_key') === 'true')
  const [role, setRole] = useState('Employee')
  const [useLlm, setUseLlm] = useState(false)
  const [topK, setTopK] = useState(3)
  const [composer, setComposer] = useState('')
  const [messages, setMessages] = useState(()=> JSON.parse(localStorage.getItem('chat_history_v1') || '[]'))
  const [inFlight, setInFlight] = useState(false)
  const [selectedFile, setSelectedFile] = useState(null)
  const [fileMeta, setFileMeta] = useState({ department:'', sensitivity:'public', tags:'' })
  const [toasts, setToasts] = useState([])
  const messagesRef = useRef(null)

  useEffect(()=>{ localStorage.setItem('chat_history_v1', JSON.stringify(messages)) }, [messages])
  useEffect(()=>{ if (rememberKey) localStorage.setItem('api_key', apiKey || ''); localStorage.setItem('remember_api_key', rememberKey ? 'true' : 'false') }, [apiKey, rememberKey])
  useEffect(()=>{ if (messagesRef.current) messagesRef.current.scrollTop = messagesRef.current.scrollHeight }, [messages])

  function addToast(text, kind='danger', title=''){ const id=Date.now(); setToasts(t=>[...t,{id,text,kind,title}]); setTimeout(()=>setToasts(t=>t.filter(x=>x.id!==id)),5000) }
  function pushMessage(msg){ setMessages(m=>[...m,{...msg, ts: new Date().toLocaleString()}]) }

  async function sendQuery(){
    if (inFlight) return
    if (!composer.trim()) return addToast('Please enter a question', 'warning', 'Validation')
    if (!apiKey) return addToast('API key required', 'warning', 'Auth')

    setInFlight(true)
    const userMsg = { role:'user', text: composer }
    pushMessage(userMsg)
    setComposer('')

    const payload = { question: composer, top_k: Number(topK||3), use_llm: !!useLlm }

    try {
      const res = await fetch(BASE_API_URL + "/api/local/query", {
        method: "POST",
        headers: { "Content-Type": "application/json", "X-API-Key": apiKey },
        body: JSON.stringify(payload),
      });
      if (!res.ok){ const txt = await res.text().catch(()=>res.statusText); throw new Error(`Server error: ${res.status} - ${txt}`) }
      const data = await res.json()
      const botMsg = {
        role:'bot', answer: data.answer || 'No answer returned', retrieved: data.retrieved||[], context: data.context||'', filtered_out_count: data.filtered_out_count||0, public_summaries: data.public_summaries||[], filtered_details: data.filtered_details||null
      }
      pushMessage(botMsg)
      if (botMsg.filtered_out_count > 0) addToast(`${botMsg.filtered_out_count} results filtered for your role`, 'warning', 'Filtered')
    } catch(err){
      addToast(err.message || 'Network error', 'danger', 'Error')
      pushMessage({ role:'bot', answer: 'Error: ' + (err.message || 'Network error') })
    } finally {
      setTimeout(()=>setInFlight(false), 2000)
    }
  }

  async function postAddJson({ source_name, text, metadata }){
    if (!apiKey) return addToast('API key required', 'warning', 'Auth')
    try {
      const res = await fetch(BASE_API_URL + '/add', { method:'POST', headers: { 'Content-Type':'application/json', 'X-API-Key': apiKey }, body: JSON.stringify({ source_name, text, metadata }) })
      if (!res.ok){ const txt = await res.text().catch(()=>res.statusText); throw new Error(txt || 'Server error') }
      const data = await res.json(); addToast('Document added', 'success'); return data
    } catch(err){ addToast(err.message || 'Add error','danger'); throw err }
  }

  async function postUploadFile(formData){
    if (!apiKey) return addToast('API key required', 'warning', 'Auth')
    try {
      const res = await fetch(BASE_API_URL + '/api/local/add-file', { method:'POST', headers: { 'X-API-Key': apiKey }, body: formData })
      if (!res.ok){ const txt = await res.text().catch(()=>res.statusText); throw new Error(txt || 'Upload error') }
      const data = await res.json(); addToast('File uploaded', 'success'); return data
    } catch(err){ addToast(err.message || 'Upload error','danger'); throw err }
  }

  async function requestAccess({ document_id, source_name, reason }){
    if (!apiKey) return addToast('API key required', 'warning', 'Auth')
    try {
      const res = await fetch(BASE_API_URL + '/request-access', { method:'POST', headers: { 'Content-Type':'application/json', 'X-API-Key': apiKey }, body: JSON.stringify({ document_id, source_name, reason }) })
      if (!res.ok){ const txt = await res.text().catch(()=>res.statusText); throw new Error(txt || 'Request error') }
      const data = await res.json(); addToast('Access request submitted', 'success'); return data
    } catch(err){ addToast(err.message || 'Request error','danger'); throw err }
  }

  async function fetchAccessRequests(){
    if (!apiKey) return addToast('API key required', 'warning', 'Auth')
    try {
      const res = await fetch(BASE_API_URL + '/access-requests', { headers: { 'X-API-Key': apiKey } })
      if (!res.ok) throw new Error('Failed to fetch'); const data = await res.json(); return data
    } catch(err){ addToast(err.message || 'Fetch error','danger'); return [] }
  }

  function truncate(text, n=200){ if (!text) return ''; return text.length<=n?text:text.slice(0,n)+'...' }

  return (
    <div>
      <div className="d-flex justify-content-between align-items-center mb-2">
        <h3>RAG Chat (React)</h3>
        <div className="d-flex gap-2 align-items-center">
          <div className="input-group input-group-sm">
            <span className="input-group-text">API Key</span>
            <input className="form-control" type="password" value={apiKey} onChange={e=>setApiKey(e.target.value)} placeholder="sk-xxxx" />
            <button className={`btn ${rememberKey?'btn-success':'btn-outline-secondary'}`} onClick={()=>setRememberKey(r=>!r)}>🔒</button>
          </div>
          <select className="form-select form-select-sm" style={{width:160}} value={role} onChange={e=>setRole(e.target.value)}>
            <option>Employee</option><option>Manager</option><option>HR</option><option>Legal</option><option>Executive</option>
          </select>
          <div className="form-check form-switch">
            <input className="form-check-input" type="checkbox" checked={useLlm} onChange={e=>setUseLlm(e.target.checked)} />
            <label className="form-check-label">Use LLM</label>
          </div>
          <div className="input-group input-group-sm" style={{width:110}}>
            <span className="input-group-text">Top K</span>
            <input type="number" className="form-control" min={1} max={20} value={topK} onChange={e=>setTopK(Number(e.target.value))} />
          </div>
        </div>
      </div>

      <div className="row g-3">
        <div className="col-md-8">
          <div className="card h-100">
            <div className="card-body d-flex flex-column">
              <div className="d-flex justify-content-between mb-2"><h5>Chat</h5><small className="text-muted">Session: local</small></div>
              <div ref={messagesRef} style={{height:'60vh', overflowY:'auto', padding:12, background:'#f8f9fa', borderRadius:8}}>
                {messages.map((m, idx)=>(
                  <div key={idx} className={`d-flex mb-3 ${m.role==='user'?'justify-content-end':'justify-content-start'}`}>
                    <div className={`p-3 rounded ${m.role==='user'?'text-white':''}`} style={{background: m.role==='user'?'#0d6efd':'#fff', maxWidth:'75%'}}>
                      <div className="small text-muted mb-1">{m.role==='user'?'You':'Assistant'} • {m.ts}</div>
                      <div style={{whiteSpace:'pre-wrap'}}>{m.role==='user'?m.text:(m.answer||m.text)}</div>
                      {m.role==='bot' && m.context && <div className="small text-muted mt-2">Context: {truncate(m.context,300)}</div>}
                      {m.role==='bot' && Array.isArray(m.retrieved) && m.retrieved.length>0 && (
                        <details className="mt-2"><summary>Retrieved ({m.retrieved.length})</summary>
                          <div className="mt-2">{m.retrieved.map((r,i)=>(
                            <div key={i} className="p-2 mb-2" style={{borderLeft:'3px solid #e9ecef', background:'#fff'}}>
                              <div className="small text-muted">Source: {r.source || (r.metadata && r.metadata.department) || 'unknown'}</div>
                              <div style={{whiteSpace:'pre-wrap'}}>{truncate(r.text||r.preview||'',600)}</div>
                              <div className="small text-muted mt-1">Metadata: {JSON.stringify(r.metadata||{})}</div>
                              {(r.metadata && (r.metadata.sensitivity==='restricted' || r.metadata.sensitivity==='confidential')) && (
                                <button className="btn btn-sm btn-outline-primary mt-2" onClick={()=>requestAccess({ document_id: r.id, source_name: r.source, reason: 'Need access via UI' })}>Request Access</button>
                              )}
                            </div>
                          ))}</div>
                        </details>
                      )}
                      {m.role==='bot' && m.filtered_out_count>0 && (
                        <div className="mt-2"><button className="btn btn-sm btn-warning" onClick={()=>{ if (Array.isArray(m.public_summaries)&&m.public_summaries.length){ addToast('Public summaries logged to console', 'info'); console.log('Public summaries:', m.public_summaries) } else addToast('No public summaries', 'warning') }}>{m.filtered_out_count} results were filtered — show public summaries</button></div>
                      )}
                    </div>
                  </div>
                ))}
              </div>

              <div className="mt-3">
                <div className="d-flex gap-2">
                  <textarea className="form-control" value={composer} onChange={e=>setComposer(e.target.value)} placeholder="Type your question..." rows={2} />
                  <div className="d-flex flex-column align-items-end">
                    <div className="mb-2">
                      <button className="btn btn-primary" disabled={inFlight} onClick={sendQuery}>{inFlight?'Sending...':'Send'}</button>
                      <label className="btn btn-outline-secondary mb-0 ms-2">Attach <input type="file" style={{display:'none'}} onChange={e=>setSelectedFile(e.target.files[0])} /></label>
                      <button className="btn btn-outline-success ms-2" data-bs-toggle="modal" data-bs-target="#addJsonModal">Add JSON</button>
                      <button className="btn btn-outline-info ms-2" data-bs-toggle="modal" data-bs-target="#uploadFileModal">Upload File</button>
                    </div>
                    <div className="small text-muted">Top K: {topK} • Use LLM: {useLlm ? 'on':'off'}</div>
                  </div>
                </div>

                {selectedFile && (<div className="mt-2"><strong>Selected:</strong> {selectedFile.name} <button className="btn btn-sm btn-outline-danger ms-2" onClick={()=>setSelectedFile(null)}>Remove</button>
                  <div className="mt-2 d-flex gap-2"><input className="form-control form-control-sm" placeholder="Department" value={fileMeta.department} onChange={e=>setFileMeta({...fileMeta, department:e.target.value})} /><select className="form-select form-select-sm" value={fileMeta.sensitivity} onChange={e=>setFileMeta({...fileMeta, sensitivity:e.target.value})}><option value="public">public</option><option value="internal">internal</option><option value="restricted">restricted</option><option value="confidential">confidential</option></select><input className="form-control form-control-sm" placeholder="tags" value={fileMeta.tags} onChange={e=>setFileMeta({...fileMeta, tags:e.target.value})} /></div></div>)}
              </div>

            </div>
          </div>
        </div>

        <div className="col-md-4">
          <div className="card mb-3"><div className="card-header">Admin Panel</div><div className="card-body">
            <div className="d-grid gap-2"><button className="btn btn-outline-primary" onClick={async ()=>{ const list = await fetchAccessRequests(); console.log('Access requests:', list); addToast('Fetched access requests (console)', 'info') }}>Fetch Access Requests</button>
              <button className="btn btn-outline-secondary" onClick={()=>addToast('Local requests shown in console','info')}>Show Local Requests</button>
              <button className="btn btn-outline-warning" data-bs-toggle="modal" data-bs-target="#updateMetadataModal">Update Metadata</button></div>
            <hr/><div className="small text-muted">Messages: {messages.length}</div><div className="small text-muted">Last activity: {messages.length ? messages[messages.length-1].ts : '-'}</div>
          </div></div>

          <div className="card"><div className="card-header">Add Document</div><div className="card-body"><button className="btn btn-success w-100 mb-2" data-bs-toggle="modal" data-bs-target="#addJsonModal">Add JSON Doc</button><button className="btn btn-info w-100" data-bs-toggle="modal" data-bs-target="#uploadFileModal">Upload File</button></div></div>
        </div>
      </div>

      {/* Modals */}
      <div className="modal fade" id="addJsonModal" tabIndex={-1} aria-hidden="true"><div className="modal-dialog modal-lg"><div className="modal-content"><div className="modal-header"><h5 className="modal-title">Add JSON Document</h5><button type="button" className="btn-close" data-bs-dismiss="modal" aria-label="Close"></button></div><AddJsonForm onSubmit={postAddJson} /></div></div></div>
      <div className="modal fade" id="uploadFileModal" tabIndex={-1} aria-hidden="true"><div className="modal-dialog"><div className="modal-content"><div className="modal-header"><h5 className="modal-title">Upload File</h5><button type="button" className="btn-close" data-bs-dismiss="modal" aria-label="Close"></button></div><UploadFileForm onSubmit={postUploadFile} /></div></div></div>
      <div className="modal fade" id="updateMetadataModal" tabIndex={-1} aria-hidden="true"><div className="modal-dialog modal-lg"><div className="modal-content"><div className="modal-header"><h5 className="modal-title">Update Metadata</h5><button type="button" className="btn-close" data-bs-dismiss="modal" aria-label="Close"></button></div><UpdateMetadataForm /></div></div></div>

      <ToastList toasts={toasts} />
    </div>
  )
}
