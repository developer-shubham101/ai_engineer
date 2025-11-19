import React from 'react'

export default function ToastList({ toasts }){
  return (
    <div className="position-fixed" style={{ right:12, top:12, zIndex:2000 }}>
      {toasts.map(t => (
        <div key={t.id} className={`toast show text-bg-${t.kind} mb-2`} role="alert"><div className="toast-body">{t.title && <strong>{t.title}: </strong>}{t.text}</div></div>
      ))}
    </div>
  )
}
