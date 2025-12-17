import React, { useState, useRef } from 'react'

export default function AudioRecorder({ onRecordingComplete }) {
  const [isRecording, setIsRecording] = useState(false)
  const [recordingTime, setRecordingTime] = useState(0)
  const mediaRecorderRef = useRef(null)
  const timerRef = useRef(null)
  const chunksRef = useRef([])

  async function startRecording() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      mediaRecorderRef.current = new MediaRecorder(stream)
      chunksRef.current = []

      mediaRecorderRef.current.ondataavailable = (e) => {
        if (e.data.size > 0) {
          chunksRef.current.push(e.data)
        }
      }

      mediaRecorderRef.current.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: 'audio/webm' }) // or audio/wav if supported
        onRecordingComplete(blob)
        chunksRef.current = []
        stream.getTracks().forEach(track => track.stop())
      }

      mediaRecorderRef.current.start()
      setIsRecording(true)

      // Timer
      setRecordingTime(0)
      timerRef.current = setInterval(() => {
        setRecordingTime(prev => prev + 1)
      }, 1000)

    } catch (err) {
      console.error("Error accessing microphone:", err)
      alert("Could not access microphone. Please ensure permission is granted.")
    }
  }

  function stopRecording() {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop()
      setIsRecording(false)
      clearInterval(timerRef.current)
    }
  }

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60)
    const secs = seconds % 60
    return `${mins}:${secs.toString().padStart(2, '0')}`
  }

  return (
    <div className="d-flex align-items-center gap-2">
      {!isRecording ? (
        <button
          type="button"
          className="btn btn-outline-secondary btn-sm"
          onClick={startRecording}
          title="Start Recording"
        >
          <i className="bi bi-mic"></i>
        </button>
      ) : (
        <button
          type="button"
          className="btn btn-danger btn-sm d-flex align-items-center gap-1"
          onClick={stopRecording}
          title="Stop Recording"
        >
          <span className="spinner-grow spinner-grow-sm" role="status" aria-hidden="true"></span>
          {formatTime(recordingTime)}
        </button>
      )}
    </div>
  )
}
