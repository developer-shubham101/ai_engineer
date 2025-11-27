import React, { useState } from 'react'

export default function PersonalizedTestModal({ onSendQuery, modelProvider }) {
  const [testScenario, setTestScenario] = useState('guest_job')
  const [customQuery, setCustomQuery] = useState('')
  const [response, setResponse] = useState(null)
  const [loading, setLoading] = useState(false)

  const testScenarios = {
    guest_job: {
      name: 'Guest User - Job Inquiry',
      query: 'Hi I want to connect with HR to check if any opening is there',
      description: 'Tests personalized job matching for guest users'
    },
    career_growth: {
      name: 'Internal Employee - Career Guidance',
      query: 'What career growth opportunities are available for me?',
      description: 'Tests career guidance for internal employees'
    },
    hr_recruitment: {
      name: 'HR Manager - Recruitment',
      query: 'How can I improve our hiring process for technical roles?',
      description: 'Tests HR-specific guidance and best practices'
    },
    profile_analysis: {
      name: 'Profile Analysis Test',
      query: 'I am a Python developer with 3 years experience. Are there any suitable positions?',
      description: 'Tests profile analysis and job matching'
    },
    onboarding_flow: {
      name: 'Guest Onboarding Flow',
      query: 'Hello, I need help',
      description: 'Tests guest onboarding question sequence'
    }
  }

  async function runTest() {
    setLoading(true)
    setResponse(null)
    
    try {
      const query = customQuery.trim() || testScenarios[testScenario].query
      
      // Send the query using the parent's sendQuery function
      const result = await fetch(`http://192.168.1.2:8000/api/rag/${modelProvider}/query`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('auth_token')}`
        },
        body: JSON.stringify({
          question: query,
          top_k: 3,
          use_llm: true,
          category: 'personalized_test'
        })
      })
      
      if (!result.ok) {
        throw new Error(`HTTP ${result.status}: ${result.statusText}`)
      }
      
      const data = await result.json()
      setResponse({ success: data })
      
      // Log detailed results to console
      console.log('=== PERSONALIZED RESPONSE TEST ===')
      console.log('Scenario:', testScenarios[testScenario].name)
      console.log('Query:', query)
      console.log('Response:', data.answer)
      console.log('Retrieved docs:', data.retrieved?.length || 0)
      console.log('Full response:', data)
      
    } catch (err) {
      setResponse({ error: err.message })
      console.error('Personalized test error:', err)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="modal-body">
      <div className="mb-3">
        <label className="form-label">Test Scenario</label>
        <select
          className="form-select"
          value={testScenario}
          onChange={e => setTestScenario(e.target.value)}
        >
          {Object.entries(testScenarios).map(([key, scenario]) => (
            <option key={key} value={key}>
              {scenario.name}
            </option>
          ))}
        </select>
        <div className="form-text">
          {testScenarios[testScenario].description}
        </div>
      </div>

      <div className="mb-3">
        <label className="form-label">Query</label>
        <textarea
          className="form-control"
          rows={3}
          value={customQuery}
          onChange={e => setCustomQuery(e.target.value)}
          placeholder={testScenarios[testScenario].query}
        />
        <div className="form-text">
          Leave empty to use default query for selected scenario
        </div>
      </div>

      <div className="mb-3">
        <button
          className="btn btn-primary"
          onClick={runTest}
          disabled={loading}
        >
          {loading ? 'Testing...' : 'Run Personalized Test'}
        </button>
      </div>

      {response && response.success && (
        <div className="alert alert-success">
          <h6>AI Response:</h6>
          <div className="border p-3 bg-light rounded">
            <pre className="mb-0" style={{ whiteSpace: 'pre-wrap' }}>
              {response.success.answer}
            </pre>
          </div>
          {response.success.retrieved && response.success.retrieved.length > 0 && (
            <div className="mt-2">
              <small className="text-muted">
                Retrieved {response.success.retrieved.length} documents
              </small>
            </div>
          )}
        </div>
      )}

      {response && response.error && (
        <div className="alert alert-danger">
          <strong>Error:</strong> {response.error}
        </div>
      )}

      <div className="mt-3">
        <h6>Expected Personalization Features:</h6>
        <ul className="small text-muted">
          <li>✅ Job Category Matching: Analyze skills → suggest relevant positions</li>
          <li>✅ Personalized Actions: Cover letters, interview prep, career guidance</li>
          <li>✅ Context Awareness: Use chat history + profile for better responses</li>
          <li>✅ Role-Based Responses: Different suggestions for employees vs guests</li>
          <li>✅ Smart Prompting: Enhanced LLM prompts with profile context</li>
        </ul>
      </div>
    </div>
  )
}