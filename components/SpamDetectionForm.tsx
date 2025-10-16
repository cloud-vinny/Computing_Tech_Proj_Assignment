'use client'

import { useState } from 'react'
import { detectSpam, checkAPIHealth } from '@/lib/api'

interface DetectionResult {
  is_spam: boolean
  confidence: number
  model_used: string
  message: string
}

export default function SpamDetectionForm() {
  const [emailText, setEmailText] = useState('')
  const [selectedModel, setSelectedModel] = useState('logistic')
  const [isLoading, setIsLoading] = useState(false)
  const [result, setResult] = useState<DetectionResult | null>(null)
  const [error, setError] = useState<string | null>(null)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    
    if (!emailText.trim()) {
      setError('Please enter some text to analyze')
      return
    }

    setIsLoading(true)
    setError(null)
    setResult(null)

    try {
      const response = await detectSpam(emailText, selectedModel)
      setResult(response)
    } catch (err) {
      const error = err as Error
      if (error.message?.includes('Network Error') || error.message?.includes('fetch')) {
        setError('Backend server is not running. Please start the FastAPI server on port 8000.')
      } else {
        setError('Failed to analyze the message. Please try again.')
      }
      console.error('Detection error:', err)
    } finally {
      setIsLoading(false)
    }
  }

  const handleCancel = () => {
    setEmailText('')
    setResult(null)
    setError(null)
  }

  const handleClear = () => {
    setEmailText('')
    setResult(null)
    setError(null)
  }

  return (
    <div className="max-w-2xl mx-auto">
      <div className="card">
        <form onSubmit={handleSubmit}>
          <div className="form-group">
            <label htmlFor="emailText" className="form-label">
              Enter email or message to analyze:
            </label>
            <textarea
              id="emailText"
              value={emailText}
              onChange={(e) => setEmailText(e.target.value)}
              className="form-input textarea"
              placeholder="Paste your email content here to check if it&apos;s spam..."
              rows={8}
              disabled={isLoading}
            />
            <div className="text-sm text-gray-600">
              {emailText.length} characters
            </div>
          </div>

          <div className="form-group">
            <label htmlFor="modelSelect" className="form-label">
              Choose AI Model:
            </label>
            <select
              id="modelSelect"
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              className="form-input"
              disabled={isLoading}
            >
              <option value="logistic">Logistic Regression (Best Performance)</option>
              <option value="naive_bayes">Naive Bayes (Text Classification)</option>
              <option value="kmeans">K-Means (Clustering)</option>
            </select>
          </div>

          {error && (
            <div className="result-card result-spam">
              <div className="text-sm font-semibold">{error}</div>
            </div>
          )}

          <div className="flex flex-col sm:flex-row gap-3">
            <button
              type="submit"
              disabled={isLoading || !emailText.trim()}
              className="btn btn-primary flex-1"
            >
              {isLoading ? (
                <>
                  <div className="loading"></div>
                  Analyzing...
                </>
              ) : (
                '🔍 Analyze for Spam'
              )}
            </button>
            
            <button
              type="button"
              onClick={handleCancel}
              disabled={isLoading}
              className="btn btn-secondary flex-1"
            >
              ❌ Cancel
            </button>
            
            <button
              type="button"
              onClick={handleClear}
              disabled={isLoading}
              className="btn btn-danger flex-1"
            >
              🗑️ Clear
            </button>
          </div>
        </form>

        {result && (
          <div className={`result-card ${result.is_spam ? 'result-spam' : 'result-ham'}`}>
            <div className="flex items-center justify-between mb-3">
              <h3 className="font-bold text-lg">
                {result.is_spam ? '🚨 Spam Detected!' : '✅ Legitimate Message'}
              </h3>
              <span className="text-sm font-bold bg-white px-3 py-1 rounded-full">
                {Math.round(result.confidence * 100)}% confidence
              </span>
            </div>
            <p className="text-sm mb-2"><strong>Model:</strong> {result.model_used}</p>
            <p className="text-sm mb-2">{result.message}</p>
            <div className="text-xs opacity-75">
              Analysis completed at {new Date().toLocaleTimeString()}
            </div>
          </div>
        )}
      </div>

      <div className="mt-8 card">
        <h3 className="font-bold text-lg mb-4">
          🤖 How it works:
        </h3>
        <ul className="text-sm space-y-2">
          <li>• Our AI analyzes the text content using advanced machine learning algorithms</li>
          <li>• The system checks for spam patterns, suspicious keywords, and content structure</li>
          <li>• Results include confidence scores to help you make informed decisions</li>
          <li>• All analysis is performed securely and your data is not stored</li>
        </ul>
      </div>
    </div>
  )
}
