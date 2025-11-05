// Prediction history storage utilities

export interface PredictionRecord {
  id: string
  timestamp: number
  text: string // Truncated text for display
  is_spam: boolean
  confidence: number
  model_used: string
}

const STORAGE_KEY = 'spam_detection_history'
const MAX_RECORDS = 100 // Store up to 100 most recent predictions

/**
 * Save a prediction to localStorage
 */
export function savePrediction(
  text: string,
  is_spam: boolean,
  confidence: number,
  model_used: string
): void {
  try {
    const history = getPredictionHistory()
    
    // Create new prediction record
    const prediction: PredictionRecord = {
      id: `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      timestamp: Date.now(),
      text: text.length > 50 ? text.substring(0, 50) + '...' : text,
      is_spam,
      confidence,
      model_used,
    }
    
    // Add to beginning and keep only last MAX_RECORDS
    const updatedHistory = [prediction, ...history].slice(0, MAX_RECORDS)
    
    localStorage.setItem(STORAGE_KEY, JSON.stringify(updatedHistory))
  } catch (error) {
    console.error('Error saving prediction:', error)
    // Don't throw - gracefully handle localStorage errors
  }
}

/**
 * Get all prediction history from localStorage
 */
export function getPredictionHistory(): PredictionRecord[] {
  try {
    const stored = localStorage.getItem(STORAGE_KEY)
    if (!stored) return []
    
    const history = JSON.parse(stored) as PredictionRecord[]
    return Array.isArray(history) ? history : []
  } catch (error) {
    console.error('Error reading prediction history:', error)
    return []
  }
}

/**
 * Clear all prediction history
 */
export function clearPredictionHistory(): void {
  try {
    localStorage.removeItem(STORAGE_KEY)
  } catch (error) {
    console.error('Error clearing prediction history:', error)
  }
}

/**
 * Get prediction statistics
 */
export function getPredictionStats() {
  const history = getPredictionHistory()
  
  if (history.length === 0) {
    return {
      total: 0,
      spamCount: 0,
      hamCount: 0,
      averageConfidence: 0,
      spamAverageConfidence: 0,
      hamAverageConfidence: 0,
    }
  }
  
  const spamCount = history.filter((p) => p.is_spam).length
  const hamCount = history.length - spamCount
  
  const totalConfidence = history.reduce((sum, p) => sum + p.confidence, 0)
  const averageConfidence = totalConfidence / history.length
  
  const spamPredictions = history.filter((p) => p.is_spam)
  const hamPredictions = history.filter((p) => !p.is_spam)
  
  const spamAverageConfidence =
    spamPredictions.length > 0
      ? spamPredictions.reduce((sum, p) => sum + p.confidence, 0) / spamPredictions.length
      : 0
  
  const hamAverageConfidence =
    hamPredictions.length > 0
      ? hamPredictions.reduce((sum, p) => sum + p.confidence, 0) / hamPredictions.length
      : 0
  
  return {
    total: history.length,
    spamCount,
    hamCount,
    averageConfidence,
    spamAverageConfidence,
    hamAverageConfidence,
  }
}

/**
 * Get predictions by model
 */
export function getPredictionsByModel() {
  const history = getPredictionHistory()
  const modelStats: Record<string, { count: number; spamCount: number; hamCount: number }> = {}
  
  history.forEach((prediction) => {
    if (!modelStats[prediction.model_used]) {
      modelStats[prediction.model_used] = {
        count: 0,
        spamCount: 0,
        hamCount: 0,
      }
    }
    
    modelStats[prediction.model_used].count++
    if (prediction.is_spam) {
      modelStats[prediction.model_used].spamCount++
    } else {
      modelStats[prediction.model_used].hamCount++
    }
  })
  
  return modelStats
}

