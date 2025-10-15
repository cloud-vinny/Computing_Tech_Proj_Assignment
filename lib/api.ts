import axios from 'axios'

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

// Create axios instance
const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Types
export interface SpamRequest {
  text: string
  model: string
}

export interface SpamResponse {
  is_spam: boolean
  confidence: number
  model_used: string
  message: string
}

export interface HealthResponse {
  status: string
  models_loaded: {
    tfidf: boolean
    logistic_regression: boolean
    naive_bayes: boolean
    kmeans: boolean
  }
}

export interface ModelsResponse {
  available_models: Array<{
    name: string
    description: string
  }>
}

// API functions
export const detectSpam = async (text: string, model: string = 'logistic'): Promise<SpamResponse> => {
  try {
    const response = await api.post<SpamResponse>('/detect', {
      text,
      model,
    })
    return response.data
  } catch (error) {
    if (axios.isAxiosError(error)) {
      throw new Error(error.response?.data?.detail || 'Failed to detect spam')
    }
    throw new Error('Network error occurred')
  }
}

export const checkAPIHealth = async (): Promise<HealthResponse> => {
  try {
    const response = await api.get<HealthResponse>('/health')
    return response.data
  } catch (error) {
    if (axios.isAxiosError(error)) {
      throw new Error(error.response?.data?.detail || 'Health check failed')
    }
    throw new Error('Network error occurred')
  }
}

export const getAvailableModels = async (): Promise<ModelsResponse> => {
  try {
    const response = await api.get<ModelsResponse>('/models')
    return response.data
  } catch (error) {
    if (axios.isAxiosError(error)) {
      throw new Error(error.response?.data?.detail || 'Failed to get models')
    }
    throw new Error('Network error occurred')
  }
}

// Utility function to check if API is available
export const isAPIAvailable = async (): Promise<boolean> => {
  try {
    await checkAPIHealth()
    return true
  } catch {
    return false
  }
}
