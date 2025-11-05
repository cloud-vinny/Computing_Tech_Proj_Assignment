import axios from 'axios'

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 
  (typeof window !== 'undefined' && window.location.hostname === 'localhost' 
    ? 'http://localhost:8000' 
    : 'https://computingtechprojassignment-production.up.railway.app')

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

export interface DatasetStatsResponse {
  total_samples: number
  spam_count: number
  ham_count: number
  spam_percentage: number
  ham_percentage: number
  balance_ratio: number
  average_text_length: number
  min_text_length: number
  max_text_length: number
  average_word_count: number
  training_samples_used: number
}

export interface DistributionItem {
  range: string
  spam: number
  ham: number
  total: number
}

export interface DatasetDistributionResponse {
  text_length_distribution: DistributionItem[]
  word_count_distribution: DistributionItem[]
}

export interface WordFrequency {
  word: string
  count: number
}

export interface DatasetFeaturesResponse {
  top_spam_words: WordFrequency[]
  top_ham_words: WordFrequency[]
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

// Dataset analysis functions
export const getDatasetStats = async (): Promise<DatasetStatsResponse> => {
  try {
    const response = await api.get<DatasetStatsResponse>('/dataset/stats')
    return response.data
  } catch (error) {
    if (axios.isAxiosError(error)) {
      throw new Error(error.response?.data?.detail || 'Failed to get dataset stats')
    }
    throw new Error('Network error occurred')
  }
}

export const getDatasetDistribution = async (): Promise<DatasetDistributionResponse> => {
  try {
    const response = await api.get<DatasetDistributionResponse>('/dataset/distribution')
    return response.data
  } catch (error) {
    if (axios.isAxiosError(error)) {
      throw new Error(error.response?.data?.detail || 'Failed to get dataset distribution')
    }
    throw new Error('Network error occurred')
  }
}

export const getDatasetFeatures = async (): Promise<DatasetFeaturesResponse> => {
  try {
    const response = await api.get<DatasetFeaturesResponse>('/dataset/features')
    return response.data
  } catch (error) {
    if (axios.isAxiosError(error)) {
      throw new Error(error.response?.data?.detail || 'Failed to get dataset features')
    }
    throw new Error('Network error occurred')
  }
}
