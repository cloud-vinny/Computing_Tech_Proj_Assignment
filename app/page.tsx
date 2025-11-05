'use client'

import { useState } from 'react'
import SpamDetectionForm from '@/components/SpamDetectionForm'
import SpamCharts from '@/components/SpamCharts'
import DatasetCharts from '@/components/DatasetCharts'
import ThemeToggle from '@/components/ThemeToggle'

export default function Home() {
  const [activeTab, setActiveTab] = useState<'predictions' | 'dataset'>('predictions')

  return (
    <main className="min-h-screen">
      <div className="container py-12">
        {/* Theme Toggle - Top Right */}
        <div className="flex justify-end mb-4">
          <ThemeToggle />
        </div>
        
        <div className="max-w-2xl mx-auto mb-12">
          <div className="text-center mb-8">
            <h1 className="text-4xl font-bold mb-2">🛡️ Spam Detection</h1>
            <p className="text-lg text-gray-600 dark:text-gray-400">
              AI-powered spam detection using machine learning
            </p>
          </div>
          
          <SpamDetectionForm />
        </div>

        {/* Data Visualization Charts with Tabs */}
        <div className="mb-8">
          {/* Tab Navigation */}
          <div className="max-w-7xl mx-auto mb-6">
            <div className="flex border-b border-gray-300 dark:border-gray-700">
              <button
                onClick={() => setActiveTab('predictions')}
                className={`px-6 py-3 font-semibold transition-colors ${
                  activeTab === 'predictions'
                    ? 'border-b-2 border-blue-500 text-blue-600 dark:text-blue-400'
                    : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-200'
                }`}
              >
                📊 Your Predictions
              </button>
              <button
                onClick={() => setActiveTab('dataset')}
                className={`px-6 py-3 font-semibold transition-colors ${
                  activeTab === 'dataset'
                    ? 'border-b-2 border-blue-500 text-blue-600 dark:text-blue-400'
                    : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-200'
                }`}
              >
                📈 Dataset Insights
              </button>
            </div>
          </div>

          {/* Tab Content */}
          {activeTab === 'predictions' && <SpamCharts />}
          {activeTab === 'dataset' && <DatasetCharts />}
        </div>
      </div>
    </main>
  )
}
