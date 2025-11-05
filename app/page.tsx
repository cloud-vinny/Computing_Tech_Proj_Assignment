'use client'

import SpamDetectionForm from '@/components/SpamDetectionForm'
import SpamCharts from '@/components/SpamCharts'
import ThemeToggle from '@/components/ThemeToggle'

export default function Home() {
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

        {/* Data Visualization Charts */}
        <div className="mb-8">
          <SpamCharts />
        </div>
      </div>
    </main>
  )
}
