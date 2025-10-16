'use client'

import SpamDetectionForm from '@/components/SpamDetectionForm'

export default function Home() {
  return (
    <main className="min-h-screen flex items-center justify-center">
      <div className="container py-12">
        <div className="max-w-2xl mx-auto">
          <div className="text-center mb-8">
            <h1 className="text-4xl font-bold text-gray-900 mb-4">
              Spam Detection
            </h1>
            <p className="text-lg text-gray-600">
              Enter your text below to check if it&apos;s spam
            </p>
          </div>
          
          <SpamDetectionForm />
        </div>
      </div>
    </main>
  )
}
