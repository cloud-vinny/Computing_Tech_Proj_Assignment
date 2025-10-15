'use client'

import SpamDetectionForm from '@/components/SpamDetectionForm'
import Header from '@/components/Header'
import Footer from '@/components/Footer'

export default function Home() {
  return (
    <main className="min-h-screen">
      <Header />
      <div className="container py-12">
        <div className="max-w-4xl mx-auto">
          <div className="text-center mb-12">
            <div className="inline-flex items-center justify-center w-16 h-16 bg-primary-100 rounded-full mb-6">
              <span className="text-2xl">🛡️</span>
            </div>
            <h1 className="text-5xl font-bold text-gray-900 mb-6">
              AI-Powered Spam Detection
            </h1>
            <p className="text-xl text-gray-600 max-w-3xl mx-auto">
              Protect your digital communications with our advanced machine learning algorithms. 
              Simply paste any email or message below to instantly detect if it's spam.
            </p>
            <div className="mt-8 flex flex-wrap justify-center gap-4 text-sm text-gray-600">
              <div className="flex items-center">
                <span className="w-2 h-2 bg-green-500 rounded-full mr-2"></span>
                High Accuracy Detection
              </div>
              <div className="flex items-center">
                <span className="w-2 h-2 bg-blue-500 rounded-full mr-2"></span>
                Real-time Analysis
              </div>
              <div className="flex items-center">
                <span className="w-2 h-2 bg-purple-500 rounded-full mr-2"></span>
                Secure & Private
              </div>
            </div>
          </div>
          
          <SpamDetectionForm />
        </div>
      </div>
      <Footer />
    </main>
  )
}
