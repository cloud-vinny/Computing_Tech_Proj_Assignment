'use client'

import { useEffect, useState } from 'react'
import dynamic from 'next/dynamic'
import { getPredictionHistory, getPredictionStats, getPredictionsByModel, clearPredictionHistory, type PredictionRecord } from '@/lib/storage'
import { useTheme } from '@/contexts/ThemeContext'

// Dynamically import Plotly to avoid SSR issues
const Plot = dynamic(() => import('react-plotly.js'), { ssr: false })

export default function SpamCharts() {
  const { theme } = useTheme()
  const [history, setHistory] = useState<PredictionRecord[]>([])
  const [stats, setStats] = useState(getPredictionStats())
  const [showCharts, setShowCharts] = useState(true)

  // Load history on mount and when component updates
  useEffect(() => {
    const loadHistory = () => {
      const predictions = getPredictionHistory()
      setHistory(predictions)
      setStats(getPredictionStats())
    }

    loadHistory()
    
    // Listen for storage changes (when new predictions are added in other tabs)
    const handleStorageChange = () => {
      loadHistory()
    }
    
    // Listen for custom events (when new predictions are added in same window)
    const handleCustomStorage = () => {
      loadHistory()
    }
    
    window.addEventListener('storage', handleStorageChange)
    window.addEventListener('localStorageUpdate', handleCustomStorage)
    
    // Also check periodically for updates (fallback)
    const interval = setInterval(loadHistory, 2000)
    
    return () => {
      window.removeEventListener('storage', handleStorageChange)
      window.removeEventListener('localStorageUpdate', handleCustomStorage)
      clearInterval(interval)
    }
  }, [])

  const handleClearHistory = () => {
    if (confirm('Are you sure you want to clear all prediction history?')) {
      clearPredictionHistory()
      setHistory([])
      setStats(getPredictionStats())
    }
  }

  // Pie chart data
  const pieChartData = [
    {
      values: [stats.hamCount, stats.spamCount],
      labels: ['Legitimate Messages', 'Spam Detected'],
      type: 'pie' as const,
      marker: {
        colors: ['#22c55e', '#ef4444'],
        line: {
          color: theme === 'dark' ? '#1f2937' : '#ffffff',
          width: 2,
        },
      },
      textinfo: 'label+percent' as const,
      textposition: 'outside' as const,
      hoverinfo: 'label+percent+name' as const,
      hole: 0.4, // Donut chart style
    },
  ]

  // Bar chart data - confidence distribution
  const confidenceRanges = [
    { range: '0-20%', min: 0, max: 0.2 },
    { range: '20-40%', min: 0.2, max: 0.4 },
    { range: '40-60%', min: 0.4, max: 0.6 },
    { range: '60-80%', min: 0.6, max: 0.8 },
    { range: '80-100%', min: 0.8, max: 1.0 },
  ]

  const barChartData = confidenceRanges.map((range) => {
    const spamCount = history.filter(
      (p) => p.is_spam && p.confidence >= range.min && p.confidence < range.max
    ).length
    const hamCount = history.filter(
      (p) => !p.is_spam && p.confidence >= range.min && p.confidence < range.max
    ).length
    
    return {
      range: range.range,
      spam: spamCount,
      ham: hamCount,
    }
  })

  const barChartTrace1 = {
    x: confidenceRanges.map((r) => r.range),
    y: barChartData.map((d) => d.spam),
    name: 'Spam',
    type: 'bar' as const,
    marker: { color: '#ef4444' },
    hovertemplate: '<b>%{x}</b><br>Spam: %{y}<extra></extra>',
  }

  const barChartTrace2 = {
    x: confidenceRanges.map((r) => r.range),
    y: barChartData.map((d) => d.ham),
    name: 'Legitimate',
    type: 'bar' as const,
    marker: { color: '#22c55e' },
    hovertemplate: '<b>%{x}</b><br>Legitimate: %{y}<extra></extra>',
  }

  const plotlyConfig: any = {
    displayModeBar: true,
    displaylogo: false,
    modeBarButtonsToRemove: ['pan2d', 'lasso2d'],
    responsive: true,
  }

  const plotlyLayoutBase: any = {
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
    font: {
      color: theme === 'dark' ? '#f9fafb' : '#111827',
      family: 'system-ui, -apple-system, sans-serif',
    },
    autosize: true,
    margin: { l: 50, r: 50, t: 30, b: 50 },
  }

  const pieLayout: any = {
    ...plotlyLayoutBase,
    title: {
      text: 'Spam vs Legitimate Messages',
      font: { size: 18, color: theme === 'dark' ? '#f9fafb' : '#111827' },
    },
    showlegend: true,
    legend: {
      x: 0.5,
      y: -0.1,
      xanchor: 'center',
      orientation: 'h',
    },
  }

  const barLayout: any = {
    ...plotlyLayoutBase,
    title: {
      text: 'Confidence Score Distribution',
      font: { size: 18, color: theme === 'dark' ? '#f9fafb' : '#111827' },
    },
    xaxis: {
      title: 'Confidence Range',
      gridcolor: theme === 'dark' ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.1)',
    },
    yaxis: {
      title: 'Number of Predictions',
      gridcolor: theme === 'dark' ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.1)',
    },
    barmode: 'stack',
    showlegend: true,
    legend: {
      x: 0.5,
      y: -0.1,
      xanchor: 'center',
      orientation: 'h',
    },
  }

  if (history.length === 0) {
    return (
      <div className="card">
        <div className="text-center py-8">
          <p className="text-lg mb-2">📊 No predictions yet</p>
          <p className="text-sm text-gray-600 dark:text-gray-400">
            Start analyzing messages to see visualization data here
          </p>
        </div>
      </div>
    )
  }

  return (
    <div className="max-w-7xl mx-auto">
      {/* Statistics Summary */}
      <div className="card mb-6">
        <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 mb-4">
          <h2 className="text-2xl font-bold">📊 Prediction Statistics</h2>
          <div className="flex gap-2">
            <button
              onClick={() => setShowCharts(!showCharts)}
              className="btn btn-secondary text-sm"
            >
              {showCharts ? '👁️ Hide Charts' : '👁️ Show Charts'}
            </button>
            <button
              onClick={handleClearHistory}
              className="btn btn-danger text-sm"
            >
              🗑️ Clear History
            </button>
          </div>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 gap-4">
          <div className="stat-card">
            <div className="text-2xl font-bold">{stats.total}</div>
            <div className="text-sm text-gray-600 dark:text-gray-400">Total Predictions</div>
          </div>
          <div className="stat-card">
            <div className="text-2xl font-bold text-red-600">{stats.spamCount}</div>
            <div className="text-sm text-gray-600 dark:text-gray-400">Spam Detected</div>
          </div>
          <div className="stat-card">
            <div className="text-2xl font-bold text-green-600">{stats.hamCount}</div>
            <div className="text-sm text-gray-600 dark:text-gray-400">Legitimate</div>
          </div>
          <div className="stat-card">
            <div className="text-2xl font-bold">
              {(stats.averageConfidence * 100).toFixed(1)}%
            </div>
            <div className="text-sm text-gray-600 dark:text-gray-400">Avg Confidence</div>
          </div>
          <div className="stat-card">
            <div className="text-2xl font-bold">
              {stats.spamCount > 0
                ? ((stats.spamCount / stats.total) * 100).toFixed(1)
                : 0}
              %
            </div>
            <div className="text-sm text-gray-600 dark:text-gray-400">Spam Rate</div>
          </div>
        </div>
      </div>

      {showCharts && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Pie Chart */}
          <div className="card">
            <div className="chart-container">
              <Plot
                data={pieChartData}
                layout={pieLayout}
                config={plotlyConfig}
                style={{ width: '100%', height: '400px' }}
              />
            </div>
          </div>

          {/* Bar Chart */}
          <div className="card">
            <div className="chart-container">
              <Plot
                data={[barChartTrace1, barChartTrace2]}
                layout={barLayout}
                config={plotlyConfig}
                style={{ width: '100%', height: '400px' }}
              />
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

