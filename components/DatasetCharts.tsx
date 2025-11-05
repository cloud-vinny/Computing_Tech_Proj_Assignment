'use client'

import { useEffect, useState } from 'react'
import dynamic from 'next/dynamic'
import { getDatasetStats, getDatasetDistribution, getDatasetFeatures, type DatasetStatsResponse, type DatasetDistributionResponse, type DatasetFeaturesResponse } from '@/lib/api'
import { useTheme } from '@/contexts/ThemeContext'

// Dynamically import Plotly to avoid SSR issues
const Plot = dynamic(() => import('react-plotly.js'), { ssr: false })

export default function DatasetCharts() {
  const { theme } = useTheme()
  const [stats, setStats] = useState<DatasetStatsResponse | null>(null)
  const [distribution, setDistribution] = useState<DatasetDistributionResponse | null>(null)
  const [features, setFeatures] = useState<DatasetFeaturesResponse | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [showCharts, setShowCharts] = useState(true)

  useEffect(() => {
    const loadDatasetData = async () => {
      setIsLoading(true)
      setError(null)
      
      try {
        // Load all dataset data in parallel
        const [statsData, distData, featuresData] = await Promise.all([
          getDatasetStats(),
          getDatasetDistribution(),
          getDatasetFeatures(),
        ])
        
        setStats(statsData)
        setDistribution(distData)
        setFeatures(featuresData)
      } catch (err) {
        const error = err as Error
        setError(error.message || 'Failed to load dataset information')
        console.error('Dataset loading error:', err)
      } finally {
        setIsLoading(false)
      }
    }

    loadDatasetData()
  }, [])

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

  if (isLoading) {
    return (
      <div className="card">
        <div className="text-center py-8">
          <div className="loading mx-auto mb-4"></div>
          <p className="text-lg">Loading dataset information...</p>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="card">
        <div className="text-center py-8">
          <p className="text-lg text-red-600 dark:text-red-400 mb-2">⚠️ Error Loading Dataset</p>
          <p className="text-sm text-gray-600 dark:text-gray-400">{error}</p>
          <p className="text-xs text-gray-500 dark:text-gray-500 mt-2">
            Make sure the backend server is running and dataset files are available.
          </p>
        </div>
      </div>
    )
  }

  if (!stats || !distribution || !features) {
    return (
      <div className="card">
        <div className="text-center py-8">
          <p className="text-lg">No dataset data available</p>
        </div>
      </div>
    )
  }

  // Pie chart data - Dataset distribution
  const datasetPieChartData = [
    {
      values: [stats.ham_count, stats.spam_count],
      labels: ['Legitimate Messages', 'Spam Messages'],
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
      hole: 0.4,
    },
  ]

  // Text length distribution bar chart
  const textLengthTrace1 = {
    x: distribution.text_length_distribution.map((d) => d.range),
    y: distribution.text_length_distribution.map((d) => d.spam),
    name: 'Spam',
    type: 'bar' as const,
    marker: { color: '#ef4444' },
    hovertemplate: '<b>%{x}</b><br>Spam: %{y}<extra></extra>',
  }

  const textLengthTrace2 = {
    x: distribution.text_length_distribution.map((d) => d.range),
    y: distribution.text_length_distribution.map((d) => d.ham),
    name: 'Legitimate',
    type: 'bar' as const,
    marker: { color: '#22c55e' },
    hovertemplate: '<b>%{x}</b><br>Legitimate: %{y}<extra></extra>',
  }

  // Word frequency charts
  const topSpamWords = features.top_spam_words.slice(0, 10).reverse()
  const topHamWords = features.top_ham_words.slice(0, 10).reverse()

  const spamWordsTrace = {
    x: topSpamWords.map((w) => w.count),
    y: topSpamWords.map((w) => w.word),
    type: 'bar' as const,
    orientation: 'h' as const,
    marker: { color: '#ef4444' },
    hovertemplate: '<b>%{y}</b><br>Count: %{x}<extra></extra>',
    name: 'Spam Words',
  }

  const hamWordsTrace = {
    x: topHamWords.map((w) => w.count),
    y: topHamWords.map((w) => w.word),
    type: 'bar' as const,
    orientation: 'h' as const,
    marker: { color: '#22c55e' },
    hovertemplate: '<b>%{y}</b><br>Count: %{x}<extra></extra>',
    name: 'Legitimate Words',
  }

  const pieLayout: any = {
    ...plotlyLayoutBase,
    title: {
      text: 'Training Dataset Distribution',
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

  const textLengthLayout: any = {
    ...plotlyLayoutBase,
    title: {
      text: 'Text Length Distribution',
      font: { size: 18, color: theme === 'dark' ? '#f9fafb' : '#111827' },
    },
    xaxis: {
      title: 'Text Length (characters)',
      gridcolor: theme === 'dark' ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.1)',
    },
    yaxis: {
      title: 'Number of Messages',
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

  const wordFrequencyLayout: any = {
    ...plotlyLayoutBase,
    title: {
      text: 'Top Words Frequency',
      font: { size: 18, color: theme === 'dark' ? '#f9fafb' : '#111827' },
    },
    xaxis: {
      title: 'Word Count',
      gridcolor: theme === 'dark' ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.1)',
    },
    yaxis: {
      title: 'Words',
      gridcolor: theme === 'dark' ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.1)',
    },
    showlegend: true,
    legend: {
      x: 0.5,
      y: -0.1,
      xanchor: 'center',
      orientation: 'h',
    },
  }

  return (
    <div className="max-w-7xl mx-auto">
      {/* Statistics Summary */}
      <div className="card mb-6">
        <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 mb-4">
          <h2 className="text-2xl font-bold">📊 Training Dataset Statistics</h2>
          <button
            onClick={() => setShowCharts(!showCharts)}
            className="btn btn-secondary text-sm"
          >
            {showCharts ? '👁️ Hide Charts' : '👁️ Show Charts'}
          </button>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 gap-4">
          <div className="stat-card">
            <div className="text-2xl font-bold">{stats.total_samples.toLocaleString()}</div>
            <div className="text-sm text-gray-600 dark:text-gray-400">Total Samples</div>
          </div>
          <div className="stat-card">
            <div className="text-2xl font-bold text-red-600">{stats.spam_count.toLocaleString()}</div>
            <div className="text-sm text-gray-600 dark:text-gray-400">Spam ({stats.spam_percentage}%)</div>
          </div>
          <div className="stat-card">
            <div className="text-2xl font-bold text-green-600">{stats.ham_count.toLocaleString()}</div>
            <div className="text-sm text-gray-600 dark:text-gray-400">Legitimate ({stats.ham_percentage}%)</div>
          </div>
          <div className="stat-card">
            <div className="text-2xl font-bold">{Math.round(stats.average_text_length)}</div>
            <div className="text-sm text-gray-600 dark:text-gray-400">Avg Text Length</div>
          </div>
          <div className="stat-card">
            <div className="text-2xl font-bold">{Math.round(stats.average_word_count)}</div>
            <div className="text-sm text-gray-600 dark:text-gray-400">Avg Word Count</div>
          </div>
        </div>

        <div className="mt-4 p-4 bg-gray-100 dark:bg-gray-800 rounded-lg">
          <p className="text-sm text-gray-700 dark:text-gray-300">
            <strong>Dataset Balance Ratio:</strong> {stats.balance_ratio} (spam:ham) | 
            <strong> Training Samples Used:</strong> {stats.training_samples_used.toLocaleString()} | 
            <strong> Text Length Range:</strong> {stats.min_text_length}-{stats.max_text_length} characters
          </p>
        </div>
      </div>

      {showCharts && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
          {/* Dataset Distribution Pie Chart */}
          <div className="card">
            <div className="chart-container">
              <Plot
                data={datasetPieChartData}
                layout={pieLayout}
                config={plotlyConfig}
                style={{ width: '100%', height: '400px' }}
              />
            </div>
          </div>

          {/* Text Length Distribution Bar Chart */}
          <div className="card">
            <div className="chart-container">
              <Plot
                data={[textLengthTrace1, textLengthTrace2]}
                layout={textLengthLayout}
                config={plotlyConfig}
                style={{ width: '100%', height: '400px' }}
              />
            </div>
          </div>
        </div>
      )}

      {showCharts && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Top Spam Words */}
          <div className="card">
            <h3 className="text-lg font-bold mb-4">Top Spam Words</h3>
            <div className="chart-container">
              <Plot
                data={[spamWordsTrace]}
                layout={{
                  ...wordFrequencyLayout,
                  title: {
                    text: 'Top 10 Spam Words',
                    font: { size: 16, color: theme === 'dark' ? '#f9fafb' : '#111827' },
                  },
                }}
                config={plotlyConfig}
                style={{ width: '100%', height: '400px' }}
              />
            </div>
          </div>

          {/* Top Legitimate Words */}
          <div className="card">
            <h3 className="text-lg font-bold mb-4">Top Legitimate Words</h3>
            <div className="chart-container">
              <Plot
                data={[hamWordsTrace]}
                layout={{
                  ...wordFrequencyLayout,
                  title: {
                    text: 'Top 10 Legitimate Words',
                    font: { size: 16, color: theme === 'dark' ? '#f9fafb' : '#111827' },
                  },
                }}
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

