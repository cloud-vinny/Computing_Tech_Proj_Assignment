'use client'

import { useTheme } from '@/contexts/ThemeContext'
import { useState, useEffect } from 'react'

export default function ThemeToggle() {
  const { theme, toggleTheme } = useTheme()
  const [mounted, setMounted] = useState(false)

  // Prevent hydration mismatch
  useEffect(() => {
    setMounted(true)
  }, [])

  if (!mounted) {
    // Return a placeholder during SSR to prevent build errors
    return (
      <button
        className="theme-toggle"
        aria-label="Toggle theme"
        disabled
      >
        <span>🌙</span>
        <span className="hidden sm:inline">Dark Mode</span>
      </button>
    )
  }

  return (
    <button
      onClick={toggleTheme}
      className="theme-toggle"
      aria-label={`Switch to ${theme === 'light' ? 'dark' : 'light'} mode`}
      title={`Switch to ${theme === 'light' ? 'dark' : 'light'} mode`}
    >
      {theme === 'light' ? (
        <>
          <span>🌙</span>
          <span className="hidden sm:inline">Dark Mode</span>
        </>
      ) : (
        <>
          <span>☀️</span>
          <span className="hidden sm:inline">Light Mode</span>
        </>
      )}
    </button>
  )
}

