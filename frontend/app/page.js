'use client'

import { useState } from 'react'

const API_URL = process.env.NEXT_PUBLIC_API_URL
const MAX_CHARS = 500  // ~125 tokens

// Simple markdown-to-HTML converter for bold and italic
function parseMarkdown(text) {
  if (!text) return text
  
  // Bold: **text** → <strong>text</strong>
  let parsed = text.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
  
  // Italic: *text* → <em>text</em> (single asterisks not already part of bold)
  parsed = parsed.replace(/\*([^*]+)\*/g, '<em>$1</em>')
  
  return parsed
}

export default function Home() {
  const [concern, setConcern] = useState('')
  const [tradition, setTradition] = useState('christian')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  const handleSubmit = async (e) => {
    e.preventDefault()
    
    if (!concern.trim()) {
      setError('Please share what you\'re going through')
      return
    }

    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const response = await fetch(`${API_URL}/recommend`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 
          issue: concern,
          tradition: tradition 
        }),
      })

      if (!response.ok) {
        throw new Error('Unable to connect. Please try again.')
      }

      const data = await response.json()
      setResult(data)
      
      // Scroll to results
      setTimeout(() => {
        document.getElementById('results')?.scrollIntoView({ 
          behavior: 'smooth', 
          block: 'start' 
        })
      }, 100)
      
    } catch (err) {
      setError(err.message || 'Something went wrong. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  const handleReset = () => {
    setConcern('')
    setResult(null)
    setError(null)
  }

  return (
    <main className="min-h-screen">
      {/* Header */}
      <header className="bg-pureWhite border-b border-warmBeige">
        <div className="max-w-4xl mx-auto px-4 py-6">
          <div className="text-center">
            <h1 className="text-3xl md:text-4xl font-serif text-deepBlue mb-2">
              Solace
            </h1>
            <p className="text-gentleGray text-sm md:text-base">
              Find comfort in the texts you love
            </p>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <div className="max-w-4xl mx-auto px-4 py-8 md:py-12">
        
        {/* Input Section */}
        <div className="bg-pureWhite rounded-2xl shadow-lg border border-warmBeige p-6 md:p-8 mb-8">
          <form onSubmit={handleSubmit}>
            {/* Tradition Selector */}
            <div className="mb-6">
              <label 
                htmlFor="tradition" 
                className="block text-deepBlue font-medium mb-3 text-sm"
              >
                Select your faith tradition
              </label>
              <select
                id="tradition"
                value={tradition}
                onChange={(e) => setTradition(e.target.value)}
                className="w-full md:w-auto bg-pureWhite border-2 border-warmBeige rounded-lg px-4 py-2 
                           focus:outline-none focus:border-softGold focus:ring-2 focus:ring-softGold focus:ring-opacity-20
                           transition-all duration-200"
                disabled={loading}
              >
                <option value="christian">Christian (Old & New Testament)</option>
                <option value="jewish">Jewish (Torah & Tanakh)</option>
              </select>
            </div>

            <label 
              htmlFor="concern" 
              className="block text-deepBlue font-medium mb-3 text-lg"
            >
              What are you going through?
            </label>
            
            <textarea
              id="concern"
              value={concern}
              onChange={(e) => {
                setConcern(e.target.value)
                setError(null)
              }}
              onKeyDown={(e) => {
                // Submit on Enter (without Shift), allow Shift+Enter for new lines
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault()
                  if (concern.trim().length > 0 && !loading) {
                    handleSubmit(e)
                  }
                }
              }}
              placeholder="Share your heart... (e.g., 'I'm anxious about work' or 'I feel alone and discouraged')"
              rows={5}
              maxLength={MAX_CHARS}
              className="input-primary"
              disabled={loading}
            />
            
            <div className="flex justify-between items-center mt-2">
              <span className={`text-sm ${
                concern.length > MAX_CHARS * 0.9 
                  ? 'text-red-600 font-medium' 
                  : 'text-gentleGray'
              }`}>
                {concern.length} / {MAX_CHARS} characters
              </span>
              {concern.length > MAX_CHARS * 0.8 && (
                <span className="text-xs text-gentleGray">
                  {MAX_CHARS - concern.length} remaining
                </span>
              )}
            </div>
            
            <div className="flex gap-3 mt-4">
              <button
                type="submit"
                disabled={loading || concern.trim().length === 0}
                className="btn-primary flex-1"
              >
                {loading ? (
                  <span className="flex items-center justify-center gap-2">
                    <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
                      <circle 
                        className="opacity-25" 
                        cx="12" 
                        cy="12" 
                        r="10" 
                        stroke="currentColor" 
                        strokeWidth="4"
                        fill="none"
                      />
                      <path 
                        className="opacity-75" 
                        fill="currentColor" 
                        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                      />
                    </svg>
                    Seeking verses...
                  </span>
                ) : (
                  'Find Comfort'
                )}
              </button>
              
              {result && (
                <button
                  type="button"
                  onClick={handleReset}
                  className="px-6 py-3 text-gentleGray hover:text-deepBlue transition-colors"
                >
                  Clear
                </button>
              )}
            </div>
          </form>

          {/* Error Message */}
          {error && (
            <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg text-red-800">
              {error}
            </div>
          )}
        </div>

        {/* Results Section */}
        {result && (
          <div id="results" className="space-y-6 animate-fade-in">
            {/* Explanation */}
            <div className="bg-warmBeige rounded-xl p-6 md:p-8 border border-warmBeige">
              <div 
                className="text-deepBlue leading-relaxed text-lg whitespace-pre-wrap"
                dangerouslySetInnerHTML={{ __html: parseMarkdown(result.explanation) }}
              />
            </div>

            {/* Verses */}
            <div>
              <h2 className="text-xl font-serif text-deepBlue mb-4">
                References
              </h2>
              <div className="space-y-4">
                {result.verses.map((verse, index) => (
                  <div key={index} className="verse-card">
                    <div className="flex items-start justify-between gap-4 mb-3">
                      <h3 className="font-serif text-lg text-deepBlue">
                        {verse.ref}
                      </h3>
                      <span className="text-xs text-gentleGray bg-warmBeige px-3 py-1 rounded-full">
                        {verse.translation}
                      </span>
                    </div>
                    <p className="text-deepBlue leading-relaxed italic">
                      "{verse.text}"
                    </p>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Empty State */}
        {!result && !loading && (
          <div className="text-center py-12">
            <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-warmBeige mb-4">
              <svg 
                className="w-8 h-8 text-softGold" 
                fill="none" 
                stroke="currentColor" 
                viewBox="0 0 24 24"
              >
                <path 
                  strokeLinecap="round" 
                  strokeLinejoin="round" 
                  strokeWidth={2} 
                  d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" 
                />
              </svg>
            </div>
            <p className="text-gentleGray">
              Share your heart, and we'll find verses to comfort you.
            </p>
          </div>
        )}
      </div>

      {/* Footer */}
      <footer className="bg-pureWhite border-t border-warmBeige mt-16">
        <div className="max-w-4xl mx-auto px-4 py-6 text-center">
          <p className="text-sm text-gentleGray">
            All verses from the World English Bible (WEB) translation
          </p>
          <p className="text-xs text-gentleGray mt-2">
            Made with care and compassion
          </p>
        </div>
      </footer>

      <style jsx>{`
        @keyframes fade-in {
          from {
            opacity: 0;
            transform: translateY(10px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }

        .animate-fade-in {
          animation: fade-in 0.5s ease-out;
        }
      `}</style>
    </main>
  )
}

