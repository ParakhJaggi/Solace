/**
 * Solace - Find comfort in the texts you love
 * 
 * Main application page
 * Supports: Bible (Christian/Jewish), Harry Potter, and Social Media
 */

'use client'

import { useState, useMemo } from 'react'

const API_URL = process.env.NEXT_PUBLIC_API_URL
const MAX_CHARS = 500
const TOAST_DURATION = 2000 // 2 seconds

export default function Home() {
  const [concern, setConcern] = useState('')
  const [tradition, setTradition] = useState('christian')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [streamingExplanation, setStreamingExplanation] = useState('')
  const [isStreaming, setIsStreaming] = useState(false)
  const [copyFeedback, setCopyFeedback] = useState({ show: false, message: '' })

  const handleSubmit = async (e) => {
    e.preventDefault()
    
    if (!concern.trim()) {
      setError('Please share what you\'re going through')
      return
    }

    setLoading(true)
    setError(null)
    setResult(null)
    setStreamingExplanation('')
    setIsStreaming(false)

    try {
      const response = await fetch(`${API_URL}/recommend/stream`, {
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

      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let tempVerses = null
      let tempExplanation = ''

      while (true) {
        const { value, done } = await reader.read()
        if (done) break

        const chunk = decoder.decode(value)
        const lines = chunk.split('\n')

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6))
              
              if (data.type === 'crisis') {
                // Crisis response - show message without verses
                setResult({ 
                  verses: [], 
                  explanation: data.content 
                })
                setLoading(false)
                
                // Scroll to results
                setTimeout(() => {
                  document.getElementById('results')?.scrollIntoView({ 
                    behavior: 'smooth', 
                    block: 'start' 
                  })
                }, 100)
              } else if (data.type === 'verses') {
                // Store verses and show them immediately
                tempVerses = data.verses
                setResult({ verses: tempVerses, explanation: '' })
                setLoading(false) // Stop loading spinner once verses appear
                
                // Scroll to results
                setTimeout(() => {
                  document.getElementById('results')?.scrollIntoView({ 
                    behavior: 'smooth', 
                    block: 'start' 
                  })
                }, 100)
              } else if (data.type === 'explanation_chunk') {
                // Stream explanation text
                tempExplanation += data.content
                setIsStreaming(true)
                setStreamingExplanation(tempExplanation)
              } else if (data.type === 'done') {
                // Finalize
                setIsStreaming(false)
                setResult({ 
                  verses: tempVerses, 
                  explanation: tempExplanation 
                })
                setStreamingExplanation('')
              } else if (data.error) {
                throw new Error(data.error)
              }
            } catch (parseError) {
              console.error('Parse error:', parseError)
            }
          }
        }
      }
      
    } catch (err) {
      setError(err.message || 'Something went wrong. Please try again.')
      setLoading(false)
      setIsStreaming(false)
    }
  }

  const handleReset = () => {
    setConcern('')
    setResult(null)
    setError(null)
    setStreamingExplanation('')
    setIsStreaming(false)
  }

  const handleShareVerse = async (verse) => {
    // Get emoji based on tradition
    const getEmoji = () => {
      if (tradition === 'christian') return '✝️'
      if (tradition === 'jewish') return '✡️'
      if (tradition === 'harry_potter') return '🪄'
      if (tradition === 'social_media') return '🐦'
      return '📖'
    }
    
    const shareText = `"${concern}"\n\n"${verse.text}"\n\n- ${verse.ref} ${getEmoji()}\n\nFind comfort in the texts you love at Solace\nsolace.parakhjaggi.com`
    
    try {
      await navigator.clipboard.writeText(shareText)
      setCopyFeedback({ show: true, message: 'Quote copied to clipboard!' })
      
      // Hide feedback after timeout
      setTimeout(() => {
        setCopyFeedback({ show: false, message: '' })
      }, TOAST_DURATION)
    } catch (err) {
      console.error('Failed to copy: ', err)
      setCopyFeedback({ show: true, message: 'Failed to copy. Please try again.' })
      
      setTimeout(() => {
        setCopyFeedback({ show: false, message: '' })
      }, TOAST_DURATION)
    }
  }

  // Memoize results section to prevent re-renders when typing
  const resultsSection = useMemo(() => {
    if (!result && !isStreaming) return null
    
    // Get the current explanation text (streaming or final)
    const explanationText = isStreaming ? streamingExplanation : (result?.explanation || '')
    
    return (
      <div id="results" className="space-y-6 animate-fade-in">
        {/* Explanation */}
        {explanationText && (
          <div className="bg-warmBeige rounded-xl p-6 md:p-8 border border-warmBeige">
            <div 
              className="prose lg:prose-lg prose-pre:p-0 prose-pre:m-0 prose-pre:bg-transparent text-deepBlue max-w-none"
              dangerouslySetInnerHTML={{ 
                __html: explanationText
                  .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
                  .replace(/\n\n/g, '</p><p>')
                  .replace(/^/, '<p>')
                  .replace(/$/, '</p>')
              }}
            />
          </div>
        )}

        {/* Verses */}
        {result?.verses && result.verses.length > 0 && (
          <div>
            <h2 className="text-xl font-serif text-deepBlue mb-4">
              References
            </h2>
            <div className="space-y-4">
              {result.verses.map((verse, index) => (
                <div 
                  key={index} 
                  className="verse-card flowtoken-slide-up"
                  style={{
                    animationDelay: `${index * 0.1}s`,
                    animationDuration: '0.4s',
                    animationTimingFunction: 'ease-out'
                  }}
                >
                  <div className="flex items-start justify-between gap-4 mb-3">
                    <h3 className="font-serif text-lg text-deepBlue">
                      {verse.url ? (
                        <a 
                          href={verse.url} 
                          target="_blank" 
                          rel="noopener noreferrer"
                          className="hover:text-softGold transition-colors underline decoration-softGold decoration-2 underline-offset-2 cursor-pointer"
                          title={`Click to view tweet: ${verse.url}`}
                        >
                          {verse.ref}
                        </a>
                      ) : (
                        verse.ref
                      )}
                    </h3>
                    <div className="flex items-center gap-2">
                      <span className="text-xs text-gentleGray bg-warmBeige px-3 py-1 rounded-full">
                        {verse.translation}
                      </span>
                      <button
                        onClick={() => handleShareVerse(verse)}
                        className="p-1 text-gentleGray hover:text-softGold transition-colors"
                        title="Share this quote"
                      >
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8.684 13.342C8.886 12.938 9 12.482 9 12c0-.482-.114-.938-.316-1.342m0 2.684a3 3 0 110-2.684m0 2.684l6.632 3.316m-6.632-6l6.632-3.316m0 0a3 3 0 105.367-2.684 3 3 0 00-5.367 2.684zm0 9.316a3 3 0 105.367 2.684 3 3 0 00-5.367-2.684z" />
                        </svg>
                      </button>
                    </div>
                  </div>
                  <p className="text-deepBlue leading-relaxed italic">
                    "{verse.text}"
                  </p>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    )
  }, [result, isStreaming, streamingExplanation])

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
                Select your source
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
                <option value="harry_potter">Harry Potter 🪄</option>
                <option value="social_media">Social Media 🐦</option>
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
                    Finding comfort...
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
        {resultsSection}

        {/* Copy Feedback Toast */}
        {copyFeedback.show && (
          <div className="fixed top-4 right-4 z-50 animate-fade-in">
            <div className="bg-softGold text-pureWhite px-4 py-3 rounded-lg shadow-lg flex items-center gap-2">
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
              </svg>
              <span className="font-medium">{copyFeedback.message}</span>
            </div>
          </div>
        )}

        {/* Empty State */}
        {!result && !loading && !isStreaming && (
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
              Share your heart, and we'll find passages to comfort you.
            </p>
          </div>
        )}
      </div>

      {/* Footer */}
      <footer className="bg-pureWhite border-t border-warmBeige mt-16">
        <div className="max-w-4xl mx-auto px-4 py-6 text-center">
          <p className="text-sm text-gentleGray">
            {tradition === 'harry_potter' 
              ? 'Harry Potter series by J.K. Rowling' 
              : tradition === 'social_media'
              ? 'Real comfort from Twitter/X users'
              : 'Biblical passages from the World English Bible (WEB) translation'}
          </p>
          <p className="text-xs text-gentleGray mt-2">
            Made with care and compassion
          </p>
          <p className="text-xs text-gentleGray mt-1">
            by <a href="https://www.linkedin.com/in/parakhjaggi/" target="_blank" rel="noopener noreferrer" className="hover:text-softGold transition-colors">Parakh Jaggi</a>
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

        @keyframes slide-up {
          from {
            opacity: 0;
            transform: translateY(20px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }

        .animate-fade-in {
          animation: fade-in 0.5s ease-out;
        }

        .flowtoken-fade-in {
          animation: fade-in 0.3s ease-out;
        }

        .flowtoken-slide-up {
          animation: slide-up 0.4s ease-out;
          animation-fill-mode: both;
        }
      `}</style>
    </main>
  )
}

