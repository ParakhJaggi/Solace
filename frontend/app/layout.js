import './globals.css'

export const metadata = {
  title: 'Solace - Find Comfort in Scripture',
  description: 'Share what you\'re going through and receive comforting Bible verses with heartfelt guidance.',
  keywords: 'bible, verses, comfort, encouragement, faith, spiritual support',
}

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap" rel="stylesheet" />
      </head>
      <body>{children}</body>
    </html>
  )
}

