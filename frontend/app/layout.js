import './globals.css'
import Script from 'next/script'

export const metadata = {
  title: 'Solace - Find Comfort in the Texts You Love',
  description: 'Share what you\'re going through and receive comforting passages from the Bible or Harry Potter with heartfelt guidance.',
  keywords: 'bible, verses, comfort, encouragement, faith, spiritual support, harry potter, wisdom',
}

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <head>
        {/* Google tag (gtag.js) */}
        <Script
          src="https://www.googletagmanager.com/gtag/js?id=G-7EYDGYRXNJ"
          strategy="afterInteractive"
        />
        <Script id="google-analytics" strategy="afterInteractive">
          {`
            window.dataLayer = window.dataLayer || [];
            function gtag(){dataLayer.push(arguments);}
            gtag('js', new Date());
            gtag('config', 'G-7EYDGYRXNJ');
          `}
        </Script>
        
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap" rel="stylesheet" />
      </head>
      <body>{children}</body>
    </html>
  )
}

