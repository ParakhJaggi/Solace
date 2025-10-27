# Solace - Frontend

Find comfort in Scripture. A warm, elegant interface for the Bible Verse Companion.

## Features

- 🕊️ Clean, spiritual design with warm colors
- 📱 Mobile-first responsive layout
- ⚡ Fast, static site generation
- 🎨 Beautiful typography and spacing
- 💫 Smooth animations and transitions
- ♿ Accessible and user-friendly

## Setup

### 1. Install Dependencies

```bash
cd frontend
npm install
```

### 2. Configure API

The API URL is hardcoded to `https://solace-q068.onrender.com`. If you want to change it for local development, update `API_URL` in `app/page.js`.

### 3. Run Development Server

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

### 4. Build for Production

```bash
npm run build
```

This creates an optimized static export in the `out/` directory.

## Deploy to Render (Static Site)

### Option 1: Via Render Dashboard (Easiest)

1. Go to https://dashboard.render.com
2. Click "New +" → "Static Site"
3. Connect your GitHub repository
4. Configure:
   - **Name**: `solace-frontend` (or your choice)
   - **Branch**: `main`
   - **Root Directory**: `frontend`
   - **Build Command**: `npm install && npm run build`
   - **Publish Directory**: `out`
5. Click "Create Static Site"

### Option 2: Via render.yaml (Infrastructure as Code)

Add this to your repo root:

```yaml
services:
  - type: web
    name: solace-frontend
    env: static
    buildCommand: cd frontend && npm install && npm run build
    staticPublishPath: frontend/out
    routes:
      - type: rewrite
        source: /*
        destination: /index.html
```

Then deploy:
```bash
git add .
git commit -m "Add frontend"
git push
```

Render will auto-deploy!

## Project Structure

```
frontend/
├── app/
│   ├── layout.js       # Root layout with metadata
│   ├── page.js         # Main page (home)
│   └── globals.css     # Global styles
├── public/             # Static assets
├── next.config.js      # Next.js config (static export)
├── tailwind.config.js  # Tailwind CSS config
├── postcss.config.js   # PostCSS config
└── package.json        # Dependencies
```

## Customization

### Colors

Edit `tailwind.config.js` to change the warm color palette:

```js
colors: {
  warmCream: '#FBF8F3',   // Background
  softGold: '#D4AF37',     // Accent
  deepBlue: '#2C3E50',     // Text
  // ... more
}
```

### Typography

Font families are in `tailwind.config.js`:
- **Serif**: Georgia (headings, elegant)
- **Sans**: Inter (body text, readable)

### API Endpoint

Change `API_URL` in `app/page.js` line 5.

## Tech Stack

- **Framework**: Next.js 14 (App Router)
- **Styling**: Tailwind CSS
- **Deployment**: Render Static Site
- **API**: FastAPI backend on Render

## License

Made with care and compassion. ✨

