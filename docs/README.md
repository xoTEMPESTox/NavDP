# NavDP Research Sandbox Documentation

This directory contains the GitHub Pages documentation site for the NavDP Research Sandbox.

## Pages

| File | URL | Description |
|------|-----|-------------|
| `index.html` | `/docs/` | Landing page with hero, contributions, tasks |
| `overview.html` | `/docs/overview.html` | Full project overview & repo structure |
| `lekiwi.html` | `/docs/lekiwi.html` | LeKiwi integration log (5 chapters) |
| `architecture.html` | `/docs/architecture.html` | System architecture & threading model |
| `installation.html` | `/docs/installation.html` | Step-by-step setup guide |
| `research_tools.html` | `/docs/research_tools.html` | Research tooling suite reference |

## Deploying to GitHub Pages

1. Go to **Settings → Pages** in the GitHub repository
2. Set **Source** to `Deploy from a branch`
3. Set **Branch** to `master` (or `main`) and **Folder** to `/docs`
4. Click **Save**

The site will be live at: `https://xoTEMPESTox.github.io/NavDP/`

## Local Preview

Open any HTML file directly in a browser, or use a local server:

```bash
cd docs/
python -m http.server 8080
# Open: http://localhost:8080
```

## Tech Stack

- Pure HTML + Vanilla CSS + Vanilla JS (no build step)
- Google Fonts: Outfit, Inter, JetBrains Mono
- No external CSS frameworks
