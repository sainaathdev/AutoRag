# 🚀 Free Deployment Guide

## Platform Comparison

| Platform | RAM | Persistent Storage | Free? | Difficulty |
|---|---|---|---|---|
| **HuggingFace Spaces** ⭐ | 16GB CPU | ✅ /data volume | ✅ Yes | Easy |
| **Streamlit Community Cloud** | 1GB | ❌ Ephemeral | ✅ Yes | Easiest |
| **Render** | 512MB | ❌ Ephemeral | ✅ Yes | Medium |

**Recommendation: HuggingFace Spaces** — best RAM (models need ~2GB), free persistent storage.

---

## Option 1: HuggingFace Spaces (Recommended) ⭐

### Prerequisites
- Free account at [huggingface.co](https://huggingface.co)
- Git installed

### Step 1: Create a New Space
1. Go to [huggingface.co/new-space](https://huggingface.co/new-space)
2. Fill in:
   - **Space name**: `autorag` (or anything you like)
   - **License**: MIT
   - **SDK**: Streamlit
   - **Visibility**: Public (required for free tier)
3. Click **Create Space**

### Step 2: Add Your API Key as a Secret
1. In your Space → go to **Settings** → **Variables and secrets**
2. Click **New secret**
3. Add: `GROQ_API_KEY` = `your_actual_groq_api_key`
4. Click **Save**

> ⚠️ NEVER put your API key in code or git. Secrets are injected as environment variables at runtime.

### Step 3: Push Your Code

```bash
# Clone your new HF Space repo
git clone https://huggingface.co/spaces/YOUR_USERNAME/autorag
cd autorag

# Copy your project files into it (exclude venv and data)
# OR initialize git in your existing project:
cd d:\sai_coding\autorag

git init
git add .
git commit -m "Initial deployment"

# Add HuggingFace as remote
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/autorag

# Push
git push hf main
```

### Step 4: Wait for Build
- HuggingFace will automatically install `requirements.txt`
- First build takes ~5-10 minutes (downloading models)
- Watch progress in the **Logs** tab of your Space

### Step 5: Access Your App
Your app will be live at:
```
https://YOUR_USERNAME-autorag.hf.space
```

---

## Option 2: Streamlit Community Cloud (Easiest, but limited RAM)

> ⚠️ Only recommended if you **disable the cross-encoder reranker** in config.yaml (`reranking_enabled: false`). The free tier has only ~1GB RAM which may not fit both models.

### Step 1: Push to GitHub
```bash
cd d:\sai_coding\autorag
git init
git add .
git commit -m "Deploy AutoRAG"

# Create a repo on github.com first, then:
git remote add origin https://github.com/YOUR_USERNAME/autorag.git
git push -u origin main
```

### Step 2: Deploy on Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click **New app**
4. Select your repo → branch: `main` → Main file: `app.py`
5. Click **Advanced settings** → add secret:
   ```
   GROQ_API_KEY = "your_key_here"
   ```
6. Click **Deploy**

---

## Important Notes for Both Platforms

### ⚡ First Startup is Slow (~3-5 min)
The embedding model (`all-MiniLM-L6-v2`, ~90MB) and cross-encoder (`ms-marco-MiniLM-L-6-v2`, ~80MB) are downloaded on first boot. Subsequent startups are fast.

### 💾 Data Persistence
- **HuggingFace**: ChromaDB and feedback data persist in `/data` — survives restarts ✅
- **Streamlit Cloud**: Data is lost on each restart ❌ (you'll need to re-upload documents)

### 🔑 Never Commit Secrets
Your `.gitignore` already excludes `.env`. Always use the platform's secret manager.

### 📦 Disabling Heavy Features to Save RAM

If you hit memory limits, edit `config.yaml`:
```yaml
retrieval:
  reranking_enabled: false   # Disables cross-encoder (~80MB RAM saved)

evaluation:
  ragas_enabled: false       # Disables RAGAS eval LLM calls (saves API calls)
  auto_evaluate: false       # Disables confidence eval (faster responses)
```

---

## Updating Your Deployment

After making code changes locally:
```bash
git add .
git commit -m "Update: describe your changes"
git push hf main      # for HuggingFace
# OR
git push origin main  # for Streamlit Cloud (GitHub)
```

The platform auto-rebuilds on every push.

---

## Cost Estimate: $0

| Component | Cost |
|---|---|
| Groq API | Free tier: 14,400 req/day |
| HuggingFace Spaces CPU | Free forever |
| ChromaDB | Self-hosted, no cost |
| Embedding models | Run locally in container, no cost |

**Total monthly cost: $0** 🎉
