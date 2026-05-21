# Deploying to Render

This guide will help you deploy the Lesson Query API to Render for free.

## Prerequisites

1. **GitHub Account** — Push code to GitHub
2. **Render Account** — Sign up at https://render.com
3. **Environment Variables** — Have your Supabase and OpenAI API keys ready

## Step-by-Step Deployment

### 1. Push Code to GitHub

```bash
# Initialize git repo (if not already done)
git init
git add .
git commit -m "Initial commit: FastAPI lesson query server"

# Create a new repo on GitHub and push
git remote add origin https://github.com/YOUR_USERNAME/lesson-query-api.git
git branch -M main
git push -u origin main
```

### 2. Create a Render Account

1. Go to https://render.com
2. Sign up with GitHub (easier for deployment)
3. Click "New +" → "Web Service"

### 3. Connect Your GitHub Repository

1. Select "Public GitHub repository"
2. Paste your repo URL: `https://github.com/YOUR_USERNAME/lesson-query-api`
3. Click "Connect"

### 4. Configure the Service

Fill in these details:

| Field | Value |
|-------|-------|
| **Name** | `lesson-query-api` |
| **Environment** | `Python 3` |
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `uvicorn main:app --host 0.0.0.0 --port $PORT` |
| **Instance Type** | `Free` |

### 5. Add Environment Variables

Click "Advanced" and add these variables:

```
OPENAI_API_KEY = your_openai_api_key_here
SUPABASE_URL = your_supabase_url_here
SUPABASE_KEY = your_supabase_key_here
```

**How to get these:**

- **OPENAI_API_KEY**: From https://platform.openai.com/api-keys
- **SUPABASE_URL & SUPABASE_KEY**: From your Supabase project settings

### 6. Deploy

Click "Create Web Service" — Render will:
1. Build the Docker image
2. Deploy the service
3. Give you a live URL (e.g., `https://lesson-query-api.onrender.com`)

## Testing Your API

Once deployed, test it:

```bash
# Health check
curl https://your-app.onrender.com/health

# Get catalog
curl https://your-app.onrender.com/catalog

# Query lessons (replace with your question)
curl -X POST https://your-app.onrender.com/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I teach family members?"}'

# Get answer
curl -X POST https://your-app.onrender.com/answer \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I teach family members?"}'
```

Or visit: `https://your-app.onrender.com/docs` for interactive API documentation

## Important Notes

### Cold Starts
On the free tier, the service goes to sleep after 15 minutes of inactivity. The first request after sleep takes ~30 seconds. This is normal.

### Auto-Deployment
Every time you push to GitHub (`git push`), Render automatically redeploys your API.

### Updating Code

```bash
# Make changes locally
nano query.py  # or edit in your editor

# Commit and push
git add .
git commit -m "Fixed section matching logic"
git push origin main

# Render auto-deploys (watch the dashboard)
```

## Troubleshooting

### Check Deployment Logs
1. Go to your service on Render dashboard
2. Click "Logs" tab
3. Look for errors

### Common Issues

**"ModuleNotFoundError: No module named 'query'"**
- Make sure `query.py`, `config.py`, `constants.py`, `supabase_client.py` are all committed to GitHub

**"OPENAI_API_KEY not found"**
- Check environment variables in Render dashboard
- Make sure the values are correctly set

**"Connection timeout to Supabase"**
- Verify `SUPABASE_URL` and `SUPABASE_KEY` are correct
- Check your Supabase project is active

## Next Steps

- Create a frontend (React, Vue, etc.) that calls your API
- Set up monitoring/alerts in Render dashboard
- Consider upgrading to paid plan for better performance (no cold starts)

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API info |
| GET | `/health` | Health check |
| GET | `/catalog` | List lessons |
| POST | `/query` | Search lessons |
| POST | `/answer` | Get AI answer |
| GET | `/docs` | Swagger UI |

## Questions?

- Render docs: https://render.com/docs
- FastAPI docs: https://fastapi.tiangolo.com
- OpenAI docs: https://platform.openai.com/docs
