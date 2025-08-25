# NitroAGI Deployment Strategy

## 🎯 Recommended Architecture: Hybrid Deployment

Given NitroAGI's requirements for AI model processing and real-time orchestration, we recommend a hybrid approach:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Frontend      │    │   API Gateway    │    │  AI Backend     │
│   (Vercel)      │────│   (Vercel)       │────│  (Railway)      │
│                 │    │                  │    │                 │
│ - React UI      │    │ - Auth           │    │ - AI Modules    │
│ - Dashboard     │    │ - Rate Limiting  │    │ - Orchestrator  │
│ - Chat Interface│    │ - Request Routing│    │ - Model Hosting │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                        │
                                │                        │
                        ┌───────▼────────┐      ┌───────▼─────────┐
                        │   Neon DB      │      │    Redis        │
                        │                │      │   (Upstash)     │
                        │ - User Data    │      │ - Session Cache │
                        │ - Conversations│      │ - Working Memory│
                        │ - System Logs  │      │ - Message Queue │
                        └────────────────┘      └─────────────────┘
```

## 🚀 Platform Choices

### Frontend: Vercel ✅
**Perfect for:**
- React/Next.js applications
- Static assets and CDN
- Automatic deployments from GitHub
- Edge functions for simple API routes

### Backend: Railway.app ✅
**Why Railway over Vercel for AI:**
- **Docker Support**: Deploy your complete Python environment
- **Persistent Storage**: Cache models and maintain state
- **No Timeout Limits**: Long-running AI inference
- **GPU Access**: Available for model acceleration
- **Real-time Communications**: WebSocket support
- **Background Jobs**: Continuous learning processes

### Database: Neon DB ✅
**Excellent choice because:**
- PostgreSQL compatibility
- Serverless scaling
- Built-in connection pooling
- Great for both platforms
- Automatic backups

### Cache/Memory: Upstash Redis ✅
**Perfect companion:**
- Serverless Redis
- Works with both Vercel and Railway
- Global edge locations
- Pay-per-request pricing

## 📁 Repository Structure for Hybrid Deployment

```
NitroAGI/
├── frontend/                    # Vercel deployment
│   ├── package.json
│   ├── next.config.js
│   ├── vercel.json
│   ├── pages/
│   │   ├── api/                # Simple API routes
│   │   │   ├── auth.js
│   │   │   └── proxy.js        # Proxy to Railway backend
│   │   └── dashboard.js
│   └── components/
│
├── backend/                     # Railway deployment
│   ├── Dockerfile
│   ├── railway.json
│   ├── requirements.txt
│   ├── src/
│   │   ├── nitroagi/
│   │   ├── modules/
│   │   └── api/
│   └── scripts/
│
├── shared/                      # Shared types/utilities
│   ├── types.py
│   ├── schemas.py
│   └── constants.py
│
└── docs/
    └── deployment/
        ├── vercel-setup.md
        ├── railway-setup.md
        └── neon-db-setup.md
```

## 🔧 Configuration Files

### Frontend: `vercel.json`
```json
{
  "framework": "nextjs",
  "functions": {
    "pages/api/**/*.js": {
      "maxDuration": 30
    }
  },
  "env": {
    "NEXT_PUBLIC_API_URL": "@api-url",
    "NEON_DATABASE_URL": "@neon-db-url",
    "UPSTASH_REDIS_URL": "@upstash-redis-url"
  },
  "headers": [
    {
      "source": "/api/(.*)",
      "headers": [
        { "key": "Access-Control-Allow-Origin", "value": "*" },
        { "key": "Access-Control-Allow-Methods", "value": "GET, POST, PUT, DELETE, OPTIONS" }
      ]
    }
  ]
}
```

### Backend: `railway.json`
```json
{
  "build": {
    "builder": "DOCKERFILE"
  },
  "deploy": {
    "startCommand": "uvicorn src.nitroagi.api.main:app --host 0.0.0.0 --port $PORT",
    "restartPolicyType": "ON_FAILURE",
    "restartPolicyMaxRetries": 10
  },
  "environments": {
    "production": {
      "variables": {
        "PYTHON_VERSION": "3.11",
        "ENVIRONMENT": "production"
      }
    }
  }
}
```

### Backend: `Dockerfile`
```dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Install the package
RUN pip install -e .

# Expose port
EXPOSE $PORT

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:$PORT/health || exit 1

# Start command (overridden by railway.json)
CMD ["uvicorn", "src.nitroagi.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 🗄️ Database Setup with Neon

### Connection Configuration
```python
# src/nitroagi/core/database.py
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

# Neon connection string
DATABASE_URL = os.getenv("NEON_DATABASE_URL")

# Configure for Neon's connection pooling
engine = create_engine(
    DATABASE_URL,
    poolclass=NullPool,  # Neon handles pooling
    echo=False,
    connect_args={
        "sslmode": "require",
        "connect_timeout": 10,
    }
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
```

### Schema Design for NitroAGI
```sql
-- User management
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    preferences JSONB
);

-- Conversations and context
CREATE TABLE conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id INTEGER REFERENCES users(id),
    title VARCHAR(500),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    metadata JSONB
);

-- Messages with AI module tracking
CREATE TABLE messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID REFERENCES conversations(id),
    role VARCHAR(20) NOT NULL, -- 'user', 'assistant', 'system'
    content TEXT NOT NULL,
    module_responses JSONB, -- Track which modules contributed
    created_at TIMESTAMP DEFAULT NOW()
);

-- AI module performance tracking
CREATE TABLE module_metrics (
    id SERIAL PRIMARY KEY,
    module_name VARCHAR(100) NOT NULL,
    response_time_ms INTEGER,
    confidence_score FLOAT,
    created_at TIMESTAMP DEFAULT NOW(),
    metadata JSONB
);

-- Knowledge base for learning
CREATE TABLE knowledge_entries (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content TEXT NOT NULL,
    embedding VECTOR(1536), -- For semantic search
    source VARCHAR(255),
    created_at TIMESTAMP DEFAULT NOW(),
    tags TEXT[]
);

-- Create indexes for performance
CREATE INDEX idx_conversations_user_id ON conversations(user_id);
CREATE INDEX idx_messages_conversation_id ON messages(conversation_id);
CREATE INDEX idx_messages_created_at ON messages(created_at);
CREATE INDEX idx_knowledge_entries_tags ON knowledge_entries USING GIN(tags);
```

## 🚀 Deployment Steps

### 1. Neon Database Setup
```bash
# 1. Create Neon account and database
# 2. Copy connection string
# 3. Run migrations
neonctl auth
neonctl databases create nitroagi-prod
```

### 2. Railway Backend Deployment
```bash
# 1. Connect Railway to GitHub
# 2. Create new project from NitroAGI repo
# 3. Set environment variables
railway login
railway link
railway env set NEON_DATABASE_URL="your_connection_string"
railway env set OPENAI_API_KEY="your_key"
railway deploy
```

### 3. Vercel Frontend Deployment
```bash
# 1. Connect Vercel to GitHub
# 2. Deploy frontend directory
vercel login
vercel link
vercel env add NEXT_PUBLIC_API_URL production
vercel deploy --prod
```

### 4. Connect Services
```bash
# Update environment variables to connect services
# Frontend → Backend → Database
```

## 💰 Cost Estimation

### Monthly Costs (Estimated)
- **Vercel Pro**: $20/month (frontend hosting)
- **Railway**: $5-50/month (depending on usage)
- **Neon**: $0-25/month (generous free tier)
- **Upstash Redis**: $0-10/month (pay-per-request)
- **Total**: ~$25-105/month

### Free Tier Limits
- **Vercel**: 100GB bandwidth, edge functions
- **Railway**: $5/month credit
- **Neon**: 10GB storage, 1 database
- **Upstash**: 10K commands/day

## 🔒 Security & Environment Variables

### Frontend (.env.local)
```bash
NEXT_PUBLIC_API_URL=https://nitroagi-backend.railway.app
NEON_DATABASE_URL=postgresql://...
NEXTAUTH_SECRET=your_secret
NEXTAUTH_URL=https://nitroagi.vercel.app
```

### Backend
```bash
DATABASE_URL=postgresql://...
REDIS_URL=redis://...
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
SECRET_KEY=your_jwt_secret
CORS_ORIGINS=https://nitroagi.vercel.app
```

## 📊 Monitoring & Analytics

### Health Checks
```python
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow(),
        "version": "1.0.0",
        "database": await check_db_connection(),
        "modules": await check_ai_modules()
    }
```

### Performance Monitoring
- **Railway**: Built-in metrics and logs
- **Vercel**: Analytics and performance insights
- **Neon**: Database performance monitoring
- **Upstash**: Redis usage analytics

## 🔄 CI/CD Pipeline

### GitHub Actions
```yaml
# .github/workflows/deploy.yml
name: Deploy NitroAGI
on:
  push:
    branches: [main]

jobs:
  deploy-frontend:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Deploy to Vercel
        uses: amondnet/vercel-action@v25
        with:
          vercel-token: ${{ secrets.VERCEL_TOKEN }}
          vercel-org-id: ${{ secrets.VERCEL_ORG_ID }}
          vercel-project-id: ${{ secrets.VERCEL_PROJECT_ID }}

  deploy-backend:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Deploy to Railway
        run: |
          npm install -g @railway/cli
          railway login --token ${{ secrets.RAILWAY_TOKEN }}
          railway deploy
```

## ✅ Advantages of This Architecture

1. **Scalability**: Frontend scales globally, backend scales with demand
2. **Performance**: AI processing on dedicated servers, fast frontend delivery
3. **Cost-Effective**: Pay only for what you use
4. **Development Experience**: Familiar tools and workflows
5. **Reliability**: Managed services with high uptime
6. **Future-Proof**: Easy to migrate or scale individual components

This hybrid approach gives you the best of both worlds: Vercel's excellent frontend experience with Railway's robust backend capabilities, all connected to Neon's modern PostgreSQL database.
