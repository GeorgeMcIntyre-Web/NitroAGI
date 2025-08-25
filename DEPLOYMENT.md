# 🚀 NitroAGI NEXUS Deployment Guide

## 🌐 Production Deployment to fractalnexusai.space

### Prerequisites

1. **Vercel Account**: Ensure you have access to the Vercel project
2. **Domain**: `fractalnexusai.space` configured in Vercel
3. **Token**: `yCezdRKOJ6UZhfOp87jVMZce`

### Quick Deployment

#### Option 1: PowerShell (Windows)
```powershell
# Run the deployment script
.\scripts\deploy-vercel.ps1

# With environment variable setup
.\scripts\deploy-vercel.ps1 -SetEnvVars

# Test existing deployment only
.\scripts\deploy-vercel.ps1 -TestOnly
```

#### Option 2: Bash (Linux/Mac)
```bash
# Make script executable
chmod +x scripts/deploy-vercel.sh

# Run deployment
./scripts/deploy-vercel.sh
```

#### Option 3: Manual Deployment
```bash
# Install Vercel CLI
npm install -g vercel

# Set token
export VERCEL_TOKEN="yCezdRKOJ6UZhfOp87jVMZce"

# Deploy
vercel --prod --confirm
```

### 🔧 Environment Variables

Set these in the Vercel dashboard at https://vercel.com/dashboard:

#### Required Variables
```env
NITROAGI_OPENAI_API_KEY=sk-your-openai-key
NITROAGI_ANTHROPIC_API_KEY=your-anthropic-key
NITROAGI_JWT_SECRET_KEY=your-super-secret-jwt-key
```

#### Database (Optional for MVP)
```env
NITROAGI_REDIS_URL=redis://your-redis-instance:6379
NITROAGI_POSTGRES_URL=postgresql://user:pass@host:5432/nitroagi
```

#### Auto-configured Variables
These are set automatically by the deployment script:
- `NITROAGI_ENV=production`
- `NITROAGI_DEBUG=false`
- `NITROAGI_DOMAIN=fractalnexusai.space`
- `NITROAGI_ENABLE_6G=true`
- `PYTHONPATH=src`

### 🧪 Testing the Deployment

#### Health Check
```bash
curl https://fractalnexusai.space/health
```

#### NEXUS System Info
```bash
curl https://fractalnexusai.space/api/v1/system/info
```

#### Chat API Test
```bash
curl -X POST https://fractalnexusai.space/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello NEXUS!"}'
```

#### Interactive API Documentation
Visit: https://fractalnexusai.space/docs

### 📊 Monitoring

#### Vercel Dashboard
- **Logs**: https://vercel.com/dashboard
- **Analytics**: Built-in Vercel analytics
- **Functions**: Monitor serverless function performance

#### Health Monitoring
```bash
# Continuous health check
watch -n 10 'curl -s https://fractalnexusai.space/health | jq'
```

### 🏗️ Architecture on Vercel

```
Internet → Vercel Edge Network → Serverless Functions
                                        ↓
                              NitroAGI NEXUS API
                                        ↓
                              Redis/PostgreSQL (External)
```

### 🔒 Security Considerations

1. **API Keys**: Store in Vercel environment variables (encrypted)
2. **CORS**: Configured for fractalnexusai.space domain
3. **Rate Limiting**: Built-in Vercel rate limiting + app-level limits
4. **HTTPS**: Automatic SSL/TLS via Vercel

### 🚨 Troubleshooting

#### Common Issues

1. **Function Timeout**
   - Vercel has 30-second timeout for serverless functions
   - Heavy AI operations might need optimization

2. **Cold Starts**
   - First request after inactivity may be slower
   - Consider warming functions for production

3. **Memory Limits**
   - Vercel Pro: 3008 MB limit per function
   - Large models (like HuggingFace) might hit limits

4. **Environment Variables Not Working**
   ```bash
   # Re-deploy after setting env vars
   vercel --prod --force
   ```

#### Debug Commands
```bash
# View deployment logs
vercel logs https://fractalnexusai.space

# Check function details
vercel inspect https://fractalnexusai.space

# Test locally with production env
vercel dev --prod
```

### 📈 Performance Optimization

#### Vercel-Specific Optimizations
1. **Edge Functions**: Static responses cached at edge
2. **Serverless Functions**: Auto-scaling based on demand
3. **CDN**: Global distribution via Vercel's CDN

#### Application Optimizations
1. **Lightweight Dependencies**: Using `requirements-vercel.txt`
2. **Lazy Loading**: AI models loaded on demand
3. **Caching**: Response caching for common queries

### 🔄 CI/CD Pipeline

#### Automatic Deployment
Vercel automatically deploys on:
- Push to `main` branch (if GitHub connected)
- Manual trigger via CLI/dashboard

#### Manual Deployment Workflow
1. Test locally: `vercel dev`
2. Deploy to preview: `vercel`
3. Deploy to production: `vercel --prod`

### 🌍 Custom Domain Setup

The domain `fractalnexusai.space` should be configured in Vercel:

1. **Domain Settings**: Vercel Dashboard → Project → Settings → Domains
2. **DNS Configuration**: Point to Vercel's nameservers
3. **SSL Certificate**: Automatic via Vercel

### 📋 Deployment Checklist

- [ ] Vercel CLI installed
- [ ] Environment variables set
- [ ] API keys configured
- [ ] Health check passes
- [ ] API endpoints responding
- [ ] CORS configured correctly
- [ ] SSL certificate active
- [ ] Monitoring setup

### 🎯 Next Steps

1. **Database Setup**: Configure Redis/PostgreSQL for full functionality
2. **Monitoring**: Set up comprehensive monitoring and alerts
3. **CDN**: Optimize static asset delivery
4. **Scaling**: Configure auto-scaling parameters
5. **Backup**: Implement backup strategy for data

### 🆘 Support

- **Vercel Docs**: https://vercel.com/docs
- **NitroAGI Issues**: GitHub Issues tab
- **Community**: Discord/Slack (if available)

---

**🧠 NitroAGI NEXUS is now live at fractalnexusai.space!**

*The Neural Executive Unit System is ready to revolutionize AI interactions.* 🚀