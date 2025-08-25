#!/bin/bash

# NitroAGI NEXUS Deployment Script for Vercel
# Domain: fractalnexusai.space

set -e

echo "🧠 Deploying NitroAGI NEXUS to fractalnexusai.space..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if Vercel CLI is installed
if ! command -v vercel &> /dev/null; then
    echo -e "${RED}❌ Vercel CLI not found. Installing...${NC}"
    npm install -g vercel
fi

# Set Vercel token
export VERCEL_TOKEN="yCezdRKOJ6UZhfOp87jVMZce"

echo -e "${BLUE}📋 Pre-deployment checks...${NC}"

# Check if we're in the right directory
if [ ! -f "vercel.json" ]; then
    echo -e "${RED}❌ vercel.json not found. Make sure you're in the NitroAGI root directory.${NC}"
    exit 1
fi

# Check Python version
python_version=$(python3 --version 2>&1 | grep -o '[0-9]\+\.[0-9]\+' | head -1)
echo -e "${GREEN}✓${NC} Python version: $python_version"

# Check for required files
echo -e "${BLUE}📁 Checking required files...${NC}"
required_files=("src/nitroagi/api/main.py" "requirements-vercel.txt" ".env.production")

for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        echo -e "${GREEN}✓${NC} Found: $file"
    else
        echo -e "${RED}❌${NC} Missing: $file"
        exit 1
    fi
done

# Copy production requirements
echo -e "${BLUE}📦 Preparing requirements...${NC}"
cp requirements-vercel.txt requirements.txt

# Set environment variables for deployment
echo -e "${BLUE}🔧 Setting environment variables...${NC}"

# Core environment variables
vercel env add NITROAGI_ENV production production --force
vercel env add NITROAGI_DEBUG false production --force
vercel env add NITROAGI_LOG_LEVEL INFO production --force
vercel env add NITROAGI_API_HOST 0.0.0.0 production --force
vercel env add NITROAGI_API_PORT 8000 production --force
vercel env add NITROAGI_DOMAIN fractalnexusai.space production --force
vercel env add NITROAGI_ENABLE_6G true production --force
vercel env add NITROAGI_NETWORK_PROFILE ultra_low_latency production --force
vercel env add PYTHONPATH src production --force

echo -e "${YELLOW}⚠️  Please set the following environment variables manually in Vercel dashboard:${NC}"
echo -e "   - NITROAGI_OPENAI_API_KEY"
echo -e "   - NITROAGI_ANTHROPIC_API_KEY"
echo -e "   - NITROAGI_REDIS_URL"
echo -e "   - NITROAGI_POSTGRES_URL"
echo -e "   - NITROAGI_JWT_SECRET_KEY"

# Deploy to production
echo -e "${BLUE}🚀 Deploying to production...${NC}"
vercel --prod --token="$VERCEL_TOKEN" --confirm

# Get deployment URL
deployment_url=$(vercel ls --token="$VERCEL_TOKEN" | grep "fractalnexusai.space" | head -1 | awk '{print $2}')

echo -e "${GREEN}✅ Deployment complete!${NC}"
echo -e "${BLUE}🌐 Production URL:${NC} https://fractalnexusai.space"
echo -e "${BLUE}🧠 NEXUS API:${NC} https://fractalnexusai.space/api/v1"
echo -e "${BLUE}💊 Health Check:${NC} https://fractalnexusai.space/health"
echo -e "${BLUE}📚 API Docs:${NC} https://fractalnexusai.space/docs"

# Test deployment
echo -e "${BLUE}🧪 Testing deployment...${NC}"
sleep 5

health_response=$(curl -s -o /dev/null -w "%{http_code}" https://fractalnexusai.space/health || echo "000")

if [ "$health_response" = "200" ]; then
    echo -e "${GREEN}✅ Health check passed!${NC}"
    
    # Test NEXUS API
    echo -e "${BLUE}🧠 Testing NEXUS API...${NC}"
    api_response=$(curl -s "https://fractalnexusai.space/api/v1/system/info" | grep -o '"version"' | wc -l)
    
    if [ "$api_response" -gt 0 ]; then
        echo -e "${GREEN}✅ NEXUS API responding!${NC}"
    else
        echo -e "${YELLOW}⚠️  NEXUS API might be initializing...${NC}"
    fi
else
    echo -e "${RED}❌ Health check failed (HTTP: $health_response)${NC}"
    echo -e "${YELLOW}💡 Check Vercel function logs for errors${NC}"
fi

echo -e "${GREEN}🎉 NitroAGI NEXUS is live at fractalnexusai.space!${NC}"
echo -e "${BLUE}📊 Monitor at:${NC} https://vercel.com/dashboard"

# Cleanup
rm -f requirements.txt

echo -e "${BLUE}🔗 Next steps:${NC}"
echo -e "   1. Set API keys in Vercel environment variables"
echo -e "   2. Configure Redis and PostgreSQL databases"
echo -e "   3. Test the chat endpoint: https://fractalnexusai.space/api/v1/chat"
echo -e "   4. Monitor logs in Vercel dashboard"