# NitroAGI NEXUS Deployment Script for Vercel (PowerShell)
# Domain: fractalnexusai.space

param(
    [switch]$SkipChecks,
    [switch]$SetEnvVars,
    [switch]$TestOnly
)

Write-Host "🧠 Deploying NitroAGI NEXUS to fractalnexusai.space..." -ForegroundColor Cyan

# Set Vercel token
$env:VERCEL_TOKEN = "yCezdRKOJ6UZhfOp87jVMZce"

if (-not $SkipChecks) {
    Write-Host "📋 Pre-deployment checks..." -ForegroundColor Blue
    
    # Check if Vercel CLI is installed
    try {
        vercel --version | Out-Null
        Write-Host "✓ Vercel CLI found" -ForegroundColor Green
    } catch {
        Write-Host "❌ Vercel CLI not found. Please install with: npm install -g vercel" -ForegroundColor Red
        exit 1
    }
    
    # Check if we're in the right directory
    if (-not (Test-Path "vercel.json")) {
        Write-Host "❌ vercel.json not found. Make sure you're in the NitroAGI root directory." -ForegroundColor Red
        exit 1
    }
    
    # Check Python version
    $pythonVersion = python --version 2>&1
    Write-Host "✓ Python: $pythonVersion" -ForegroundColor Green
    
    # Check for required files
    Write-Host "📁 Checking required files..." -ForegroundColor Blue
    $requiredFiles = @("src/nitroagi/api/main.py", "requirements-vercel.txt", ".env.production")
    
    foreach ($file in $requiredFiles) {
        if (Test-Path $file) {
            Write-Host "✓ Found: $file" -ForegroundColor Green
        } else {
            Write-Host "❌ Missing: $file" -ForegroundColor Red
            exit 1
        }
    }
}

# Copy production requirements
Write-Host "📦 Preparing requirements..." -ForegroundColor Blue
Copy-Item "requirements-vercel.txt" "requirements.txt" -Force

if ($SetEnvVars) {
    # Set environment variables for deployment
    Write-Host "🔧 Setting environment variables..." -ForegroundColor Blue
    
    $envVars = @{
        "NITROAGI_ENV" = "production"
        "NITROAGI_DEBUG" = "false"
        "NITROAGI_LOG_LEVEL" = "INFO"
        "NITROAGI_API_HOST" = "0.0.0.0"
        "NITROAGI_API_PORT" = "8000"
        "NITROAGI_DOMAIN" = "fractalnexusai.space"
        "NITROAGI_ENABLE_6G" = "true"
        "NITROAGI_NETWORK_PROFILE" = "ultra_low_latency"
        "PYTHONPATH" = "src"
    }
    
    foreach ($key in $envVars.Keys) {
        try {
            vercel env add $key $envVars[$key] production --force
            Write-Host "✓ Set: $key" -ForegroundColor Green
        } catch {
            Write-Host "⚠️  Failed to set: $key" -ForegroundColor Yellow
        }
    }
    
    Write-Host "⚠️  Please set the following environment variables manually in Vercel dashboard:" -ForegroundColor Yellow
    Write-Host "   - NITROAGI_OPENAI_API_KEY" -ForegroundColor White
    Write-Host "   - NITROAGI_ANTHROPIC_API_KEY" -ForegroundColor White
    Write-Host "   - NITROAGI_REDIS_URL" -ForegroundColor White
    Write-Host "   - NITROAGI_POSTGRES_URL" -ForegroundColor White
    Write-Host "   - NITROAGI_JWT_SECRET_KEY" -ForegroundColor White
}

if ($TestOnly) {
    Write-Host "🧪 Testing existing deployment..." -ForegroundColor Blue
} else {
    # Deploy to production
    Write-Host "🚀 Deploying to production..." -ForegroundColor Blue
    try {
        vercel --prod --token="$env:VERCEL_TOKEN" --confirm
        Write-Host "✅ Deployment complete!" -ForegroundColor Green
    } catch {
        Write-Host "❌ Deployment failed!" -ForegroundColor Red
        Write-Host $_.Exception.Message -ForegroundColor Red
        exit 1
    }
}

Write-Host "🌐 Production URL: https://fractalnexusai.space" -ForegroundColor Blue
Write-Host "🧠 NEXUS API: https://fractalnexusai.space/api/v1" -ForegroundColor Blue
Write-Host "💊 Health Check: https://fractalnexusai.space/health" -ForegroundColor Blue
Write-Host "📚 API Docs: https://fractalnexusai.space/docs" -ForegroundColor Blue

# Test deployment
Write-Host "🧪 Testing deployment..." -ForegroundColor Blue
Start-Sleep -Seconds 5

try {
    $healthResponse = Invoke-WebRequest -Uri "https://fractalnexusai.space/health" -Method GET -TimeoutSec 10
    if ($healthResponse.StatusCode -eq 200) {
        Write-Host "✅ Health check passed!" -ForegroundColor Green
        
        # Test NEXUS API
        Write-Host "🧠 Testing NEXUS API..." -ForegroundColor Blue
        try {
            $apiResponse = Invoke-WebRequest -Uri "https://fractalnexusai.space/api/v1/system/info" -Method GET -TimeoutSec 10
            if ($apiResponse.StatusCode -eq 200) {
                Write-Host "✅ NEXUS API responding!" -ForegroundColor Green
            } else {
                Write-Host "⚠️  NEXUS API returned: $($apiResponse.StatusCode)" -ForegroundColor Yellow
            }
        } catch {
            Write-Host "⚠️  NEXUS API might be initializing..." -ForegroundColor Yellow
        }
    } else {
        Write-Host "❌ Health check failed (HTTP: $($healthResponse.StatusCode))" -ForegroundColor Red
    }
} catch {
    Write-Host "❌ Health check failed: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "💡 Check Vercel function logs for errors" -ForegroundColor Yellow
}

Write-Host "🎉 NitroAGI NEXUS is live at fractalnexusai.space!" -ForegroundColor Green
Write-Host "📊 Monitor at: https://vercel.com/dashboard" -ForegroundColor Blue

# Cleanup
if (Test-Path "requirements.txt") {
    Remove-Item "requirements.txt" -Force
}

Write-Host "🔗 Next steps:" -ForegroundColor Blue
Write-Host "   1. Set API keys in Vercel environment variables" -ForegroundColor White
Write-Host "   2. Configure Redis and PostgreSQL databases" -ForegroundColor White
Write-Host "   3. Test the chat endpoint: https://fractalnexusai.space/api/v1/chat" -ForegroundColor White
Write-Host "   4. Monitor logs in Vercel dashboard" -ForegroundColor White