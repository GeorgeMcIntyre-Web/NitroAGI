# NitroAGI NEXUS Vercel Setup Script
# This script helps you set up the project on Vercel

Write-Host "🧠 Setting up NitroAGI NEXUS on Vercel..." -ForegroundColor Cyan

# Check if Vercel CLI is installed
try {
    vercel --version | Out-Null
    Write-Host "✅ Vercel CLI found" -ForegroundColor Green
} catch {
    Write-Host "❌ Installing Vercel CLI..." -ForegroundColor Yellow
    npm install -g vercel
}

# Set token
$env:VERCEL_TOKEN = "yCezdRKOJ6UZhfOp87jVMZce"
Write-Host "✅ Vercel token configured" -ForegroundColor Green

Write-Host "`n📋 Next steps:" -ForegroundColor Blue
Write-Host "1. Run: vercel" -ForegroundColor White
Write-Host "2. Choose: 'Set up and deploy NitroAGI? [Y/n]' → Y" -ForegroundColor White
Write-Host "3. Project name: 'nitroagi-nexus'" -ForegroundColor White
Write-Host "4. Directory: './' (current directory)" -ForegroundColor White
Write-Host "`n🔧 After project creation, set environment variables:" -ForegroundColor Blue
Write-Host "   vercel env add NITROAGI_OPENAI_API_KEY your-key-here" -ForegroundColor White
Write-Host "   vercel env add NITROAGI_JWT_SECRET_KEY random-secret-key" -ForegroundColor White
Write-Host "`n🌐 Add custom domain:" -ForegroundColor Blue
Write-Host "   vercel domains add fractalnexusai.space" -ForegroundColor White
Write-Host "   vercel alias set [project-url] fractalnexusai.space" -ForegroundColor White
Write-Host "`n🚀 Deploy to production:" -ForegroundColor Blue
Write-Host "   vercel --prod" -ForegroundColor White

Write-Host "`n🎯 Ready to start? Run 'vercel' now!" -ForegroundColor Green