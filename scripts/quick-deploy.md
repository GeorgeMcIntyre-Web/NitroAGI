# ⚡ Quick Deploy Guide for NitroAGI NEXUS

## 🚀 **5-Minute Deployment to fractalnexusai.space**

### **Prerequisites**
- Node.js installed (for Vercel CLI)
- Your Vercel token: `yCezdRKOJ6UZhfOp87jVMZce`

### **Commands to Run (Copy & Paste)**

```bash
# 1. Install Vercel CLI
npm install -g vercel

# 2. Set token
set VERCEL_TOKEN=yCezdRKOJ6UZhfOp87jVMZce

# 3. Initialize project (in NitroAGI directory)
vercel

# 4. Set essential environment variables
vercel env add NITROAGI_ENV production
vercel env add NITROAGI_DEBUG false
vercel env add PYTHONPATH src
vercel env add NITROAGI_ENABLE_6G true

# 5. Deploy to production
vercel --prod
```

### **During `vercel` initialization:**
- ✅ Set up and deploy "NitroAGI"? → **Y**
- ✅ Which scope? → **[choose your scope]**
- ✅ Link to existing project? → **N**
- ✅ Project name? → **nitroagi-nexus**
- ✅ Directory? → **./** (just press Enter)

### **After Deployment:**

1. **Add API Keys** (required for chat functionality):
```bash
vercel env add NITROAGI_OPENAI_API_KEY sk-your-openai-key-here
vercel env add NITROAGI_JWT_SECRET_KEY your-random-secret-key-here
```

2. **Configure Domain**:
```bash
vercel domains add fractalnexusai.space
vercel alias set [your-vercel-url] fractalnexusai.space
```

3. **Test Your Deployment**:
```bash
curl https://fractalnexusai.space/health
```

### **🧠 Expected Results:**

- **✅ Health Check**: https://fractalnexusai.space/health
- **🔗 API Docs**: https://fractalnexusai.space/docs  
- **🧠 NEXUS API**: https://fractalnexusai.space/api/v1
- **💬 Chat**: https://fractalnexusai.space/api/v1/chat

### **🚨 If You Get Errors:**

1. **"Command not found"** → Install Node.js first
2. **"Token invalid"** → Check token spelling
3. **"Build failed"** → Check requirements-vercel.txt exists
4. **"Function timeout"** → Normal for first cold start

### **🎯 Success Indicators:**

```bash
# Should return HTTP 200
curl -I https://fractalnexusai.space/health

# Should return JSON with version
curl https://fractalnexusai.space/api/v1/system/info
```

### **📱 Monitor Your Deployment:**
- **Dashboard**: https://vercel.com/dashboard
- **Logs**: Click on your project → Functions tab
- **Analytics**: Built-in traffic analytics

---

**🧠 That's it! NitroAGI NEXUS will be live on fractalnexusai.space in under 5 minutes!**