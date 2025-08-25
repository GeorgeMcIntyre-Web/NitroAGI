"""
Ultra-simple Vercel entry point for NitroAGI NEXUS
"""

def handler(request):
    """Simple handler for Vercel"""
    
    # Get the path
    path = request.get('path', '/')
    
    if path == '/health':
        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": '{"status": "healthy", "service": "NitroAGI NEXUS", "engine": "NEXUS Core"}'
        }
    
    elif path.startswith('/api/v1/system/info'):
        return {
            "statusCode": 200, 
            "headers": {"Content-Type": "application/json"},
            "body": '{"name": "NitroAGI NEXUS", "version": "1.0.0", "engine": "NEXUS", "6g_ready": true}'
        }
    
    else:
        # Default root response
        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"}, 
            "body": '{"message": "🧠 NitroAGI NEXUS is online!", "service": "Neural Executive Unit System", "status": "active", "engine": "NEXUS Core"}'
        }

# Also export as 'app' in case Vercel expects it
app = handler