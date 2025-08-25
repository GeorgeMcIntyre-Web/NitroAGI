# NitroAGI API Reference

This document provides a comprehensive reference for the NitroAGI API. The API enables developers to interact with the multi-modal AI system through RESTful endpoints.

## 📚 Table of Contents

- [Overview](#overview)
- [Authentication](#authentication)
- [Base URL](#base-url)
- [Rate Limiting](#rate-limiting)
- [Error Handling](#error-handling)
- [Data Formats](#data-formats)
- [Core Endpoints](#core-endpoints)
- [AI Modules](#ai-modules)
- [Memory System](#memory-system)
- [Monitoring](#monitoring)
- [WebSocket API](#websocket-api)
- [Examples](#examples)

## Overview

The NitroAGI API is built with FastAPI and provides:

- **RESTful endpoints** for AI processing
- **WebSocket connections** for real-time communication
- **Multi-modal processing** (text, image, audio)
- **Memory management** for conversation context
- **Module orchestration** for complex workflows

### API Versioning

- **Current Version**: v1
- **API Path**: `/api/v1/`
- **Versioning Strategy**: URL path versioning

## Authentication

### API Key Authentication

```http
Authorization: Bearer your_api_key_here
```

### Example Request
```bash
curl -H "Authorization: Bearer sk-nitro-your-key" \
     -H "Content-Type: application/json" \
     https://api.nitroagi.com/api/v1/chat/completions
```

### Getting an API Key
```python
# Coming Soon: API key generation endpoint
POST /api/v1/auth/keys
```

## Base URL

### Production
```
https://api.nitroagi.com/api/v1/
```

### Development
```
http://localhost:8000/api/v1/
```

## Rate Limiting

| Tier | Requests/Hour | Concurrent Requests |
|------|---------------|-------------------|
| Free | 100 | 5 |
| Pro | 1,000 | 10 |
| Enterprise | 10,000 | 50 |

### Rate Limit Headers
```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1640995200
```

## Error Handling

### HTTP Status Codes

| Status Code | Description |
|-------------|-------------|
| 200 | Success |
| 400 | Bad Request |
| 401 | Unauthorized |
| 403 | Forbidden |
| 404 | Not Found |
| 422 | Validation Error |
| 429 | Rate Limited |
| 500 | Internal Server Error |

### Error Response Format
```json
{
  "error": {
    "type": "validation_error",
    "message": "Input validation failed",
    "details": {
      "field": "content",
      "issue": "Content cannot be empty"
    },
    "code": "VALIDATION_FAILED"
  }
}
```

## Data Formats

### Standard Response Format
```json
{
  "id": "req_abc123",
  "object": "chat.completion",
  "created": 1640995200,
  "model": "nitroagi-v1",
  "data": {
    // Response data here
  },
  "metadata": {
    "processing_time_ms": 1250,
    "modules_used": ["language", "reasoning"],
    "tokens_used": 150
  }
}
```

### Pagination Format
```json
{
  "data": [...],
  "pagination": {
    "page": 1,
    "per_page": 20,
    "total": 100,
    "pages": 5,
    "has_next": true,
    "has_prev": false
  }
}
```

## Core Endpoints

### Health Check

Check API availability and system status.

```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-08-25T10:30:00Z",
  "version": "1.0.0",
  "uptime": 86400,
  "services": {
    "database": "healthy",
    "redis": "healthy",
    "ai_modules": {
      "language": "healthy",
      "vision": "healthy",
      "reasoning": "degraded"
    }
  }
}
```

### System Information

Get system capabilities and configuration.

```http
GET /system/info
```

**Response:**
```json
{
  "version": "1.0.0",
  "capabilities": {
    "text_processing": true,
    "image_processing": true,
    "audio_processing": false,
    "multi_modal": true,
    "memory_persistence": true
  },
  "modules": [
    {
      "name": "language",
      "version": "1.0.0",
      "status": "active",
      "models": ["gpt-4", "claude-3"]
    },
    {
      "name": "vision",
      "version": "1.0.0", 
      "status": "active",
      "models": ["clip", "resnet"]
    }
  ],
  "limits": {
    "max_tokens": 4096,
    "max_image_size": "10MB",
    "max_conversation_length": 100
  }
}
```

## AI Modules

### Chat Completions

Process text through the language module with multi-modal capabilities.

```http
POST /chat/completions
```

**Request Body:**
```json
{
  "messages": [
    {
      "role": "user",
      "content": "Explain quantum computing in simple terms"
    }
  ],
  "model": "nitroagi-v1",
  "temperature": 0.7,
  "max_tokens": 1000,
  "stream": false,
  "memory_enabled": true,
  "modules": ["language", "reasoning"]
}
```

**Response:**
```json
{
  "id": "comp_abc123",
  "object": "chat.completion",
  "created": 1640995200,
  "model": "nitroagi-v1",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Quantum computing is like having a super-powered calculator...",
        "module_contributions": {
          "language": {
            "confidence": 0.92,
            "processing_time_ms": 800
          },
          "reasoning": {
            "confidence": 0.88,
            "processing_time_ms": 450
          }
        }
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 12,
    "completion_tokens": 150,
    "total_tokens": 162
  }
}
```

### Image Analysis

Process images through the vision module.

```http
POST /vision/analyze
```

**Request Body (multipart/form-data):**
```bash
curl -X POST \
  -H "Authorization: Bearer your_api_key" \
  -F "image=@path/to/image.jpg" \
  -F "prompt=What objects do you see in this image?" \
  -F "detail=high" \
  https://api.nitroagi.com/api/v1/vision/analyze
```

**Response:**
```json
{
  "id": "img_abc123",
  "object": "vision.analysis",
  "created": 1640995200,
  "analysis": {
    "description": "The image shows a modern office space with...",
    "objects": [
      {"name": "desk", "confidence": 0.95, "bounding_box": [10, 20, 100, 80]},
      {"name": "chair", "confidence": 0.89, "bounding_box": [50, 60, 90, 120]}
    ],
    "scene": "office_interior",
    "colors": ["#2C3E50", "#3498DB", "#ECF0F1"],
    "text_detected": ["Welcome", "Office Hours: 9-5"]
  },
  "metadata": {
    "image_size": [1920, 1080],
    "processing_time_ms": 2100,
    "model_used": "vision-v1"
  }
}
```

### Multi-Modal Processing

Process multiple types of input simultaneously.

```http
POST /multimodal/process
```

**Request Body:**
```json
{
  "inputs": [
    {
      "type": "text",
      "content": "Analyze this image and explain what makes it effective"
    },
    {
      "type": "image",
      "url": "https://example.com/image.jpg"
    }
  ],
  "task": "analysis",
  "modules": ["language", "vision", "reasoning"],
  "context": {
    "conversation_id": "conv_123",
    "user_preferences": {
      "detail_level": "high"
    }
  }
}
```

**Response:**
```json
{
  "id": "multi_abc123",
  "object": "multimodal.process",
  "created": 1640995200,
  "result": {
    "analysis": "This image is effective because...",
    "reasoning_chain": [
      "Analyzed image composition and visual elements",
      "Considered principles of effective visual design",
      "Synthesized findings into comprehensive explanation"
    ],
    "module_outputs": {
      "vision": {
        "objects": ["..."],
        "composition": "rule_of_thirds",
        "quality": "high"
      },
      "reasoning": {
        "effectiveness_score": 0.87,
        "key_factors": ["composition", "color_harmony", "clarity"]
      }
    }
  }
}
```

## Memory System

### Conversations

Manage conversation history and context.

#### Create Conversation
```http
POST /conversations
```

**Request Body:**
```json
{
  "title": "Quantum Computing Discussion",
  "metadata": {
    "topic": "physics",
    "language": "en"
  }
}
```

#### Get Conversation
```http
GET /conversations/{conversation_id}
```

**Response:**
```json
{
  "id": "conv_abc123",
  "title": "Quantum Computing Discussion",
  "created_at": "2025-08-25T10:30:00Z",
  "updated_at": "2025-08-25T11:45:00Z",
  "message_count": 12,
  "metadata": {
    "topic": "physics",
    "language": "en"
  },
  "messages": [
    {
      "id": "msg_def456",
      "role": "user",
      "content": "What is quantum entanglement?",
      "timestamp": "2025-08-25T10:30:00Z"
    }
  ]
}
```

#### List Conversations
```http
GET /conversations?page=1&per_page=20&sort=updated_at&order=desc
```

### Memory Search

Search through conversation history and knowledge base.

```http
POST /memory/search
```

**Request Body:**
```json
{
  "query": "quantum computing basics",
  "filters": {
    "conversation_id": "conv_abc123",
    "date_range": {
      "start": "2025-08-01",
      "end": "2025-08-25"
    },
    "message_types": ["user", "assistant"]
  },
  "limit": 10
}
```

**Response:**
```json
{
  "results": [
    {
      "id": "msg_abc123",
      "content": "Quantum computing uses quantum mechanical phenomena...",
      "relevance_score": 0.95,
      "conversation_id": "conv_def456",
      "timestamp": "2025-08-20T14:30:00Z",
      "context": {
        "previous_message": "Can you explain quantum computing?",
        "next_message": "That's fascinating! What about quantum entanglement?"
      }
    }
  ],
  "total": 5,
  "query_time_ms": 45
}
```

## Monitoring

### Usage Statistics

Get API usage statistics and analytics.

```http
GET /monitoring/usage
```

**Query Parameters:**
- `start_date`: Start date (ISO 8601)
- `end_date`: End date (ISO 8601)
- `granularity`: `hour`, `day`, `week`, `month`

**Response:**
```json
{
  "period": {
    "start": "2025-08-01T00:00:00Z",
    "end": "2025-08-25T23:59:59Z"
  },
  "summary": {
    "total_requests": 15420,
    "successful_requests": 15180,
    "failed_requests": 240,
    "avg_response_time_ms": 1250,
    "tokens_processed": 2450000
  },
  "by_endpoint": {
    "/chat/completions": 12000,
    "/vision/analyze": 2800,
    "/multimodal/process": 620
  },
  "by_day": [
    {
      "date": "2025-08-25",
      "requests": 650,
      "avg_response_time_ms": 1100
    }
  ]
}
```

### Performance Metrics

Get system performance metrics.

```http
GET /monitoring/performance
```

**Response:**
```json
{
  "system": {
    "cpu_usage": 45.2,
    "memory_usage": 68.7,
    "disk_usage": 23.1,
    "uptime_seconds": 86400
  },
  "modules": {
    "language": {
      "status": "healthy",
      "avg_response_time_ms": 800,
      "requests_per_minute": 45,
      "error_rate": 0.02
    },
    "vision": {
      "status": "healthy",
      "avg_response_time_ms": 1500,
      "requests_per_minute": 12,
      "error_rate": 0.01
    }
  },
  "database": {
    "connection_pool_usage": 12,
    "avg_query_time_ms": 25,
    "active_connections": 8
  }
}
```

## WebSocket API

### Real-time Chat

Connect to real-time chat functionality.

**Connection:**
```javascript
const ws = new WebSocket('wss://api.nitroagi.com/ws/chat?token=your_api_key');
```

**Message Format:**
```json
{
  "type": "chat_message",
  "data": {
    "conversation_id": "conv_abc123",
    "message": "Tell me about artificial intelligence",
    "stream": true,
    "modules": ["language", "reasoning"]
  },
  "id": "req_123"
}
```

**Response Streaming:**
```json
{
  "type": "chat_response",
  "data": {
    "id": "req_123",
    "delta": "Artificial intelligence is",
    "done": false
  }
}
```

### System Events

Subscribe to system events and notifications.

**Subscription:**
```json
{
  "type": "subscribe",
  "events": ["module_status", "performance_alerts", "rate_limit_warnings"]
}
```

**Event Example:**
```json
{
  "type": "event",
  "event": "module_status",
  "data": {
    "module": "vision",
    "status": "degraded",
    "message": "High response times detected",
    "timestamp": "2025-08-25T12:30:00Z"
  }
}
```

## Examples

### Python SDK Example

```python
import asyncio
from nitroagi import NitroAGI

async def main():
    # Initialize client
    client = NitroAGI(api_key="your_api_key")
    
    # Simple chat completion
    response = await client.chat.completions.create(
        messages=[
            {"role": "user", "content": "Explain machine learning"}
        ],
        model="nitroagi-v1",
        temperature=0.7
    )
    
    print(response.choices[0].message.content)
    
    # Multi-modal processing
    result = await client.multimodal.process(
        inputs=[
            {"type": "text", "content": "What's in this image?"},
            {"type": "image", "path": "./image.jpg"}
        ],
        modules=["language", "vision"]
    )
    
    print(result.analysis)

asyncio.run(main())
```

### JavaScript/Node.js Example

```javascript
const NitroAGI = require('nitroagi');

const client = new NitroAGI({
  apiKey: 'your_api_key'
});

async function example() {
  // Chat completion
  const completion = await client.chat.completions.create({
    messages: [
      { role: 'user', content: 'Write a haiku about AI' }
    ],
    model: 'nitroagi-v1'
  });
  
  console.log(completion.choices[0].message.content);
  
  // Image analysis
  const analysis = await client.vision.analyze({
    image: './photo.jpg',
    prompt: 'Describe this scene'
  });
  
  console.log(analysis.description);
}

example();
```

### cURL Examples

**Chat Completion:**
```bash
curl -X POST https://api.nitroagi.com/api/v1/chat/completions \
  -H "Authorization: Bearer your_api_key" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello, NitroAGI!"}],
    "model": "nitroagi-v1"
  }'
```

**Image Upload and Analysis:**
```bash
curl -X POST https://api.nitroagi.com/api/v1/vision/analyze \
  -H "Authorization: Bearer your_api_key" \
  -F "image=@image.jpg" \
  -F "prompt=What do you see?"
```

## SDKs and Libraries

### Official SDKs
- **Python**: `pip install nitroagi`
- **JavaScript/Node.js**: `npm install nitroagi`
- **Go**: `go get github.com/nitroagi/nitroagi-go`

### Community SDKs
- **Ruby**: `gem install nitroagi-ruby`
- **Java**: Coming Soon
- **C#/.NET**: Coming Soon

## Support

### API Documentation
- **Interactive Docs**: https://api.nitroagi.com/docs
- **OpenAPI Spec**: https://api.nitroagi.com/openapi.json

### Community
- **GitHub Issues**: Bug reports and feature requests
- **Discord**: Real-time support and discussions
- **Documentation**: https://docs.nitroagi.com

### Contact
- **Support Email**: [support@nitroagi.dev]
- **API Issues**: [api@nitroagi.dev]

---

**Note**: This API is under active development. Breaking changes will be communicated in advance and documented in the changelog.
