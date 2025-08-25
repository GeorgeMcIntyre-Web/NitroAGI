# Getting Started with NitroAGI NEXUS

Welcome to **NitroAGI NEXUS** - a brain-inspired multi-modal AI system with advanced reasoning, learning, and creative thinking capabilities.

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- Redis (for development)
- PostgreSQL (optional)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/GeorgeMcIntyre-Web/NitroAGI.git
   cd NitroAGI
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

4. **Start with Docker Compose**
   ```bash
   docker-compose up -d
   ```

5. **Access the system**
   - **API**: http://localhost:8000
   - **Chat Interface**: http://localhost:8000/chat
   - **Dashboard**: http://localhost:8000/dashboard
   - **API Docs**: http://localhost:8000/docs

## 🧠 Understanding NEXUS

NEXUS (Neural Executive Unit System) is the core orchestrator that coordinates all AI modules:

### Core Components

1. **Prefrontal Cortex** - Executive control and planning
2. **Multi-Modal Processor** - Handles text, images, audio, video
3. **Learning System** - Reinforcement learning and adaptation  
4. **Reasoning Engine** - Abstract, mathematical, scientific reasoning
5. **Creative Engine** - Innovative problem solving

### Architecture Overview

```
┌─────────────────────────────────────────┐
│              NEXUS Core                 │
├─────────────────────────────────────────┤
│  Prefrontal Cortex (Executive Control) │
├─────────────────────────────────────────┤
│  Multi-Modal Processor                  │
├─────────────────────────────────────────┤
│  Learning & Adaptation System           │
├─────────────────────────────────────────┤
│  Advanced Reasoning Modules             │
│  ├─ Abstract Reasoning                  │
│  ├─ Mathematical Solver                 │
│  ├─ Scientific Reasoner                 │
│  └─ Creative Thinking                   │
└─────────────────────────────────────────┘
```

## 🔧 Basic Usage

### API Examples

#### Execute a Task
```python
import requests

response = requests.post("http://localhost:8000/v1/nexus/execute", json={
    "goal": "Analyze the sentiment of customer reviews",
    "context": {"data_source": "reviews.csv"},
    "modules": ["language", "reasoning"]
})

print(response.json())
```

#### Solve Math Problems
```python
response = requests.post("http://localhost:8000/v1/reasoning/solve-math", json={
    "problem": "Find the derivative of x^3 + 2x^2 - 5x + 7",
    "show_steps": True
})

print(response.json()["solution"])
```

#### Creative Problem Solving
```python
response = requests.post("http://localhost:8000/v1/reasoning/creative-solve", json={
    "problem_description": "How to reduce energy consumption in data centers?",
    "constraints": ["cost-effective", "environmentally friendly"],
    "strategy": "design_thinking"
})

print(response.json()["solution"])
```

### Chat Interface

The web chat interface provides an intuitive way to interact with NEXUS:

1. Open http://localhost:8000/chat
2. Type your message or question
3. NEXUS will automatically select the appropriate modules
4. Get real-time responses with reasoning steps

### WebSocket Real-time Communication

```javascript
const ws = new WebSocket('ws://localhost:8000/ws');

ws.onmessage = (event) => {
    const message = JSON.parse(event.data);
    console.log('NEXUS response:', message);
};

// Send a task request
ws.send(JSON.stringify({
    type: 'task_request',
    data: {
        goal: 'Generate creative marketing ideas',
        context: { product: 'AI assistant' }
    }
}));
```

## 📊 Monitoring & Analytics

### Developer Dashboard

Access the dashboard at http://localhost:8000/dashboard to monitor:

- System performance metrics
- Module status and usage
- Real-time connection monitoring
- Error rates and response times

### Performance Monitoring

```python
from config.monitoring import get_performance_monitor

monitor = get_performance_monitor()
monitor.start_monitoring()

# Get performance report
report = monitor.get_performance_report(duration_minutes=60)
print(report)
```

### Health Checks

```bash
curl http://localhost:8000/health
curl http://localhost:8000/v1/nexus/status
```

## 🎯 Use Cases

### 1. Content Analysis & Generation
- Sentiment analysis of text
- Content summarization
- Creative writing assistance
- Multi-language translation

### 2. Data Analysis & Insights
- Pattern recognition in datasets
- Statistical analysis and visualization
- Predictive modeling
- Business intelligence

### 3. Problem Solving & Innovation
- Creative brainstorming sessions
- Technical problem diagnosis
- Process optimization
- Innovation consulting

### 4. Education & Research
- Mathematical problem solving
- Scientific hypothesis generation
- Research assistance
- Educational content creation

### 5. Automation & Workflows
- Intelligent task routing
- Decision support systems
- Process automation
- Quality assurance

## 🔌 Integration Examples

### FastAPI Integration

```python
from fastapi import FastAPI
from src.nitroagi.core.orchestrator import Orchestrator

app = FastAPI()
orchestrator = Orchestrator()

@app.post("/analyze")
async def analyze_data(data: dict):
    result = await orchestrator.process_request(
        goal="Analyze this data for insights",
        context=data
    )
    return result
```

### Jupyter Notebook

```python
import asyncio
from src.nitroagi.core.orchestrator import Orchestrator

# Initialize NEXUS
orchestrator = Orchestrator()

# Run analysis
async def analyze():
    result = await orchestrator.process_request(
        goal="Perform statistical analysis on dataset",
        context={"data": your_data}
    )
    return result

# Execute
result = await analyze()
print(result)
```

## 🛠️ Configuration

### Environment Variables

Key configuration options:

```bash
# Core Settings
PYTHON_ENV=development
LOG_LEVEL=info
DEBUG=true

# Database
REDIS_URL=redis://localhost:6379
DATABASE_URL=postgresql://user:pass@localhost/nexus

# Security
JWT_SECRET_KEY=your-secret-key
RATE_LIMIT_REQUESTS=1000
SESSION_TIMEOUT=3600

# Performance
WORKERS=4
MAX_CONNECTIONS=1000
ENABLE_METRICS=true
```

### Module Configuration

```python
# config/modules.py
LANGUAGE_CONFIG = {
    "provider": "openai",
    "model": "gpt-4",
    "max_tokens": 2000,
    "temperature": 0.7
}

VISION_CONFIG = {
    "enable_ocr": True,
    "object_detection": True,
    "face_recognition": False
}

LEARNING_CONFIG = {
    "learning_rate": 0.001,
    "batch_size": 32,
    "enable_online_learning": True
}
```

## 🔍 Troubleshooting

### Common Issues

1. **Module not responding**
   ```bash
   # Check module status
   curl http://localhost:8000/v1/modules/list
   
   # Restart specific module
   curl -X POST http://localhost:8000/v1/modules/language/enable
   ```

2. **High response times**
   ```bash
   # Check performance metrics
   curl http://localhost:8000/v1/nexus/metrics
   
   # View system resources
   curl http://localhost:8000/health
   ```

3. **Memory issues**
   ```bash
   # Clear learning system cache
   curl -X POST http://localhost:8000/v1/learning/reset
   
   # Restart services
   docker-compose restart
   ```

### Logging

Logs are available in:
- Application logs: `./logs/app.log`
- Error logs: `./logs/error.log`
- Access logs: `./logs/access.log`

### Debug Mode

Enable debug mode for detailed logging:

```bash
export DEBUG=true
export LOG_LEVEL=debug
```

## 📚 Next Steps

1. **Explore the API**: Check out the [API Reference](API_REFERENCE.md)
2. **Learn the Architecture**: Read the [Architecture Guide](ARCHITECTURE.md)
3. **Deploy to Production**: Follow the [Deployment Guide](DEPLOYMENT.md)
4. **Contribute**: See [Contributing Guidelines](CONTRIBUTING.md)
5. **Join Community**: Visit our [Discord](https://discord.gg/nitroagi) or [Forum](https://forum.nitroagi.com)

## 🆘 Need Help?

- 📖 **Documentation**: https://docs.nitroagi.com
- 💬 **Discord Community**: https://discord.gg/nitroagi
- 🐛 **Report Issues**: https://github.com/GeorgeMcIntyre-Web/NitroAGI/issues
- 📧 **Email Support**: support@nitroagi.com
- 🎥 **Video Tutorials**: https://youtube.com/@nitroagi

---

**Welcome to the future of AI orchestration with NitroAGI NEXUS!** 🚀