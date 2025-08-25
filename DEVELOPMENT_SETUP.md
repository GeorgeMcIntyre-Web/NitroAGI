# NitroAGI Development Setup Guide

This guide will help you set up your development environment for NitroAGI from scratch.

## 🏗️ Prerequisites

### Required Software
- **Python 3.9+** ([Download](https://www.python.org/downloads/))
- **Docker & Docker Compose** ([Download](https://www.docker.com/products/docker-desktop/))
- **Git** ([Download](https://git-scm.com/downloads))
- **Node.js 16+** (for UI components) ([Download](https://nodejs.org/))

### Recommended Tools
- **VS Code** with Python extension ([Download](https://code.visualstudio.com/))
- **Postman** or **Insomnia** for API testing
- **Redis CLI** for debugging memory systems

## 🚀 Quick Start

### 1. Clone and Setup
```bash
# Clone the repository
git clone https://github.com/GeorgeMcIntyre-Web/NitroAGI.git
cd NitroAGI

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Configuration
```bash
# Copy environment template
cp .env.example .env

# Edit with your settings
# Windows:
notepad .env
# macOS/Linux:
nano .env
```

### 3. Start Development Services
```bash
# Start all services with Docker Compose
docker-compose -f docker-compose.dev.yml up -d

# Verify services are running
docker-compose ps
```

### 4. Run Tests
```bash
# Install test dependencies
pip install -r requirements-dev.txt

# Run test suite
pytest tests/ -v

# Run with coverage
pytest --cov=nitroagi tests/
```

## 📁 Project Structure

```
NitroAGI/
├── .env.example              # Environment variables template
├── .gitignore               # Git ignore rules
├── docker-compose.dev.yml   # Development Docker setup
├── docker-compose.prod.yml  # Production Docker setup
├── requirements.txt         # Python dependencies
├── requirements-dev.txt     # Development dependencies
├── pytest.ini             # Test configuration
├── pyproject.toml          # Project metadata and tool configs
│
├── src/
│   ├── nitroagi/           # Main application package
│   │   ├── __init__.py
│   │   ├── core/           # Core framework
│   │   │   ├── __init__.py
│   │   │   ├── orchestrator.py    # Executive controller
│   │   │   ├── message_bus.py     # Communication system
│   │   │   └── memory.py          # Memory management
│   │   │
│   │   ├── modules/        # AI modules
│   │   │   ├── __init__.py
│   │   │   ├── language/   # Language processing
│   │   │   ├── vision/     # Computer vision
│   │   │   ├── reasoning/  # Logic and reasoning
│   │   │   └── learning/   # RL and adaptation
│   │   │
│   │   ├── api/           # REST API endpoints
│   │   │   ├── __init__.py
│   │   │   ├── routes/
│   │   │   └── middleware/
│   │   │
│   │   └── utils/         # Utilities and helpers
│   │       ├── __init__.py
│   │       ├── config.py
│   │       └── logging.py
│   │
├── tests/                 # Test suite
│   ├── unit/             # Unit tests
│   ├── integration/      # Integration tests
│   ├── e2e/             # End-to-end tests
│   └── fixtures/        # Test data
│
├── docs/                 # Documentation
│   ├── api/             # API documentation
│   ├── architecture/    # System design docs
│   └── tutorials/       # How-to guides
│
├── scripts/             # Utility scripts
│   ├── setup.sh        # Initial setup script
│   ├── test.sh         # Test runner
│   └── deploy.sh       # Deployment script
│
├── config/             # Configuration files
│   ├── development/
│   ├── production/
│   └── testing/
│
└── data/              # Sample data and models
    ├── models/        # AI model files (gitignored)
    ├── samples/       # Sample inputs/outputs
    └── benchmarks/    # Performance test data
```

## ⚙️ Configuration Files

### `.env.example`
```env
# API Keys
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
HUGGINGFACE_API_KEY=your_huggingface_api_key_here

# Database Configuration
REDIS_URL=redis://localhost:6379
POSTGRES_URL=postgresql://nitroagi:password@localhost:5432/nitroagi_dev
MONGODB_URL=mongodb://localhost:27017/nitroagi

# Vector Database
CHROMA_HOST=localhost
CHROMA_PORT=8000

# Message Queue
KAFKA_BOOTSTRAP_SERVERS=localhost:9092

# Application Settings
DEBUG=true
LOG_LEVEL=INFO
API_HOST=0.0.0.0
API_PORT=8000

# Security
SECRET_KEY=your_secret_key_here
JWT_SECRET=your_jwt_secret_here
```

### `requirements.txt`
```txt
# Core Framework
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.4.2
pydantic-settings==2.0.3

# AI/ML Libraries
torch==2.1.0
transformers==4.35.0
sentence-transformers==2.2.2
openai==1.3.0
anthropic==0.7.0

# Computer Vision
opencv-python==4.8.1.78
pillow==10.0.1

# Data Processing
numpy==1.24.3
pandas==2.0.3
scipy==1.11.4

# Database & Storage
redis==5.0.1
psycopg2-binary==2.9.7
pymongo==4.5.0
chromadb==0.4.15

# Message Queue
kafka-python==2.0.2
celery==5.3.4

# Utilities
python-dotenv==1.0.0
pyyaml==6.0.1
click==8.1.7
rich==13.6.0
```

### `requirements-dev.txt`
```txt
# Testing
pytest==7.4.3
pytest-cov==4.1.0
pytest-asyncio==0.21.1
pytest-mock==3.12.0

# Code Quality
black==23.10.1
isort==5.12.0
flake8==6.1.0
mypy==1.7.0

# Development Tools
pre-commit==3.5.0
jupyter==1.0.0
ipython==8.17.2

# Documentation
sphinx==7.2.6
sphinx-rtd-theme==1.3.0
```

### `docker-compose.dev.yml`
```yaml
version: '3.8'

services:
  # Redis for working memory
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes

  # PostgreSQL for structured data
  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: nitroagi_dev
      POSTGRES_USER: nitroagi
      POSTGRES_PASSWORD: password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

  # MongoDB for document storage
  mongodb:
    image: mongo:7
    ports:
      - "27017:27017"
    volumes:
      - mongodb_data:/data/db

  # ChromaDB for vector storage
  chromadb:
    image: chromadb/chroma:latest
    ports:
      - "8000:8000"
    volumes:
      - chromadb_data:/chroma/chroma

  # Kafka for message queuing
  zookeeper:
    image: confluentinc/cp-zookeeper:latest
    environment:
      ZOOKEEPER_CLIENT_PORT: 2181
      ZOOKEEPER_TICK_TIME: 2000

  kafka:
    image: confluentinc/cp-kafka:latest
    depends_on:
      - zookeeper
    ports:
      - "9092:9092"
    environment:
      KAFKA_BROKER_ID: 1
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://localhost:9092
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1

volumes:
  redis_data:
  postgres_data:
  mongodb_data:
  chromadb_data:
```

### `pytest.ini`
```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    --strict-markers
    --strict-config
    --verbose
    --tb=short
    --cov=nitroagi
    --cov-report=term-missing
    --cov-report=html:htmlcov
    --cov-report=xml
markers =
    unit: Unit tests
    integration: Integration tests
    e2e: End-to-end tests
    slow: Slow running tests
```

## 🔧 Development Workflow

### 1. Feature Development
```bash
# Create feature branch
git checkout -b feature/your-feature-name

# Make changes and commit
git add .
git commit -m "feat: add your feature description"

# Run tests before pushing
pytest tests/ -v

# Push to GitHub
git push origin feature/your-feature-name
```

### 2. Code Quality Checks
```bash
# Format code
black src/ tests/
isort src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/
```

### 3. Running Specific Components
```bash
# Start only the API server
python -m nitroagi.api.main

# Run specific module tests
pytest tests/unit/modules/test_language.py -v

# Debug with specific log level
LOG_LEVEL=DEBUG python -m nitroagi.api.main
```

## 🐛 Debugging

### Common Issues

#### Docker Services Won't Start
```bash
# Check if ports are already in use
netstat -tulpn | grep :6379

# Stop existing services
docker-compose down
docker system prune -f

# Restart services
docker-compose -f docker-compose.dev.yml up -d
```

#### Python Import Errors
```bash
# Ensure you're in the virtual environment
which python

# Install package in development mode
pip install -e .

# Check Python path
python -c "import sys; print(sys.path)"
```

#### Memory/Performance Issues
```bash
# Monitor resource usage
docker stats

# Check Redis memory usage
redis-cli info memory

# Monitor Python memory usage
pip install memory-profiler
python -m memory_profiler your_script.py
```

### Logging and Monitoring

```python
# Example logging setup in your modules
import logging
from nitroagi.utils.logging import get_logger

logger = get_logger(__name__)

# Use throughout your code
logger.info("Starting language module")
logger.error("Failed to process input", exc_info=True)
```

## 🧪 Testing Strategy

### Test Types
- **Unit Tests**: Test individual functions/classes
- **Integration Tests**: Test module interactions
- **E2E Tests**: Test complete workflows
- **Performance Tests**: Benchmark response times

### Running Tests
```bash
# All tests
pytest

# Specific test type
pytest -m unit
pytest -m integration

# With coverage
pytest --cov=nitroagi --cov-report=html

# Watch mode for development
pytest-watch
```

## 📚 Additional Resources

### Documentation
- [Architecture Overview](docs/architecture/README.md)
- [API Reference](docs/api/README.md)
- [Contributing Guidelines](CONTRIBUTING.md)

### External Links
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Docker Best Practices](https://docs.docker.com/develop/best-practices/)

## 🆘 Getting Help

1. **Check the documentation** in the `docs/` directory
2. **Search existing issues** on GitHub
3. **Create a new issue** with detailed information
4. **Join our Discord** [Coming Soon]

## ✅ Verification Checklist

After setup, verify everything works:

- [ ] Virtual environment activated
- [ ] All dependencies installed
- [ ] Docker services running
- [ ] Environment variables configured
- [ ] Tests pass
- [ ] API server starts successfully
- [ ] Can import `nitroagi` modules

**Happy coding! 🚀**
