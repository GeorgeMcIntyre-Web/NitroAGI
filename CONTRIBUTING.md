# Contributing to NitroAGI

Thank you for your interest in contributing to NitroAGI! This document provides guidelines and information for contributors.

## 🎯 Ways to Contribute

### Code Contributions
- **New AI Modules**: Implement vision, reasoning, or learning modules
- **Core Framework**: Improve the orchestration and communication systems
- **Performance**: Optimize inference speed and memory usage
- **Bug Fixes**: Fix reported issues and improve stability
- **Testing**: Add unit tests, integration tests, and benchmarks

### Non-Code Contributions
- **Documentation**: Improve guides, API docs, and tutorials
- **Research**: Share papers, techniques, and architectural insights
- **Community**: Help answer questions and support other users
- **Design**: Create UI/UX improvements and visual assets
- **Testing**: Report bugs and suggest improvements

## 🚀 Getting Started

### Prerequisites
- Python 3.9+ development experience
- Familiarity with AI/ML concepts and libraries
- Understanding of distributed systems (helpful but not required)
- Git and GitHub workflow knowledge

### Development Setup
1. **Fork the repository** on GitHub
2. **Clone your fork** locally
3. **Follow the setup guide** in [DEVELOPMENT_SETUP.md](DEVELOPMENT_SETUP.md)
4. **Create a feature branch** for your work
5. **Make your changes** following our coding standards
6. **Test thoroughly** before submitting

```bash
git clone https://github.com/YOUR_USERNAME/NitroAGI.git
cd NitroAGI
git remote add upstream https://github.com/GeorgeMcIntyre-Web/NitroAGI.git
```

## 📋 Development Process

### Branch Strategy
- **main**: Production-ready code
- **develop**: Integration branch for features
- **feature/[name]**: Individual feature development
- **hotfix/[name]**: Critical bug fixes
- **release/[version]**: Release preparation

### Workflow
1. **Create an issue** or claim an existing one
2. **Create a feature branch** from `develop`
3. **Implement your changes** with tests
4. **Update documentation** if needed
5. **Submit a pull request** to `develop`
6. **Address review feedback**
7. **Celebrate** when merged! 🎉

## 🏗️ Project Architecture

### Core Components
```
src/nitroagi/
├── core/           # Framework foundation
│   ├── orchestrator.py    # Executive controller
│   ├── message_bus.py     # Communication system
│   └── memory.py          # Memory management
├── modules/        # AI processing modules
│   ├── language/   # LLM integration
│   ├── vision/     # Computer vision
│   ├── reasoning/  # Logic and inference
│   └── learning/   # RL and adaptation
├── api/           # REST API endpoints
└── utils/         # Shared utilities
```

### Adding a New Module
When contributing a new AI module:

1. **Create module directory**: `src/nitroagi/modules/your_module/`
2. **Implement base interface**: Extend `AIModule` class
3. **Add configuration**: Update module registry
4. **Write tests**: Unit and integration tests
5. **Document API**: Add docstrings and examples

Example module structure:
```python
from nitroagi.core.base import AIModule

class YourModule(AIModule):
    def __init__(self, config):
        super().__init__(config)
        # Initialize your AI model
        
    async def process(self, input_data):
        # Process input and return results
        return {"output": "processed_data"}
        
    async def health_check(self):
        # Return module health status
        return {"status": "healthy"}
```

## 💻 Coding Standards

### Python Style Guide
- **Follow PEP 8** with 100 character line limit
- **Use Black** for code formatting
- **Use isort** for import sorting  
- **Use type hints** for all function signatures
- **Write docstrings** for all public functions and classes

### Code Formatting
```bash
# Format code before committing
black src/ tests/
isort src/ tests/
flake8 src/ tests/
mypy src/
```

### Example Code Style
```python
from typing import Dict, List, Optional
import asyncio
from nitroagi.core.base import AIModule


class ExampleModule(AIModule):
    """Example AI module demonstrating coding standards.
    
    This module shows the expected code style and structure
    for NitroAGI AI modules.
    
    Args:
        config: Module configuration dictionary
        model_name: Name of the AI model to use
        
    Attributes:
        model: The loaded AI model instance
        config: Module configuration
    """
    
    def __init__(
        self, 
        config: Dict[str, any], 
        model_name: Optional[str] = None
    ) -> None:
        super().__init__(config)
        self.model_name = model_name or "default-model"
        self.model = self._load_model()
    
    async def process(self, input_data: Dict[str, any]) -> Dict[str, any]:
        """Process input data through the AI model.
        
        Args:
            input_data: Dictionary containing input to process
            
        Returns:
            Dictionary with processed results
            
        Raises:
            ProcessingError: If input processing fails
        """
        try:
            # Process the input
            result = await self._run_inference(input_data)
            return {"output": result, "confidence": 0.95}
        except Exception as e:
            self.logger.error(f"Processing failed: {e}")
            raise ProcessingError(f"Failed to process input: {e}")
    
    def _load_model(self):
        """Load the AI model (private method)."""
        # Implementation details
        pass
```

## 🧪 Testing Guidelines

### Test Types
- **Unit Tests**: Test individual functions and classes
- **Integration Tests**: Test module interactions
- **End-to-End Tests**: Test complete workflows
- **Performance Tests**: Benchmark response times and memory usage

### Writing Tests
```python
import pytest
from nitroagi.modules.example import ExampleModule

class TestExampleModule:
    @pytest.fixture
    def module(self):
        config = {"model_name": "test-model"}
        return ExampleModule(config)
    
    def test_initialization(self, module):
        assert module.model_name == "test-model"
        assert module.model is not None
    
    @pytest.mark.asyncio
    async def test_process_valid_input(self, module):
        input_data = {"text": "Hello world"}
        result = await module.process(input_data)
        
        assert "output" in result
        assert "confidence" in result
        assert result["confidence"] > 0.0
    
    @pytest.mark.asyncio
    async def test_process_invalid_input(self, module):
        with pytest.raises(ProcessingError):
            await module.process({})
```

### Running Tests
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/modules/test_example.py

# Run with coverage
pytest --cov=nitroagi --cov-report=html

# Run performance tests
pytest -m performance --benchmark-only
```

## 📚 Documentation Standards

### Docstring Format
Use Google-style docstrings:
```python
def example_function(param1: str, param2: int = 0) -> Dict[str, any]:
    """Brief description of the function.
    
    Longer description explaining the function's purpose,
    behavior, and any important details.
    
    Args:
        param1: Description of the first parameter
        param2: Description of the second parameter with default value
        
    Returns:
        Dictionary containing the results with keys:
            - key1: Description of return value
            - key2: Description of another return value
            
    Raises:
        ValueError: If param1 is empty
        ProcessingError: If processing fails
        
    Example:
        >>> result = example_function("hello", 42)
        >>> print(result["key1"])
        processed_hello
    """
```

### README Updates
When adding features, update:
- Feature list in main README
- Installation instructions if needed
- Usage examples
- API documentation links

## 🐛 Issue Reporting

### Bug Reports
When reporting bugs, include:
- **Environment**: OS, Python version, dependency versions
- **Steps to reproduce**: Clear, minimal example
- **Expected behavior**: What should happen
- **Actual behavior**: What actually happens  
- **Error messages**: Full stack traces
- **Additional context**: Screenshots, logs, etc.

### Feature Requests
When requesting features:
- **Problem description**: What problem does this solve?
- **Proposed solution**: How should it work?
- **Alternatives considered**: Other approaches you've thought of
- **Use cases**: Specific examples of when this would be useful

### Issue Templates
Use our issue templates:
- 🐛 Bug Report
- ✨ Feature Request  
- 📚 Documentation Improvement
- 🚀 Performance Issue

## 📝 Pull Request Guidelines

### PR Checklist
Before submitting a pull request:

- [ ] **Code follows** style guidelines and passes linting
- [ ] **Tests added** for new functionality
- [ ] **All tests pass** locally
- [ ] **Documentation updated** (docstrings, README, etc.)
- [ ] **Commits are atomic** and have clear messages
- [ ] **Branch is up-to-date** with develop branch
- [ ] **No merge conflicts** exist

### PR Description Template
```markdown
## Description
Brief description of changes and motivation.

## Type of Change
- [ ] Bug fix (non-breaking change fixing an issue)
- [ ] New feature (non-breaking change adding functionality)
- [ ] Breaking change (fix/feature causing existing functionality to break)
- [ ] Documentation update

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] All tests pass locally
- [ ] Performance impact assessed

## Screenshots (if applicable)
Add screenshots or GIFs showing the changes.

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes or breaking changes documented
```

## 🏆 Recognition

### Contributors
All contributors are recognized in:
- **AUTHORS.md**: List of all contributors
- **Release notes**: Major contributors per release  
- **README**: Top contributors section
- **Annual report**: Yearly contribution highlights

### Contribution Types
We recognize various contribution types:
- 💻 Code
- 📖 Documentation  
- 🎨 Design
- 🤔 Ideas
- 🐛 Bug reports
- 💬 Answering questions
- ⚠️ Tests
- 🔧 Tools
- 🌍 Translation

## 📞 Community

### Communication Channels
- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and ideas
- **Discord** [Coming Soon]: Real-time chat and support
- **Email**: maintainers@nitroagi.dev [Coming Soon]

### Code of Conduct
Please read our [Code of Conduct](CODE_OF_CONDUCT.md) before participating in our community.

### Getting Help
- Check existing documentation and issues first
- Use GitHub Discussions for questions
- Tag maintainers only for urgent issues
- Be patient and respectful with responses

## 🎓 Learning Resources

### AI/ML Resources
- [Hugging Face Course](https://huggingface.co/course)
- [Fast.ai Practical Deep Learning](https://course.fast.ai/)
- [Papers With Code](https://paperswithcode.com/)

### Python/Development Resources
- [Real Python](https://realpython.com/)
- [Python Testing 101](https://realpython.com/python-testing/)
- [FastAPI Tutorial](https://fastapi.tiangolo.com/tutorial/)

### System Design Resources
- [System Design Primer](https://github.com/donnemartin/system-design-primer)
- [Microservices Patterns](https://microservices.io/patterns/)

## ❓ FAQ

### Q: How do I get started contributing?
**A**: Check out "good first issue" labels, read the development setup guide, and don't hesitate to ask questions!

### Q: Can I work on multiple issues at once?
**A**: Focus on one issue at a time for better quality and faster reviews.

### Q: How long do PR reviews take?
**A**: We aim for initial review within 48 hours, but complex PRs may take longer.

### Q: What if my PR is rejected?
**A**: We'll provide constructive feedback. Address the concerns and resubmit!

### Q: Can I contribute if I'm new to AI/ML?
**A**: Absolutely! There are many ways to contribute beyond AI code - documentation, testing, UI, and more.

---

Thank you for contributing to NitroAGI! Together, we're building the future of artificial intelligence. 🚀
