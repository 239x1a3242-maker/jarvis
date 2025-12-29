# Web Automation Module for JARVIS

A comprehensive, production-ready web automation system with fault-tolerant execution, human-in-the-loop verification, and Docker containerization support.

## Overview

The Web Automation Module provides JARVIS with the ability to perform complex web automation tasks including:

- **Login & Registration**: Automated form filling with adaptive field detection
- **Data Extraction**: Structured data scraping from dynamic and static websites
- **Screenshot Capture**: High-quality webpage screenshots for documentation
- **Verification Handling**: CAPTCHA, OTP, and email verification with human-in-the-loop support
- **Error Recovery**: Comprehensive error handling with multiple recovery strategies
- **Data Persistence**: SQLite-based storage with Docker volume support

## Architecture

The module follows a modular, fault-tolerant architecture with 10 core components:

### Core Components
- **AutomationOrchestrator**: Main task lifecycle management and coordination
- **TaskParser**: Natural language processing and execution plan generation
- **BrowserController**: Selenium WebDriver management with retry logic

### Handler Components
- **WebInteractionLayer**: Multi-strategy element interaction with robust waiting
- **FormAutomationEngine**: Adaptive form detection and automated filling
- **VerificationHandler**: Human-in-the-loop CAPTCHA/OTP/email verification
- **DataAcquisitionLayer**: Structured data extraction and screenshot capture
- **ErrorHandlerRecoveryEngine**: Comprehensive error classification and recovery

### Data & Utils
- **DataPersistenceLayer**: SQLite-based data storage and retrieval
- **ResponseBuilder**: Structured response generation with voice interface support

## Key Features

### Fault-Tolerant Execution
- Exponential backoff retry mechanisms
- Multiple recovery strategies per error type
- Graceful degradation for partial failures
- Comprehensive error logging and tracking

### Human-in-the-Loop Verification
- CAPTCHA detection and human intervention requests
- OTP and email verification workflows
- Voice/text interface integration for user prompts
- Ethical automation without illegal bypass techniques

### Adaptive Web Automation
- Dynamic selector strategies (CSS, XPath, text, attributes)
- Pattern-based form field detection
- Website structure adaptation without hardcoded logic
- Real-time element interaction validation

### Production-Ready Design
- Async/await pattern for non-blocking operations
- Docker containerization with volume mounting
- CPU-compatible headless browser operation
- Structured logging and monitoring
- Comprehensive testing and validation

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. For Docker deployment:
```bash
docker build -t jarvis-web-automation .
docker run -v ./data:/app/data jarvis-web-automation
```

## Usage

### Basic Usage

```python
from web_automation_module import WebAutomationModule

async def main():
    # Initialize the module
    module = WebAutomationModule()
    await module.initialize()

    # Execute a web automation task
    result = await module.execute_task(
        "Login to example.com with username 'user' and password 'pass', "
        "then extract the user profile data"
    )

    print(f"Task completed: {result['status']}")

    # Cleanup
    await module.shutdown()
```

### Advanced Configuration

```python
config = {
    'browser': {
        'headless': True,
        'window_size': '1920x1080',
        'timeout': 30
    },
    'data': {
        'directory': './automation_data',
        'screenshots_dir': './automation_data/screenshots'
    },
    'recovery': {
        'max_retries': 3,
        'base_delay': 1.0,
        'human_timeout': 300
    }
}

module = WebAutomationModule(config)
```

## Task Examples

The module supports natural language task descriptions:

- **Login Tasks**: "Login to gmail.com with my credentials"
- **Registration**: "Create a new account on example.com with email user@example.com"
- **Data Extraction**: "Scrape product prices from amazon.com for 'laptop'"
- **Screenshots**: "Take a screenshot of the current page on github.com"
- **Form Filling**: "Fill out the contact form on website.com with name 'John Doe'"

## Docker Deployment

The module includes Docker support for containerized deployment:

```dockerfile
FROM python:3.11-slim

# Install Chrome and dependencies
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    && wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add - \
    && echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google.list \
    && apt-get update \
    && apt-get install -y google-chrome-stable \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create data directory
RUN mkdir -p /app/data

# Run the application
CMD ["python", "main.py"]
```

## Error Handling

The module implements comprehensive error handling:

- **Network Errors**: Automatic retry with exponential backoff
- **Element Not Found**: Alternative selector strategies
- **Timeout Errors**: Page refresh and extended wait times
- **CAPTCHA Detection**: Human intervention requests
- **Rate Limiting**: Adaptive delay strategies

## Data Persistence

All automation results are persisted to SQLite database:

- Task metadata and execution history
- Extracted data in JSON/CSV formats
- Screenshots with metadata
- Error logs and recovery attempts
- Docker volume mounting for data persistence

## Security & Ethics

- **No CAPTCHA Bypass**: Ethical automation without illegal techniques
- **Human Verification**: Transparent human-in-the-loop for verification steps
- **Data Privacy**: Secure credential handling and data storage
- **Rate Limiting**: Respectful automation with appropriate delays
- **Legal Compliance**: Designed for legitimate automation use cases

## Testing

Run the module validation test:

```bash
python test_module.py
```

For full functionality testing with dependencies:

```bash
pip install -r requirements.txt
python -m pytest tests/
```

## Contributing

The module is designed with extensibility in mind:

- Modular architecture allows easy addition of new handlers
- Strategy pattern enables custom selector and recovery implementations
- Comprehensive logging facilitates debugging and monitoring
- Async design supports high-throughput automation

## License

This module is part of the JARVIS AI Assistant system.

## Version

Current Version: 1.0.0

## Support

For issues and questions, please refer to the main JARVIS documentation or create an issue in the project repository.