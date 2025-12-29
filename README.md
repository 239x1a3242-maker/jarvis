# JARVIS - Advanced Personal AI Assistant

A sophisticated, enterprise-grade personal AI assistant inspired by J.A.R.V.I.S. from Iron Man, featuring advanced autonomous web automation capabilities and modular architecture.

## 🏗️ Architecture

JARVIS follows a clean, modular architecture with clear separation of concerns:

### Core Components
- **Orchestrator**: Central coordination and task management
- **Intent Classifier**: Natural language understanding and routing
- **Planner**: Task planning and execution strategy
- **LLM Interface**: Local language model integration (Phi-3, GPT-2, etc.)

### Interface Layer
- **Text Interface**: Command-line interaction
- **Voice Interface**: Speech-to-text and text-to-speech capabilities

### Memory System
- **Short-term Memory**: Recent conversation context
- **Working Memory**: Active task state
- **Long-term Memory**: SQLite-based persistent storage
- **Vector Store**: ChromaDB for semantic search

### Tool Ecosystem
- **System Commands**: Safe execution of shell commands
- **File Manager**: File system operations
- **Web Automation**: Advanced autonomous web interaction

### Advanced Web Automation Module
- **Core Components**: Automation orchestrator, browser controller, task parser
- **Handler Layer**: Form automation, data acquisition, verification, error recovery
- **Data Layer**: Persistence, response building, utilities
- **Advanced Features**:
  - **State Machine**: Workflow state management
  - **Observability**: Metrics collection and monitoring
  - **Browser Abstraction**: Multi-browser support (Selenium, Playwright, Remote)
  - **Planner Feedback**: Execution feedback and optimization
  - **Policy Engine**: Safety policies and compliance

## ✨ Features

### Core Capabilities
- **Modular Design**: Easy to extend with new tools and interfaces
- **Local Operation**: Runs entirely locally, prioritizing privacy
- **Intent-Based Routing**: Classifies requests and routes to appropriate modules
- **Memory System**: Maintains conversation context and long-term knowledge
- **Tool Execution**: Safe execution of system commands and operations
- **Async Architecture**: Supports concurrent task execution

### Advanced Web Automation
- **Autonomous Operation**: Self-directed web task execution
- **Multi-Browser Support**: Selenium, Playwright, and remote browsers
- **Intelligent Recovery**: Error handling with fallback strategies
- **Crash-Safe Execution**: Policy enforcement and safety measures
- **Metrics & Monitoring**: Comprehensive observability and analytics
- **State Management**: Formal workflow state machines

### Web Automation Commands
- **Navigation**: Open websites and navigate pages
- **Interaction**: Click elements, fill forms, handle dropdowns
- **Data Extraction**: Scrape content, extract structured data
- **Authentication**: Login automation with credential management
- **Verification**: Element presence and content validation
- **Screenshots**: Visual documentation and debugging
- **Advanced Workflows**: Multi-step automation sequences

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Chrome browser (for web automation)
- Microphone and speakers (for voice interface)

### Installation

1. **Clone and setup**:
   ```bash
   git clone <repository-url>
   cd jarvis
   pip install -r requirements.txt
   ```

2. **Install Playwright browsers** (for advanced web automation):
   ```bash
   playwright install
   ```

3. **Configure settings**:
   ```bash
   # Edit config/settings.yaml as needed
   nano config/settings.yaml
   ```

4. **Run JARVIS**:
   ```bash
   python main.py
   ```

## 🐳 Docker Deployment

### Using Docker Compose (Recommended)
```bash
cd docker
docker-compose up --build
```

### Manual Docker Build
```bash
# Build the image
docker build -f docker/Dockerfile -t jarvis .

# Run the container
docker run -it --rm \
  -v $(pwd)/memory:/app/memory \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/models:/app/models \
  --device /dev/snd \
  jarvis
```

## 📋 Usage Examples

### Basic Commands
```
"Hello JARVIS" → Greeting and interface selection
"Check system status" → System information
"Open file manager" → File system operations
```

### Web Automation
```
"automate web open https://example.com"
"automate web click button.login"
"automate web fill_form {'input.email': 'user@example.com', 'input.password': 'pass'}"
"automate web extract_data {'title': 'h1', 'content': '.article'}"
"automate web screenshot dashboard.png"
```

### Advanced Workflows
```
"automate web login https://site.com {'username': 'user', 'password': 'pass'}"
"automate web scrape https://news.com {'headlines': 'h2.article-title'}"
"automate web workflow shopping_cart {'add_item': 'product-123', 'checkout': true}"
```

## 🛠️ Development

### Project Structure
```
jarvis/
├── main.py                 # Application entry point
├── config/
│   └── settings.yaml       # Configuration file
├── core/                   # Core AI components
│   ├── orchestrator.py     # Central coordinator
│   ├── intent_classifier.py # Intent classification
│   ├── planner.py          # Task planning
│   └── llm_interface.py    # LLM integration
├── interfaces/             # User interfaces
│   ├── text_interface.py   # Text-based interaction
│   └── voice_interface.py  # Voice interaction
├── memory/                 # Memory management
│   └── memory_manager.py   # Memory operations
├── tools/                  # Tool ecosystem
│   ├── tool_registry.py    # Tool management
│   ├── file_manager.py     # File operations
│   ├── system_commands.py  # System commands
│   └── web_automation.py   # Basic web automation
├── web_automation_module/  # Advanced web automation
│   ├── core/               # Core automation components
│   ├── handlers/           # Automation handlers
│   ├── data/               # Data management
│   ├── utils/              # Utilities
│   └── advanced/           # Advanced features
│       ├── state_machine/      # Workflow states
│       ├── observability/      # Metrics & monitoring
│       ├── browser_abstraction/ # Multi-browser support
│       ├── planner_feedback/   # Execution feedback
│       └── policies/           # Safety policies
├── logs/                   # Application logs
├── models/                 # Local LLM models
├── docker/                 # Docker configuration
└── requirements.txt        # Python dependencies
```

### Adding New Tools
1. Create a new tool class in `tools/`
2. Register it in `tools/tool_registry.py`
3. Add configuration in `config/settings.yaml`

### Extending Web Automation
1. Add new handlers in `web_automation_module/handlers/`
2. Update the orchestrator in `web_automation_module/core/`
3. Add advanced features in `web_automation_module/advanced/`

## 📊 Development Roadmap

- ✅ **Phase 1**: Basic project structure and core orchestrator
- ✅ **Phase 2**: Text interface and command loop
- ✅ **Phase 3**: Intent classification and routing
- ✅ **Phase 4**: Memory system (SQLite + ChromaDB)
- ✅ **Phase 5**: Tool framework and system integration
- ✅ **Phase 6**: Local LLM integration (Phi-3 Mini)
- ✅ **Phase 7**: Voice capabilities (STT/TTS)
- ✅ **Phase 8**: Basic web automation (Selenium)
- ✅ **Phase 9**: Advanced web automation module
- ✅ **Phase 10**: Docker containerization
- 🔄 **Phase 11**: Advanced planning and reasoning
- 🔄 **Phase 12**: Computer vision integration
- 🔄 **Phase 13**: Proactive monitoring and behavior
- 🔄 **Phase 14**: Multi-modal interfaces

## 🔧 Configuration

Edit `config/settings.yaml` to customize:
- LLM model settings
- Memory configuration
- Tool permissions
- Web automation policies
- Logging levels

## 🤝 Contributing

This is a personal project, but the modular design makes it easy to add new features. Follow the existing patterns for tools and interfaces. Key principles:

- **Clean Architecture**: Separation of concerns
- **Async First**: Asynchronous operations throughout
- **Error Resilience**: Comprehensive error handling
- **Observability**: Logging and metrics collection
- **Safety First**: Policy enforcement and validation

## 📄 License

Personal project - see individual file headers for licensing information.