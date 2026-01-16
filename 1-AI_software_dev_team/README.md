# 🧑‍💻 AI Software Development Team

A complete AI-powered software development team using **CrewAI** that can take a project description and deliver working code with tests, documentation, and deployment configurations.

![CrewAI](https://img.shields.io/badge/Framework-CrewAI-blue)
![Multi-LLM](https://img.shields.io/badge/LLM-Gemini%20|%20Claude%20|%20GPT-green)
![Architecture](https://img.shields.io/badge/Architecture-Hierarchical-purple)

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🎭 **8 Specialized Agents** | Product Manager, Tech Lead, Backend Dev, Frontend Dev, DevOps, QA, Security, Docs |
| 🏗️ **Hierarchical Structure** | PM → Tech Lead → Developers (realistic team dynamics) |
| 📝 **Complete Deliverables** | Working code, tests, docs, Docker configs, CI/CD |
| 🔄 **Multi-LLM Support** | Gemini, Claude, or GPT - switch with one config |
| 💻 **Multiple Interfaces** | CLI, Python API, Web UI |
| 🔒 **Security Review** | OWASP-based vulnerability scanning |

## 🏗️ Team Structure

```
┌─────────────────────────────────────────────────────────────┐
│                    📋 PRODUCT MANAGER                        │
│         (Interprets requirements, creates specs)            │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                     🏗️ TECH LEAD                             │
│    (Designs architecture, assigns tasks, reviews code)      │
└─────────────────────────┬───────────────────────────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        ▼                 ▼                 ▼
┌───────────────┐ ┌───────────────┐ ┌───────────────┐
│   ⚙️ BACKEND   │ │   🎨 FRONTEND  │ │   🚀 DEVOPS    │
│   DEVELOPER   │ │   DEVELOPER   │ │   ENGINEER    │
└───────────────┘ └───────────────┘ └───────────────┘
        │                 │                 │
        ▼                 ▼                 ▼
┌───────────────┐ ┌───────────────┐ ┌───────────────┐
│   🧪 QA        │ │   📝 DOCS      │ │   🔒 SECURITY  │
│   ENGINEER    │ │   WRITER      │ │   REVIEWER    │
└───────────────┘ └───────────────┘ └───────────────┘
```

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/yourusername/ai-dev-team.git
cd ai-dev-team

pip install -r requirements.txt

# Set API key for your provider:
export GOOGLE_API_KEY="your-key"     # For Gemini (recommended)
export ANTHROPIC_API_KEY="your-key"  # For Claude
export OPENAI_API_KEY="your-key"     # For OpenAI
```

### CLI Usage

```bash
# Quick build (minimal team)
python main.py build "Build a REST API for a todo app with user authentication"

# Full build with all agents
python main.py build "Build a todo app" --full

# Include frontend development
python main.py build "Build a full-stack todo app" --frontend --full

# Interactive mode
python main.py interactive

# Run a single task
python main.py task requirements "Build a user authentication system"
```

### Python API

```python
from ai_dev_team import AIDevTeam, DevTeamConfig

# Quick build
from ai_dev_team import quick_build
result = quick_build("Build a REST API for a todo app")

# Full configuration
config = DevTeamConfig(
    llm_provider="gemini",    # or "anthropic", "openai"
    team_type="full",         # "full", "minimal", "backend"
    include_frontend=True,
    output_dir="./my_project",
)
team = AIDevTeam(config)
result = team.run("Build a REST API for a todo app with user authentication")

# Check results
print(f"✅ Project created at: {result.output_dir}")
print(f"📄 Files created: {len(result.files_created)}")
print(f"⏱️ Time: {result.total_time_seconds:.1f}s")

# Run single task
output = team.run_single_task("architecture", "Design a microservices system")
print(output)
```

### Web UI

```bash
python main.py serve
# Open http://localhost:8000
```

## 📁 Output Structure

When you build a project, the team creates:

```
output/
└── todo_api_20240115_143052/
    ├── main.py              # Application entry point
    ├── models.py            # Database models
    ├── routes.py            # API endpoints
    ├── services.py          # Business logic
    ├── auth.py              # Authentication
    ├── config.py            # Configuration
    ├── database.py          # Database setup
    │
    ├── tests/
    │   ├── conftest.py      # Test fixtures
    │   ├── test_unit.py     # Unit tests
    │   └── test_integration.py
    │
    ├── docs/
    │   ├── README.md        # Project documentation
    │   ├── API.md           # API documentation
    │   ├── ARCHITECTURE.md  # Architecture docs
    │   └── DEPLOYMENT.md    # Deployment guide
    │
    ├── deployment/
    │   ├── Dockerfile       # Multi-stage Docker build
    │   ├── docker-compose.yml
    │   ├── docker-compose.prod.yml
    │   └── k8s/
    │       ├── deployment.yaml
    │       ├── service.yaml
    │       └── configmap.yaml
    │
    ├── .github/
    │   └── workflows/
    │       └── ci.yml       # CI/CD pipeline
    │
    ├── requirements.txt
    ├── .env.example
    └── SECURITY_REVIEW.md   # Security audit results
```

## 🎭 Agent Roles

### Product Manager 📋
- Transforms requirements into technical specs
- Creates user stories with acceptance criteria
- Prioritizes features (MoSCoW method)
- Identifies edge cases and error scenarios

### Tech Lead 🏗️
- Designs system architecture
- Makes technology decisions
- Creates API specifications
- Writes Architecture Decision Records (ADRs)

### Backend Developer ⚙️
- Implements server-side code
- Creates database models
- Builds API endpoints
- Handles authentication/authorization

### Frontend Developer 🎨
- Creates UI components
- Implements responsive design
- Handles state management
- Ensures accessibility

### DevOps Engineer 🚀
- Creates Docker configurations
- Sets up Kubernetes manifests
- Builds CI/CD pipelines
- Configures monitoring

### QA Engineer 🧪
- Writes unit tests
- Creates integration tests
- Tests edge cases
- Ensures code coverage

### Security Engineer 🔒
- Reviews code for vulnerabilities
- Checks OWASP Top 10
- Identifies security misconfigurations
- Suggests remediation

### Technical Writer 📝
- Creates README documentation
- Writes API documentation
- Documents architecture
- Creates deployment guides

## 🔧 Configuration

### Team Types

| Type | Agents | Best For |
|------|--------|----------|
| `minimal` | PM, Tech Lead, Backend, QA, Docs | Quick prototypes |
| `backend` | + DevOps, Security | Backend services |
| `full` | + Frontend | Full-stack apps |

### LLM Providers

| Provider | Model | API Key |
|----------|-------|---------|
| `gemini` | gemini-1.5-flash | `GOOGLE_API_KEY` |
| `anthropic` | claude-sonnet-4-20250514 | `ANTHROPIC_API_KEY` |
| `openai` | gpt-4o-mini | `OPENAI_API_KEY` |

### Full Configuration

```python
from ai_dev_team import AIDevTeam, DevTeamConfig

config = DevTeamConfig(
    # LLM Settings
    llm_provider="gemini",
    llm_model=None,  # Uses default
    temperature=0.1,
    
    # Team Settings
    team_type="full",
    include_frontend=True,
    
    # Output Settings
    output_dir="./output",
    project_name="my_project",
    
    # Execution Settings
    verbose=True,
    max_iterations=15,
)
```

## 📊 Example Projects

### 1. REST API for Todo App

```bash
python main.py build "Build a REST API for a todo app with user authentication, 
    JWT tokens, CRUD operations for tasks, task categories, and due dates"
```

**Output includes:**
- FastAPI application with SQLAlchemy models
- JWT authentication with refresh tokens
- Full CRUD endpoints
- Comprehensive test suite
- Docker deployment

### 2. URL Shortener Service

```bash
python main.py build "Create a URL shortener service with:
    - Custom short codes
    - Click analytics
    - Rate limiting
    - API key authentication"
```

### 3. Blog API

```bash
python main.py build "Build a blog API with posts, comments, tags, 
    user authentication, and markdown support" --full
```

## 🛠️ Tools Available to Agents

The agents have access to these tools:

| Tool | Description |
|------|-------------|
| `file_writer` | Write code files to output directory |
| `file_reader` | Read existing files |
| `code_validator` | Validate Python syntax and check issues |
| `linter` | Run linting checks (ruff/pylint) |
| `code_executor` | Execute Python code in sandbox |
| `test_runner` | Run pytest tests |
| `security_scanner` | Scan for security vulnerabilities |

## 📈 Performance Tips

1. **Use `minimal` team** for quick prototypes (5 agents vs 8)
2. **Gemini is fastest** and has a free tier
3. **Be specific** in requirements for better results
4. **Run single tasks** for quick iterations:
   ```bash
   python main.py task architecture "Design a caching system"
   ```

## 🔒 Security

- Code execution is sandboxed
- No secrets stored in generated code (uses env vars)
- Security review identifies vulnerabilities
- OWASP Top 10 checks included

## 📝 License

MIT License

## 🙏 Acknowledgments

- [CrewAI](https://github.com/joaomdmoura/crewAI) - Multi-agent framework
- [LangChain](https://github.com/langchain-ai/langchain) - LLM orchestration
