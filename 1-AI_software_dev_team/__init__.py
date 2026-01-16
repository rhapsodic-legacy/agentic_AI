"""
AI Dev Team - Task Definitions

Defines the tasks for each stage of the development process.
Tasks are chained together to form a complete development workflow.
"""

from crewai import Task
from typing import Optional


def create_requirements_task(agent, user_input: str) -> Task:
    """
    Task: Convert user requirements into technical specifications.
    """
    return Task(
        description=f"""Analyze the following user requirement and create a detailed 
        Product Requirements Document (PRD):

        USER REQUIREMENT:
        {user_input}

        Your PRD should include:
        1. **Project Overview**: Brief description and goals
        2. **User Stories**: In the format "As a [user], I want [feature], so that [benefit]"
        3. **Functional Requirements**: Detailed list of features
        4. **Non-Functional Requirements**: Performance, security, scalability needs
        5. **Acceptance Criteria**: How to verify each feature works
        6. **Out of Scope**: What this project will NOT include
        7. **Technical Constraints**: Any limitations or requirements (languages, frameworks)
        8. **Priority**: MoSCoW prioritization (Must/Should/Could/Won't have)

        Be thorough but practical. Focus on what's achievable in a reasonable timeframe.
        Consider edge cases and error scenarios.""",
        expected_output="""A comprehensive PRD document in markdown format containing:
        - Clear project goals
        - 3-10 user stories with acceptance criteria
        - Prioritized feature list
        - Technical requirements and constraints""",
        agent=agent,
    )


def create_architecture_task(agent, context: Optional[list] = None) -> Task:
    """
    Task: Design the system architecture based on requirements.
    """
    return Task(
        description="""Based on the Product Requirements Document, design a robust 
        and scalable software architecture.

        Your architecture document should include:

        1. **System Overview**
           - High-level architecture diagram (describe in text/ASCII)
           - Key components and their responsibilities

        2. **Technology Stack**
           - Programming language(s) and frameworks
           - Database choice with justification
           - External services/APIs needed

        3. **API Design**
           - REST endpoints (method, path, request/response)
           - Authentication approach
           - Error handling strategy

        4. **Data Model**
           - Entity definitions with fields and types
           - Relationships between entities
           - Database schema overview

        5. **Project Structure**
           - Directory layout
           - File organization
           - Module responsibilities

        6. **Architecture Decision Records (ADRs)**
           - Key decisions and their rationale
           - Alternatives considered
           - Trade-offs made

        Design for:
        - Maintainability: Easy to understand and modify
        - Scalability: Can grow with usage
        - Security: Protected against common vulnerabilities
        - Testability: Easy to write tests for

        Be pragmatic - choose simple solutions over complex ones when appropriate.""",
        expected_output="""A comprehensive architecture document containing:
        - System design with component diagram
        - Complete technology stack
        - API endpoint specifications
        - Data model with relationships
        - Project structure
        - At least 2 ADRs for key decisions""",
        agent=agent,
        context=context,
    )


def create_backend_task(agent, context: Optional[list] = None) -> Task:
    """
    Task: Implement the backend code.
    """
    return Task(
        description="""Based on the architecture design, implement the complete backend code.

        You must create ALL of the following files with COMPLETE, WORKING code:

        1. **Main Application** (main.py or app.py)
           - Application setup and configuration
           - Route registration
           - Middleware setup
           - Error handlers

        2. **Models** (models.py)
           - Database models/schemas
           - Pydantic models for request/response validation
           - Relationships and constraints

        3. **Routes/Controllers** (routes.py or api/)
           - All API endpoints
           - Request validation
           - Response formatting
           - Proper HTTP status codes

        4. **Services/Business Logic** (services.py)
           - Core business logic
           - Data processing
           - External service integrations

        5. **Database** (database.py)
           - Connection setup
           - Session management
           - Migration utilities if needed

        6. **Authentication** (auth.py) - if required
           - User authentication
           - Token management
           - Password hashing

        7. **Configuration** (config.py)
           - Environment variables
           - App settings
           - Constants

        8. **Requirements** (requirements.txt)
           - All Python dependencies with versions

        IMPORTANT:
        - Write COMPLETE, PRODUCTION-READY code, not placeholders
        - Include proper error handling and logging
        - Add docstrings and type hints
        - Follow PEP 8 style guidelines
        - Use environment variables for secrets
        - Include input validation""",
        expected_output="""Complete backend implementation with:
        - All files listed above with full, working code
        - Proper error handling throughout
        - Type hints and docstrings
        - Requirements.txt with all dependencies""",
        agent=agent,
        context=context,
    )


def create_frontend_task(agent, context: Optional[list] = None) -> Task:
    """
    Task: Implement the frontend code (when requested).
    """
    return Task(
        description="""Based on the architecture design, implement the frontend code.

        Create a complete, functional frontend with:

        1. **Main Application Setup**
           - React/Vue/Svelte app structure (as specified)
           - Routing configuration
           - State management setup

        2. **Components**
           - Reusable UI components
           - Form components with validation
           - Layout components

        3. **Pages/Views**
           - All required pages
           - Proper routing
           - Loading and error states

        4. **API Integration**
           - API client/service
           - Request/response handling
           - Error handling

        5. **Styling**
           - CSS/Tailwind styles
           - Responsive design
           - Dark mode if appropriate

        6. **Configuration**
           - Environment variables
           - Build configuration

        IMPORTANT:
        - Write complete, working code
        - Ensure responsive design
        - Include form validation
        - Handle loading and error states
        - Follow accessibility best practices (ARIA labels, semantic HTML)""",
        expected_output="""Complete frontend implementation with:
        - All components and pages
        - API integration
        - Styling and responsive design
        - Package.json with dependencies""",
        agent=agent,
        context=context,
    )


def create_testing_task(agent, context: Optional[list] = None) -> Task:
    """
    Task: Write comprehensive tests.
    """
    return Task(
        description="""Write comprehensive tests for the implemented code.

        Create the following test files:

        1. **Unit Tests** (test_unit.py or tests/unit/)
           - Test individual functions and methods
           - Test edge cases and boundary conditions
           - Test error handling
           - Mock external dependencies

        2. **Integration Tests** (test_integration.py or tests/integration/)
           - Test API endpoints end-to-end
           - Test database operations
           - Test authentication flows
           - Test with realistic data

        3. **Test Fixtures** (conftest.py)
           - Shared fixtures for tests
           - Test database setup
           - Mock services

        Test Coverage Goals:
        - All API endpoints tested
        - All business logic functions tested
        - Error cases and edge cases covered
        - At least 80% code coverage target

        For each test:
        - Use descriptive test names (test_user_can_create_account)
        - Include docstrings explaining what's being tested
        - Follow Arrange-Act-Assert pattern
        - Use appropriate assertions

        IMPORTANT:
        - Tests should be RUNNABLE with pytest
        - Include both happy path and error cases
        - Test validation and error messages
        - Mock external services properly""",
        expected_output="""Complete test suite with:
        - Unit tests for all functions/methods
        - Integration tests for all API endpoints
        - Test fixtures and conftest.py
        - At least 20 meaningful test cases""",
        agent=agent,
        context=context,
    )


def create_devops_task(agent, context: Optional[list] = None) -> Task:
    """
    Task: Create deployment and infrastructure configuration.
    """
    return Task(
        description="""Create production-ready deployment configuration.

        Create ALL of the following:

        1. **Dockerfile**
           - Multi-stage build for smaller images
           - Non-root user for security
           - Proper layer caching
           - Health check

        2. **docker-compose.yml**
           - All services (app, database, cache, etc.)
           - Proper networking
           - Volume mounts
           - Environment variables
           - Health checks

        3. **docker-compose.prod.yml** (production overrides)
           - Production-specific settings
           - Resource limits
           - Restart policies

        4. **.dockerignore**
           - Exclude unnecessary files

        5. **Kubernetes Manifests** (k8s/ directory)
           - deployment.yaml
           - service.yaml
           - configmap.yaml
           - secret.yaml (template)
           - ingress.yaml (if applicable)
           - hpa.yaml (horizontal pod autoscaler)

        6. **CI/CD Pipeline** (.github/workflows/ci.yml)
           - Lint and type check
           - Run tests
           - Build Docker image
           - Push to registry
           - Deploy (staging/production)

        7. **Environment Files**
           - .env.example (template with all variables)
           - Document required environment variables

        IMPORTANT:
        - All configs should be PRODUCTION-READY
        - Include proper resource limits
        - Use secrets for sensitive data
        - Include health checks
        - Follow 12-factor app principles""",
        expected_output="""Complete deployment configuration with:
        - Dockerfile with multi-stage build
        - Docker Compose files (dev and prod)
        - Kubernetes manifests
        - CI/CD pipeline configuration
        - Environment variable documentation""",
        agent=agent,
        context=context,
    )


def create_security_review_task(agent, context: Optional[list] = None) -> Task:
    """
    Task: Perform security review of the codebase.
    """
    return Task(
        description="""Perform a comprehensive security review of the generated code.

        Review for the following vulnerability categories:

        1. **Injection Attacks**
           - SQL Injection
           - Command Injection
           - XSS (Cross-Site Scripting)
           - Template Injection

        2. **Authentication & Authorization**
           - Weak password policies
           - Session management issues
           - Missing authorization checks
           - JWT vulnerabilities

        3. **Sensitive Data Exposure**
           - Hardcoded secrets
           - Logging sensitive data
           - Insecure data transmission
           - PII handling

        4. **Security Misconfiguration**
           - Debug mode in production
           - Default credentials
           - Missing security headers
           - CORS misconfiguration

        5. **Cryptographic Issues**
           - Weak algorithms
           - Poor key management
           - Missing encryption

        6. **Dependency Vulnerabilities**
           - Check requirements.txt versions
           - Known CVEs in dependencies

        For each finding:
        - Describe the vulnerability
        - Explain the risk (with severity: Critical/High/Medium/Low)
        - Provide a specific code fix
        - Reference relevant security standards (OWASP, CWE)

        Also provide:
        - Security recommendations for deployment
        - Suggested security headers
        - Rate limiting recommendations""",
        expected_output="""Security review report containing:
        - List of vulnerabilities found with severity
        - Specific code fixes for each issue
        - Security recommendations
        - Deployment hardening checklist""",
        agent=agent,
        context=context,
    )


def create_documentation_task(agent, context: Optional[list] = None) -> Task:
    """
    Task: Create comprehensive project documentation.
    """
    return Task(
        description="""Create comprehensive documentation for the project.

        Generate ALL of the following:

        1. **README.md**
           - Project title and description
           - Features list
           - Quick start guide (< 5 steps to run locally)
           - Prerequisites
           - Installation instructions
           - Configuration (environment variables)
           - Usage examples
           - API overview
           - Contributing guidelines
           - License

        2. **API Documentation** (API.md or docs/api.md)
           - All endpoints with:
             - Method and path
             - Description
             - Request parameters/body
             - Response format
             - Error codes
             - Example requests/responses (curl)

        3. **Architecture Documentation** (ARCHITECTURE.md)
           - System overview
           - Component descriptions
           - Data flow diagrams (text/ASCII)
           - Technology choices and rationale

        4. **Development Guide** (DEVELOPMENT.md)
           - Development setup
           - Code style guidelines
           - Testing instructions
           - Debugging tips
           - Common issues and solutions

        5. **Deployment Guide** (DEPLOYMENT.md)
           - Deployment options
           - Environment configuration
           - Docker deployment
           - Kubernetes deployment
           - Monitoring and logging

        6. **CHANGELOG.md**
           - Initial release notes

        IMPORTANT:
        - Documentation should be clear and beginner-friendly
        - Include practical examples
        - Keep code examples up-to-date
        - Use proper markdown formatting""",
        expected_output="""Complete documentation set with:
        - README.md with quick start
        - API documentation with examples
        - Architecture documentation
        - Development and deployment guides
        - CHANGELOG.md""",
        agent=agent,
        context=context,
    )


# =============================================================================
# Task Chains
# =============================================================================

def create_full_development_workflow(
    agents: dict,
    user_input: str,
    include_frontend: bool = False,
) -> list[Task]:
    """
    Create the full chain of tasks for a complete development workflow.
    """
    tasks = []
    
    # 1. Requirements analysis
    requirements_task = create_requirements_task(
        agents["product_manager"],
        user_input,
    )
    tasks.append(requirements_task)
    
    # 2. Architecture design
    architecture_task = create_architecture_task(
        agents["tech_lead"],
        context=[requirements_task],
    )
    tasks.append(architecture_task)
    
    # 3. Backend implementation
    backend_task = create_backend_task(
        agents["backend_dev"],
        context=[requirements_task, architecture_task],
    )
    tasks.append(backend_task)
    
    # 4. Frontend implementation (optional)
    if include_frontend and "frontend_dev" in agents:
        frontend_task = create_frontend_task(
            agents["frontend_dev"],
            context=[requirements_task, architecture_task, backend_task],
        )
        tasks.append(frontend_task)
    
    # 5. Testing
    qa_context = [backend_task]
    if include_frontend and "frontend_dev" in agents:
        qa_context.append(frontend_task)
    
    testing_task = create_testing_task(
        agents["qa"],
        context=[architecture_task] + qa_context,
    )
    tasks.append(testing_task)
    
    # 6. DevOps configuration
    if "devops" in agents:
        devops_task = create_devops_task(
            agents["devops"],
            context=[architecture_task, backend_task],
        )
        tasks.append(devops_task)
    
    # 7. Security review
    if "security" in agents:
        security_context = [backend_task]
        if include_frontend:
            security_context.append(frontend_task)
        
        security_task = create_security_review_task(
            agents["security"],
            context=security_context,
        )
        tasks.append(security_task)
    
    # 8. Documentation
    docs_task = create_documentation_task(
        agents["docs"],
        context=tasks[:-1] if "security" in agents else tasks,  # All previous tasks
    )
    tasks.append(docs_task)
    
    return tasks


def create_quick_prototype_workflow(
    agents: dict,
    user_input: str,
) -> list[Task]:
    """
    Create a minimal task chain for quick prototyping.
    """
    tasks = []
    
    # 1. Quick requirements
    requirements_task = create_requirements_task(
        agents["product_manager"],
        user_input,
    )
    tasks.append(requirements_task)
    
    # 2. Architecture (simplified)
    architecture_task = create_architecture_task(
        agents["tech_lead"],
        context=[requirements_task],
    )
    tasks.append(architecture_task)
    
    # 3. Backend
    backend_task = create_backend_task(
        agents["backend_dev"],
        context=[requirements_task, architecture_task],
    )
    tasks.append(backend_task)
    
    # 4. Basic tests
    testing_task = create_testing_task(
        agents["qa"],
        context=[backend_task],
    )
    tasks.append(testing_task)
    
    # 5. README
    docs_task = create_documentation_task(
        agents["docs"],
        context=[architecture_task, backend_task],
    )
    tasks.append(docs_task)
    
    return tasks
