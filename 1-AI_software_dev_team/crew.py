"""
AI Software Development Team - Main Orchestrator

Brings together agents and tasks using CrewAI's hierarchical process.
"""

import os
from dataclasses import dataclass, field
from typing import Optional, Callable
from datetime import datetime
from pathlib import Path

from crewai import Crew, Process


@dataclass
class DevTeamConfig:
    """Configuration for the AI Dev Team."""
    
    # LLM Settings
    llm_provider: str = "gemini"  # "gemini", "anthropic", "openai"
    llm_model: Optional[str] = None
    temperature: float = 0.1
    
    # Team Settings
    team_type: str = "full"  # "full", "minimal", "backend"
    include_frontend: bool = False
    
    # Output Settings
    output_dir: str = "./output"
    project_name: Optional[str] = None
    
    # Execution Settings
    verbose: bool = True
    max_iterations: int = 15
    
    # Callbacks
    on_task_complete: Optional[Callable] = None
    on_step: Optional[Callable] = None


@dataclass
class DevTeamResult:
    """Result from a development run."""
    success: bool
    project_name: str
    output_dir: str
    
    # Task outputs
    requirements: Optional[str] = None
    architecture: Optional[str] = None
    backend_code: Optional[str] = None
    frontend_code: Optional[str] = None
    tests: Optional[str] = None
    devops_config: Optional[str] = None
    security_review: Optional[str] = None
    documentation: Optional[str] = None
    
    # Metadata
    total_time_seconds: float = 0
    tasks_completed: int = 0
    errors: list = field(default_factory=list)
    
    # Files created
    files_created: list = field(default_factory=list)


def create_llm(provider: str, model: Optional[str] = None, temperature: float = 0.1):
    """Create an LLM instance for the specified provider."""
    
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=model or "claude-sonnet-4-20250514",
            temperature=temperature,
        )
    
    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=model or "gemini-1.5-flash",
            temperature=temperature,
        )
    
    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model or "gpt-4o-mini",
            temperature=temperature,
        )
    
    else:
        raise ValueError(f"Unknown provider: {provider}")


class AIDevTeam:
    """
    AI Software Development Team
    
    A complete AI-powered development team that can take a project description
    and deliver working code with tests, documentation, and deployment configs.
    
    Usage:
        team = AIDevTeam(DevTeamConfig(llm_provider="gemini"))
        result = team.run("Build a REST API for a todo app with user authentication")
        
        print(result.output_dir)  # ./output/todo_api
    """
    
    def __init__(self, config: Optional[DevTeamConfig] = None):
        self.config = config or DevTeamConfig()
        
        # Set up output directory
        os.environ["AI_DEV_OUTPUT_DIR"] = self.config.output_dir
        
        # Create LLM
        self.llm = create_llm(
            self.config.llm_provider,
            self.config.llm_model,
            self.config.temperature,
        )
        
        # Initialize team
        self._agents = None
        self._crew = None
    
    def _create_team(self):
        """Create the development team based on config."""
        from .agents.team import DevTeam
        
        team_factory = DevTeam(self.llm)
        
        if self.config.team_type == "full":
            self._agents = team_factory.create_fullstack_team()
        elif self.config.team_type == "minimal":
            self._agents = team_factory.create_minimal_team()
        elif self.config.team_type == "backend":
            self._agents = team_factory.create_backend_team()
        else:
            self._agents = team_factory.create_fullstack_team()
        
        return self._agents
    
    def _create_tasks(self, user_input: str):
        """Create tasks for the workflow."""
        from .tasks import (
            create_full_development_workflow,
            create_quick_prototype_workflow,
        )
        
        if not self._agents:
            self._create_team()
        
        if self.config.team_type == "minimal":
            return create_quick_prototype_workflow(self._agents, user_input)
        else:
            return create_full_development_workflow(
                self._agents,
                user_input,
                include_frontend=self.config.include_frontend,
            )
    
    def _setup_output_directory(self, user_input: str) -> str:
        """Set up the output directory for the project."""
        # Generate project name from input if not provided
        if self.config.project_name:
            project_name = self.config.project_name
        else:
            # Extract meaningful name from input
            words = user_input.lower().split()
            keywords = [w for w in words if w not in 
                       ['a', 'an', 'the', 'build', 'create', 'make', 'develop', 'for', 'with', 'and']]
            project_name = '_'.join(keywords[:3]) if keywords else 'project'
            project_name = ''.join(c if c.isalnum() or c == '_' else '_' for c in project_name)
        
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(self.config.output_dir, f"{project_name}_{timestamp}")
        os.makedirs(output_path, exist_ok=True)
        
        # Update environment variable
        os.environ["AI_DEV_OUTPUT_DIR"] = output_path
        
        return output_path, project_name
    
    def run(self, user_input: str) -> DevTeamResult:
        """
        Run the development team on a project.
        
        Args:
            user_input: Description of what to build
        
        Returns:
            DevTeamResult with all outputs
        """
        import time
        
        start_time = time.time()
        
        # Set up output directory
        output_path, project_name = self._setup_output_directory(user_input)
        
        # Create team and tasks
        self._create_team()
        tasks = self._create_tasks(user_input)
        
        # Create crew
        self._crew = Crew(
            agents=list(self._agents.values()),
            tasks=tasks,
            process=Process.sequential,  # Use sequential for predictable output
            verbose=self.config.verbose,
            max_rpm=10,  # Rate limit
        )
        
        # Run the crew
        try:
            result = self._crew.kickoff()
            
            # Collect results
            dev_result = DevTeamResult(
                success=True,
                project_name=project_name,
                output_dir=output_path,
                total_time_seconds=time.time() - start_time,
                tasks_completed=len(tasks),
            )
            
            # Parse task outputs
            if hasattr(result, 'tasks_output'):
                for i, task_output in enumerate(result.tasks_output):
                    task_name = tasks[i].description[:50] if i < len(tasks) else f"task_{i}"
                    
                    # Save each task output to a file
                    output_file = os.path.join(output_path, f"task_{i+1}_output.md")
                    with open(output_file, 'w') as f:
                        f.write(f"# Task {i+1}\n\n")
                        f.write(str(task_output))
                    dev_result.files_created.append(output_file)
            
            # Save final result
            if hasattr(result, 'raw'):
                final_output = os.path.join(output_path, "final_output.md")
                with open(final_output, 'w') as f:
                    f.write(result.raw)
                dev_result.files_created.append(final_output)
            
            # List all created files
            for root, dirs, files in os.walk(output_path):
                for file in files:
                    filepath = os.path.join(root, file)
                    if filepath not in dev_result.files_created:
                        dev_result.files_created.append(filepath)
            
            return dev_result
            
        except Exception as e:
            return DevTeamResult(
                success=False,
                project_name=project_name,
                output_dir=output_path,
                total_time_seconds=time.time() - start_time,
                errors=[str(e)],
            )
    
    def run_single_task(self, task_type: str, user_input: str, context: str = "") -> str:
        """
        Run a single task type.
        
        Args:
            task_type: "requirements", "architecture", "backend", "frontend",
                      "tests", "devops", "security", "docs"
            user_input: Description of what to build
            context: Optional context from previous tasks
        
        Returns:
            Task output as string
        """
        from .tasks import (
            create_requirements_task,
            create_architecture_task,
            create_backend_task,
            create_frontend_task,
            create_testing_task,
            create_devops_task,
            create_security_review_task,
            create_documentation_task,
        )
        
        if not self._agents:
            self._create_team()
        
        task_map = {
            "requirements": (create_requirements_task, "product_manager", [user_input]),
            "architecture": (create_architecture_task, "tech_lead", []),
            "backend": (create_backend_task, "backend_dev", []),
            "frontend": (create_frontend_task, "frontend_dev", []),
            "tests": (create_testing_task, "qa", []),
            "devops": (create_devops_task, "devops", []),
            "security": (create_security_review_task, "security", []),
            "docs": (create_documentation_task, "docs", []),
        }
        
        if task_type not in task_map:
            raise ValueError(f"Unknown task type: {task_type}")
        
        task_creator, agent_key, args = task_map[task_type]
        
        if agent_key not in self._agents:
            raise ValueError(f"Agent {agent_key} not available in team type {self.config.team_type}")
        
        # Create task
        if args:
            task = task_creator(self._agents[agent_key], *args)
        else:
            task = task_creator(self._agents[agent_key])
        
        # Create minimal crew
        crew = Crew(
            agents=[self._agents[agent_key]],
            tasks=[task],
            process=Process.sequential,
            verbose=self.config.verbose,
        )
        
        result = crew.kickoff()
        return str(result)
    
    def get_team_info(self) -> dict:
        """Get information about the current team configuration."""
        if not self._agents:
            self._create_team()
        
        return {
            "team_type": self.config.team_type,
            "llm_provider": self.config.llm_provider,
            "agents": [
                {
                    "role": agent.role,
                    "goal": agent.goal[:100] + "...",
                }
                for agent in self._agents.values()
            ],
        }


# =============================================================================
# Convenience Functions
# =============================================================================

def quick_build(description: str, provider: str = "gemini") -> DevTeamResult:
    """
    Quickly build a project from a description.
    
    Args:
        description: What to build
        provider: LLM provider ("gemini", "anthropic", "openai")
    
    Returns:
        DevTeamResult
    """
    config = DevTeamConfig(
        llm_provider=provider,
        team_type="minimal",
        verbose=True,
    )
    team = AIDevTeam(config)
    return team.run(description)


def full_build(description: str, provider: str = "gemini", with_frontend: bool = False) -> DevTeamResult:
    """
    Full build with all team members.
    
    Args:
        description: What to build
        provider: LLM provider
        with_frontend: Include frontend development
    
    Returns:
        DevTeamResult
    """
    config = DevTeamConfig(
        llm_provider=provider,
        team_type="full",
        include_frontend=with_frontend,
        verbose=True,
    )
    team = AIDevTeam(config)
    return team.run(description)
