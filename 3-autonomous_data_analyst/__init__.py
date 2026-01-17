"""
Autonomous Data Analyst - AutoGen Agents

Multi-agent system using AutoGen for conversational data analysis:
- UserProxy: Human-in-the-loop interface
- DataAnalyst: Main orchestrator
- SQLAgent: Database queries
- PythonExecutor: Code execution
- VizAgent: Visualization generation
- Reporter: Report generation
"""

from typing import Optional, Callable, Any
import os
import json
from dataclasses import dataclass

# Try to import autogen, fall back to pyautogen
try:
    import autogen
    from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
except ImportError:
    try:
        import pyautogen as autogen
        from pyautogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
    except ImportError:
        raise ImportError("Please install autogen: pip install pyautogen")


@dataclass
class AgentConfig:
    """Configuration for the agent system."""
    llm_provider: str = "gemini"
    llm_model: Optional[str] = None
    temperature: float = 0.1
    human_input_mode: str = "NEVER"  # ALWAYS, TERMINATE, NEVER
    max_consecutive_auto_reply: int = 10
    code_execution_enabled: bool = True


def get_llm_config(provider: str, model: Optional[str] = None, temperature: float = 0.1) -> dict:
    """Get LLM configuration for AutoGen."""
    
    if provider == "gemini":
        return {
            "config_list": [{
                "model": model or "gemini-1.5-flash",
                "api_key": os.environ.get("GOOGLE_API_KEY"),
                "api_type": "google",
            }],
            "temperature": temperature,
        }
    
    elif provider == "anthropic":
        return {
            "config_list": [{
                "model": model or "claude-sonnet-4-20250514",
                "api_key": os.environ.get("ANTHROPIC_API_KEY"),
                "api_type": "anthropic",
            }],
            "temperature": temperature,
        }
    
    elif provider == "openai":
        return {
            "config_list": [{
                "model": model or "gpt-4o-mini",
                "api_key": os.environ.get("OPENAI_API_KEY"),
            }],
            "temperature": temperature,
        }
    
    else:
        raise ValueError(f"Unknown provider: {provider}")


class DataAnalystAgents:
    """
    Multi-agent system for data analysis.
    
    Architecture:
        UserProxy (human interface)
            ↓
        DataAnalyst (orchestrator)
            ↓
        ┌───────────────────────┐
        │   SQLAgent           │ → Execute SQL queries
        │   PythonExecutor     │ → Run analysis code
        │   VizAgent           │ → Create visualizations
        │   Reporter           │ → Generate reports
        └───────────────────────┘
    """
    
    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig()
        self.llm_config = get_llm_config(
            self.config.llm_provider,
            self.config.llm_model,
            self.config.temperature,
        )
        
        # Data context
        self.data_source = None
        self.current_df = None
        self.analysis_history = []
        
        # Create agents
        self._create_agents()
    
    def _create_agents(self):
        """Create all agents."""
        
        # User Proxy - Interface for human interaction
        self.user_proxy = UserProxyAgent(
            name="User",
            human_input_mode=self.config.human_input_mode,
            max_consecutive_auto_reply=self.config.max_consecutive_auto_reply,
            code_execution_config={
                "work_dir": "workspace",
                "use_docker": False,
            } if self.config.code_execution_enabled else False,
            system_message="You are a user interacting with a data analysis system. Provide clear requests and feedback.",
        )
        
        # Data Analyst - Main orchestrator
        self.analyst = AssistantAgent(
            name="DataAnalyst",
            llm_config=self.llm_config,
            system_message="""You are an expert Data Analyst. Your role is to:

1. Understand user requests about their data
2. Explore and profile datasets
3. Write SQL queries to extract relevant data
4. Perform statistical analysis
5. Generate insights and visualizations
6. Create comprehensive reports

When analyzing data:
- Always start by understanding the data structure (tables, columns, types)
- Use SQL for data extraction and aggregation
- Use Python for complex analysis and visualization
- Provide clear explanations of your findings
- Suggest follow-up analyses

Available tools:
- SQL queries for data extraction
- Python for analysis (pandas, numpy, scipy)
- Plotly/Matplotlib for visualization

When you need to execute code, write it in a code block.
When you're done with analysis, summarize key findings.""",
        )
        
        # SQL Agent - Database queries
        self.sql_agent = AssistantAgent(
            name="SQLAgent",
            llm_config=self.llm_config,
            system_message="""You are an expert SQL developer. Your role is to:

1. Write efficient SQL queries based on natural language requests
2. Explain query logic
3. Optimize queries for performance
4. Handle different SQL dialects (SQLite, PostgreSQL, MySQL)

Guidelines:
- Always use proper SQL formatting
- Add comments explaining complex logic
- Use CTEs for complex queries
- Consider NULL handling
- Use appropriate aggregations and window functions

When writing queries:
```sql
-- Your SQL query here
SELECT ...
FROM ...
WHERE ...
```

After writing a query, explain what it does.""",
        )
        
        # Python Executor - Analysis code
        self.python_executor = AssistantAgent(
            name="PythonExecutor",
            llm_config=self.llm_config,
            system_message="""You are a Python data analysis expert. Your role is to:

1. Write Python code for data analysis
2. Perform statistical tests
3. Create data transformations
4. Build models when needed

Available libraries:
- pandas for data manipulation
- numpy for numerical operations
- scipy for statistical tests
- sklearn for machine learning

Guidelines:
- Write clean, documented code
- Handle errors gracefully
- Print results clearly
- Use vectorized operations for efficiency

When writing code:
```python
import pandas as pd
import numpy as np
# Your code here
```""",
        )
        
        # Visualization Agent
        self.viz_agent = AssistantAgent(
            name="VizAgent",
            llm_config=self.llm_config,
            system_message="""You are a data visualization expert. Your role is to:

1. Choose appropriate chart types for data
2. Write code to create visualizations
3. Ensure charts are clear and informative
4. Follow visualization best practices

Available libraries:
- plotly for interactive charts
- matplotlib for static charts

Chart selection guide:
- Time series → Line chart
- Comparison → Bar chart
- Distribution → Histogram, Box plot
- Relationship → Scatter plot
- Composition → Pie chart, Stacked bar
- Correlation → Heatmap

Guidelines:
- Use clear titles and labels
- Choose appropriate colors
- Don't overcrowd charts
- Add annotations for key points

When creating visualizations:
```python
import plotly.express as px
# Your visualization code
fig.show()
```""",
        )
        
        # Reporter - Report generation
        self.reporter = AssistantAgent(
            name="Reporter",
            llm_config=self.llm_config,
            system_message="""You are a report writer. Your role is to:

1. Compile analysis findings into clear reports
2. Write executive summaries
3. Create narrative explanations of data
4. Format reports professionally

Report structure:
1. Executive Summary (key findings, 2-3 sentences)
2. Overview (context and scope)
3. Key Findings (with supporting data)
4. Visualizations (reference charts)
5. Recommendations (actionable insights)
6. Appendix (methodology, data sources)

Guidelines:
- Use clear, non-technical language when possible
- Support claims with data
- Highlight actionable insights
- Use bullet points for clarity
- Include relevant metrics""",
        )
    
    def create_group_chat(self) -> GroupChat:
        """Create a group chat with all agents."""
        return GroupChat(
            agents=[
                self.user_proxy,
                self.analyst,
                self.sql_agent,
                self.python_executor,
                self.viz_agent,
                self.reporter,
            ],
            messages=[],
            max_round=20,
        )
    
    def set_data_source(self, data_source):
        """Set the data source for analysis."""
        self.data_source = data_source
        
        # Update analyst with schema information
        if data_source.is_connected:
            schema_info = data_source.describe()
            self.analyst.update_system_message(
                self.analyst.system_message + f"\n\nCurrent Data Source:\n{schema_info}"
            )
    
    def set_dataframe(self, df, name: str = "data"):
        """Set a DataFrame for analysis."""
        import pandas as pd
        self.current_df = df
        
        # Create description
        desc = f"\n\nCurrent DataFrame '{name}':\n"
        desc += f"Shape: {df.shape[0]} rows × {df.shape[1]} columns\n"
        desc += f"Columns: {list(df.columns)}\n"
        desc += f"Types:\n{df.dtypes.to_string()}\n"
        
        self.analyst.update_system_message(
            self.analyst.system_message + desc
        )
    
    def analyze(self, query: str) -> str:
        """
        Start an analysis based on user query.
        
        Uses a simple two-agent chat for basic queries,
        or group chat for complex analyses.
        """
        # For simple queries, use direct chat
        self.user_proxy.initiate_chat(
            self.analyst,
            message=query,
        )
        
        # Get the conversation history
        messages = self.user_proxy.chat_messages.get(self.analyst, [])
        
        # Extract the final response
        if messages:
            last_message = messages[-1]
            return last_message.get("content", "Analysis complete.")
        
        return "Analysis complete."
    
    def analyze_with_group(self, query: str) -> str:
        """
        Perform analysis using group chat with all agents.
        """
        group_chat = self.create_group_chat()
        manager = GroupChatManager(
            groupchat=group_chat,
            llm_config=self.llm_config,
        )
        
        self.user_proxy.initiate_chat(
            manager,
            message=query,
        )
        
        # Compile all messages
        all_messages = group_chat.messages
        
        return "\n\n".join([
            f"**{m.get('name', 'Agent')}**: {m.get('content', '')}"
            for m in all_messages
        ])


# =============================================================================
# Function Tools for Agents
# =============================================================================

def create_analysis_functions(data_source, chart_generator):
    """Create functions that agents can use for analysis."""
    
    def execute_sql(query: str) -> str:
        """Execute a SQL query and return results."""
        result = data_source.query(query)
        if result.success:
            if result.data:
                # Format as table
                import pandas as pd
                df = pd.DataFrame(result.data)
                return f"Query returned {result.row_count} rows:\n{df.to_string()}"
            return f"Query executed. Rows affected: {result.row_count}"
        return f"Error: {result.error}"
    
    def get_schema(table: str) -> str:
        """Get schema for a table."""
        schema = data_source.get_schema(table)
        result = f"Table: {schema.name}\n"
        result += f"Rows: {schema.row_count}\n"
        result += "Columns:\n"
        for col in schema.columns:
            result += f"  - {col['name']}: {col['type']}\n"
        return result
    
    def list_tables() -> str:
        """List all tables in the database."""
        tables = data_source.get_tables()
        return f"Tables: {', '.join(tables)}"
    
    def create_chart(chart_type: str, x: str, y: str, title: str = "") -> str:
        """Create a chart."""
        # This would need the current DataFrame
        return f"Chart created: {chart_type} of {y} vs {x}"
    
    def profile_data(table: str) -> str:
        """Profile a table's data."""
        from .tools.analysis import DataProfiler
        
        result = data_source.query(f"SELECT * FROM {table} LIMIT 1000")
        if result.success:
            import pandas as pd
            df = pd.DataFrame(result.data)
            profiler = DataProfiler()
            profile = profiler.profile_dataframe(df, table)
            return profiler.profile_to_text(profile)
        return f"Error profiling: {result.error}"
    
    return {
        "execute_sql": execute_sql,
        "get_schema": get_schema,
        "list_tables": list_tables,
        "create_chart": create_chart,
        "profile_data": profile_data,
    }
