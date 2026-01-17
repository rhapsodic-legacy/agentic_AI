"""
Autonomous Data Analyst - Main Orchestrator

Brings together all components:
- Database connectors
- Analysis tools
- Visualization
- AutoGen agents
- Report generation
"""

import os
from typing import Optional, Union
from dataclasses import dataclass, field
from pathlib import Path
import json

from .connectors import DataSource, ConnectionConfig
from .tools.analysis import (
    DataProfiler, 
    StatisticalAnalyzer, 
    TimeSeriesAnalyzer,
    InsightGenerator,
)
from .visualizations import ChartGenerator, ChartConfig
from .reports import ReportGenerator, Report


@dataclass
class AnalystConfig:
    """Configuration for the Autonomous Data Analyst."""
    
    # LLM Settings
    llm_provider: str = "gemini"  # "gemini", "anthropic", "openai"
    llm_model: Optional[str] = None
    temperature: float = 0.1
    
    # Output directories
    output_dir: str = "./output"
    charts_dir: str = "./output/charts"
    reports_dir: str = "./output/reports"
    
    # Analysis settings
    auto_profile: bool = True
    auto_insights: bool = True
    max_rows_display: int = 100
    
    # Agent settings
    human_input_mode: str = "NEVER"  # ALWAYS, TERMINATE, NEVER
    use_group_chat: bool = False


@dataclass
class AnalysisResult:
    """Result from an analysis session."""
    success: bool
    query: str
    
    # Results
    data: Optional[list] = None
    row_count: int = 0
    columns: list = field(default_factory=list)
    
    # Analysis
    profile: Optional[dict] = None
    insights: list = field(default_factory=list)
    statistics: dict = field(default_factory=dict)
    
    # Outputs
    charts: list = field(default_factory=list)
    report_path: Optional[str] = None
    
    # Conversation
    conversation: list = field(default_factory=list)
    
    # Errors
    error: Optional[str] = None


class AutonomousDataAnalyst:
    """
    Autonomous Data Analyst - Conversational AI for Data Analysis
    
    A multi-agent system that can:
    - Connect to various data sources (SQL databases, CSV, Excel, etc.)
    - Understand natural language queries
    - Execute SQL and Python code
    - Generate insights and visualizations
    - Create comprehensive reports
    
    Usage:
        analyst = AutonomousDataAnalyst()
        
        # Connect to data source
        analyst.connect_sqlite("path/to/database.db")
        # or
        analyst.connect_csv("path/to/data.csv")
        
        # Analyze with natural language
        result = analyst.analyze("What are the top 10 customers by revenue?")
        
        # Generate visualizations
        charts = analyst.visualize("revenue", "date")
        
        # Create report
        report = analyst.create_report("Q4 Analysis")
    """
    
    def __init__(self, config: Optional[AnalystConfig] = None):
        self.config = config or AnalystConfig()
        
        # Create output directories
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.charts_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.reports_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.data_source: Optional[DataSource] = None
        self.current_df = None
        
        # Analysis tools
        self.profiler = DataProfiler()
        self.stats = StatisticalAnalyzer()
        self.timeseries = TimeSeriesAnalyzer()
        self.insight_generator = InsightGenerator()
        
        # Visualization
        self.chart_gen = ChartGenerator(output_dir=self.config.charts_dir)
        
        # Reports
        self.report_gen = ReportGenerator(output_dir=self.config.reports_dir)
        
        # History
        self.analysis_history = []
        self.conversation_history = []
    
    # =========================================================================
    # Data Source Connection
    # =========================================================================
    
    def connect_sqlite(self, database: str) -> bool:
        """Connect to a SQLite database."""
        self.data_source = DataSource.from_sqlite(database)
        return self.data_source.connect()
    
    def connect_csv(self, filepath: str) -> bool:
        """Connect to a CSV file or directory of CSV files."""
        self.data_source = DataSource.from_csv(filepath)
        return self.data_source.connect()
    
    def connect_excel(self, filepath: str) -> bool:
        """Connect to an Excel file."""
        self.data_source = DataSource.from_excel(filepath)
        return self.data_source.connect()
    
    def connect_parquet(self, filepath: str) -> bool:
        """Connect to a Parquet file or directory."""
        self.data_source = DataSource.from_parquet(filepath)
        return self.data_source.connect()
    
    def connect_postgresql(
        self,
        host: str = "localhost",
        port: int = 5432,
        database: str = "",
        username: str = "",
        password: str = "",
    ) -> bool:
        """Connect to a PostgreSQL database."""
        self.data_source = DataSource.from_postgresql(
            host=host,
            port=port,
            database=database,
            username=username,
            password=password,
        )
        return self.data_source.connect()
    
    def load_dataframe(self, df, name: str = "data"):
        """Load a pandas DataFrame directly."""
        import pandas as pd
        from .connectors import CSVConnector, ConnectionConfig
        
        self.current_df = df
        
        # Create a connector that wraps the DataFrame
        config = ConnectionConfig(type="csv", filepath="")
        self.data_source = DataSource(config)
        self.data_source.connector = CSVConnector(config)
        self.data_source.connector.dataframes = {name: df}
        self.data_source._connected = True
        
        # Set up DuckDB for SQL queries
        try:
            import duckdb
            self.data_source.connector.duckdb_conn = duckdb.connect(":memory:")
            self.data_source.connector.duckdb_conn.register(name, df)
        except ImportError:
            pass
    
    def disconnect(self):
        """Disconnect from data source."""
        if self.data_source:
            self.data_source.disconnect()
    
    # =========================================================================
    # Data Exploration
    # =========================================================================
    
    def get_tables(self) -> list[str]:
        """Get list of available tables."""
        if not self.data_source or not self.data_source.is_connected:
            return []
        return self.data_source.get_tables()
    
    def describe(self) -> str:
        """Get description of the data source."""
        if not self.data_source or not self.data_source.is_connected:
            return "No data source connected."
        return self.data_source.describe()
    
    def get_schema(self, table: str) -> dict:
        """Get schema for a table."""
        if not self.data_source or not self.data_source.is_connected:
            return {}
        schema = self.data_source.get_schema(table)
        return {
            "name": schema.name,
            "columns": schema.columns,
            "row_count": schema.row_count,
        }
    
    def preview(self, table: str, limit: int = 10):
        """Preview data from a table."""
        import pandas as pd
        
        result = self.query(f"SELECT * FROM {table} LIMIT {limit}")
        if result.success and result.data:
            return pd.DataFrame(result.data)
        return None
    
    # =========================================================================
    # Query Execution
    # =========================================================================
    
    def query(self, sql: str) -> 'AnalysisResult':
        """Execute a SQL query."""
        if not self.data_source or not self.data_source.is_connected:
            return AnalysisResult(
                success=False,
                query=sql,
                error="No data source connected.",
            )
        
        result = self.data_source.query(sql)
        
        analysis_result = AnalysisResult(
            success=result.success,
            query=sql,
            data=result.data,
            row_count=result.row_count,
            columns=result.columns or [],
            error=result.error,
        )
        
        # Auto-generate insights if enabled
        if result.success and result.data and self.config.auto_insights:
            import pandas as pd
            df = pd.DataFrame(result.data)
            self.current_df = df
            
            insights = self.insight_generator.generate_insights(df)
            analysis_result.insights = [
                {"title": i.title, "description": i.description, "importance": i.importance}
                for i in insights
            ]
        
        self.analysis_history.append(analysis_result)
        return analysis_result
    
    # =========================================================================
    # Analysis
    # =========================================================================
    
    def profile(self, table_or_df=None) -> str:
        """Profile a table or DataFrame."""
        import pandas as pd
        
        if table_or_df is None:
            df = self.current_df
            table_name = "current_data"
        elif isinstance(table_or_df, str):
            result = self.query(f"SELECT * FROM {table_or_df}")
            if not result.success:
                return f"Error: {result.error}"
            df = pd.DataFrame(result.data)
            table_name = table_or_df
        else:
            df = table_or_df
            table_name = "data"
        
        if df is None:
            return "No data to profile."
        
        profile = self.profiler.profile_dataframe(df, table_name)
        return self.profiler.profile_to_text(profile)
    
    def get_insights(self, table_or_df=None) -> list:
        """Generate insights for a table or DataFrame."""
        import pandas as pd
        
        if table_or_df is None:
            df = self.current_df
        elif isinstance(table_or_df, str):
            result = self.query(f"SELECT * FROM {table_or_df}")
            if not result.success:
                return []
            df = pd.DataFrame(result.data)
        else:
            df = table_or_df
        
        if df is None:
            return []
        
        return self.insight_generator.generate_insights(df)
    
    def correlations(self, table_or_df=None) -> dict:
        """Compute correlations."""
        import pandas as pd
        
        df = self._get_df(table_or_df)
        if df is None:
            return {"error": "No data"}
        
        return self.stats.compute_correlations(df)
    
    def detect_outliers(self, column: str, table_or_df=None, method: str = "iqr") -> dict:
        """Detect outliers in a column."""
        df = self._get_df(table_or_df)
        if df is None:
            return {"error": "No data"}
        
        return self.stats.detect_outliers(df, column, method)
    
    def time_series_analysis(
        self, 
        date_column: str, 
        value_column: str, 
        table_or_df=None
    ) -> dict:
        """Analyze a time series."""
        df = self._get_df(table_or_df)
        if df is None:
            return {"error": "No data"}
        
        return self.timeseries.analyze_time_series(df, date_column, value_column)
    
    def _get_df(self, table_or_df):
        """Helper to get DataFrame from various inputs."""
        import pandas as pd
        
        if table_or_df is None:
            return self.current_df
        elif isinstance(table_or_df, str):
            result = self.query(f"SELECT * FROM {table_or_df}")
            if result.success:
                return pd.DataFrame(result.data)
            return None
        else:
            return table_or_df
    
    # =========================================================================
    # Visualization
    # =========================================================================
    
    def visualize(
        self,
        chart_type: str,
        x: Optional[str] = None,
        y: Optional[str] = None,
        table_or_df=None,
        **kwargs
    ) -> dict:
        """Create a visualization."""
        df = self._get_df(table_or_df)
        if df is None:
            return {"error": "No data"}
        
        chart_methods = {
            "line": self.chart_gen.line_chart,
            "bar": self.chart_gen.bar_chart,
            "histogram": self.chart_gen.histogram,
            "scatter": self.chart_gen.scatter_plot,
            "pie": self.chart_gen.pie_chart,
            "box": self.chart_gen.box_plot,
            "heatmap": self.chart_gen.heatmap,
            "timeseries": self.chart_gen.time_series,
        }
        
        method = chart_methods.get(chart_type.lower())
        if not method:
            return {"error": f"Unknown chart type: {chart_type}"}
        
        if chart_type in ["heatmap"]:
            return method(df, **kwargs)
        elif chart_type in ["histogram", "box"]:
            return method(df, x or y, **kwargs)
        else:
            return method(df, x, y, **kwargs)
    
    def auto_visualize(self, table_or_df=None, max_charts: int = 5) -> list[dict]:
        """Automatically generate appropriate visualizations."""
        df = self._get_df(table_or_df)
        if df is None:
            return []
        
        return self.chart_gen.auto_visualize(df, max_charts)
    
    # =========================================================================
    # Reporting
    # =========================================================================
    
    def create_report(
        self,
        title: str,
        table_or_df=None,
        include_profile: bool = True,
        include_insights: bool = True,
        include_charts: bool = True,
        format: str = "html",
    ) -> str:
        """Create a comprehensive analysis report."""
        df = self._get_df(table_or_df)
        
        # Gather data
        data_summary = self.profile(df) if include_profile else ""
        insights = self.get_insights(df) if include_insights else []
        charts = self.auto_visualize(df) if include_charts else []
        
        # Generate recommendations based on insights
        recommendations = []
        for insight in insights[:5]:
            if insight.importance == "high":
                recommendations.append(f"Investigate: {insight.title}")
        
        # Create report
        report = self.report_gen.create_analysis_report(
            title=title,
            data_summary=data_summary,
            insights=insights,
            charts=charts,
            recommendations=recommendations,
        )
        
        # Save
        filename = title.lower().replace(" ", "_")
        return self.report_gen.save(report, filename, format=format)
    
    # =========================================================================
    # Conversational Interface
    # =========================================================================
    
    def analyze(self, question: str) -> AnalysisResult:
        """
        Analyze data based on a natural language question.
        
        This method uses LLM to:
        1. Understand the question
        2. Generate appropriate SQL/Python code
        3. Execute the analysis
        4. Generate insights
        
        Args:
            question: Natural language question about the data
        
        Returns:
            AnalysisResult with data, insights, and visualizations
        """
        # Get schema context
        schema_context = self.describe()
        
        # Use LLM to generate SQL
        sql = self._generate_sql(question, schema_context)
        
        # Execute query
        result = self.query(sql)
        
        # Add to conversation history
        self.conversation_history.append({
            "role": "user",
            "content": question,
        })
        self.conversation_history.append({
            "role": "assistant",
            "content": f"Executed: {sql}\nRows: {result.row_count}",
        })
        
        return result
    
    def _generate_sql(self, question: str, schema_context: str) -> str:
        """Generate SQL from natural language using LLM."""
        
        # Try to use LLM
        try:
            if self.config.llm_provider == "gemini":
                from langchain_google_genai import ChatGoogleGenerativeAI
                llm = ChatGoogleGenerativeAI(
                    model=self.config.llm_model or "gemini-1.5-flash",
                    temperature=0.1,
                )
            elif self.config.llm_provider == "anthropic":
                from langchain_anthropic import ChatAnthropic
                llm = ChatAnthropic(
                    model=self.config.llm_model or "claude-sonnet-4-20250514",
                    temperature=0.1,
                )
            elif self.config.llm_provider == "openai":
                from langchain_openai import ChatOpenAI
                llm = ChatOpenAI(
                    model=self.config.llm_model or "gpt-4o-mini",
                    temperature=0.1,
                )
            else:
                raise ValueError(f"Unknown provider: {self.config.llm_provider}")
            
            prompt = f"""You are a SQL expert. Generate a SQL query to answer this question.

Schema:
{schema_context}

Question: {question}

Return ONLY the SQL query, nothing else. Do not include markdown code blocks."""

            response = llm.invoke(prompt)
            sql = response.content.strip()
            
            # Clean up response
            if sql.startswith("```"):
                sql = sql.split("\n", 1)[1].rsplit("```", 1)[0]
            
            return sql
            
        except Exception as e:
            # Fallback: try to extract a simple query from the question
            tables = self.get_tables()
            if tables:
                return f"SELECT * FROM {tables[0]} LIMIT 100"
            return "SELECT 'No tables found' as error"
    
    def chat(self, message: str) -> str:
        """
        Chat interface for conversational analysis.
        
        Args:
            message: User message
        
        Returns:
            Assistant response
        """
        # Add to history
        self.conversation_history.append({
            "role": "user",
            "content": message,
        })
        
        # Analyze
        result = self.analyze(message)
        
        # Format response
        if result.success:
            response = f"I found {result.row_count} results.\n\n"
            
            if result.insights:
                response += "Key insights:\n"
                for insight in result.insights[:3]:
                    response += f"• {insight['title']}: {insight['description']}\n"
            
            if result.data and len(result.data) <= 10:
                import pandas as pd
                df = pd.DataFrame(result.data)
                response += f"\nData:\n{df.to_string()}"
        else:
            response = f"Error: {result.error}"
        
        # Add response to history
        self.conversation_history.append({
            "role": "assistant",
            "content": response,
        })
        
        return response
    
    # =========================================================================
    # Memory
    # =========================================================================
    
    def get_history(self) -> list:
        """Get conversation history."""
        return self.conversation_history
    
    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []
        self.analysis_history = []
    
    def save_session(self, filepath: str):
        """Save session state to file."""
        state = {
            "conversation_history": self.conversation_history,
            "analysis_count": len(self.analysis_history),
        }
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_session(self, filepath: str):
        """Load session state from file."""
        with open(filepath, 'r') as f:
            state = json.load(f)
        self.conversation_history = state.get("conversation_history", [])


# =============================================================================
# Convenience Functions
# =============================================================================

def quick_analyze(filepath: str, question: str) -> AnalysisResult:
    """
    Quick analysis of a file.
    
    Args:
        filepath: Path to data file (CSV, Excel, etc.)
        question: Natural language question
    
    Returns:
        AnalysisResult
    """
    analyst = AutonomousDataAnalyst()
    
    if filepath.endswith('.csv'):
        analyst.connect_csv(filepath)
    elif filepath.endswith(('.xlsx', '.xls')):
        analyst.connect_excel(filepath)
    elif filepath.endswith('.parquet'):
        analyst.connect_parquet(filepath)
    elif filepath.endswith(('.db', '.sqlite', '.sqlite3')):
        analyst.connect_sqlite(filepath)
    else:
        raise ValueError(f"Unsupported file type: {filepath}")
    
    return analyst.analyze(question)


def analyze_dataframe(df, question: str) -> AnalysisResult:
    """
    Quick analysis of a DataFrame.
    
    Args:
        df: pandas DataFrame
        question: Natural language question
    
    Returns:
        AnalysisResult
    """
    analyst = AutonomousDataAnalyst()
    analyst.load_dataframe(df)
    return analyst.analyze(question)
