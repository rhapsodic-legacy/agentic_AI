# 📊 Autonomous Data Analyst

A conversational AI data analyst using **AutoGen** that can connect to databases, explore data, generate insights, create visualizations, and produce reports—all through natural language.

![AutoGen](https://img.shields.io/badge/Framework-AutoGen-blue)
![Multi-LLM](https://img.shields.io/badge/LLM-Gemini%20|%20Claude%20|%20GPT-green)
![Architecture](https://img.shields.io/badge/Architecture-Conversational-purple)

## ✨ Features

| Feature | Description |
|---------|-------------|
| 💬 **Natural Language Queries** | Ask questions in plain English |
| 🔌 **Multiple Data Sources** | SQLite, PostgreSQL, CSV, Excel, Parquet |
| 📊 **Auto Data Profiling** | Automatic data quality and statistics |
| 🔍 **Insight Generation** | Find patterns, outliers, correlations |
| 📈 **Visualizations** | Auto-generate charts with Plotly |
| 📝 **Report Generation** | Create HTML/Markdown reports |
| 🧠 **Context Memory** | Remembers conversation history |
| 🤖 **Multi-Agent System** | Specialized agents for SQL, viz, analysis |

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      USER PROXY                             │
│              (Human-in-the-loop interface)                  │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    DATA ANALYST                             │
│         (Orchestrates analysis, writes queries)             │
└─────────────────────────┬───────────────────────────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        ▼                 ▼                 ▼
┌───────────────┐ ┌───────────────┐ ┌───────────────┐
│   SQL AGENT   │ │  PYTHON EXEC  │ │   VIZ AGENT   │
│ (Query data)  │ │  (Analysis)   │ │  (Charts)     │
└───────────────┘ └───────────────┘ └───────────────┘
        │                 │                 │
        └─────────────────┼─────────────────┘
                          ▼
                   ┌─────────────┐
                   │  REPORTER   │
                   │(Narratives) │
                   └─────────────┘
```

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt

# Set API key for your provider:
export GOOGLE_API_KEY="your-key"     # For Gemini (recommended)
export ANTHROPIC_API_KEY="your-key"  # For Claude
export OPENAI_API_KEY="your-key"     # For OpenAI
```

### Generate Sample Data

```bash
cd sample_data
python generate.py
```

### CLI Usage

```bash
# Connect and explore
python main.py connect sample_data/sales.csv

# Analyze with natural language
python main.py analyze sample_data/sales.csv "What are the top products by revenue?"

# Interactive mode
python main.py interactive

# Start web server
python main.py serve
```

### Python API

```python
from data_analyst import AutonomousDataAnalyst, AnalystConfig

# Create analyst
analyst = AutonomousDataAnalyst()

# Connect to data
analyst.connect_csv("sales.csv")
# or
analyst.connect_sqlite("database.db")
# or
analyst.connect_postgresql(host="localhost", database="mydb")

# Natural language analysis
result = analyst.analyze("What are the top 10 customers by revenue?")
print(result.data)
print(result.insights)

# Chat interface
response = analyst.chat("Tell me about Q4 performance")
print(response)

# Direct SQL
result = analyst.query("SELECT category, SUM(revenue) FROM sales GROUP BY category")

# Profile data
print(analyst.profile("sales"))

# Generate visualizations
chart = analyst.visualize("bar", x="category", y="revenue")

# Create report
report_path = analyst.create_report("Q4 Analysis Report")
```

### Quick Functions

```python
from data_analyst import quick_analyze, analyze_dataframe

# Analyze any file
result = quick_analyze("data.csv", "What's the average order value?")

# Analyze a DataFrame
import pandas as pd
df = pd.read_csv("data.csv")
result = analyze_dataframe(df, "Show me sales trends")
```

## 📁 Project Structure

```
data-analyst/
├── data_analyst/
│   ├── __init__.py           # Package exports
│   ├── analyst.py            # Main AutonomousDataAnalyst class
│   ├── agents/
│   │   └── __init__.py       # AutoGen agent definitions
│   ├── connectors/
│   │   └── __init__.py       # Database connectors
│   ├── tools/
│   │   ├── __init__.py
│   │   └── analysis.py       # Profiling, statistics, insights
│   ├── visualizations/
│   │   └── __init__.py       # Chart generation
│   ├── reports/
│   │   └── __init__.py       # Report generation
│   └── memory/
│       └── __init__.py       # Conversation memory
├── api.py                     # FastAPI backend
├── frontend/
│   └── index.html            # React web UI
├── main.py                    # CLI
├── sample_data/
│   └── generate.py           # Sample data generator
├── requirements.txt
└── README.md
```

## 🔌 Supported Data Sources

| Source | Connection |
|--------|------------|
| **CSV** | `analyst.connect_csv("data.csv")` |
| **Excel** | `analyst.connect_excel("data.xlsx")` |
| **SQLite** | `analyst.connect_sqlite("db.sqlite")` |
| **Parquet** | `analyst.connect_parquet("data.parquet")` |
| **PostgreSQL** | `analyst.connect_postgresql(host, database, ...)` |
| **DataFrame** | `analyst.load_dataframe(df)` |

## 📊 Analysis Capabilities

### Data Profiling
```python
profile = analyst.profile("sales")
# Returns:
# - Row count, column count
# - Data types
# - Null percentages
# - Unique values
# - Statistical summaries
# - Quality score
```

### Statistical Analysis
```python
# Correlations
correlations = analyst.correlations()

# Outlier detection
outliers = analyst.detect_outliers("revenue", method="iqr")

# Time series analysis
ts = analyst.time_series_analysis("date", "revenue")
```

### Insight Generation
```python
insights = analyst.get_insights()
# Automatically finds:
# - Trends
# - Anomalies
# - Correlations
# - Distribution patterns
# - Data quality issues
```

## 📈 Visualization

```python
# Line chart
analyst.visualize("line", x="date", y="revenue")

# Bar chart
analyst.visualize("bar", x="category", y="sales")

# Scatter plot
analyst.visualize("scatter", x="price", y="quantity")

# Histogram
analyst.visualize("histogram", x="revenue")

# Box plot
analyst.visualize("box", y="revenue", x="category")

# Correlation heatmap
analyst.visualize("heatmap")

# Time series with trend
analyst.visualize("timeseries", x="date", y="revenue")

# Auto-generate best visualizations
charts = analyst.auto_visualize(max_charts=5)
```

## 📝 Report Generation

```python
# Generate full report
report_path = analyst.create_report(
    title="Q4 Sales Analysis",
    include_profile=True,
    include_insights=True,
    include_charts=True,
    format="html"  # or "markdown"
)

# Report includes:
# - Executive summary
# - Data overview
# - Key insights
# - Visualizations
# - Recommendations
```

## 💬 Sample Conversation

```
User: "Connect to our sales database and tell me about Q4 performance"

Analyst: "I'll connect to the database and explore the sales data..."
         [Executes: SHOW TABLES, describes schema]
         
         "I found the sales_transactions table. Let me analyze Q4..."
         [Executes: SELECT queries for Q4 data]
         
         "Here's what I found:
          
          📈 Revenue: $2.4M (+15% vs Q3)
          📦 Orders: 12,847 (+8% vs Q3)
          👥 New Customers: 1,203
          
          Top insights:
          1. Electronics category drove 40% of growth
          2. Weekend sales increased by 25%
          3. Average order value up $12
          
          Would you like me to:
          - Create visualizations?
          - Deep dive into any category?
          - Compare to Q4 last year?"

User: "Create visualizations for revenue by category"

Analyst: "Creating a bar chart of revenue by category..."
         [Generates Plotly chart]
         
         "Here's the visualization. Electronics leads at $960K,
          followed by Furniture at $720K..."
```

## 🔧 Configuration

```python
from data_analyst import AutonomousDataAnalyst, AnalystConfig

config = AnalystConfig(
    # LLM Settings
    llm_provider="gemini",      # "gemini", "anthropic", "openai"
    llm_model=None,             # Uses provider default
    temperature=0.1,
    
    # Output directories
    output_dir="./output",
    charts_dir="./output/charts",
    reports_dir="./output/reports",
    
    # Analysis settings
    auto_profile=True,          # Auto-profile on connect
    auto_insights=True,         # Auto-generate insights
    max_rows_display=100,       # Max rows in output
    
    # Agent settings
    human_input_mode="NEVER",   # ALWAYS, TERMINATE, NEVER
    use_group_chat=False,       # Use multi-agent group chat
)

analyst = AutonomousDataAnalyst(config)
```

## 🧠 Memory & Context

The analyst maintains conversation context:

```python
# History is preserved across queries
analyst.chat("What's the total revenue?")
analyst.chat("Break that down by month")  # Remembers context
analyst.chat("Now show me the trend")     # Still in context

# Save/load session
analyst.save_session("session.json")
analyst.load_session("session.json")

# Clear history
analyst.clear_history()
```

## 🌐 Web Interface

```bash
python main.py serve --port 8000
```

Then open http://localhost:8000

Features:
- Drag & drop file upload
- Chat interface
- Data preview
- Interactive charts
- Report download

## 📚 API Reference

### AutonomousDataAnalyst

| Method | Description |
|--------|-------------|
| `connect_csv(path)` | Connect to CSV file |
| `connect_sqlite(path)` | Connect to SQLite database |
| `connect_postgresql(...)` | Connect to PostgreSQL |
| `get_tables()` | List available tables |
| `describe()` | Describe data source |
| `query(sql)` | Execute SQL query |
| `analyze(question)` | Natural language analysis |
| `chat(message)` | Chat interface |
| `profile(table)` | Profile a table |
| `get_insights()` | Generate insights |
| `visualize(type, ...)` | Create visualization |
| `create_report(title)` | Generate report |

### AnalysisResult

| Field | Type | Description |
|-------|------|-------------|
| `success` | bool | Query success |
| `data` | list | Result data |
| `columns` | list | Column names |
| `row_count` | int | Number of rows |
| `insights` | list | Auto-generated insights |
| `error` | str | Error message if failed |

## 📈 Tips for Best Results

1. **Be specific** - "Show Q4 revenue by category" works better than "show me data"
2. **Start broad** - Ask for overview first, then drill down
3. **Use follow-ups** - The analyst remembers context
4. **Check profile first** - Understand your data structure
5. **Request visualizations** - Charts help understand patterns

## 📝 License

MIT License
