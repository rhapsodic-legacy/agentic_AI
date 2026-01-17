# 📚 Research Paper Synthesizer

An agentic system using **LangGraph** that researches topics, finds relevant papers, synthesizes findings, and produces literature reviews or research proposals.

![LangGraph](https://img.shields.io/badge/Framework-LangGraph-blue)
![Multi-Source](https://img.shields.io/badge/Sources-Arxiv%20|%20S2%20|%20PubMed-green)
![Architecture](https://img.shields.io/badge/Architecture-Cyclic%20Graph-purple)

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🔄 **Cyclic Refinement** | Iteratively searches until sufficient coverage |
| 📚 **Multi-Source Search** | Arxiv, Semantic Scholar, PubMed |
| 🔬 **Theme Extraction** | Automatically groups findings into themes |
| ⚖️ **Contradiction Detection** | Identifies where papers disagree |
| 🔍 **Gap Analysis** | Highlights missing research areas |
| 📝 **Multiple Outputs** | Literature review, research proposal, blog post, etc. |
| 📖 **Auto Citations** | APA and BibTeX citation generation |

## 🏗️ Architecture

```
                    ┌─────────────┐
                    │   START     │
                    └──────┬──────┘
                           │
                           ▼
                    ┌─────────────┐
            ┌───────│   PLANNER   │◄───────┐
            │       │  (Queries)  │        │
            │       └──────┬──────┘        │
            │              │               │
            │              ▼               │
            │       ┌─────────────┐        │
            │       │  SEARCHER   │        │
            │       │(Arxiv, S2)  │        │
            │       └──────┬──────┘        │
            │              │               │
            │              ▼               │
            │       ┌─────────────┐        │
            │       │   READER    │        │
            │       │  (Analyze)  │        │
            │       └──────┬──────┘        │
            │              │               │
            │              ▼               │
            │       ┌─────────────┐        │
            │       │  EVALUATOR  │────────┘
            │       │(Sufficient?)│   NO
            │       └──────┬──────┘
            │              │
            │          YES │
            │              ▼
            │       ┌─────────────┐
            │       │ SYNTHESIZER │
            │       │  (Themes)   │
            │       └──────┬──────┘
            │              │
            │              ▼
            │       ┌─────────────┐
            └──────►│   WRITER    │
                    │  (Output)   │
                    └──────┬──────┘
                           │
                           ▼
                    ┌─────────────┐
                    │    END      │
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

### CLI Usage

```bash
# Research a topic
python main.py research "Large Language Models in Healthcare"

# Create a research proposal
python main.py research "AI Ethics" --format research_proposal

# Search papers only
python main.py search "transformer architecture" --source arxiv

# Interactive mode
python main.py interactive

# Web server
python main.py serve
```

### Python API

```python
from research_synth import ResearchSynthesizer, SynthesizerConfig

# Quick usage
from research_synth import create_literature_review
output = create_literature_review("Large Language Models")

# Full configuration
config = SynthesizerConfig(
    llm_provider="gemini",       # or "anthropic", "openai"
    max_iterations=3,            # Max refinement cycles
    enable_arxiv=True,
    enable_semantic_scholar=True,
    enable_pubmed=True,
)

synthesizer = ResearchSynthesizer(config)
result = synthesizer.run(
    topic="Transformer Architectures in NLP",
    output_format="literature_review"
)

# Access results
print(result.output)                    # The generated document
print(f"Papers: {result.papers_found}") # Number of papers found
print(f"Themes: {result.themes_identified}")  # Themes extracted

# Get papers data
for paper in result.papers:
    print(f"- {paper['title']} ({paper['year']})")

# Get citations
for cite in result.citations:
    print(cite)  # APA format
```

## 📁 Project Structure

```
research-synthesizer/
├── research_synth/
│   ├── __init__.py           # Package exports
│   ├── state.py              # State schema and data models
│   ├── graph.py              # LangGraph workflow
│   ├── nodes/
│   │   └── __init__.py       # Graph nodes (planner, searcher, etc.)
│   ├── sources/
│   │   └── __init__.py       # Paper sources (Arxiv, S2, PubMed)
│   └── tools/
├── api.py                     # FastAPI backend
├── frontend/
│   └── index.html            # React web UI
├── main.py                    # CLI
├── requirements.txt
└── README.md
```

## 🔀 Graph Nodes

| Node | Role | Output |
|------|------|--------|
| **Planner** | Generate search queries from topic | `search_queries` |
| **Searcher** | Search academic sources | `papers` |
| **Reader** | Analyze papers with LLM | `summaries`, `key_findings` |
| **Evaluator** | Check if coverage is sufficient | `is_sufficient`, `gaps` |
| **Synthesizer** | Group findings into themes | `themes`, `contradictions` |
| **Writer** | Generate final output | `final_output`, `citations` |

## 📚 Paper Sources

### Arxiv
- Computer Science, Physics, Math, and more
- Free access to millions of preprints
- Real-time API

### Semantic Scholar
- 200M+ papers across all fields
- Citation data and influence metrics
- Optional API key for higher limits

### PubMed
- Biomedical and life sciences
- 35M+ citations
- NCBI E-utilities API

## 📝 Output Formats

| Format | Description | Best For |
|--------|-------------|----------|
| `literature_review` | Academic review with themes and analysis | Research papers |
| `research_proposal` | Proposal with methodology and timeline | Grant applications |
| `blog_post` | Accessible summary for general audience | Science communication |
| `executive_summary` | Concise key points and recommendations | Decision makers |
| `annotated_bibliography` | Papers with annotations | Literature surveys |

## 📊 State Schema

```python
class ResearchState(TypedDict):
    # Input
    topic: str
    research_question: Optional[str]
    output_format: str
    
    # Search
    search_queries: list[str]
    papers: list[Paper]
    papers_analyzed: list[str]
    
    # Analysis
    summaries: dict[str, str]
    key_findings: list[Finding]
    themes: list[Theme]
    contradictions: list[Contradiction]
    gaps: list[Gap]
    
    # Output
    final_output: str
    citations: list[str]
    
    # Control flow
    iteration: int
    max_iterations: int
    is_sufficient: bool
```

## 🔧 Configuration

```python
from research_synth import SynthesizerConfig

config = SynthesizerConfig(
    # LLM Settings
    llm_provider="gemini",      # "gemini", "anthropic", "openai"
    llm_model=None,             # Uses provider default
    
    # Search Settings
    max_results_per_source=5,   # Papers per source per query
    enable_arxiv=True,
    enable_semantic_scholar=True,
    enable_pubmed=True,
    
    # Analysis Settings
    max_papers_per_iteration=5, # Papers to analyze per cycle
    min_papers=5,               # Minimum papers needed
    min_coverage=0.6,           # Coverage threshold
    max_iterations=3,           # Max refinement cycles
    
    # Output
    default_format="literature_review",
)
```

## 🎯 Example Output

```markdown
# Literature Review: Large Language Models in Healthcare

## Executive Summary
Large language models (LLMs) are transforming healthcare through clinical 
decision support, medical documentation, and patient communication...

## 1. Introduction
The application of artificial intelligence in healthcare has accelerated 
dramatically with the advent of large language models...

## 2. Key Themes

### 2.1 Clinical Decision Support
[Chen et al., 2023] demonstrated that LLMs can achieve diagnostic accuracy 
comparable to physicians in specific domains...

### 2.2 Medical Documentation
Automated clinical note generation has shown promise in reducing physician 
burnout [Smith et al., 2024]...

## 3. Contradictions and Debates
While some studies report high accuracy [Lee et al., 2023], others raise 
concerns about hallucination risks in medical contexts [Wang et al., 2024]...

## 4. Gaps in the Literature
- Limited research on long-term deployment effects
- Few studies in non-English healthcare settings
- Sparse evidence on patient outcomes

## 5. Future Directions
Future research should focus on developing robust evaluation frameworks 
and addressing bias in medical LLMs...

## References
[1] Chen, Y. et al. (2023). Diagnostic accuracy of GPT-4...
[2] Smith, J. et al. (2024). Automated clinical documentation...
```

## 🔒 Rate Limits

| Source | Rate Limit | Notes |
|--------|------------|-------|
| Arxiv | 1 req/sec | Be respectful |
| Semantic Scholar | 100/5min | API key recommended |
| PubMed | 3 req/sec | NCBI guidelines |

## 📈 Tips

1. **Be specific** - More specific topics yield better results
2. **Use research questions** - Helps focus the search
3. **Iterate** - Run multiple times with different queries
4. **Check sources** - Verify important findings manually
5. **Enable all sources** - More sources = better coverage

## 📝 License

MIT License
