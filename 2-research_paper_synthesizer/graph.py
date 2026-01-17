"""
Research Paper Synthesizer - Main Graph

Defines the LangGraph workflow that orchestrates the research process
with cyclic refinement for iterative improvement.

Graph Structure:
    START → Planner → Searcher → Reader → Evaluator
                                              │
                                    ┌─────────┴──────────┐
                                    │                    │
                                NOT SUFFICIENT      SUFFICIENT
                                    │                    │
                                    └────► Planner       │
                                                         │
                                              ┌──────────┘
                                              ▼
                                         Synthesizer → Writer → END
"""

from typing import Optional, Callable, Literal
from dataclasses import dataclass

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from .state import ResearchState, create_initial_state, OutputFormat
from .nodes import (
    PlannerNode,
    SearcherNode,
    ReaderNode,
    EvaluatorNode,
    SynthesizerNode,
    WriterNode,
    create_llm,
)


@dataclass
class SynthesizerConfig:
    """Configuration for the Research Synthesizer."""
    
    # LLM settings
    llm_provider: str = "gemini"  # "gemini", "anthropic", "openai"
    llm_model: Optional[str] = None
    
    # Search settings
    max_results_per_source: int = 5
    enable_arxiv: bool = True
    enable_semantic_scholar: bool = True
    enable_pubmed: bool = True
    
    # Analysis settings
    max_papers_per_iteration: int = 5
    min_papers: int = 5
    min_coverage: float = 0.6
    max_iterations: int = 3
    
    # Output settings
    default_format: str = "literature_review"
    
    # Callbacks
    on_step: Optional[Callable[[str, dict], None]] = None


@dataclass
class SynthesizerResult:
    """Result from a research synthesis run."""
    success: bool
    topic: str
    output: str
    output_format: str
    
    # Details
    papers_found: int
    papers_analyzed: int
    themes_identified: int
    gaps_found: int
    contradictions_found: int
    iterations: int
    
    # Data
    papers: list
    themes: list
    gaps: list
    contradictions: list
    citations: list
    
    # Errors
    errors: list


def should_continue(state: ResearchState) -> Literal["planner", "synthesizer"]:
    """
    Conditional edge: decide whether to continue searching or synthesize.
    
    Returns:
        "planner" to search for more papers
        "synthesizer" to proceed to synthesis
    """
    if state.get("is_sufficient", False):
        return "synthesizer"
    
    iteration = state.get("iteration", 0)
    max_iterations = state.get("max_iterations", 3)
    
    if iteration >= max_iterations:
        return "synthesizer"
    
    return "planner"


class ResearchSynthesizer:
    """
    Research Paper Synthesizer using LangGraph.
    
    Takes a research topic and produces a comprehensive literature review
    or other academic output by:
    1. Searching multiple academic sources
    2. Analyzing papers with LLM
    3. Iteratively refining until sufficient coverage
    4. Synthesizing findings into themes
    5. Generating formatted output
    
    Usage:
        synthesizer = ResearchSynthesizer()
        result = synthesizer.run(
            topic="Large Language Models for Scientific Discovery",
            output_format="literature_review"
        )
        
        print(result.output)
        print(f"Papers analyzed: {result.papers_analyzed}")
    """
    
    def __init__(self, config: Optional[SynthesizerConfig] = None):
        self.config = config or SynthesizerConfig()
        
        # Create LLM
        self.llm = create_llm(
            provider=self.config.llm_provider,
            model=self.config.llm_model,
        )
        
        # Create nodes
        self._create_nodes()
        
        # Build graph
        self._build_graph()
    
    def _create_nodes(self):
        """Create all the workflow nodes."""
        self.planner = PlannerNode(self.llm)
        
        self.searcher = SearcherNode(
            max_results_per_source=self.config.max_results_per_source,
            enable_arxiv=self.config.enable_arxiv,
            enable_semantic_scholar=self.config.enable_semantic_scholar,
            enable_pubmed=self.config.enable_pubmed,
        )
        
        self.reader = ReaderNode(
            self.llm,
            max_papers_per_iteration=self.config.max_papers_per_iteration,
        )
        
        self.evaluator = EvaluatorNode(
            self.llm,
            min_papers=self.config.min_papers,
            min_coverage=self.config.min_coverage,
        )
        
        self.synthesizer_node = SynthesizerNode(self.llm)
        self.writer = WriterNode(self.llm)
    
    def _build_graph(self):
        """Build the LangGraph workflow."""
        # Create graph with state schema
        workflow = StateGraph(ResearchState)
        
        # Add nodes
        workflow.add_node("planner", self._wrap_node(self.planner, "planner"))
        workflow.add_node("searcher", self._wrap_node(self.searcher, "searcher"))
        workflow.add_node("reader", self._wrap_node(self.reader, "reader"))
        workflow.add_node("evaluator", self._wrap_node(self.evaluator, "evaluator"))
        workflow.add_node("synthesizer", self._wrap_node(self.synthesizer_node, "synthesizer"))
        workflow.add_node("writer", self._wrap_node(self.writer, "writer"))
        
        # Add edges
        workflow.set_entry_point("planner")
        workflow.add_edge("planner", "searcher")
        workflow.add_edge("searcher", "reader")
        workflow.add_edge("reader", "evaluator")
        
        # Conditional edge for cyclic refinement
        workflow.add_conditional_edges(
            "evaluator",
            should_continue,
            {
                "planner": "planner",
                "synthesizer": "synthesizer",
            }
        )
        
        workflow.add_edge("synthesizer", "writer")
        workflow.add_edge("writer", END)
        
        # Compile with memory for checkpointing
        self.memory = MemorySaver()
        self.graph = workflow.compile(checkpointer=self.memory)
    
    def _wrap_node(self, node, name: str):
        """Wrap a node to add logging/callbacks."""
        def wrapped(state: ResearchState) -> dict:
            if self.config.on_step:
                self.config.on_step(name, {"state": state})
            
            result = node(state)
            
            # Log messages
            for msg in result.get("messages", []):
                print(f"  [{name}] {msg}")
            
            return result
        
        return wrapped
    
    def run(
        self,
        topic: str,
        research_question: Optional[str] = None,
        output_format: str = None,
        max_iterations: int = None,
    ) -> SynthesizerResult:
        """
        Run the research synthesis workflow.
        
        Args:
            topic: Research topic to investigate
            research_question: Optional specific research question
            output_format: Output format (literature_review, research_proposal, etc.)
            max_iterations: Maximum search iterations
        
        Returns:
            SynthesizerResult with the synthesis output and metadata
        """
        # Create initial state
        initial_state = create_initial_state(
            topic=topic,
            research_question=research_question,
            output_format=output_format or self.config.default_format,
            max_iterations=max_iterations or self.config.max_iterations,
        )
        
        print(f"\n🔬 Starting research synthesis: {topic}")
        print(f"   Format: {initial_state['output_format']}")
        print(f"   Max iterations: {initial_state['max_iterations']}")
        print()
        
        # Run the graph
        config = {"configurable": {"thread_id": "research_1"}}
        
        try:
            final_state = None
            for event in self.graph.stream(initial_state, config):
                # Track the latest state
                for node_name, node_state in event.items():
                    final_state = {**initial_state, **node_state} if final_state is None else {**final_state, **node_state}
            
            # Build result
            papers = final_state.get("papers", [])
            themes = final_state.get("themes", [])
            gaps = final_state.get("gaps", [])
            contradictions = final_state.get("contradictions", [])
            
            return SynthesizerResult(
                success=True,
                topic=topic,
                output=final_state.get("final_output", ""),
                output_format=output_format or self.config.default_format,
                papers_found=len(papers),
                papers_analyzed=len(final_state.get("papers_analyzed", [])),
                themes_identified=len(themes),
                gaps_found=len(gaps),
                contradictions_found=len(contradictions),
                iterations=final_state.get("iteration", 0),
                papers=[p.to_dict() if hasattr(p, 'to_dict') else p for p in papers],
                themes=themes,
                gaps=gaps,
                contradictions=contradictions,
                citations=final_state.get("citations", []),
                errors=final_state.get("errors", []),
            )
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return SynthesizerResult(
                success=False,
                topic=topic,
                output="",
                output_format=output_format or self.config.default_format,
                papers_found=0,
                papers_analyzed=0,
                themes_identified=0,
                gaps_found=0,
                contradictions_found=0,
                iterations=0,
                papers=[],
                themes=[],
                gaps=[],
                contradictions=[],
                citations=[],
                errors=[str(e)],
            )
    
    def run_step_by_step(
        self,
        topic: str,
        research_question: Optional[str] = None,
        output_format: str = None,
    ):
        """
        Generator that yields results after each step.
        
        Useful for streaming progress to a UI.
        """
        initial_state = create_initial_state(
            topic=topic,
            research_question=research_question,
            output_format=output_format or self.config.default_format,
            max_iterations=self.config.max_iterations,
        )
        
        config = {"configurable": {"thread_id": "research_stream"}}
        
        current_state = initial_state
        
        for event in self.graph.stream(initial_state, config):
            for node_name, node_state in event.items():
                current_state = {**current_state, **node_state}
                yield {
                    "node": node_name,
                    "state": current_state,
                    "messages": node_state.get("messages", []),
                }
    
    def get_graph_visualization(self) -> str:
        """Get a text visualization of the graph."""
        return """
Research Synthesis Graph:
========================

    ┌─────────────┐
    │   START     │
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐
    │   PLANNER   │◄───────────────┐
    │  (Queries)  │                │
    └──────┬──────┘                │
           │                       │
           ▼                       │
    ┌─────────────┐                │
    │  SEARCHER   │                │
    │ (Arxiv etc) │                │
    └──────┬──────┘                │
           │                       │
           ▼                       │
    ┌─────────────┐                │
    │   READER    │                │
    │  (Analyze)  │                │
    └──────┬──────┘                │
           │                       │
           ▼                       │
    ┌─────────────┐                │
    │  EVALUATOR  │────── NO ──────┘
    │(Sufficient?)│
    └──────┬──────┘
           │
        YES│
           ▼
    ┌─────────────┐
    │ SYNTHESIZER │
    │  (Themes)   │
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐
    │   WRITER    │
    │  (Output)   │
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐
    │    END      │
    └─────────────┘
"""


# Convenience functions

def synthesize_topic(
    topic: str,
    output_format: str = "literature_review",
    provider: str = "gemini",
) -> SynthesizerResult:
    """
    Quick function to synthesize research on a topic.
    
    Args:
        topic: Research topic
        output_format: Output format
        provider: LLM provider
    
    Returns:
        SynthesizerResult
    """
    config = SynthesizerConfig(llm_provider=provider)
    synthesizer = ResearchSynthesizer(config)
    return synthesizer.run(topic, output_format=output_format)


def create_literature_review(topic: str, provider: str = "gemini") -> str:
    """Create a literature review for a topic."""
    result = synthesize_topic(topic, "literature_review", provider)
    return result.output


def create_research_proposal(topic: str, provider: str = "gemini") -> str:
    """Create a research proposal for a topic."""
    result = synthesize_topic(topic, "research_proposal", provider)
    return result.output
