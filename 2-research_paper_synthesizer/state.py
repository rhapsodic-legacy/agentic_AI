"""
Research Paper Synthesizer - State Schema and Data Models

Defines the state that flows through the LangGraph workflow
and the data models for papers, citations, and findings.
"""

from typing import TypedDict, Optional, Annotated
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import operator


class OutputFormat(Enum):
    LITERATURE_REVIEW = "literature_review"
    RESEARCH_PROPOSAL = "research_proposal"
    BLOG_POST = "blog_post"
    EXECUTIVE_SUMMARY = "executive_summary"
    ANNOTATED_BIBLIOGRAPHY = "annotated_bibliography"


class PaperSource(Enum):
    ARXIV = "arxiv"
    SEMANTIC_SCHOLAR = "semantic_scholar"
    PUBMED = "pubmed"
    LOCAL = "local"
    WEB = "web"


@dataclass
class Author:
    """Paper author."""
    name: str
    affiliation: Optional[str] = None
    
    def __str__(self):
        return self.name


@dataclass
class Paper:
    """A research paper with metadata."""
    id: str
    title: str
    abstract: str
    authors: list[Author]
    year: int
    source: PaperSource
    
    # Optional fields
    url: Optional[str] = None
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    pdf_url: Optional[str] = None
    citations_count: int = 0
    venue: Optional[str] = None
    
    # Content (may be fetched later)
    full_text: Optional[str] = None
    sections: dict[str, str] = field(default_factory=dict)
    
    # Analysis results
    summary: Optional[str] = None
    key_points: list[str] = field(default_factory=list)
    methodology: Optional[str] = None
    findings: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    relevance_score: float = 0.0
    
    def citation_key(self) -> str:
        """Generate a citation key."""
        first_author = self.authors[0].name.split()[-1] if self.authors else "Unknown"
        return f"{first_author}{self.year}"
    
    def to_bibtex(self) -> str:
        """Generate BibTeX entry."""
        authors_str = " and ".join(str(a) for a in self.authors)
        entry_type = "article"
        
        bibtex = f"@{entry_type}{{{self.citation_key()},\n"
        bibtex += f"  title = {{{self.title}}},\n"
        bibtex += f"  author = {{{authors_str}}},\n"
        bibtex += f"  year = {{{self.year}}},\n"
        
        if self.venue:
            bibtex += f"  journal = {{{self.venue}}},\n"
        if self.doi:
            bibtex += f"  doi = {{{self.doi}}},\n"
        if self.url:
            bibtex += f"  url = {{{self.url}}},\n"
        
        bibtex += "}"
        return bibtex
    
    def to_apa(self) -> str:
        """Generate APA citation."""
        if len(self.authors) == 1:
            authors_str = str(self.authors[0])
        elif len(self.authors) == 2:
            authors_str = f"{self.authors[0]} & {self.authors[1]}"
        elif len(self.authors) > 2:
            authors_str = f"{self.authors[0]} et al."
        else:
            authors_str = "Unknown"
        
        citation = f"{authors_str} ({self.year}). {self.title}."
        if self.venue:
            citation += f" {self.venue}."
        if self.doi:
            citation += f" https://doi.org/{self.doi}"
        
        return citation
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "title": self.title,
            "abstract": self.abstract[:500] + "..." if len(self.abstract) > 500 else self.abstract,
            "authors": [str(a) for a in self.authors],
            "year": self.year,
            "source": self.source.value,
            "url": self.url,
            "citations_count": self.citations_count,
            "relevance_score": self.relevance_score,
        }


@dataclass
class Finding:
    """A key finding from the research."""
    claim: str
    evidence: str
    source_papers: list[str]  # Paper IDs
    confidence: float  # 0-1
    category: Optional[str] = None
    
    # For contradiction detection
    contradicted_by: list[str] = field(default_factory=list)
    supports: list[str] = field(default_factory=list)


@dataclass  
class Theme:
    """A thematic grouping of findings."""
    name: str
    description: str
    findings: list[Finding]
    papers: list[str]  # Paper IDs


@dataclass
class Gap:
    """An identified gap in the literature."""
    description: str
    importance: str  # "high", "medium", "low"
    suggested_research: str
    related_papers: list[str]


@dataclass
class Contradiction:
    """A contradiction between papers."""
    topic: str
    position_a: str
    papers_a: list[str]
    position_b: str
    papers_b: list[str]
    possible_explanation: Optional[str] = None


class ResearchState(TypedDict):
    """
    State schema for the research workflow.
    
    This flows through all nodes in the LangGraph.
    """
    # Input
    topic: str
    research_question: Optional[str]
    output_format: str
    
    # Search
    search_queries: Annotated[list[str], operator.add]
    papers: Annotated[list[Paper], operator.add]
    papers_analyzed: list[str]  # Paper IDs that have been analyzed
    
    # Analysis
    summaries: dict[str, str]  # paper_id -> summary
    key_findings: list[Finding]
    themes: list[Theme]
    contradictions: list[Contradiction]
    gaps: list[Gap]
    
    # Synthesis
    synthesis: str
    outline: dict
    
    # Output
    final_output: str
    citations: list[str]  # Formatted citations
    
    # Control flow
    iteration: int
    max_iterations: int
    is_sufficient: bool
    
    # Metadata
    messages: Annotated[list[str], operator.add]
    errors: Annotated[list[str], operator.add]


def create_initial_state(
    topic: str,
    research_question: Optional[str] = None,
    output_format: str = "literature_review",
    max_iterations: int = 3,
) -> ResearchState:
    """Create the initial state for a research workflow."""
    return ResearchState(
        topic=topic,
        research_question=research_question,
        output_format=output_format,
        search_queries=[],
        papers=[],
        papers_analyzed=[],
        summaries={},
        key_findings=[],
        themes=[],
        contradictions=[],
        gaps=[],
        synthesis="",
        outline={},
        final_output="",
        citations=[],
        iteration=0,
        max_iterations=max_iterations,
        is_sufficient=False,
        messages=[],
        errors=[],
    )
