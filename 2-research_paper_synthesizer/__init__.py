"""
Research Paper Synthesizer - LangGraph Nodes

Defines the nodes that make up the research workflow graph:
- Planner: Generate search queries
- Searcher: Find papers from sources
- Reader: Analyze papers
- Evaluator: Check if sufficient
- Synthesizer: Create themes
- Writer: Generate output
"""

from typing import Optional
import json
import re

from langchain_core.messages import HumanMessage, SystemMessage

from ..state import (
    ResearchState, Paper, Finding, Theme, Gap, Contradiction,
    OutputFormat, create_initial_state
)
from ..sources import MultiSourceSearch


def create_llm(provider: str = "gemini", model: Optional[str] = None):
    """Create an LLM instance."""
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=model or "claude-sonnet-4-20250514",
            temperature=0.1,
        )
    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=model or "gemini-1.5-flash",
            temperature=0.1,
        )
    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model or "gpt-4o-mini",
            temperature=0.1,
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")


class PlannerNode:
    """
    Planner Node: Generate search queries from the research topic.
    
    This is the starting point that creates targeted search queries
    to find relevant papers.
    """
    
    SYSTEM_PROMPT = """You are a research planning expert. Given a research topic, 
generate effective search queries to find relevant academic papers.

Your queries should:
1. Cover different aspects of the topic
2. Include synonyms and related terms
3. Be specific enough to find relevant papers
4. Consider different research angles (theoretical, empirical, applied)

Respond with a JSON object:
{
    "queries": ["query 1", "query 2", ...],
    "research_angles": ["angle 1", "angle 2", ...],
    "key_concepts": ["concept 1", "concept 2", ...]
}

Generate 3-5 search queries."""

    def __init__(self, llm):
        self.llm = llm
    
    def __call__(self, state: ResearchState) -> dict:
        """Generate search queries from the topic."""
        topic = state["topic"]
        research_question = state.get("research_question", "")
        iteration = state.get("iteration", 0)
        
        # Build prompt
        prompt = f"Research Topic: {topic}"
        if research_question:
            prompt += f"\nResearch Question: {research_question}"
        
        if iteration > 0:
            # Refine queries based on previous results
            existing_queries = state.get("search_queries", [])
            papers_count = len(state.get("papers", []))
            prompt += f"\n\nPrevious queries: {existing_queries}"
            prompt += f"\nPapers found so far: {papers_count}"
            prompt += "\nGenerate DIFFERENT queries to find additional perspectives."
        
        # Call LLM
        response = self.llm.invoke([
            SystemMessage(content=self.SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ])
        
        # Parse response
        try:
            # Extract JSON from response
            content = response.content
            json_match = re.search(r'\{[^{}]*\}', content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                queries = data.get("queries", [topic])
            else:
                queries = [topic]
        except Exception:
            queries = [topic]
        
        return {
            "search_queries": queries,
            "iteration": iteration + 1,
            "messages": [f"📋 Generated {len(queries)} search queries"],
        }


class SearcherNode:
    """
    Searcher Node: Find papers from academic sources.
    
    Searches Arxiv, Semantic Scholar, and PubMed for relevant papers.
    """
    
    def __init__(
        self,
        max_results_per_source: int = 5,
        enable_arxiv: bool = True,
        enable_semantic_scholar: bool = True,
        enable_pubmed: bool = True,
    ):
        self.search = MultiSourceSearch(
            enable_arxiv=enable_arxiv,
            enable_semantic_scholar=enable_semantic_scholar,
            enable_pubmed=enable_pubmed,
        )
        self.max_results = max_results_per_source
    
    def __call__(self, state: ResearchState) -> dict:
        """Search for papers using the generated queries."""
        queries = state.get("search_queries", [])
        existing_paper_ids = {p.id for p in state.get("papers", [])}
        
        new_papers = []
        
        for query in queries[-3:]:  # Use last 3 queries
            papers = self.search.search(
                query,
                max_results_per_source=self.max_results,
            )
            
            for paper in papers:
                if paper.id not in existing_paper_ids:
                    new_papers.append(paper)
                    existing_paper_ids.add(paper.id)
        
        return {
            "papers": new_papers,
            "messages": [f"🔍 Found {len(new_papers)} new papers"],
        }


class ReaderNode:
    """
    Reader Node: Analyze papers and extract key information.
    
    Uses LLM to analyze abstracts and extract:
    - Summary
    - Key findings
    - Methodology
    - Limitations
    """
    
    SYSTEM_PROMPT = """You are a research paper analyst. Analyze the given paper 
and extract key information.

Respond with a JSON object:
{
    "summary": "2-3 sentence summary",
    "key_findings": ["finding 1", "finding 2", ...],
    "methodology": "brief methodology description or null",
    "limitations": ["limitation 1", ...],
    "relevance_to_topic": 0.0-1.0
}"""

    def __init__(self, llm, max_papers_per_iteration: int = 5):
        self.llm = llm
        self.max_papers = max_papers_per_iteration
    
    def __call__(self, state: ResearchState) -> dict:
        """Analyze papers that haven't been analyzed yet."""
        topic = state["topic"]
        papers = state.get("papers", [])
        analyzed_ids = set(state.get("papers_analyzed", []))
        summaries = dict(state.get("summaries", {}))
        
        # Find papers to analyze
        to_analyze = [p for p in papers if p.id not in analyzed_ids][:self.max_papers]
        
        new_findings = []
        new_analyzed = []
        
        for paper in to_analyze:
            # Build prompt
            prompt = f"""Topic: {topic}

Paper: {paper.title}
Authors: {', '.join(str(a) for a in paper.authors)}
Year: {paper.year}
Abstract: {paper.abstract}"""

            try:
                response = self.llm.invoke([
                    SystemMessage(content=self.SYSTEM_PROMPT),
                    HumanMessage(content=prompt),
                ])
                
                # Parse response
                content = response.content
                json_match = re.search(r'\{[^{}]*\}', content, re.DOTALL)
                
                if json_match:
                    data = json.loads(json_match.group())
                    
                    # Update paper
                    paper.summary = data.get("summary", "")
                    paper.key_points = data.get("key_findings", [])
                    paper.methodology = data.get("methodology")
                    paper.limitations = data.get("limitations", [])
                    paper.relevance_score = float(data.get("relevance_to_topic", 0.5))
                    
                    summaries[paper.id] = paper.summary
                    
                    # Create findings
                    for finding_text in data.get("key_findings", []):
                        new_findings.append(Finding(
                            claim=finding_text,
                            evidence=paper.abstract[:200],
                            source_papers=[paper.id],
                            confidence=paper.relevance_score,
                        ))
                
                new_analyzed.append(paper.id)
                
            except Exception as e:
                print(f"Error analyzing paper {paper.id}: {e}")
                new_analyzed.append(paper.id)
        
        return {
            "papers_analyzed": state.get("papers_analyzed", []) + new_analyzed,
            "summaries": summaries,
            "key_findings": state.get("key_findings", []) + new_findings,
            "messages": [f"📖 Analyzed {len(new_analyzed)} papers"],
        }


class EvaluatorNode:
    """
    Evaluator Node: Determine if we have sufficient information.
    
    Checks:
    - Number of papers
    - Coverage of key aspects
    - Presence of diverse perspectives
    """
    
    SYSTEM_PROMPT = """You are evaluating research coverage. Given a topic and 
the current research findings, determine if we have sufficient information 
for a comprehensive analysis.

Consider:
1. Do we have enough papers (at least 5-10)?
2. Are different perspectives represented?
3. Are there major gaps in coverage?
4. Is the topic adequately explored?

Respond with a JSON object:
{
    "is_sufficient": true/false,
    "coverage_score": 0.0-1.0,
    "gaps": ["gap 1", "gap 2", ...],
    "recommendation": "explanation"
}"""

    def __init__(self, llm, min_papers: int = 5, min_coverage: float = 0.6):
        self.llm = llm
        self.min_papers = min_papers
        self.min_coverage = min_coverage
    
    def __call__(self, state: ResearchState) -> dict:
        """Evaluate if we have sufficient research coverage."""
        topic = state["topic"]
        papers = state.get("papers", [])
        findings = state.get("key_findings", [])
        iteration = state.get("iteration", 0)
        max_iterations = state.get("max_iterations", 3)
        
        # Quick checks
        if len(papers) < self.min_papers and iteration < max_iterations:
            return {
                "is_sufficient": False,
                "messages": [f"⚠️ Only {len(papers)} papers found, need more"],
            }
        
        if iteration >= max_iterations:
            return {
                "is_sufficient": True,
                "messages": [f"✓ Max iterations ({max_iterations}) reached, proceeding"],
            }
        
        # Build prompt for LLM evaluation
        paper_summaries = "\n".join([
            f"- {p.title} ({p.year}): {p.summary or p.abstract[:200]}"
            for p in papers[:10]
        ])
        
        findings_text = "\n".join([f"- {f.claim}" for f in findings[:15]])
        
        prompt = f"""Topic: {topic}

Papers found ({len(papers)} total):
{paper_summaries}

Key findings so far:
{findings_text}

Iteration: {iteration}/{max_iterations}

Evaluate if this is sufficient for a comprehensive literature review."""

        try:
            response = self.llm.invoke([
                SystemMessage(content=self.SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ])
            
            content = response.content
            json_match = re.search(r'\{[^{}]*\}', content, re.DOTALL)
            
            if json_match:
                data = json.loads(json_match.group())
                is_sufficient = data.get("is_sufficient", False)
                coverage = data.get("coverage_score", 0.5)
                gaps = data.get("gaps", [])
                
                if coverage >= self.min_coverage or iteration >= max_iterations - 1:
                    is_sufficient = True
                
                return {
                    "is_sufficient": is_sufficient,
                    "gaps": [Gap(
                        description=g,
                        importance="medium",
                        suggested_research=f"Investigate: {g}",
                        related_papers=[],
                    ) for g in gaps],
                    "messages": [f"📊 Coverage: {coverage:.0%}, Sufficient: {is_sufficient}"],
                }
        
        except Exception as e:
            print(f"Evaluation error: {e}")
        
        # Default: continue if under max iterations
        return {
            "is_sufficient": iteration >= max_iterations - 1,
            "messages": ["⚠️ Could not evaluate, continuing"],
        }


class SynthesizerNode:
    """
    Synthesizer Node: Create themes and identify patterns.
    
    Groups findings into themes, identifies contradictions,
    and highlights gaps in the literature.
    """
    
    SYSTEM_PROMPT = """You are a research synthesizer. Analyze the findings 
and create a coherent synthesis.

Tasks:
1. Group findings into 3-5 major themes
2. Identify any contradictions between papers
3. Note gaps in the literature
4. Highlight key debates in the field

Respond with a JSON object:
{
    "themes": [
        {
            "name": "Theme Name",
            "description": "Theme description",
            "findings": ["finding 1", "finding 2"]
        }
    ],
    "contradictions": [
        {
            "topic": "Topic of disagreement",
            "position_a": "One view",
            "position_b": "Opposing view"
        }
    ],
    "gaps": ["Gap 1", "Gap 2"],
    "key_debates": ["Debate 1", "Debate 2"],
    "synthesis_summary": "Overall synthesis paragraph"
}"""

    def __init__(self, llm):
        self.llm = llm
    
    def __call__(self, state: ResearchState) -> dict:
        """Synthesize findings into themes."""
        topic = state["topic"]
        papers = state.get("papers", [])
        findings = state.get("key_findings", [])
        
        # Build context
        papers_context = "\n".join([
            f"[{p.citation_key()}] {p.title} ({p.year})\n  {p.summary or p.abstract[:200]}"
            for p in papers[:15]
        ])
        
        findings_context = "\n".join([
            f"- {f.claim} (confidence: {f.confidence:.0%})"
            for f in findings[:20]
        ])
        
        prompt = f"""Topic: {topic}

Papers:
{papers_context}

Findings:
{findings_context}

Synthesize these into coherent themes and identify patterns."""

        try:
            response = self.llm.invoke([
                SystemMessage(content=self.SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ])
            
            content = response.content
            # Try to find JSON in response
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            
            if json_match:
                try:
                    data = json.loads(json_match.group())
                except json.JSONDecodeError:
                    # Try to clean the JSON
                    cleaned = json_match.group().replace('\n', ' ')
                    data = json.loads(cleaned)
                
                # Create themes
                themes = []
                for t in data.get("themes", []):
                    themes.append(Theme(
                        name=t.get("name", "Theme"),
                        description=t.get("description", ""),
                        findings=[Finding(
                            claim=f, evidence="", source_papers=[], confidence=0.7
                        ) for f in t.get("findings", [])],
                        papers=[],
                    ))
                
                # Create contradictions
                contradictions = []
                for c in data.get("contradictions", []):
                    contradictions.append(Contradiction(
                        topic=c.get("topic", ""),
                        position_a=c.get("position_a", ""),
                        papers_a=[],
                        position_b=c.get("position_b", ""),
                        papers_b=[],
                    ))
                
                # Create gaps
                gaps = [
                    Gap(
                        description=g,
                        importance="medium",
                        suggested_research=f"Investigate {g}",
                        related_papers=[],
                    )
                    for g in data.get("gaps", [])
                ]
                
                synthesis = data.get("synthesis_summary", "")
                
                return {
                    "themes": themes,
                    "contradictions": contradictions,
                    "gaps": state.get("gaps", []) + gaps,
                    "synthesis": synthesis,
                    "messages": [f"🔬 Created {len(themes)} themes, found {len(contradictions)} contradictions"],
                }
        
        except Exception as e:
            print(f"Synthesis error: {e}")
        
        return {
            "synthesis": "Unable to synthesize findings.",
            "messages": ["⚠️ Synthesis had issues"],
        }


class WriterNode:
    """
    Writer Node: Generate the final output document.
    
    Creates a well-formatted literature review, research proposal,
    or other output format based on the synthesis.
    """
    
    TEMPLATES = {
        "literature_review": """You are writing a literature review. Create a comprehensive 
academic document with the following structure:

# Literature Review: {topic}

## Executive Summary
(2-3 paragraph overview of the field)

## 1. Introduction
(Background, scope, and objectives of this review)

## 2. Methodology
(How papers were selected and analyzed)

## 3. Key Themes
(For each theme, discuss findings with citations)

## 4. Contradictions and Debates
(Where researchers disagree)

## 5. Gaps in the Literature
(What's missing from current research)

## 6. Future Directions
(Suggested research directions)

## 7. Conclusion
(Summary and implications)

## References
(APA format citations)

Use proper academic language. Cite sources using author names and years.""",

        "research_proposal": """You are writing a research proposal. Create a document with:

# Research Proposal: {topic}

## Abstract
(Brief summary of proposed research)

## 1. Introduction and Background
(Context and motivation)

## 2. Literature Review
(Current state of research)

## 3. Research Gap
(What this research addresses)

## 4. Research Questions
(Specific questions to answer)

## 5. Proposed Methodology
(How the research will be conducted)

## 6. Expected Contributions
(Significance and impact)

## 7. Timeline
(Proposed schedule)

## References
(APA format)""",

        "blog_post": """You are writing an accessible blog post about research. Create:

# {topic}: What the Research Says

(Engaging introduction for general audience)

## The Big Picture
(What researchers are studying and why it matters)

## Key Findings
(Main discoveries, explained simply)

## The Debates
(Where experts disagree)

## What We Still Don't Know
(Gaps in knowledge)

## The Bottom Line
(Key takeaways)

---
*Sources: (List main papers referenced)*

Use engaging, accessible language. Avoid jargon.""",

        "executive_summary": """Create a concise executive summary:

# Executive Summary: {topic}

## Overview
(1 paragraph summary)

## Key Findings
(Bullet points)

## Critical Gaps
(Bullet points)

## Recommendations
(Action items)

## Sources
(Brief list)

Keep it under 2 pages. Focus on actionable insights.""",

        "annotated_bibliography": """Create an annotated bibliography:

# Annotated Bibliography: {topic}

For each paper:
---
**Citation** (APA format)

*Summary*: (2-3 sentence summary)

*Key Findings*: (Bullet points)

*Relevance*: (How it relates to the topic)

*Limitations*: (Any noted limitations)

---

List papers in order of relevance.""",
    }
    
    def __init__(self, llm):
        self.llm = llm
    
    def __call__(self, state: ResearchState) -> dict:
        """Generate the final output document."""
        topic = state["topic"]
        output_format = state.get("output_format", "literature_review")
        papers = state.get("papers", [])
        themes = state.get("themes", [])
        contradictions = state.get("contradictions", [])
        gaps = state.get("gaps", [])
        synthesis = state.get("synthesis", "")
        
        # Get template
        template = self.TEMPLATES.get(output_format, self.TEMPLATES["literature_review"])
        
        # Build context
        papers_context = "\n".join([
            f"- {p.to_apa()}\n  Summary: {p.summary or p.abstract[:200]}"
            for p in papers[:20]
        ])
        
        themes_context = "\n".join([
            f"Theme: {t.name}\nDescription: {t.description}\nFindings: {[f.claim for f in t.findings]}"
            for t in themes
        ])
        
        contradictions_context = "\n".join([
            f"- {c.topic}: {c.position_a} vs {c.position_b}"
            for c in contradictions
        ])
        
        gaps_context = "\n".join([f"- {g.description}" for g in gaps])
        
        prompt = f"""Create the document based on this research:

TOPIC: {topic}

PAPERS REVIEWED:
{papers_context}

THEMES IDENTIFIED:
{themes_context}

SYNTHESIS:
{synthesis}

CONTRADICTIONS:
{contradictions_context}

GAPS:
{gaps_context}

Follow this template:
{template.format(topic=topic)}

Write a complete, well-structured document. Include proper citations."""

        try:
            response = self.llm.invoke([
                SystemMessage(content="You are an expert academic writer."),
                HumanMessage(content=prompt),
            ])
            
            final_output = response.content
            
            # Generate citations
            citations = [p.to_apa() for p in papers[:20]]
            
            return {
                "final_output": final_output,
                "citations": citations,
                "messages": [f"📝 Generated {output_format} ({len(final_output)} chars)"],
            }
        
        except Exception as e:
            print(f"Writer error: {e}")
            return {
                "final_output": f"Error generating output: {e}",
                "citations": [],
                "errors": [str(e)],
            }
