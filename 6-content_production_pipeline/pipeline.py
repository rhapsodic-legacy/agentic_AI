"""
Content Production Pipeline - LangGraph Pipeline

DAG-based content production system using LangGraph.

Pipeline Flow:
    Brief Intake → Research → Outline
                                ↓
              ┌─────────────────┼─────────────────┐
              ↓                 ↓                 ↓
         Blog Writer      Social Writer     Video Script
              ↓                 ↓                 ↓
        SEO Optimizer    Image Generator    Voice Over
              └─────────────────┼─────────────────┘
                                ↓
                            Editor
                                ↓
                           Publisher
"""

from typing import Optional, Literal
from dataclasses import dataclass

try:
    from langgraph.graph import StateGraph, END
except ImportError:
    raise ImportError("Install langgraph: pip install langgraph")

from .models import (
    ContentBrief, ContentType, Platform, BlogLength, Tone,
    ContentPackage, PipelineState
)
from .nodes import (
    brief_intake_node, research_node, outline_node,
    blog_writer_node, social_writer_node, video_script_node,
    seo_optimizer_node, image_generator_node, voiceover_node,
    editor_node, publisher_node
)


@dataclass
class PipelineConfig:
    """Configuration for the content pipeline."""
    llm_provider: str = "gemini"
    llm_model: Optional[str] = None
    
    # Content types to generate
    generate_blog: bool = True
    generate_social: bool = True
    generate_video: bool = False
    
    # Options
    run_seo: bool = True
    run_editor_review: bool = True
    verbose: bool = True


def should_generate_blog(state: PipelineState) -> bool:
    """Check if blog should be generated."""
    content_types = state.get("content_types", [])
    return "blog" in content_types


def should_generate_social(state: PipelineState) -> bool:
    """Check if social should be generated."""
    content_types = state.get("content_types", [])
    return "social" in content_types


def should_generate_video(state: PipelineState) -> bool:
    """Check if video should be generated."""
    content_types = state.get("content_types", [])
    return "video" in content_types


def route_after_outline(state: PipelineState) -> list[str]:
    """Route to appropriate content writers after outline."""
    content_types = state.get("content_types", ["blog"])
    routes = []
    
    if "blog" in content_types:
        routes.append("blog_writer")
    if "social" in content_types:
        routes.append("social_writer")
    if "video" in content_types:
        routes.append("video_script")
    
    # Default to blog if nothing specified
    if not routes:
        routes.append("blog_writer")
    
    return routes


def route_after_blog(state: PipelineState) -> str:
    """Route after blog writing."""
    return "seo_optimizer"


def route_after_social(state: PipelineState) -> str:
    """Route after social writing."""
    return "image_generator"


def route_after_video(state: PipelineState) -> str:
    """Route after video script."""
    return "voiceover"


def create_pipeline_graph(config: PipelineConfig = None) -> StateGraph:
    """
    Create the content production pipeline graph.
    
    DAG Structure:
        brief_intake → research → outline
                                    ↓
              ┌─────────────────────┼─────────────────────┐
              ↓                     ↓                     ↓
         blog_writer          social_writer         video_script
              ↓                     ↓                     ↓
        seo_optimizer        image_generator         voiceover
              └─────────────────────┼─────────────────────┘
                                    ↓
                                 editor
                                    ↓
                                publisher
    """
    config = config or PipelineConfig()
    
    # Create graph
    graph = StateGraph(PipelineState)
    
    # Add nodes
    graph.add_node("brief_intake", brief_intake_node)
    graph.add_node("research", research_node)
    graph.add_node("outline", outline_node)
    graph.add_node("blog_writer", blog_writer_node)
    graph.add_node("social_writer", social_writer_node)
    graph.add_node("video_script", video_script_node)
    graph.add_node("seo_optimizer", seo_optimizer_node)
    graph.add_node("image_generator", image_generator_node)
    graph.add_node("voiceover", voiceover_node)
    graph.add_node("editor", editor_node)
    graph.add_node("publisher", publisher_node)
    
    # Set entry point
    graph.set_entry_point("brief_intake")
    
    # Linear flow: brief → research → outline
    graph.add_edge("brief_intake", "research")
    graph.add_edge("research", "outline")
    
    # Parallel branches after outline (conditional)
    # For simplicity, we'll use sequential execution with conditional nodes
    
    # Outline → Content writers (we'll run all, they check internally)
    graph.add_edge("outline", "blog_writer")
    graph.add_edge("blog_writer", "seo_optimizer")
    graph.add_edge("seo_optimizer", "social_writer")
    graph.add_edge("social_writer", "image_generator")
    graph.add_edge("image_generator", "video_script")
    graph.add_edge("video_script", "voiceover")
    
    # All paths converge at editor
    graph.add_edge("voiceover", "editor")
    
    # Editor → Publisher → End
    graph.add_edge("editor", "publisher")
    graph.add_edge("publisher", END)
    
    return graph


class ContentPipeline:
    """
    Content Production Pipeline
    
    Creates blog posts, social media content, video scripts, and newsletters
    from a single topic or brief.
    
    Usage:
        pipeline = ContentPipeline()
        result = pipeline.run(
            topic="AI in Marketing",
            content_types=["blog", "social", "video"],
            platforms=["twitter", "linkedin"],
        )
        
        # Access results
        print(result["blog"]["content"])
        for post in result["social_posts"]:
            print(post["text"])
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        
        # Build graph
        self.graph = create_pipeline_graph(self.config)
        self.app = self.graph.compile()
    
    def run(
        self,
        topic: str,
        description: str = "",
        content_types: list[str] = None,
        platforms: list[str] = None,
        tone: str = "professional",
        target_audience: str = "general",
        blog_length: str = "medium",
        primary_keyword: str = "",
        call_to_action: str = "",
        key_points: list[str] = None,
    ) -> dict:
        """
        Run the content production pipeline.
        
        Args:
            topic: Main topic/subject
            description: Optional detailed description
            content_types: List of content types ["blog", "social", "video"]
            platforms: Social platforms ["twitter", "linkedin", "instagram"]
            tone: Content tone
            target_audience: Target audience description
            blog_length: "short", "medium", "long"
            primary_keyword: SEO keyword
            call_to_action: Desired CTA
            key_points: Specific points to cover
        
        Returns:
            Content package with all generated content
        """
        # Default content types
        if content_types is None:
            content_types = ["blog"]
        
        # Default platforms for social
        if platforms is None and "social" in content_types:
            platforms = ["twitter", "linkedin"]
        elif platforms is None:
            platforms = []
        
        # Build brief
        brief = {
            "topic": topic,
            "description": description or f"Create content about {topic}",
            "content_types": content_types,
            "platforms": platforms,
            "tone": tone,
            "target_audience": target_audience,
            "blog_length": blog_length,
            "primary_keyword": primary_keyword or topic.lower(),
            "call_to_action": call_to_action or "Learn more today!",
            "key_points": key_points or [],
        }
        
        # Initial state
        initial_state = {
            "brief": brief,
            "content_types": content_types,
            "messages": [],
        }
        
        # Run pipeline
        if self.config.verbose:
            print(f"🚀 Starting content pipeline for: {topic}")
            print(f"   Content types: {', '.join(content_types)}")
            if platforms:
                print(f"   Platforms: {', '.join(platforms)}")
        
        result = self.app.invoke(initial_state)
        
        if self.config.verbose:
            print("\n📋 Pipeline Messages:")
            for msg in result.get("messages", []):
                print(f"   {msg}")
        
        return result.get("content_package", {})
    
    def run_from_brief(self, brief: ContentBrief) -> dict:
        """Run pipeline from a ContentBrief object."""
        return self.run(
            topic=brief.topic,
            description=brief.description,
            content_types=[ct.value for ct in brief.content_types],
            platforms=[p.value for p in brief.platforms],
            tone=brief.tone.value,
            target_audience=brief.target_audience,
            blog_length=brief.blog_length.value,
            primary_keyword=brief.primary_keyword,
            call_to_action=brief.call_to_action,
            key_points=brief.key_points,
        )
    
    def get_graph_visualization(self) -> str:
        """Get mermaid diagram of the pipeline."""
        return """
```mermaid
graph TD
    A[Brief Intake] --> B[Research Agent]
    B --> C[Outline Creator]
    
    C --> D[Blog Writer]
    C --> E[Social Writer]
    C --> F[Video Script]
    
    D --> G[SEO Optimizer]
    E --> H[Image Generator]
    F --> I[Voice Over]
    
    G --> J[Editor]
    H --> J
    I --> J
    
    J --> K[Publisher]
```
"""


# =============================================================================
# Convenience Functions
# =============================================================================

def create_content(
    topic: str,
    content_types: list[str] = None,
    platforms: list[str] = None,
    **kwargs
) -> dict:
    """
    Quick function to create content.
    
    Args:
        topic: Content topic
        content_types: Types to generate
        platforms: Social platforms
        **kwargs: Additional brief options
    
    Returns:
        Content package
    """
    pipeline = ContentPipeline()
    return pipeline.run(
        topic=topic,
        content_types=content_types,
        platforms=platforms,
        **kwargs
    )


def create_blog_post(topic: str, length: str = "medium", **kwargs) -> dict:
    """Create a blog post only."""
    pipeline = ContentPipeline()
    result = pipeline.run(
        topic=topic,
        content_types=["blog"],
        blog_length=length,
        **kwargs
    )
    return result.get("blog", {})


def create_social_content(
    topic: str,
    platforms: list[str] = None,
    **kwargs
) -> list[dict]:
    """Create social media content only."""
    platforms = platforms or ["twitter", "linkedin"]
    pipeline = ContentPipeline()
    result = pipeline.run(
        topic=topic,
        content_types=["social"],
        platforms=platforms,
        **kwargs
    )
    return result.get("social_posts", [])


def create_video_script(topic: str, **kwargs) -> dict:
    """Create video script only."""
    pipeline = ContentPipeline()
    result = pipeline.run(
        topic=topic,
        content_types=["video"],
        **kwargs
    )
    return result.get("video", {})
