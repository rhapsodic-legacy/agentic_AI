"""
Content Production Pipeline - Data Models

Models for content briefs, outputs, and pipeline state.
"""

from typing import Optional, Any, Literal
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json


class ContentType(Enum):
    BLOG = "blog"
    SOCIAL = "social"
    VIDEO = "video"
    NEWSLETTER = "newsletter"


class Platform(Enum):
    TWITTER = "twitter"
    LINKEDIN = "linkedin"
    INSTAGRAM = "instagram"
    TIKTOK = "tiktok"
    YOUTUBE = "youtube"
    MEDIUM = "medium"
    WORDPRESS = "wordpress"
    SUBSTACK = "substack"


class BlogLength(Enum):
    SHORT = "short"      # ~500 words
    MEDIUM = "medium"    # ~1500 words
    LONG = "long"        # ~3000 words


class Tone(Enum):
    PROFESSIONAL = "professional"
    CASUAL = "casual"
    HUMOROUS = "humorous"
    INSPIRATIONAL = "inspirational"
    EDUCATIONAL = "educational"
    PERSUASIVE = "persuasive"


@dataclass
class ContentBrief:
    """Input brief for content creation."""
    topic: str
    description: str = ""
    
    # Target content types
    content_types: list[ContentType] = field(default_factory=lambda: [ContentType.BLOG])
    
    # Target platforms
    platforms: list[Platform] = field(default_factory=list)
    
    # Style
    tone: Tone = Tone.PROFESSIONAL
    target_audience: str = "general"
    
    # Blog settings
    blog_length: BlogLength = BlogLength.MEDIUM
    
    # Keywords
    primary_keyword: str = ""
    secondary_keywords: list[str] = field(default_factory=list)
    
    # Brand
    brand_name: str = ""
    brand_voice: str = ""
    
    # CTA
    call_to_action: str = ""
    
    # Additional context
    reference_urls: list[str] = field(default_factory=list)
    key_points: list[str] = field(default_factory=list)
    avoid_topics: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "topic": self.topic,
            "description": self.description,
            "content_types": [ct.value for ct in self.content_types],
            "platforms": [p.value for p in self.platforms],
            "tone": self.tone.value,
            "target_audience": self.target_audience,
            "blog_length": self.blog_length.value,
            "primary_keyword": self.primary_keyword,
        }


@dataclass
class ResearchResult:
    """Research findings for content creation."""
    topic: str
    
    # Key findings
    key_facts: list[str] = field(default_factory=list)
    statistics: list[str] = field(default_factory=list)
    quotes: list[dict] = field(default_factory=list)
    
    # Sources
    sources: list[dict] = field(default_factory=list)
    
    # Trends
    trending_angles: list[str] = field(default_factory=list)
    competitor_content: list[dict] = field(default_factory=list)
    
    # SEO insights
    related_keywords: list[str] = field(default_factory=list)
    search_intent: str = ""
    
    # Audience insights
    audience_questions: list[str] = field(default_factory=list)
    pain_points: list[str] = field(default_factory=list)


@dataclass
class ContentOutline:
    """Content outline structure."""
    title: str
    hook: str = ""
    
    # Structure
    sections: list[dict] = field(default_factory=list)  # {heading, key_points, word_count}
    
    # Metadata
    estimated_word_count: int = 0
    estimated_read_time: int = 0
    
    # Headlines
    headline_options: list[str] = field(default_factory=list)
    
    # Key messages
    main_argument: str = ""
    supporting_points: list[str] = field(default_factory=list)
    
    def to_markdown(self) -> str:
        md = f"# {self.title}\n\n"
        md += f"**Hook:** {self.hook}\n\n"
        md += f"**Main Argument:** {self.main_argument}\n\n"
        
        for i, section in enumerate(self.sections, 1):
            md += f"## {i}. {section.get('heading', 'Section')}\n"
            for point in section.get('key_points', []):
                md += f"- {point}\n"
            md += "\n"
        
        return md


@dataclass
class BlogPost:
    """Blog post content."""
    title: str
    content: str  # Markdown or HTML
    
    # Meta
    meta_title: str = ""
    meta_description: str = ""
    excerpt: str = ""
    
    # SEO
    slug: str = ""
    primary_keyword: str = ""
    keywords: list[str] = field(default_factory=list)
    
    # Structure
    word_count: int = 0
    read_time: int = 0
    headings: list[str] = field(default_factory=list)
    
    # Media
    featured_image_prompt: str = ""
    image_prompts: list[str] = field(default_factory=list)
    
    # CTA
    call_to_action: str = ""
    
    # Formats
    html_content: str = ""
    wordpress_content: str = ""
    
    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "content": self.content,
            "meta_title": self.meta_title,
            "meta_description": self.meta_description,
            "word_count": self.word_count,
            "read_time": self.read_time,
            "slug": self.slug,
        }


@dataclass
class SocialPost:
    """Social media post content."""
    platform: Platform
    
    # Content
    text: str = ""
    hashtags: list[str] = field(default_factory=list)
    mentions: list[str] = field(default_factory=list)
    
    # Media
    image_prompt: str = ""
    video_description: str = ""
    
    # Engagement
    call_to_action: str = ""
    link: str = ""
    
    # Timing
    suggested_post_time: str = ""
    
    # Platform-specific
    character_count: int = 0
    thread_parts: list[str] = field(default_factory=list)  # For Twitter threads
    
    # A/B variants
    variants: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "platform": self.platform.value,
            "text": self.text,
            "hashtags": self.hashtags,
            "character_count": self.character_count,
            "image_prompt": self.image_prompt,
        }
    
    def format_post(self) -> str:
        """Format the complete post."""
        post = self.text
        
        if self.hashtags:
            post += "\n\n" + " ".join(f"#{h}" for h in self.hashtags)
        
        if self.link:
            post += f"\n\n{self.link}"
        
        return post


@dataclass
class VideoScript:
    """Video script content."""
    title: str
    platform: Platform  # YouTube, TikTok, etc.
    
    # Script
    hook: str = ""  # First 3 seconds
    script_sections: list[dict] = field(default_factory=list)  # {timestamp, narration, visual, b_roll}
    outro: str = ""
    
    # Full script
    full_script: str = ""
    
    # Duration
    estimated_duration: int = 0  # seconds
    
    # Production notes
    b_roll_suggestions: list[str] = field(default_factory=list)
    music_mood: str = ""
    transitions: list[str] = field(default_factory=list)
    
    # Thumbnails
    thumbnail_ideas: list[str] = field(default_factory=list)
    
    # YouTube specific
    video_title: str = ""
    video_description: str = ""
    tags: list[str] = field(default_factory=list)
    
    # Voice over
    voice_over_script: str = ""
    voice_over_markers: list[dict] = field(default_factory=list)  # {time, text, emphasis}
    
    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "platform": self.platform.value,
            "estimated_duration": self.estimated_duration,
            "hook": self.hook,
            "music_mood": self.music_mood,
        }
    
    def to_formatted_script(self) -> str:
        """Format script for reading."""
        script = f"# {self.title}\n\n"
        script += f"**Duration:** ~{self.estimated_duration} seconds\n"
        script += f"**Music Mood:** {self.music_mood}\n\n"
        script += "---\n\n"
        
        script += f"## HOOK (0:00-0:03)\n{self.hook}\n\n"
        
        for section in self.script_sections:
            time = section.get('timestamp', '')
            script += f"## {time}\n"
            script += f"**Narration:** {section.get('narration', '')}\n"
            script += f"**Visual:** {section.get('visual', '')}\n"
            if section.get('b_roll'):
                script += f"**B-Roll:** {section.get('b_roll', '')}\n"
            script += "\n"
        
        script += f"## OUTRO\n{self.outro}\n"
        
        return script


@dataclass
class Newsletter:
    """Newsletter/email content."""
    # Subject
    subject_line: str = ""
    subject_line_variants: list[str] = field(default_factory=list)
    preview_text: str = ""
    
    # Content
    greeting: str = ""
    body: str = ""
    sections: list[dict] = field(default_factory=list)
    
    # CTA
    primary_cta: str = ""
    cta_button_text: str = ""
    cta_url: str = ""
    
    # Footer
    footer: str = ""
    
    # Formats
    html_content: str = ""
    plain_text: str = ""
    
    def to_dict(self) -> dict:
        return {
            "subject_line": self.subject_line,
            "preview_text": self.preview_text,
            "primary_cta": self.primary_cta,
        }


@dataclass
class SEOAnalysis:
    """SEO optimization results."""
    # Keywords
    primary_keyword: str = ""
    keyword_density: float = 0.0
    secondary_keywords_used: list[str] = field(default_factory=list)
    
    # Meta
    title_score: int = 0
    meta_description_score: int = 0
    
    # Structure
    heading_structure_score: int = 0
    readability_score: int = 0
    
    # Recommendations
    improvements: list[str] = field(default_factory=list)
    
    # Overall
    overall_score: int = 0


@dataclass
class ContentPackage:
    """Complete content package from pipeline."""
    brief: ContentBrief
    created_at: str = ""
    
    # Research
    research: Optional[ResearchResult] = None
    
    # Outline
    outline: Optional[ContentOutline] = None
    
    # Content pieces
    blog_post: Optional[BlogPost] = None
    social_posts: list[SocialPost] = field(default_factory=list)
    video_script: Optional[VideoScript] = None
    newsletter: Optional[Newsletter] = None
    
    # SEO
    seo_analysis: Optional[SEOAnalysis] = None
    
    # Image prompts
    image_prompts: list[str] = field(default_factory=list)
    
    # Status
    status: str = "draft"  # draft, review, approved, published
    
    # Editor notes
    editor_notes: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "topic": self.brief.topic,
            "created_at": self.created_at,
            "status": self.status,
            "has_blog": self.blog_post is not None,
            "social_post_count": len(self.social_posts),
            "has_video": self.video_script is not None,
            "has_newsletter": self.newsletter is not None,
        }


# =============================================================================
# LangGraph State
# =============================================================================

from typing import TypedDict, Annotated
from operator import add


class PipelineState(TypedDict, total=False):
    """State for the content production pipeline."""
    
    # Input
    brief: dict
    
    # Research phase
    research: dict
    
    # Outline phase
    outline: dict
    
    # Content pieces
    blog_draft: str
    social_drafts: list[dict]
    video_draft: dict
    newsletter_draft: dict
    
    # Post-processing
    blog_seo: dict
    social_images: list[str]
    video_voiceover: dict
    
    # Final content
    blog_final: dict
    social_final: list[dict]
    video_final: dict
    newsletter_final: dict
    
    # Editor review
    editor_feedback: list[str]
    
    # Final package
    content_package: dict
    
    # Control flow
    content_types: list[str]
    current_step: str
    errors: list[str]
    
    # Messages for debugging
    messages: Annotated[list[str], add]
