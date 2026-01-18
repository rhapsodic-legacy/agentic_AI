# 🎬 Content Production Pipeline

An end-to-end content production system using **LangGraph** that creates blog posts, social media content, video scripts, and newsletters from a single topic or brief.

![LangGraph](https://img.shields.io/badge/Framework-LangGraph-blue)
![Architecture](https://img.shields.io/badge/Architecture-DAG-green)
![Complexity](https://img.shields.io/badge/Complexity-⭐⭐⭐⭐-yellow)

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🔍 **SEO Optimization** | Keyword research, meta optimization, content scoring |
| 📱 **Multi-Platform Adaptation** | Auto-adapt content for Twitter, LinkedIn, Instagram, TikTok |
| 🖼️ **AI Image Prompts** | Generate DALL-E/Midjourney prompts for each piece |
| 🎙️ **Voice-Over Scripts** | Timing markers and production notes for video |
| #️⃣ **Hashtag Generation** | Platform-optimized hashtag suggestions |
| ⏰ **Posting Times** | Optimal posting time recommendations |
| ✏️ **Editor Review** | Readability and quality analysis |
| 📊 **Content Analytics** | Word count, reading time, SEO scores |

## 🏗️ Pipeline Architecture (DAG)

```
                         ┌─────────────┐
                         │   BRIEF     │
                         │   INTAKE    │
                         └──────┬──────┘
                                │
                                ▼
                         ┌─────────────┐
                         │  RESEARCH   │
                         │   AGENT     │
                         └──────┬──────┘
                                │
                                ▼
                         ┌─────────────┐
                         │   OUTLINE   │
                         │   CREATOR   │
                         └──────┬──────┘
                                │
         ┌──────────────────────┼──────────────────────┐
         │                      │                      │
         ▼                      ▼                      ▼
  ┌─────────────┐       ┌─────────────┐       ┌─────────────┐
  │   BLOG      │       │   SOCIAL    │       │   VIDEO     │
  │   WRITER    │       │   WRITER    │       │   SCRIPT    │
  └──────┬──────┘       └──────┬──────┘       └──────┬──────┘
         │                     │                      │
         ▼                     ▼                      ▼
  ┌─────────────┐       ┌─────────────┐       ┌─────────────┐
  │   SEO       │       │   IMAGE     │       │   VOICE     │
  │   OPTIMIZER │       │   GENERATOR │       │   OVER      │
  └──────┬──────┘       └──────┬──────┘       └──────┬──────┘
         │                     │                      │
         └─────────────────────┼──────────────────────┘
                               │
                               ▼
                        ┌─────────────┐
                        │   EDITOR    │
                        │   (Review)  │
                        └──────┬──────┘
                               │
                               ▼
                        ┌─────────────┐
                        │  PUBLISHER  │
                        └─────────────┘
```

## 📦 Output Formats

```python
OUTPUT_FORMATS = {
    "blog": {
        "formats": ["markdown", "html", "wordpress"],
        "lengths": ["short (500w)", "medium (1500w)", "long (3000w)"],
    },
    "social": {
        "platforms": ["Twitter/X", "LinkedIn", "Instagram", "TikTok"],
        "includes": ["text", "hashtags", "image_prompt", "posting_time"],
    },
    "video": {
        "formats": ["youtube_script", "tiktok_script", "shorts_script"],
        "includes": ["script", "b_roll_suggestions", "music_mood"],
    },
    "newsletter": {
        "formats": ["html_email", "plain_text"],
        "includes": ["subject_lines", "preview_text", "cta"],
    },
}
```

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt

# Set API key for LLM
export GOOGLE_API_KEY="your-key"  # For Gemini
```

### CLI Usage

```bash
# Create all content types
python main.py create "AI in Marketing" --types blog,social,video

# Blog post only
python main.py blog "Getting Started with Python" --length long

# Social posts only
python main.py social "Product Launch" --platforms twitter,linkedin,instagram

# Video script
python main.py video "How to Build a Startup"

# Interactive mode
python main.py interactive

# Web server
python main.py serve
```

### Python API

```python
from content_pipeline import ContentPipeline, create_content

# Quick content creation
result = create_content(
    topic="AI in Marketing",
    content_types=["blog", "social", "video"],
    platforms=["twitter", "linkedin"],
)

# Access results
print(result["blog"]["content"])        # Blog post markdown
print(result["blog"]["meta_title"])     # SEO title
print(result["seo"]["overall_score"])   # SEO score

for post in result["social_posts"]:
    print(f"[{post['platform']}] {post['text']}")
    print(f"Hashtags: {post['hashtags']}")
    print(f"Post at: {post['suggested_post_time']}")

print(result["video"]["hook"])          # Video hook
print(result["video"]["script_sections"])

# Or use the full pipeline class
pipeline = ContentPipeline()
result = pipeline.run(
    topic="AI in Marketing",
    content_types=["blog", "social"],
    tone="professional",
    target_audience="marketers",
    blog_length="medium",
)
```

## 📁 Project Structure

```
content-pipeline/
├── content_pipeline/
│   ├── __init__.py           # Package exports
│   ├── pipeline.py           # Main ContentPipeline class
│   ├── nodes/
│   │   └── __init__.py       # 11 LangGraph nodes
│   ├── models/
│   │   └── __init__.py       # Content models & state
│   ├── tools/
│   │   └── __init__.py       # SEO, Social, Image tools
│   └── outputs/
├── api.py                     # FastAPI backend
├── frontend/
│   └── index.html            # React dashboard
├── main.py                    # CLI
├── requirements.txt
└── README.md
```

## 🔧 Pipeline Nodes

| Node | Purpose | Output |
|------|---------|--------|
| **Brief Intake** | Parse and validate input | Structured brief |
| **Research** | Gather topic information | Key facts, stats, questions |
| **Outline** | Create content structure | Sections with key points |
| **Blog Writer** | Write blog post | Markdown content |
| **Social Writer** | Create social posts | Platform-adapted posts |
| **Video Script** | Write video script | Script with timestamps |
| **SEO Optimizer** | Optimize for search | Meta tags, improvements |
| **Image Generator** | Create image prompts | DALL-E prompts |
| **Voice Over** | Format for recording | Timed script markers |
| **Editor** | Review and feedback | Quality analysis |
| **Publisher** | Package all content | Final content bundle |

## 🔍 SEO Features

```python
# Title optimization
seo_tool.optimize_title("My Title", "keyword")
# Returns: score, suggestions, optimized variants

# Meta description
seo_tool.optimize_meta_description(description, keyword)
# Returns: score, character count, suggestions

# Content analysis
seo_tool.analyze_content(content, keyword)
# Returns: word_count, keyword_density, heading_structure, improvements
```

## 📱 Social Media Features

```python
# Platform-specific hashtags
social_tool.generate_hashtags("AI Marketing", "twitter", count=5)

# Optimal posting times
social_tool.get_optimal_posting_time("linkedin")
# Returns: "7:00 AM - 8:00 AM, 12:00 PM (Tue-Thu)"

# Content adaptation
social_tool.adapt_content(long_text, "twitter")
# Returns: truncated text within character limit

# Twitter threads
social_tool.create_twitter_thread(long_content)
# Returns: list of tweets with thread markers
```

## 🎬 Video Script Features

```python
video_script = {
    "title": "Topic Explained",
    "hook": "Opening 3 seconds",
    "script_sections": [
        {
            "timestamp": "0:30 - 1:30",
            "narration": "What to say",
            "visual": "What to show",
            "b_roll": "Supporting footage",
        }
    ],
    "music_mood": "upbeat, motivational",
    "thumbnail_ideas": ["Idea 1", "Idea 2"],
}
```

## ⚙️ Configuration

```python
from content_pipeline import ContentPipeline, PipelineConfig

config = PipelineConfig(
    llm_provider="gemini",     # "gemini", "anthropic", "openai"
    llm_model=None,            # Uses provider default
    generate_blog=True,
    generate_social=True,
    generate_video=False,
    run_seo=True,
    run_editor_review=True,
    verbose=True,
)

pipeline = ContentPipeline(config)
```

## 🌐 Web API

```bash
python main.py serve
```

Endpoints:
- `POST /api/create` - Create content package (async)
- `GET /api/job/{job_id}` - Get job status
- `POST /api/blog` - Create blog only
- `POST /api/social` - Create social only
- `POST /api/video` - Create video only
- `GET /api/pipeline` - Get pipeline structure

## 📊 Content Package Output

```json
{
  "topic": "AI in Marketing",
  "created_at": "2024-01-15T10:30:00",
  "status": "ready",
  
  "blog": {
    "title": "The Complete Guide to AI in Marketing",
    "content": "# The Complete Guide...",
    "meta_title": "AI in Marketing: Complete Guide 2024",
    "meta_description": "Learn how AI is transforming...",
    "word_count": 1500,
    "slug": "ai-in-marketing-guide"
  },
  
  "seo": {
    "overall_score": 85,
    "keyword_density": 1.8,
    "improvements": ["Add more H2 headings"]
  },
  
  "social_posts": [
    {
      "platform": "twitter",
      "text": "🚀 AI is revolutionizing marketing...",
      "hashtags": ["AI", "Marketing", "Tech"],
      "character_count": 245,
      "suggested_post_time": "9:00 AM - 12:00 PM"
    }
  ],
  
  "video": {
    "title": "AI in Marketing Explained",
    "hook": "Want to know how AI can 10x your marketing?",
    "estimated_duration": 300,
    "music_mood": "upbeat, modern"
  },
  
  "image_prompts": [
    {
      "type": "featured",
      "prompt": "Create a featured image...",
      "use": "Blog header"
    }
  ],
  
  "summary": {
    "blog_word_count": 1500,
    "social_post_count": 4,
    "video_duration": 300,
    "seo_score": 85
  }
}
```

## 📝 License

MIT License
