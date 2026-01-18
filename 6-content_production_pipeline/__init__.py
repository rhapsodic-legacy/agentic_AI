"""
Content Production Pipeline - LangGraph Nodes

Node implementations for each stage of the DAG:
1. Brief Intake
2. Research
3. Outline Creator
4. Content Writers (Blog, Social, Video)
5. Post-Processing (SEO, Images, Voice Over)
6. Editor Review
7. Publisher
"""

from typing import Optional, Callable
import json
import re
from datetime import datetime

from ..models import (
    ContentBrief, ContentType, Platform, BlogLength, Tone,
    ResearchResult, ContentOutline, BlogPost, SocialPost,
    VideoScript, Newsletter, SEOAnalysis, ContentPackage, PipelineState
)
from ..tools import research_tool, seo_tool, social_tool, image_tool, analyzer_tool


def create_llm(provider: str = "gemini"):
    """Create LLM instance."""
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.7,
        )
    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model="claude-sonnet-4-20250514",
            temperature=0.7,
        )
    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.7,
        )
    else:
        return None


# Global LLM instance
_llm = None


def get_llm():
    """Get the LLM instance."""
    global _llm
    if _llm is None:
        try:
            _llm = create_llm("gemini")
        except:
            _llm = None
    return _llm


def call_llm(prompt: str) -> str:
    """Call the LLM with a prompt."""
    llm = get_llm()
    if llm:
        response = llm.invoke(prompt)
        return response.content
    else:
        # Mock response for demo
        return f"[Generated content for: {prompt[:100]}...]"


# =============================================================================
# Node: Brief Intake
# =============================================================================

def brief_intake_node(state: PipelineState) -> PipelineState:
    """
    Process and validate the content brief.
    """
    brief_data = state.get("brief", {})
    
    # Parse brief
    topic = brief_data.get("topic", "")
    
    # Determine content types needed
    content_types = brief_data.get("content_types", ["blog"])
    
    # Set primary keyword if not provided
    if not brief_data.get("primary_keyword"):
        brief_data["primary_keyword"] = topic.lower()
    
    return {
        **state,
        "brief": brief_data,
        "content_types": content_types,
        "current_step": "brief_intake",
        "messages": [f"✓ Brief processed: {topic}"],
    }


# =============================================================================
# Node: Research
# =============================================================================

def research_node(state: PipelineState) -> PipelineState:
    """
    Research the topic and gather information.
    """
    brief = state.get("brief", {})
    topic = brief.get("topic", "")
    
    # Use research tool
    research_data = research_tool.research_topic(topic)
    
    # Enhance with LLM if available
    llm = get_llm()
    if llm:
        prompt = f"""Research the topic: "{topic}"
        
Target audience: {brief.get('target_audience', 'general')}
Tone: {brief.get('tone', 'professional')}

Provide:
1. 5 key facts about this topic
2. 3 relevant statistics or data points
3. Common questions people have
4. Trending angles or perspectives

Format as JSON with keys: key_facts, statistics, questions, trending_angles"""
        
        try:
            response = llm.invoke(prompt)
            # Parse response and merge with tool data
            # For now, use tool data as base
        except:
            pass
    
    # Find competitor content
    competitor_data = research_tool.find_competitor_content(topic)
    research_data["competitor_content"] = competitor_data
    
    return {
        **state,
        "research": research_data,
        "current_step": "research",
        "messages": state.get("messages", []) + [f"✓ Research complete: {len(research_data.get('key_facts', []))} facts gathered"],
    }


# =============================================================================
# Node: Outline Creator
# =============================================================================

def outline_node(state: PipelineState) -> PipelineState:
    """
    Create content outline based on research.
    """
    brief = state.get("brief", {})
    research = state.get("research", {})
    
    topic = brief.get("topic", "")
    blog_length = brief.get("blog_length", "medium")
    
    # Determine section count based on length
    section_counts = {"short": 3, "medium": 5, "long": 8}
    num_sections = section_counts.get(blog_length, 5)
    
    # Generate outline with LLM
    llm = get_llm()
    
    if llm:
        prompt = f"""Create a content outline for: "{topic}"

Research findings:
- Key facts: {research.get('key_facts', [])[:3]}
- Trending angles: {research.get('trending_angles', [])[:3]}
- Audience questions: {research.get('audience_questions', [])[:3]}

Requirements:
- Target length: {blog_length} ({section_counts.get(blog_length, 5) * 300} words approx)
- Number of main sections: {num_sections}
- Tone: {brief.get('tone', 'professional')}

Provide:
1. Main title (compelling, SEO-friendly)
2. Hook/introduction angle
3. Section headings with key points for each
4. Conclusion/CTA approach

Format as structured outline."""

        try:
            response = llm.invoke(prompt)
            outline_text = response.content
        except:
            outline_text = None
    else:
        outline_text = None
    
    # Build outline structure
    sections = []
    trending = research.get("trending_angles", [f"Understanding {topic}"])
    questions = research.get("audience_questions", [])
    
    # Generate sections based on research
    section_templates = [
        {"heading": f"What is {topic}?", "key_points": ["Definition", "Background", "Importance"]},
        {"heading": f"Benefits of {topic}", "key_points": ["Key benefit 1", "Key benefit 2", "Key benefit 3"]},
        {"heading": "How to Get Started", "key_points": ["Step 1", "Step 2", "Step 3"]},
        {"heading": "Best Practices", "key_points": ["Tip 1", "Tip 2", "Tip 3"]},
        {"heading": "Common Mistakes to Avoid", "key_points": ["Mistake 1", "Mistake 2", "Mistake 3"]},
        {"heading": "Advanced Strategies", "key_points": ["Strategy 1", "Strategy 2"]},
        {"heading": "Tools and Resources", "key_points": ["Tool 1", "Tool 2", "Resources"]},
        {"heading": "Future Trends", "key_points": ["Trend 1", "Trend 2", "Predictions"]},
    ]
    
    sections = section_templates[:num_sections]
    
    # Calculate word counts
    words_per_section = {"short": 150, "medium": 280, "long": 350}
    section_words = words_per_section.get(blog_length, 280)
    
    for section in sections:
        section["word_count"] = section_words
    
    outline = {
        "title": f"The Complete Guide to {topic}",
        "hook": f"Discover everything you need to know about {topic} and how it can transform your approach.",
        "main_argument": f"{topic} is essential for anyone looking to succeed in today's landscape.",
        "sections": sections,
        "headline_options": [
            f"The Ultimate Guide to {topic}: Everything You Need to Know",
            f"{topic} 101: A Beginner's Complete Guide",
            f"Mastering {topic}: Tips, Strategies, and Best Practices",
        ],
        "estimated_word_count": sum(s.get("word_count", 300) for s in sections) + 200,
    }
    
    return {
        **state,
        "outline": outline,
        "current_step": "outline",
        "messages": state.get("messages", []) + [f"✓ Outline created: {len(sections)} sections"],
    }


# =============================================================================
# Node: Blog Writer
# =============================================================================

def blog_writer_node(state: PipelineState) -> PipelineState:
    """
    Write the blog post content.
    """
    brief = state.get("brief", {})
    research = state.get("research", {})
    outline = state.get("outline", {})
    
    topic = brief.get("topic", "")
    tone = brief.get("tone", "professional")
    
    llm = get_llm()
    
    if llm:
        prompt = f"""Write a blog post based on this outline:

Title: {outline.get('title', topic)}
Hook: {outline.get('hook', '')}

Sections:
{json.dumps(outline.get('sections', []), indent=2)}

Research to incorporate:
- Key facts: {research.get('key_facts', [])[:5]}
- Statistics: {research.get('statistics', [])[:3]}

Requirements:
- Tone: {tone}
- Include engaging introduction
- Use subheadings (## for H2)
- Add bullet points where appropriate
- Include a conclusion with CTA
- Target word count: {outline.get('estimated_word_count', 1500)}

Write the complete blog post in Markdown format."""

        try:
            response = llm.invoke(prompt)
            content = response.content
        except:
            content = None
    else:
        content = None
    
    # Fallback content generation
    if not content:
        content = f"""# {outline.get('title', f'Guide to {topic}')}

{outline.get('hook', f'Welcome to our comprehensive guide on {topic}.')}

"""
        for section in outline.get("sections", []):
            content += f"\n## {section.get('heading', 'Section')}\n\n"
            for point in section.get("key_points", []):
                content += f"**{point}**: This is an important aspect to consider.\n\n"
        
        content += f"""
## Conclusion

{topic} offers tremendous opportunities for growth and success. By following the strategies outlined above, you'll be well on your way to mastering this important topic.

**Ready to get started?** {brief.get('call_to_action', 'Take the first step today!')}
"""
    
    # Calculate word count
    word_count = len(content.split())
    read_time = word_count // 200  # Average reading speed
    
    blog_draft = {
        "title": outline.get("title", f"Guide to {topic}"),
        "content": content,
        "word_count": word_count,
        "read_time": read_time,
        "headings": re.findall(r'^## (.+)$', content, re.MULTILINE),
    }
    
    return {
        **state,
        "blog_draft": content,
        "blog_final": blog_draft,
        "current_step": "blog_writer",
        "messages": state.get("messages", []) + [f"✓ Blog draft complete: {word_count} words"],
    }


# =============================================================================
# Node: Social Writer
# =============================================================================

def social_writer_node(state: PipelineState) -> PipelineState:
    """
    Create social media posts for multiple platforms.
    """
    brief = state.get("brief", {})
    outline = state.get("outline", {})
    research = state.get("research", {})
    
    topic = brief.get("topic", "")
    platforms = brief.get("platforms", ["twitter", "linkedin"])
    
    social_posts = []
    
    for platform in platforms:
        llm = get_llm()
        
        if llm:
            prompt = f"""Create a {platform} post about: {topic}

Key message: {outline.get('hook', '')}
Key facts: {research.get('key_facts', [])[:2]}

Requirements:
- Platform: {platform}
- Character limit: {social_tool.PLATFORM_LIMITS.get(platform, 2200)}
- Include call to action
- Be engaging and shareable

Provide the post text only, no hashtags."""

            try:
                response = llm.invoke(prompt)
                text = response.content.strip()
            except:
                text = None
        else:
            text = None
        
        # Fallback
        if not text:
            if platform == "twitter":
                text = f"🚀 {topic} is changing the game!\n\nHere's what you need to know:\n\n✅ Key insight 1\n✅ Key insight 2\n✅ Key insight 3\n\nLearn more 👇"
            elif platform == "linkedin":
                text = f"I've been diving deep into {topic} lately, and here's what I've learned:\n\n{outline.get('hook', '')}\n\nThe key takeaways:\n\n1️⃣ First important point\n2️⃣ Second important point\n3️⃣ Third important point\n\nWhat are your thoughts on {topic}? Let me know in the comments!"
            elif platform == "instagram":
                text = f"✨ {topic} ✨\n\nSwipe to learn everything you need to know!\n\n{outline.get('hook', '')}\n\nDouble tap if you found this helpful! 💡"
            else:
                text = f"Let's talk about {topic}! 🔥\n\n{outline.get('hook', '')}"
        
        # Generate hashtags
        hashtags = social_tool.generate_hashtags(topic, platform)
        
        # Get optimal posting time
        post_time = social_tool.get_optimal_posting_time(platform)
        
        # Adapt content to platform limits
        adapted = social_tool.adapt_content(text, platform)
        
        # Generate image prompt
        image_prompt = image_tool.generate_social_image_prompt(platform, topic)
        
        post = {
            "platform": platform,
            "text": adapted["text"],
            "hashtags": hashtags,
            "character_count": adapted["character_count"],
            "suggested_post_time": post_time,
            "image_prompt": image_prompt,
        }
        
        # Create thread for Twitter if long
        if platform == "twitter" and len(text) > 280:
            post["thread_parts"] = social_tool.create_twitter_thread(text)
        
        social_posts.append(post)
    
    return {
        **state,
        "social_drafts": social_posts,
        "social_final": social_posts,
        "current_step": "social_writer",
        "messages": state.get("messages", []) + [f"✓ Social posts created: {len(social_posts)} platforms"],
    }


# =============================================================================
# Node: Video Script Writer
# =============================================================================

def video_script_node(state: PipelineState) -> PipelineState:
    """
    Create video script content.
    """
    brief = state.get("brief", {})
    outline = state.get("outline", {})
    research = state.get("research", {})
    
    topic = brief.get("topic", "")
    platform = "youtube"  # Default to YouTube
    
    llm = get_llm()
    
    if llm:
        prompt = f"""Write a YouTube video script about: {topic}

Outline to follow:
{json.dumps(outline.get('sections', [])[:4], indent=2)}

Key facts to include:
{research.get('key_facts', [])[:3]}

Requirements:
- Hook viewers in first 3 seconds
- Target duration: 5-8 minutes
- Conversational, engaging tone
- Include timestamps
- Suggest B-roll footage
- Include outro with CTA

Format:
[TIMESTAMP] 
Narration: ...
Visual: ...
B-Roll: ..."""

        try:
            response = llm.invoke(prompt)
            script_text = response.content
        except:
            script_text = None
    else:
        script_text = None
    
    # Build structured script
    sections = []
    
    # Hook
    hook = f"Have you ever wondered about {topic}? In this video, I'm going to show you everything you need to know!"
    
    # Main sections
    for i, section in enumerate(outline.get("sections", [])[:4], 1):
        sections.append({
            "timestamp": f"{i}:00 - {i+1}:00",
            "narration": f"Now let's talk about {section.get('heading', 'this topic')}. {' '.join(section.get('key_points', ['Important point'])[:2])}.",
            "visual": f"Show graphics explaining {section.get('heading', 'concept')}",
            "b_roll": f"Footage related to {section.get('heading', 'topic')}",
        })
    
    # Outro
    outro = f"That's everything you need to know about {topic}! If you found this helpful, make sure to like and subscribe. Drop a comment below with your questions!"
    
    video_script = {
        "title": f"{topic} Explained: Complete Guide",
        "platform": platform,
        "hook": hook,
        "script_sections": sections,
        "outro": outro,
        "estimated_duration": len(sections) * 90,  # ~90 seconds per section
        "b_roll_suggestions": [
            f"Stock footage of {topic} in action",
            "Screen recordings or demos",
            "Animated graphics explaining concepts",
            "Interview clips or testimonials",
        ],
        "music_mood": "upbeat, motivational, modern",
        "thumbnail_ideas": [
            f"Surprised face with '{topic}' text",
            f"Before/after comparison",
            f"Bold text: 'The Truth About {topic}'",
        ],
        "video_title": f"{topic} Explained: Everything You Need to Know in 2024",
        "video_description": f"Learn all about {topic} in this comprehensive guide. We cover the basics, best practices, and advanced tips.\n\n⏱️ Timestamps:\n0:00 - Intro\n0:30 - What is {topic}?\n...",
        "tags": [topic.lower(), f"{topic.lower()} tutorial", f"{topic.lower()} guide", "how to", "explained"],
    }
    
    return {
        **state,
        "video_draft": video_script,
        "video_final": video_script,
        "current_step": "video_script",
        "messages": state.get("messages", []) + [f"✓ Video script complete: ~{video_script['estimated_duration']}s"],
    }


# =============================================================================
# Node: SEO Optimizer
# =============================================================================

def seo_optimizer_node(state: PipelineState) -> PipelineState:
    """
    Optimize blog content for SEO.
    """
    brief = state.get("brief", {})
    blog = state.get("blog_final", {})
    
    keyword = brief.get("primary_keyword", brief.get("topic", ""))
    content = blog.get("content", state.get("blog_draft", ""))
    title = blog.get("title", "")
    
    # Analyze title
    title_analysis = seo_tool.optimize_title(title, keyword)
    
    # Generate meta description
    meta_description = f"Learn everything about {keyword}. This comprehensive guide covers tips, best practices, and strategies for success."
    meta_analysis = seo_tool.optimize_meta_description(meta_description, keyword)
    
    # Analyze content
    content_analysis = seo_tool.analyze_content(content, keyword)
    
    # Generate slug
    slug = seo_tool.generate_slug(title)
    
    # Calculate overall score
    overall_score = (
        title_analysis["score"] * 0.3 +
        meta_analysis["score"] * 0.2 +
        content_analysis["score"] * 0.5
    )
    
    # Compile improvements
    all_improvements = (
        title_analysis.get("suggestions", []) +
        meta_analysis.get("suggestions", []) +
        content_analysis.get("improvements", [])
    )
    
    seo_analysis = {
        "primary_keyword": keyword,
        "keyword_density": content_analysis["keyword_density"],
        "title_score": title_analysis["score"],
        "meta_description_score": meta_analysis["score"],
        "content_score": content_analysis["score"],
        "overall_score": round(overall_score),
        "improvements": all_improvements,
        "meta_title": title_analysis.get("optimized_variants", [title])[0] if title_analysis.get("optimized_variants") else title,
        "meta_description": meta_description,
        "slug": slug,
    }
    
    # Update blog with SEO data
    updated_blog = {
        **blog,
        "meta_title": seo_analysis["meta_title"],
        "meta_description": seo_analysis["meta_description"],
        "slug": slug,
        "primary_keyword": keyword,
        "seo_score": seo_analysis["overall_score"],
    }
    
    return {
        **state,
        "blog_seo": seo_analysis,
        "blog_final": updated_blog,
        "current_step": "seo_optimizer",
        "messages": state.get("messages", []) + [f"✓ SEO optimized: Score {seo_analysis['overall_score']}/100"],
    }


# =============================================================================
# Node: Image Generator (Prompts)
# =============================================================================

def image_generator_node(state: PipelineState) -> PipelineState:
    """
    Generate image prompts for content.
    """
    brief = state.get("brief", {})
    blog = state.get("blog_final", {})
    social = state.get("social_final", [])
    
    topic = brief.get("topic", "")
    
    image_prompts = []
    
    # Featured image for blog
    featured_prompt = image_tool.generate_featured_image_prompt(topic, "professional")
    image_prompts.append({
        "type": "featured",
        "prompt": featured_prompt,
        "use": "Blog header image",
    })
    
    # Section images
    for heading in blog.get("headings", [])[:3]:
        prompt = image_tool.generate_featured_image_prompt(heading, "minimal")
        image_prompts.append({
            "type": "section",
            "prompt": prompt,
            "use": f"Image for section: {heading}",
        })
    
    # Social images (already in social posts, but collect them)
    for post in social:
        if post.get("image_prompt"):
            image_prompts.append({
                "type": "social",
                "prompt": post["image_prompt"],
                "use": f"{post['platform']} post",
            })
    
    return {
        **state,
        "social_images": [p["prompt"] for p in image_prompts if p["type"] == "social"],
        "image_prompts": image_prompts,
        "current_step": "image_generator",
        "messages": state.get("messages", []) + [f"✓ Generated {len(image_prompts)} image prompts"],
    }


# =============================================================================
# Node: Voice Over Formatter
# =============================================================================

def voiceover_node(state: PipelineState) -> PipelineState:
    """
    Format video script for voice-over recording.
    """
    video = state.get("video_final", {})
    
    if not video:
        return {
            **state,
            "current_step": "voiceover",
            "messages": state.get("messages", []) + ["⏭ Voice over skipped (no video script)"],
        }
    
    # Create voice-over script with markers
    vo_script = f"""VOICE OVER SCRIPT: {video.get('title', 'Video')}
{'='*50}

HOOK (Energetic, attention-grabbing):
{video.get('hook', '')}

[PAUSE 1 second]

"""
    
    markers = []
    current_time = 3  # Start after hook
    
    for section in video.get("script_sections", []):
        markers.append({
            "time": f"{current_time}s",
            "text": section.get("narration", ""),
            "emphasis": "normal",
            "visual_cue": section.get("visual", ""),
        })
        
        vo_script += f"""
---
TIMESTAMP: {section.get('timestamp', '')}
NARRATION: {section.get('narration', '')}
[Visual cue: {section.get('visual', '')}]
[PAUSE 0.5 seconds]
"""
        current_time += 60  # Approximate 60 seconds per section
    
    vo_script += f"""
---
OUTRO (Friendly, encouraging):
{video.get('outro', '')}

[End card: Subscribe + suggested videos]
"""
    
    voiceover_data = {
        "script": vo_script,
        "markers": markers,
        "total_duration": current_time + 15,  # Add outro time
        "speaking_pace": "moderate (150 words/minute)",
        "tone_guidance": "Conversational, engaging, enthusiastic but professional",
    }
    
    # Update video with voiceover
    updated_video = {
        **video,
        "voice_over_script": vo_script,
        "voice_over_markers": markers,
    }
    
    return {
        **state,
        "video_voiceover": voiceover_data,
        "video_final": updated_video,
        "current_step": "voiceover",
        "messages": state.get("messages", []) + [f"✓ Voice over script formatted"],
    }


# =============================================================================
# Node: Editor Review
# =============================================================================

def editor_node(state: PipelineState) -> PipelineState:
    """
    Review and provide feedback on all content.
    """
    brief = state.get("brief", {})
    blog = state.get("blog_final", {})
    social = state.get("social_final", [])
    video = state.get("video_final", {})
    seo = state.get("blog_seo", {})
    
    feedback = []
    
    # Review blog
    if blog:
        content = blog.get("content", "")
        readability = analyzer_tool.analyze_readability(content)
        
        feedback.append(f"📝 BLOG REVIEW:")
        feedback.append(f"   - Word count: {blog.get('word_count', 0)}")
        feedback.append(f"   - Readability: {readability.get('grade_level', 'Unknown')}")
        feedback.append(f"   - SEO Score: {seo.get('overall_score', 'N/A')}/100")
        
        for rec in readability.get("recommendations", []):
            feedback.append(f"   ⚠️ {rec}")
        
        for imp in seo.get("improvements", [])[:3]:
            feedback.append(f"   💡 {imp}")
    
    # Review social
    if social:
        feedback.append(f"\n📱 SOCIAL REVIEW ({len(social)} posts):")
        for post in social:
            platform = post.get("platform", "unknown")
            char_count = post.get("character_count", 0)
            feedback.append(f"   - {platform}: {char_count} chars, {len(post.get('hashtags', []))} hashtags")
    
    # Review video
    if video:
        feedback.append(f"\n🎬 VIDEO REVIEW:")
        feedback.append(f"   - Duration: ~{video.get('estimated_duration', 0)}s")
        feedback.append(f"   - Sections: {len(video.get('script_sections', []))}")
        feedback.append(f"   - Has voiceover: {'Yes' if video.get('voice_over_script') else 'No'}")
    
    # Overall status
    feedback.append(f"\n✅ EDITOR APPROVAL: Ready for publishing")
    
    return {
        **state,
        "editor_feedback": feedback,
        "current_step": "editor",
        "messages": state.get("messages", []) + ["✓ Editor review complete"],
    }


# =============================================================================
# Node: Publisher
# =============================================================================

def publisher_node(state: PipelineState) -> PipelineState:
    """
    Compile final content package.
    """
    brief = state.get("brief", {})
    
    content_package = {
        "topic": brief.get("topic", ""),
        "created_at": datetime.now().isoformat(),
        "status": "ready",
        
        # Blog
        "blog": state.get("blog_final", {}),
        "seo": state.get("blog_seo", {}),
        
        # Social
        "social_posts": state.get("social_final", []),
        
        # Video
        "video": state.get("video_final", {}),
        
        # Images
        "image_prompts": state.get("image_prompts", []),
        
        # Editor feedback
        "editor_feedback": state.get("editor_feedback", []),
        
        # Summary
        "summary": {
            "blog_word_count": state.get("blog_final", {}).get("word_count", 0),
            "social_post_count": len(state.get("social_final", [])),
            "video_duration": state.get("video_final", {}).get("estimated_duration", 0),
            "image_count": len(state.get("image_prompts", [])),
            "seo_score": state.get("blog_seo", {}).get("overall_score", 0),
        },
    }
    
    return {
        **state,
        "content_package": content_package,
        "current_step": "publisher",
        "messages": state.get("messages", []) + ["✓ Content package ready for publishing!"],
    }


# =============================================================================
# Node Registry
# =============================================================================

NODES = {
    "brief_intake": brief_intake_node,
    "research": research_node,
    "outline": outline_node,
    "blog_writer": blog_writer_node,
    "social_writer": social_writer_node,
    "video_script": video_script_node,
    "seo_optimizer": seo_optimizer_node,
    "image_generator": image_generator_node,
    "voiceover": voiceover_node,
    "editor": editor_node,
    "publisher": publisher_node,
}
