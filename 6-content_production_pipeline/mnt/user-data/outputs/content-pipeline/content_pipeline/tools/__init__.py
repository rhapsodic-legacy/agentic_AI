"""
Content Production Pipeline - Tools

Tools for:
- Topic research
- SEO optimization
- Hashtag generation
- Image prompt creation
- Content analysis
"""

from typing import Optional
import re
from datetime import datetime
import random


class ResearchTool:
    """
    Research tool for gathering information on topics.
    """
    
    def research_topic(self, topic: str, depth: str = "medium") -> dict:
        """
        Research a topic and gather relevant information.
        
        In production, this would call search APIs, news APIs, etc.
        """
        # Mock research results for demo
        return {
            "key_facts": [
                f"{topic} is a rapidly evolving field with significant recent developments",
                "Industry experts predict continued growth in this area",
                "Recent studies show increasing consumer interest",
                "Technology advances are driving innovation",
                "Market trends indicate strong future potential",
            ],
            "statistics": [
                "73% of professionals consider this topic highly relevant",
                "The market has grown 45% year-over-year",
                "Investment in this sector reached $50B in 2024",
                "Over 80% of companies are exploring solutions in this space",
            ],
            "quotes": [
                {
                    "text": "This represents a fundamental shift in how we approach the problem.",
                    "source": "Industry Expert",
                },
                {
                    "text": "We're seeing unprecedented levels of adoption.",
                    "source": "Market Analyst",
                },
            ],
            "trending_angles": [
                f"The future of {topic}",
                f"How {topic} is changing industries",
                f"Beginner's guide to {topic}",
                f"{topic} best practices for 2024",
                f"Common mistakes with {topic}",
            ],
            "audience_questions": [
                f"What is {topic}?",
                f"How do I get started with {topic}?",
                f"What are the benefits of {topic}?",
                f"How much does {topic} cost?",
                f"Is {topic} right for my business?",
            ],
            "related_keywords": [
                topic.lower(),
                f"{topic.lower()} guide",
                f"{topic.lower()} tips",
                f"{topic.lower()} tutorial",
                f"best {topic.lower()}",
                f"{topic.lower()} for beginners",
            ],
            "search_intent": "informational",
        }
    
    def find_competitor_content(self, topic: str) -> list[dict]:
        """Find existing content on the topic."""
        return [
            {
                "title": f"Complete Guide to {topic}",
                "url": "https://example.com/guide",
                "word_count": 2500,
                "key_points": ["Comprehensive coverage", "Includes examples"],
            },
            {
                "title": f"10 Tips for {topic} Success",
                "url": "https://example.com/tips",
                "word_count": 1200,
                "key_points": ["List format", "Actionable advice"],
            },
        ]


class SEOTool:
    """
    SEO optimization tool.
    """
    
    def analyze_keywords(self, keyword: str) -> dict:
        """Analyze keyword potential."""
        return {
            "keyword": keyword,
            "search_volume": random.randint(1000, 50000),
            "difficulty": random.randint(20, 80),
            "cpc": round(random.uniform(0.5, 5.0), 2),
            "related_keywords": [
                f"{keyword} guide",
                f"best {keyword}",
                f"{keyword} tips",
                f"{keyword} tutorial",
                f"how to {keyword}",
            ],
            "questions": [
                f"What is {keyword}?",
                f"How does {keyword} work?",
                f"Why use {keyword}?",
            ],
        }
    
    def optimize_title(self, title: str, keyword: str) -> dict:
        """Optimize a title for SEO."""
        title_lower = title.lower()
        keyword_lower = keyword.lower()
        
        score = 50
        suggestions = []
        
        # Check keyword presence
        if keyword_lower in title_lower:
            score += 20
        else:
            suggestions.append(f"Include primary keyword '{keyword}' in title")
        
        # Check length
        if 50 <= len(title) <= 60:
            score += 15
        elif len(title) < 50:
            suggestions.append("Title is too short. Aim for 50-60 characters.")
        else:
            suggestions.append("Title is too long. Keep under 60 characters.")
        
        # Check for power words
        power_words = ["ultimate", "complete", "guide", "best", "top", "essential", "proven"]
        if any(pw in title_lower for pw in power_words):
            score += 10
        else:
            suggestions.append("Consider adding power words like 'Ultimate', 'Complete', or 'Essential'")
        
        # Check for numbers
        if any(char.isdigit() for char in title):
            score += 5
        else:
            suggestions.append("Consider adding numbers (e.g., '10 Tips', '5 Ways')")
        
        return {
            "original": title,
            "score": min(score, 100),
            "suggestions": suggestions,
            "optimized_variants": [
                f"The Ultimate Guide to {keyword}: Everything You Need to Know",
                f"{keyword}: A Complete Guide for Beginners",
                f"10 {keyword} Tips That Actually Work in 2024",
            ],
        }
    
    def optimize_meta_description(self, description: str, keyword: str) -> dict:
        """Optimize meta description."""
        score = 50
        suggestions = []
        
        if keyword.lower() in description.lower():
            score += 20
        else:
            suggestions.append("Include primary keyword in meta description")
        
        if 150 <= len(description) <= 160:
            score += 20
        elif len(description) < 150:
            suggestions.append("Meta description is too short. Aim for 150-160 characters.")
        else:
            suggestions.append("Meta description is too long. Keep under 160 characters.")
        
        # Check for CTA
        cta_words = ["learn", "discover", "find out", "get", "start", "try"]
        if any(cta in description.lower() for cta in cta_words):
            score += 10
        else:
            suggestions.append("Add a call-to-action like 'Learn more' or 'Discover'")
        
        return {
            "original": description,
            "score": min(score, 100),
            "character_count": len(description),
            "suggestions": suggestions,
        }
    
    def analyze_content(self, content: str, keyword: str) -> dict:
        """Analyze content for SEO."""
        word_count = len(content.split())
        keyword_count = content.lower().count(keyword.lower())
        keyword_density = (keyword_count / word_count * 100) if word_count > 0 else 0
        
        # Count headings
        h2_count = len(re.findall(r'^## ', content, re.MULTILINE))
        h3_count = len(re.findall(r'^### ', content, re.MULTILINE))
        
        score = 50
        improvements = []
        
        # Keyword density check
        if 1 <= keyword_density <= 2.5:
            score += 15
        elif keyword_density < 1:
            improvements.append("Increase keyword usage slightly")
        else:
            improvements.append("Reduce keyword stuffing - aim for 1-2.5% density")
        
        # Word count
        if word_count >= 1500:
            score += 15
        else:
            improvements.append("Consider expanding content to 1500+ words")
        
        # Headings
        if h2_count >= 3:
            score += 10
        else:
            improvements.append("Add more H2 subheadings for better structure")
        
        # Paragraphs
        paragraphs = content.split('\n\n')
        long_paragraphs = sum(1 for p in paragraphs if len(p.split()) > 150)
        if long_paragraphs > 0:
            improvements.append("Break up long paragraphs for better readability")
        
        return {
            "word_count": word_count,
            "keyword_count": keyword_count,
            "keyword_density": round(keyword_density, 2),
            "h2_count": h2_count,
            "h3_count": h3_count,
            "score": min(score, 100),
            "improvements": improvements,
        }
    
    def generate_slug(self, title: str) -> str:
        """Generate URL slug from title."""
        slug = title.lower()
        slug = re.sub(r'[^a-z0-9\s-]', '', slug)
        slug = re.sub(r'[\s_]+', '-', slug)
        slug = re.sub(r'-+', '-', slug)
        slug = slug.strip('-')
        return slug[:60]


class SocialMediaTool:
    """
    Social media content optimization tool.
    """
    
    PLATFORM_LIMITS = {
        "twitter": 280,
        "linkedin": 3000,
        "instagram": 2200,
        "tiktok": 2200,
    }
    
    OPTIMAL_HASHTAG_COUNTS = {
        "twitter": 3,
        "linkedin": 5,
        "instagram": 15,
        "tiktok": 5,
    }
    
    def generate_hashtags(self, topic: str, platform: str, count: int = None) -> list[str]:
        """Generate relevant hashtags."""
        if count is None:
            count = self.OPTIMAL_HASHTAG_COUNTS.get(platform, 5)
        
        # Generate hashtags based on topic
        base_hashtags = [
            topic.replace(" ", ""),
            f"{topic.replace(' ', '')}Tips",
            f"{topic.replace(' ', '')}Guide",
            topic.split()[0] if " " in topic else topic,
        ]
        
        # Add common engagement hashtags
        engagement_hashtags = {
            "twitter": ["trending", "tips", "howto", "mustread"],
            "linkedin": ["business", "leadership", "growth", "innovation", "career"],
            "instagram": ["instagood", "photooftheday", "trending", "viral", "explore"],
            "tiktok": ["fyp", "foryou", "viral", "trending", "learnontiktok"],
        }
        
        platform_tags = engagement_hashtags.get(platform, [])
        all_tags = base_hashtags + platform_tags
        
        return [tag.lower() for tag in all_tags[:count]]
    
    def get_optimal_posting_time(self, platform: str) -> str:
        """Get optimal posting time for platform."""
        optimal_times = {
            "twitter": "9:00 AM - 12:00 PM (weekdays)",
            "linkedin": "7:00 AM - 8:00 AM, 12:00 PM (Tue-Thu)",
            "instagram": "11:00 AM - 1:00 PM, 7:00 PM - 9:00 PM",
            "tiktok": "7:00 PM - 9:00 PM",
            "youtube": "2:00 PM - 4:00 PM (Thu-Fri)",
        }
        return optimal_times.get(platform, "9:00 AM - 5:00 PM")
    
    def adapt_content(self, content: str, platform: str) -> dict:
        """Adapt content for specific platform."""
        limit = self.PLATFORM_LIMITS.get(platform, 2200)
        
        adapted = content[:limit]
        if len(content) > limit:
            adapted = content[:limit-3] + "..."
        
        return {
            "text": adapted,
            "character_count": len(adapted),
            "limit": limit,
            "truncated": len(content) > limit,
        }
    
    def create_twitter_thread(self, content: str, max_tweets: int = 10) -> list[str]:
        """Split content into Twitter thread."""
        sentences = content.replace('\n', ' ').split('. ')
        
        tweets = []
        current_tweet = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            if not sentence.endswith('.'):
                sentence += '.'
            
            if len(current_tweet) + len(sentence) + 1 <= 270:  # Leave room for thread marker
                current_tweet += (" " if current_tweet else "") + sentence
            else:
                if current_tweet:
                    tweets.append(current_tweet)
                current_tweet = sentence
        
        if current_tweet:
            tweets.append(current_tweet)
        
        # Add thread markers
        total = min(len(tweets), max_tweets)
        threaded = []
        for i, tweet in enumerate(tweets[:max_tweets], 1):
            threaded.append(f"{tweet}\n\n🧵 {i}/{total}")
        
        return threaded


class ImagePromptTool:
    """
    AI image prompt generation tool.
    """
    
    def generate_featured_image_prompt(self, topic: str, style: str = "professional") -> str:
        """Generate a featured image prompt."""
        styles = {
            "professional": "clean, modern, professional photography style, high quality, corporate aesthetic",
            "creative": "creative, artistic, vibrant colors, unique composition, eye-catching",
            "minimal": "minimalist, clean background, simple composition, elegant, sophisticated",
            "tech": "futuristic, technology-focused, digital elements, blue tones, innovative",
            "warm": "warm lighting, inviting, friendly, natural colors, approachable",
        }
        
        style_desc = styles.get(style, styles["professional"])
        
        return f"Create a featured image for an article about {topic}. Style: {style_desc}. No text overlays. High resolution, suitable for blog header."
    
    def generate_social_image_prompt(self, platform: str, topic: str, text: str = "") -> str:
        """Generate social media image prompt."""
        dimensions = {
            "twitter": "1200x675 aspect ratio",
            "linkedin": "1200x627 aspect ratio",
            "instagram": "square 1080x1080",
            "tiktok": "vertical 1080x1920",
        }
        
        dim = dimensions.get(platform, "1200x675")
        
        prompt = f"Create a {platform} post image about {topic}. {dim}. Engaging, scroll-stopping visual."
        
        if text:
            prompt += f" Space for text overlay: '{text[:30]}...'"
        
        return prompt
    
    def generate_thumbnail_prompt(self, title: str, style: str = "youtube") -> str:
        """Generate video thumbnail prompt."""
        if style == "youtube":
            return f"YouTube thumbnail for video titled '{title}'. Bold, eye-catching, high contrast, expressive. Space for large text. 1280x720."
        elif style == "tiktok":
            return f"TikTok cover image for '{title}'. Vertical format, attention-grabbing, trendy aesthetic. 1080x1920."
        else:
            return f"Video thumbnail for '{title}'. Engaging, professional, click-worthy."


class ContentAnalyzer:
    """
    Content quality analyzer.
    """
    
    def analyze_readability(self, content: str) -> dict:
        """Analyze content readability."""
        words = content.split()
        sentences = re.split(r'[.!?]+', content)
        sentences = [s for s in sentences if s.strip()]
        
        word_count = len(words)
        sentence_count = len(sentences)
        avg_words_per_sentence = word_count / sentence_count if sentence_count > 0 else 0
        
        # Simplified Flesch-Kincaid approximation
        syllables = sum(self._count_syllables(word) for word in words)
        avg_syllables = syllables / word_count if word_count > 0 else 0
        
        # Flesch Reading Ease (simplified)
        reading_ease = 206.835 - (1.015 * avg_words_per_sentence) - (84.6 * avg_syllables)
        reading_ease = max(0, min(100, reading_ease))
        
        # Grade level
        if reading_ease >= 80:
            grade = "Easy (6th grade)"
        elif reading_ease >= 60:
            grade = "Standard (8th-9th grade)"
        elif reading_ease >= 40:
            grade = "Difficult (College level)"
        else:
            grade = "Very Difficult (Professional)"
        
        return {
            "word_count": word_count,
            "sentence_count": sentence_count,
            "avg_words_per_sentence": round(avg_words_per_sentence, 1),
            "reading_ease": round(reading_ease, 1),
            "grade_level": grade,
            "recommendations": self._get_readability_recommendations(avg_words_per_sentence, reading_ease),
        }
    
    def _count_syllables(self, word: str) -> int:
        """Approximate syllable count."""
        word = word.lower()
        count = 0
        vowels = "aeiouy"
        prev_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_was_vowel:
                count += 1
            prev_was_vowel = is_vowel
        
        if word.endswith('e'):
            count -= 1
        
        return max(1, count)
    
    def _get_readability_recommendations(self, avg_words: float, reading_ease: float) -> list[str]:
        """Get readability recommendations."""
        recs = []
        
        if avg_words > 20:
            recs.append("Break up long sentences. Aim for 15-20 words per sentence.")
        
        if reading_ease < 60:
            recs.append("Simplify language. Use shorter words and clearer phrasing.")
        
        if reading_ease > 80:
            recs.append("Content is very simple. Consider adding more depth if targeting professionals.")
        
        return recs


# Create tool instances
research_tool = ResearchTool()
seo_tool = SEOTool()
social_tool = SocialMediaTool()
image_tool = ImagePromptTool()
analyzer_tool = ContentAnalyzer()
