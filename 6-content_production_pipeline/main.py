#!/usr/bin/env python3
"""
Content Production Pipeline - Command Line Interface

Usage:
    python main.py create "AI in Marketing" --types blog,social
    python main.py blog "Topic" --length medium
    python main.py social "Topic" --platforms twitter,linkedin
    python main.py video "Topic"
    python main.py serve
"""

import argparse
import sys
import json
from pathlib import Path
from datetime import datetime

# Rich console
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.markdown import Markdown
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich import print as rprint
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    console = None


def print_banner():
    """Print the CLI banner."""
    if RICH_AVAILABLE:
        console.print(Panel.fit(
            "[bold magenta]🎬 Content Production Pipeline[/bold magenta]\n"
            "[dim]Multi-Platform Content from One Topic[/dim]",
            border_style="magenta"
        ))
    else:
        print("\n" + "="*50)
        print("🎬 Content Production Pipeline")
        print("Multi-Platform Content from One Topic")
        print("="*50 + "\n")


def print_pipeline():
    """Print the pipeline visualization."""
    if RICH_AVAILABLE:
        console.print("""
[dim]Pipeline Flow:[/dim]
    Brief → Research → Outline
                         ↓
           ┌─────────────┼─────────────┐
           ↓             ↓             ↓
        [cyan]Blog[/cyan]        [green]Social[/green]       [yellow]Video[/yellow]
           ↓             ↓             ↓
        [cyan]SEO[/cyan]        [green]Images[/green]       [yellow]VoiceOver[/yellow]
           └─────────────┼─────────────┘
                         ↓
                      Editor
                         ↓
                     Publisher
""")


def save_output(result: dict, output_dir: str, topic: str):
    """Save results to files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = topic.lower().replace(" ", "-")[:30]
    
    saved_files = []
    
    # Save blog
    if result.get("blog", {}).get("content"):
        blog_file = output_path / f"{slug}_blog_{timestamp}.md"
        with open(blog_file, 'w') as f:
            f.write(f"# {result['blog'].get('title', topic)}\n\n")
            f.write(result['blog']['content'])
        saved_files.append(str(blog_file))
    
    # Save social posts
    if result.get("social_posts"):
        social_file = output_path / f"{slug}_social_{timestamp}.json"
        with open(social_file, 'w') as f:
            json.dump(result['social_posts'], f, indent=2)
        saved_files.append(str(social_file))
    
    # Save video script
    if result.get("video", {}).get("script_sections"):
        video_file = output_path / f"{slug}_video_{timestamp}.md"
        video = result['video']
        with open(video_file, 'w') as f:
            f.write(f"# Video Script: {video.get('title', topic)}\n\n")
            f.write(f"**Duration:** ~{video.get('estimated_duration', 0)} seconds\n\n")
            f.write(f"## HOOK\n{video.get('hook', '')}\n\n")
            for section in video.get('script_sections', []):
                f.write(f"## {section.get('timestamp', '')}\n")
                f.write(f"**Narration:** {section.get('narration', '')}\n")
                f.write(f"**Visual:** {section.get('visual', '')}\n\n")
        saved_files.append(str(video_file))
    
    # Save full package
    package_file = output_path / f"{slug}_package_{timestamp}.json"
    with open(package_file, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    saved_files.append(str(package_file))
    
    return saved_files


def run_create(args):
    """Run full content creation."""
    from content_pipeline import ContentPipeline
    
    print_banner()
    print_pipeline()
    
    # Parse content types
    content_types = args.types.split(",") if args.types else ["blog"]
    
    # Parse platforms
    platforms = args.platforms.split(",") if args.platforms else ["twitter", "linkedin"]
    
    if RICH_AVAILABLE:
        console.print(f"\n[bold]Creating content for:[/bold] {args.topic}")
        console.print(f"[dim]Types: {', '.join(content_types)} | Platforms: {', '.join(platforms)}[/dim]\n")
    else:
        print(f"\nCreating content for: {args.topic}")
    
    pipeline = ContentPipeline()
    
    if RICH_AVAILABLE:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Running pipeline...", total=None)
            result = pipeline.run(
                topic=args.topic,
                content_types=content_types,
                platforms=platforms,
                tone=args.tone,
                blog_length=args.length,
            )
    else:
        result = pipeline.run(
            topic=args.topic,
            content_types=content_types,
            platforms=platforms,
            tone=args.tone,
            blog_length=args.length,
        )
    
    # Display summary
    if RICH_AVAILABLE:
        summary = result.get("summary", {})
        
        table = Table(title="Content Package Summary")
        table.add_column("Content Type", style="cyan")
        table.add_column("Details", style="white")
        
        if summary.get("blog_word_count"):
            table.add_row("📝 Blog", f"{summary['blog_word_count']} words, SEO: {summary.get('seo_score', 0)}/100")
        
        if summary.get("social_post_count"):
            table.add_row("📱 Social", f"{summary['social_post_count']} posts")
        
        if summary.get("video_duration"):
            table.add_row("🎬 Video", f"~{summary['video_duration']}s script")
        
        if summary.get("image_count"):
            table.add_row("🖼️ Images", f"{summary['image_count']} prompts")
        
        console.print(table)
    
    # Save outputs
    if args.output:
        saved = save_output(result, args.output, args.topic)
        if RICH_AVAILABLE:
            console.print(f"\n[green]✓ Saved {len(saved)} files to {args.output}[/green]")
        else:
            print(f"\n✓ Saved {len(saved)} files")
    
    # Show blog preview
    if result.get("blog", {}).get("content") and RICH_AVAILABLE:
        console.print("\n[bold]Blog Preview:[/bold]")
        console.print(Panel(
            Markdown(result["blog"]["content"][:1000] + "..."),
            title=result["blog"].get("title", "Blog Post"),
            border_style="cyan"
        ))
    
    # Show social preview
    if result.get("social_posts") and RICH_AVAILABLE:
        console.print("\n[bold]Social Posts:[/bold]")
        for post in result["social_posts"][:2]:
            console.print(Panel(
                f"{post.get('text', '')[:200]}...\n\n[dim]#{' #'.join(post.get('hashtags', [])[:5])}[/dim]",
                title=f"📱 {post.get('platform', 'Social').title()}",
                border_style="green"
            ))


def run_blog(args):
    """Create blog post only."""
    from content_pipeline import create_blog_post
    
    print_banner()
    
    if RICH_AVAILABLE:
        console.print(f"\n[bold]Creating blog post:[/bold] {args.topic}")
    
    result = create_blog_post(args.topic, length=args.length, tone=args.tone)
    
    if RICH_AVAILABLE and result.get("content"):
        console.print(Panel(
            Markdown(result["content"]),
            title=result.get("title", "Blog Post"),
            border_style="cyan"
        ))
    elif result.get("content"):
        print(result["content"])
    
    if args.output:
        saved = save_output({"blog": result}, args.output, args.topic)
        print(f"✓ Saved to {saved[0]}")


def run_social(args):
    """Create social content only."""
    from content_pipeline import create_social_content
    
    print_banner()
    
    platforms = args.platforms.split(",") if args.platforms else ["twitter", "linkedin"]
    
    if RICH_AVAILABLE:
        console.print(f"\n[bold]Creating social posts:[/bold] {args.topic}")
        console.print(f"[dim]Platforms: {', '.join(platforms)}[/dim]")
    
    posts = create_social_content(args.topic, platforms=platforms)
    
    for post in posts:
        if RICH_AVAILABLE:
            text = post.get("text", "")
            hashtags = " ".join(f"#{h}" for h in post.get("hashtags", []))
            
            console.print(Panel(
                f"{text}\n\n[dim]{hashtags}[/dim]\n\n[italic]Best time: {post.get('suggested_post_time', 'N/A')}[/italic]",
                title=f"📱 {post.get('platform', '').title()} ({post.get('character_count', 0)} chars)",
                border_style="green"
            ))
        else:
            print(f"\n--- {post.get('platform', '')} ---")
            print(post.get("text", ""))
            print(f"Hashtags: {' '.join(f'#{h}' for h in post.get('hashtags', []))}")


def run_video(args):
    """Create video script only."""
    from content_pipeline import create_video_script
    
    print_banner()
    
    if RICH_AVAILABLE:
        console.print(f"\n[bold]Creating video script:[/bold] {args.topic}")
    
    result = create_video_script(args.topic)
    
    if result:
        if RICH_AVAILABLE:
            console.print(f"\n[bold yellow]🎬 {result.get('title', 'Video Script')}[/bold yellow]")
            console.print(f"[dim]Duration: ~{result.get('estimated_duration', 0)} seconds | Music: {result.get('music_mood', 'N/A')}[/dim]\n")
            
            console.print("[bold]HOOK:[/bold]")
            console.print(Panel(result.get("hook", ""), border_style="yellow"))
            
            console.print("\n[bold]SCRIPT:[/bold]")
            for section in result.get("script_sections", []):
                console.print(f"\n[cyan]{section.get('timestamp', '')}[/cyan]")
                console.print(f"[bold]Narration:[/bold] {section.get('narration', '')}")
                console.print(f"[dim]Visual: {section.get('visual', '')}[/dim]")
            
            console.print("\n[bold]OUTRO:[/bold]")
            console.print(Panel(result.get("outro", ""), border_style="yellow"))
        else:
            print(f"\n{result.get('title', 'Video Script')}")
            print(f"Hook: {result.get('hook', '')}")


def run_interactive(args):
    """Run interactive mode."""
    from content_pipeline import ContentPipeline
    
    print_banner()
    print_pipeline()
    
    print("\nCommands:")
    print("  create TOPIC     - Full content creation")
    print("  blog TOPIC       - Blog post only")
    print("  social TOPIC     - Social posts only")
    print("  video TOPIC      - Video script only")
    print("  quit             - Exit")
    print()
    
    pipeline = ContentPipeline()
    
    while True:
        try:
            if RICH_AVAILABLE:
                user_input = console.input("[bold magenta]>>> [/bold magenta]")
            else:
                user_input = input(">>> ")
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye! 👋")
            break
        
        parts = user_input.strip().split(maxsplit=1)
        if not parts:
            continue
        
        cmd = parts[0].lower()
        topic = parts[1] if len(parts) > 1 else ""
        
        if cmd in ("quit", "exit", "q"):
            print("Goodbye! 👋")
            break
        
        elif cmd == "create" and topic:
            result = pipeline.run(topic, content_types=["blog", "social"])
            if RICH_AVAILABLE:
                console.print(f"[green]✓ Created content package[/green]")
        
        elif cmd == "blog" and topic:
            from content_pipeline import create_blog_post
            result = create_blog_post(topic)
            print(result.get("content", "")[:500] + "...")
        
        elif cmd == "social" and topic:
            from content_pipeline import create_social_content
            posts = create_social_content(topic)
            for post in posts:
                print(f"\n[{post['platform']}] {post['text'][:100]}...")
        
        elif cmd == "video" and topic:
            from content_pipeline import create_video_script
            result = create_video_script(topic)
            print(f"Hook: {result.get('hook', '')}")
        
        else:
            print(f"Unknown command or missing topic: {cmd}")


def run_serve(args):
    """Run the web server."""
    import uvicorn
    
    print_banner()
    
    print(f"🌐 Starting server at http://localhost:{args.port}")
    print(f"📖 API docs at http://localhost:{args.port}/docs")
    print("\nPress Ctrl+C to stop\n")
    
    uvicorn.run(
        "api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Content Production Pipeline - Multi-Platform Content Creator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
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
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Create command
    create_parser = subparsers.add_parser("create", help="Full content creation")
    create_parser.add_argument("topic", help="Content topic")
    create_parser.add_argument("--types", "-t", default="blog,social", help="Content types (comma-separated)")
    create_parser.add_argument("--platforms", "-p", default="twitter,linkedin", help="Social platforms")
    create_parser.add_argument("--tone", default="professional", help="Content tone")
    create_parser.add_argument("--length", "-l", default="medium", choices=["short", "medium", "long"])
    create_parser.add_argument("--output", "-o", default="./output", help="Output directory")
    
    # Blog command
    blog_parser = subparsers.add_parser("blog", help="Create blog post")
    blog_parser.add_argument("topic", help="Blog topic")
    blog_parser.add_argument("--length", "-l", default="medium", choices=["short", "medium", "long"])
    blog_parser.add_argument("--tone", default="professional")
    blog_parser.add_argument("--output", "-o", help="Output file")
    
    # Social command
    social_parser = subparsers.add_parser("social", help="Create social posts")
    social_parser.add_argument("topic", help="Content topic")
    social_parser.add_argument("--platforms", "-p", default="twitter,linkedin")
    
    # Video command
    video_parser = subparsers.add_parser("video", help="Create video script")
    video_parser.add_argument("topic", help="Video topic")
    
    # Interactive command
    interactive_parser = subparsers.add_parser("interactive", help="Interactive mode")
    
    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Start web server")
    serve_parser.add_argument("--host", default="0.0.0.0")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.add_argument("--reload", action="store_true")
    
    args = parser.parse_args()
    
    if args.command == "create":
        run_create(args)
    elif args.command == "blog":
        run_blog(args)
    elif args.command == "social":
        run_social(args)
    elif args.command == "video":
        run_video(args)
    elif args.command == "interactive":
        run_interactive(args)
    elif args.command == "serve":
        run_serve(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
