#!/usr/bin/env python3
"""
Customer Support Swarm - Command Line Interface

Usage:
    python main.py chat
    python main.py chat --customer CUST-001
    python main.py demo
    python main.py serve
"""

import argparse
import sys

# Rich console
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.markdown import Markdown
    from rich.live import Live
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
            "[bold blue]🎧 Customer Support Swarm[/bold blue]\n"
            "[dim]Multi-Agent Support with Seamless Handoffs[/dim]",
            border_style="blue"
        ))
    else:
        print("\n" + "="*50)
        print("🎧 Customer Support Swarm")
        print("Multi-Agent Support with Seamless Handoffs")
        print("="*50 + "\n")


def print_agents():
    """Print the agent swarm structure."""
    if RICH_AVAILABLE:
        table = Table(title="Support Agents")
        table.add_column("Agent", style="cyan")
        table.add_column("Role", style="white")
        table.add_column("Handoffs To", style="green")
        
        table.add_row(
            "🎯 Triage", 
            "Routes queries to specialists",
            "Billing, Technical, Sales, Retention"
        )
        table.add_row(
            "💳 Billing",
            "Billing & payment issues",
            "Refunds, Escalation"
        )
        table.add_row(
            "🔧 Technical",
            "Technical problems",
            "Billing, Escalation"
        )
        table.add_row(
            "💼 Sales",
            "Sales & upgrades",
            "Billing, Retention"
        )
        table.add_row(
            "💰 Refunds",
            "Processes refunds",
            "Billing, Escalation"
        )
        table.add_row(
            "🚨 Escalation",
            "Complex issues",
            "All agents"
        )
        table.add_row(
            "🤝 Retention",
            "Prevents cancellations",
            "Billing, Sales, Escalation"
        )
        
        console.print(table)
    else:
        print("Agents: Triage → Billing, Technical, Sales, Retention")
        print("        Billing → Refunds, Escalation")
        print("        etc.")


def print_response(agent: str, message: str):
    """Print an agent response."""
    if RICH_AVAILABLE:
        # Agent name with emoji
        agent_emojis = {
            "Triage Agent": "🎯",
            "Billing Specialist": "💳",
            "Technical Support": "🔧",
            "Sales Agent": "💼",
            "Refunds Specialist": "💰",
            "Escalation Agent": "🚨",
            "Retention Agent": "🤝",
        }
        emoji = agent_emojis.get(agent, "🤖")
        
        console.print(f"\n[bold cyan]{emoji} {agent}:[/bold cyan]")
        console.print(Panel(message, border_style="dim"))
    else:
        print(f"\n[{agent}]:")
        print(message)
        print()


def run_chat(args):
    """Run interactive chat mode."""
    from support_swarm import SupportSwarm, SwarmConfig
    
    print_banner()
    print_agents()
    
    config = SwarmConfig(
        llm_provider=args.provider,
        debug=args.debug,
    )
    swarm = SupportSwarm(config)
    
    # Set customer if provided
    if args.customer:
        if swarm.set_customer(args.customer):
            print(f"\n✓ Customer loaded: {args.customer}")
        else:
            print(f"\n⚠ Customer not found: {args.customer}")
    
    print("\nType your message or:")
    print("  /agent     - Show current agent")
    print("  /history   - Show conversation history")
    print("  /context   - Show context")
    print("  /reset     - Reset conversation")
    print("  /quit      - Exit")
    print()
    
    while True:
        try:
            if RICH_AVAILABLE:
                user_input = console.input("[bold green]You:[/bold green] ")
            else:
                user_input = input("You: ")
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye! 👋")
            break
        
        user_input = user_input.strip()
        
        if not user_input:
            continue
        
        # Commands
        if user_input.startswith("/"):
            cmd = user_input.lower()
            
            if cmd in ("/quit", "/exit", "/q"):
                print("Goodbye! 👋")
                break
            
            elif cmd == "/agent":
                print(f"Current agent: {swarm.get_current_agent()}")
            
            elif cmd == "/history":
                history = swarm.get_history()
                for turn in history[-10:]:
                    role = turn["role"]
                    content = turn["content"][:100]
                    agent = turn.get("agent", "")
                    print(f"  [{role}] {agent}: {content}...")
            
            elif cmd == "/context":
                context = swarm.get_context()
                import json
                print(json.dumps(context, indent=2))
            
            elif cmd == "/reset":
                swarm.reset()
                print("✓ Conversation reset")
            
            else:
                print(f"Unknown command: {cmd}")
            
            continue
        
        # Chat
        response = swarm.chat(user_input)
        print_response(swarm.get_current_agent(), response)


def run_demo(args):
    """Run a demonstration conversation."""
    from support_swarm import SupportSwarm, SwarmConfig
    
    print_banner()
    print("\n🎬 Running demo conversation...\n")
    print("-" * 50)
    
    config = SwarmConfig(llm_provider=args.provider)
    swarm = SupportSwarm(config)
    
    # Set demo customer
    swarm.set_customer("CUST-001")
    
    # Demo conversation
    demo_messages = [
        "Hi, I was charged twice for my subscription last week",
        "Yes, I can see there are two charges of $29.99 on the 15th",
        "Can I get a refund for the duplicate charge?",
    ]
    
    for msg in demo_messages:
        print(f"\n👤 Customer: {msg}")
        
        response = swarm.chat(msg)
        print_response(swarm.get_current_agent(), response)
        
        print("-" * 50)
    
    print("\n🎬 Demo complete!")
    print(f"Final agent: {swarm.get_current_agent()}")
    print(f"Handoffs: {swarm.handoff_count}")


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
        description="Customer Support Swarm - Multi-Agent Support System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Interactive chat
    python main.py chat
    
    # Chat with customer context
    python main.py chat --customer CUST-001
    
    # Run demo
    python main.py demo
    
    # Start web server
    python main.py serve
        """
    )
    
    parser.add_argument(
        "--provider", "-p",
        choices=["openai", "gemini", "anthropic", "mock"],
        default="mock",
        help="LLM provider (default: mock for demo)"
    )
    
    parser.add_argument(
        "--debug", "-d",
        action="store_true",
        help="Enable debug mode"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Chat command
    chat_parser = subparsers.add_parser("chat", help="Interactive chat")
    chat_parser.add_argument("--customer", "-c", help="Customer ID")
    
    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run demonstration")
    
    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Start web server")
    serve_parser.add_argument("--host", default="0.0.0.0")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.add_argument("--reload", action="store_true")
    
    args = parser.parse_args()
    
    if args.command == "chat":
        run_chat(args)
    elif args.command == "demo":
        run_demo(args)
    elif args.command == "serve":
        run_serve(args)
    else:
        # Default to chat
        args.customer = None
        run_chat(args)


if __name__ == "__main__":
    main()
