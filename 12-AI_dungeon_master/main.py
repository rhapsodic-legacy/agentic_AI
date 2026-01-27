#!/usr/bin/env python3
"""
🎲 AI Game Master - Command Line Interface

An immersive D&D experience in your terminal!

Usage:
    python main.py play
    python main.py play --name "Thorin" --class Fighter --race Dwarf
    python main.py serve
"""

import argparse
import sys
import os
from pathlib import Path

# Rich console for beautiful output
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.markdown import Markdown
    from rich.text import Text
    from rich.live import Live
    from rich.layout import Layout
    from rich import box
    from rich.prompt import Prompt
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    console = None


def print_title():
    """Print the epic title screen."""
    if RICH_AVAILABLE:
        title = """
[bold red]
    ██████╗ ██╗   ██╗███╗   ██╗ ██████╗ ███████╗ ██████╗ ███╗   ██╗
    ██╔══██╗██║   ██║████╗  ██║██╔════╝ ██╔════╝██╔═══██╗████╗  ██║
    ██║  ██║██║   ██║██╔██╗ ██║██║  ███╗█████╗  ██║   ██║██╔██╗ ██║
    ██║  ██║██║   ██║██║╚██╗██║██║   ██║██╔══╝  ██║   ██║██║╚██╗██║
    ██████╔╝╚██████╔╝██║ ╚████║╚██████╔╝███████╗╚██████╔╝██║ ╚████║
    ╚═════╝  ╚═════╝ ╚═╝  ╚═══╝ ╚═════╝ ╚══════╝ ╚═════╝ ╚═╝  ╚═══╝
[/bold red]
[bold yellow]
    ███╗   ███╗ █████╗ ███████╗████████╗███████╗██████╗ 
    ████╗ ████║██╔══██╗██╔════╝╚══██╔══╝██╔════╝██╔══██╗
    ██╔████╔██║███████║███████╗   ██║   █████╗  ██████╔╝
    ██║╚██╔╝██║██╔══██║╚════██║   ██║   ██╔══╝  ██╔══██╗
    ██║ ╚═╝ ██║██║  ██║███████║   ██║   ███████╗██║  ██║
    ╚═╝     ╚═╝╚═╝  ╚═╝╚══════╝   ╚═╝   ╚══════╝╚═╝  ╚═╝
[/bold yellow]
        """
        console.print(title)
        console.print(Panel.fit(
            "[dim]AI-Powered Dungeons & Dragons[/dim]\n"
            "[cyan]🎲 Roll for Initiative! 🐉[/cyan]",
            border_style="yellow"
        ))
    else:
        print("\n" + "="*60)
        print("       🎲 AI DUNGEON MASTER 🐉")
        print("    AI-Powered Dungeons & Dragons")
        print("="*60 + "\n")


def print_architecture():
    """Print the system architecture."""
    if RICH_AVAILABLE:
        console.print("""
[dim]Event-Driven Architecture:[/dim]
                    ┌─────────────────────┐
                    │    [bold]GAME STATE[/bold]       │
                    │  (World, Players)   │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
         ┌─────────│   [bold cyan]EVENT ROUTER[/bold cyan]     │─────────┐
         │         └──────────┬──────────┘         │
         │                    │                    │
         ▼                    ▼                    ▼
[green]NARRATIVE[/green]           [red]COMBAT[/red]             [yellow]NPC[/yellow]
[green]ENGINE[/green]             [red]SYSTEM[/red]            [yellow]MANAGER[/yellow]
         │                    │                    │
         ▼                    ▼                    ▼
[dim]Story/Atmosphere[/dim]  [dim]D&D 5e Rules[/dim]      [dim]Dialogue/Memory[/dim]
""")


def character_creation():
    """Interactive character creation."""
    if RICH_AVAILABLE:
        console.print("\n[bold cyan]═══ CHARACTER CREATION ═══[/bold cyan]\n")
        
        name = Prompt.ask("[yellow]What is your character's name?[/yellow]", default="Hero")
        
        # Class selection
        classes = ["Fighter", "Rogue", "Wizard", "Cleric", "Ranger", "Barbarian"]
        console.print("\n[yellow]Choose your class:[/yellow]")
        for i, c in enumerate(classes, 1):
            console.print(f"  {i}. {c}")
        
        class_choice = Prompt.ask("Class", choices=[str(i) for i in range(1, len(classes)+1)], default="1")
        character_class = classes[int(class_choice) - 1]
        
        # Race selection
        races = ["Human", "Elf", "Dwarf", "Halfling", "Half-Orc", "Tiefling"]
        console.print("\n[yellow]Choose your race:[/yellow]")
        for i, r in enumerate(races, 1):
            console.print(f"  {i}. {r}")
        
        race_choice = Prompt.ask("Race", choices=[str(i) for i in range(1, len(races)+1)], default="1")
        race = races[int(race_choice) - 1]
        
        return name, character_class, race
    else:
        name = input("Character name: ") or "Hero"
        character_class = input("Class (Fighter/Rogue/Wizard/Cleric): ") or "Fighter"
        race = input("Race (Human/Elf/Dwarf/Halfling): ") or "Human"
        return name, character_class, race


def run_game(name: str, character_class: str, race: str):
    """Run the main game loop."""
    from game_master import AIGameMaster
    
    gm = AIGameMaster()
    
    # Start new game
    opening = gm.new_game(name, character_class, race)
    
    if RICH_AVAILABLE:
        console.print(Markdown(opening))
    else:
        print(opening)
    
    # Game loop
    if RICH_AVAILABLE:
        console.print("\n[dim]Type 'help' for commands, 'quit' to save and exit.[/dim]\n")
    else:
        print("\nType 'help' for commands, 'quit' to save and exit.\n")
    
    while True:
        try:
            if RICH_AVAILABLE:
                player_input = console.input("\n[bold green]🎲 What do you do? >[/bold green] ").strip()
            else:
                player_input = input("\n🎲 What do you do? > ").strip()
            
            if not player_input:
                continue
            
            response, continue_game = gm.process_input(player_input)
            
            if RICH_AVAILABLE:
                # Format response nicely
                if "⚔️ COMBAT" in response:
                    console.print(Panel(response, border_style="red", title="⚔️ Combat"))
                elif "**" in response:
                    console.print(Markdown(response))
                else:
                    console.print(f"\n{response}")
            else:
                print(f"\n{response}")
            
            if not continue_game:
                break
                
        except KeyboardInterrupt:
            if RICH_AVAILABLE:
                console.print("\n\n[yellow]Game interrupted. Saving...[/yellow]")
            else:
                print("\n\nGame interrupted. Saving...")
            gm.process_input("save")
            break
        except Exception as e:
            if RICH_AVAILABLE:
                console.print(f"\n[red]Error: {e}[/red]")
            else:
                print(f"\nError: {e}")


def run_play(args):
    """Start a new game."""
    print_title()
    print_architecture()
    
    if args.name and args.character_class:
        name = args.name
        character_class = args.character_class
        race = args.race or "Human"
    else:
        name, character_class, race = character_creation()
    
    if RICH_AVAILABLE:
        console.print(f"\n[bold green]Creating {name} the {race} {character_class}...[/bold green]\n")
    else:
        print(f"\nCreating {name} the {race} {character_class}...\n")
    
    run_game(name, character_class, race)


def run_quick(args):
    """Quick start with defaults."""
    print_title()
    
    if RICH_AVAILABLE:
        console.print("[dim]Starting quick game with default character...[/dim]\n")
    
    run_game("Adventurer", "Fighter", "Human")


def run_dice(args):
    """Roll dice."""
    from game_master import roll_dice
    
    dice_str = args.dice or "1d20"
    
    result = roll_dice(dice_str)
    
    if RICH_AVAILABLE:
        if result.is_critical:
            console.print(f"[bold green]🎲 {dice_str}: {result} - CRITICAL![/bold green]")
        elif result.is_fumble:
            console.print(f"[bold red]🎲 {dice_str}: {result} - FUMBLE![/bold red]")
        else:
            console.print(f"🎲 {dice_str}: {result}")
    else:
        print(f"🎲 {dice_str}: {result}")


def run_serve(args):
    """Start the web server."""
    import uvicorn
    
    print_title()
    
    if RICH_AVAILABLE:
        console.print(f"[cyan]🌐 Starting server at http://localhost:{args.port}[/cyan]")
        console.print(f"[cyan]📖 API docs at http://localhost:{args.port}/docs[/cyan]")
        console.print("\n[dim]Press Ctrl+C to stop[/dim]\n")
    else:
        print(f"Starting server at http://localhost:{args.port}")
    
    uvicorn.run(
        "api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


def main():
    parser = argparse.ArgumentParser(
        description="🎲 AI Dungeon Master - AI-Powered D&D",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Start a new game (interactive character creation)
    python main.py play
    
    # Start with specific character
    python main.py play --name "Thorin" --class Fighter --race Dwarf
    
    # Quick start with default character
    python main.py quick
    
    # Roll some dice
    python main.py roll 2d6+3
    python main.py roll 1d20
    
    # Start web server
    python main.py serve

Character Classes:
    Fighter, Rogue, Wizard, Cleric, Ranger, Barbarian
    
Races:
    Human, Elf, Dwarf, Halfling, Half-Orc, Tiefling, Gnome, Dragonborn
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Play command
    play_parser = subparsers.add_parser("play", help="Start a new game")
    play_parser.add_argument("--name", "-n", help="Character name")
    play_parser.add_argument("--class", dest="character_class", help="Character class")
    play_parser.add_argument("--race", "-r", default="Human", help="Character race")
    
    # Quick command
    quick_parser = subparsers.add_parser("quick", help="Quick start with defaults")
    
    # Roll command
    roll_parser = subparsers.add_parser("roll", help="Roll dice")
    roll_parser.add_argument("dice", nargs="?", default="1d20", help="Dice notation (e.g., 2d6+3)")
    
    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Start web server")
    serve_parser.add_argument("--host", default="0.0.0.0")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.add_argument("--reload", action="store_true")
    
    args = parser.parse_args()
    
    if args.command == "play":
        run_play(args)
    elif args.command == "quick":
        run_quick(args)
    elif args.command == "roll":
        run_dice(args)
    elif args.command == "serve":
        run_serve(args)
    else:
        print_title()
        parser.print_help()


if __name__ == "__main__":
    main()
