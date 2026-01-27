"""
AI Game Master - Event-Driven Engine

LangGraph-based reactive game engine that routes player actions
to appropriate systems (narrative, combat, dialogue, etc.)

Architecture:
                    ┌─────────────────────┐
                    │    GAME STATE       │
                    │  (World, Players,   │
                    │   NPCs, Inventory)  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
         ┌─────────│   EVENT ROUTER      │─────────┐
         │         └──────────┬──────────┘         │
         │                    │                    │
         ▼                    ▼                    ▼
┌─────────────┐       ┌─────────────┐       ┌─────────────┐
│  NARRATIVE  │       │   COMBAT    │       │    NPC      │
│   ENGINE    │       │   SYSTEM    │       │  MANAGER    │
└─────────────┘       └─────────────┘       └─────────────┘
"""

from typing import Optional, TypedDict, Annotated, Literal
from dataclasses import dataclass, field
from datetime import datetime
import uuid
import re
import os

from langgraph.graph import StateGraph, END

from .models import (
    GameState, Campaign, WorldMap, GameTime, PlayerCharacter, NPC,
    Location, Quest, QuestStatus, QuestObjective, EventType,
    Skill, AbilityScore, DamageType, Condition, roll_dice
)
from .systems import DiceSystem, CombatSystem, DialogueSystem, NarrativeEngine
from .data import (
    LOCATIONS, NPCS, MONSTERS, STARTER_QUESTS,
    create_starter_campaign, create_starter_character, create_monster,
    generate_loot, WEAPONS, CONSUMABLES
)


# =============================================================================
# Engine State
# =============================================================================

class EngineState(TypedDict):
    """State passed through the LangGraph."""
    game_state: GameState
    player_input: str
    action_type: str  # "narrative", "combat", "dialogue", "skill_check", "movement", "inventory", "quit"
    target: str
    response: str
    continue_game: bool


# =============================================================================
# Action Parser
# =============================================================================

class ActionParser:
    """Parses player input into game actions."""
    
    MOVEMENT_WORDS = ["go", "move", "walk", "run", "travel", "head", "enter", "exit", "leave"]
    COMBAT_WORDS = ["attack", "hit", "strike", "fight", "kill", "shoot"]
    SKILL_WORDS = ["check", "roll", "try", "attempt", "search", "look", "examine", "investigate", "sneak", "hide", "persuade", "intimidate"]
    DIALOGUE_WORDS = ["talk", "speak", "say", "ask", "tell", "chat", "greet"]
    INVENTORY_WORDS = ["inventory", "items", "equipment", "bag", "use", "equip", "drop", "take", "pick", "loot"]
    
    @staticmethod
    def parse(input_str: str, state: GameState) -> tuple[str, str]:
        """
        Parse player input and return (action_type, target).
        """
        input_lower = input_str.lower().strip()
        
        # Check for quit
        if input_lower in ["quit", "exit", "q", "save", "save and quit"]:
            return "quit", ""
        
        # Check for status commands
        if input_lower in ["status", "stats", "character", "sheet", "hp"]:
            return "status", ""
        
        if input_lower in ["map", "where", "location"]:
            return "location", ""
        
        if input_lower in ["quests", "quest", "journal"]:
            return "quests", ""
        
        if input_lower in ["help", "?", "commands"]:
            return "help", ""
        
        # Check if in combat
        if state.combat_state and state.combat_state.is_active:
            # Combat-specific parsing
            for word in ActionParser.COMBAT_WORDS:
                if word in input_lower:
                    # Extract target
                    target = input_lower.split(word)[-1].strip()
                    return "combat", target
            
            if "end turn" in input_lower or "pass" in input_lower:
                return "combat_end_turn", ""
            
            # Default to combat action
            return "combat", input_lower
        
        # Movement
        for word in ActionParser.MOVEMENT_WORDS:
            if input_lower.startswith(word):
                direction = input_lower.replace(word, "").strip()
                return "movement", direction
        
        # Dialogue
        for word in ActionParser.DIALOGUE_WORDS:
            if word in input_lower:
                # Extract NPC name
                target = input_lower.split(word)[-1].strip()
                # Remove "to" if present
                if target.startswith("to "):
                    target = target[3:]
                return "dialogue", target
        
        # Skill checks
        for word in ActionParser.SKILL_WORDS:
            if word in input_lower:
                return "skill_check", input_lower
        
        # Inventory
        for word in ActionParser.INVENTORY_WORDS:
            if word in input_lower:
                return "inventory", input_lower
        
        # Combat initiation
        for word in ActionParser.COMBAT_WORDS:
            if word in input_lower:
                target = input_lower.split(word)[-1].strip()
                return "combat_start", target
        
        # Default to narrative action
        return "narrative", input_lower


# =============================================================================
# Event Router Node
# =============================================================================

def route_event(state: EngineState) -> str:
    """Route to appropriate handler based on action type."""
    action = state["action_type"]
    
    if action == "quit":
        return "quit_handler"
    elif action in ["status", "location", "quests", "help"]:
        return "info_handler"
    elif action == "movement":
        return "movement_handler"
    elif action in ["combat", "combat_start", "combat_end_turn"]:
        return "combat_handler"
    elif action == "dialogue":
        return "dialogue_handler"
    elif action == "skill_check":
        return "skill_handler"
    elif action == "inventory":
        return "inventory_handler"
    else:
        return "narrative_handler"


# =============================================================================
# Handler Nodes
# =============================================================================

def parse_input(state: EngineState) -> EngineState:
    """Parse player input and determine action type."""
    player_input = state["player_input"]
    game_state = state["game_state"]
    
    action_type, target = ActionParser.parse(player_input, game_state)
    
    return {
        **state,
        "action_type": action_type,
        "target": target,
    }


def info_handler(state: EngineState) -> EngineState:
    """Handle information requests."""
    game_state = state["game_state"]
    action = state["action_type"]
    
    if action == "status":
        if game_state.players:
            player = game_state.players[0]
            response = player.to_character_sheet()
        else:
            response = "No character loaded."
    
    elif action == "location":
        location = game_state.world_map.get_current_location()
        if location:
            response = f"📍 **{location.name}**\n\n{location.description}"
            if location.connections:
                exits = [f"{direction} → {dest}" for direction, dest in location.connections.items()]
                response += f"\n\n*Exits: {', '.join(exits)}*"
        else:
            response = "You're not sure where you are."
    
    elif action == "quests":
        active = game_state.get_active_quests()
        if active:
            response = "📜 **Active Quests:**\n"
            for quest in active:
                response += f"\n**{quest.name}**\n{quest.description}\n"
                for obj in quest.objectives:
                    checkbox = "☑" if obj.completed else "☐"
                    response += f"  {checkbox} {obj.description}\n"
        else:
            response = "You have no active quests."
    
    elif action == "help":
        response = """
🎮 **Available Commands:**

**Movement:** go [direction], enter, leave
**Look:** look, examine [object], search
**Dialogue:** talk to [NPC], ask [NPC] about [topic]
**Combat:** attack [target], cast [spell]
**Inventory:** inventory, use [item], equip [item]
**Info:** status, map, quests
**Other:** rest, save, quit

*You can also just describe what you want to do!*
"""
    else:
        response = "Unknown command."
    
    return {**state, "response": response}


def movement_handler(state: EngineState) -> EngineState:
    """Handle player movement."""
    game_state = state["game_state"]
    direction = state["target"].lower()
    
    current_location = game_state.world_map.get_current_location()
    
    if not current_location:
        return {**state, "response": "You're lost in the void..."}
    
    # Normalize direction
    direction_map = {
        "n": "north", "s": "south", "e": "east", "w": "west",
        "u": "up", "d": "down", "i": "in", "o": "out",
    }
    direction = direction_map.get(direction, direction)
    
    if direction in current_location.connections:
        new_location_id = current_location.connections[direction]
        
        if new_location_id in game_state.world_map.locations:
            game_state.world_map.move_to(new_location_id)
            new_location = game_state.world_map.get_current_location()
            
            # Generate description
            narrative = NarrativeEngine(game_state)
            entering = not new_location.is_discovered
            description = narrative.describe_scene(new_location, entering)
            
            game_state.add_event(
                EventType.LOCATION_CHANGE,
                f"Moved to {new_location.name}",
                location_id=new_location_id,
            )
            
            # Check for random encounter
            if not new_location.is_safe and random.random() < 0.2:
                # Trigger encounter
                description += "\n\n⚠️ *Something stirs in the shadows...*"
            
            return {**state, "response": description, "game_state": game_state}
        else:
            return {**state, "response": f"The path {direction} seems blocked or leads nowhere."}
    else:
        available = list(current_location.connections.keys())
        return {**state, "response": f"You can't go that way. Available exits: {', '.join(available)}"}


def combat_handler(state: EngineState) -> EngineState:
    """Handle combat actions."""
    game_state = state["game_state"]
    action = state["action_type"]
    target = state["target"]
    
    combat_system = CombatSystem(game_state)
    
    if action == "combat_start":
        # Find or create enemy
        enemy = None
        
        # Check for NPC in location
        location = game_state.world_map.get_current_location()
        if location:
            for npc_id in location.npcs:
                npc = game_state.get_npc(npc_id)
                if npc and target.lower() in npc.name.lower():
                    npc.is_hostile = True
                    enemy = npc
                    break
        
        # Create monster if no NPC found
        if not enemy:
            for monster_type in MONSTERS:
                if target.lower() in monster_type.lower():
                    enemy = create_monster(monster_type)
                    game_state.npcs[enemy.character_id] = enemy
                    break
        
        if not enemy:
            return {**state, "response": "You don't see anything to attack."}
        
        # Start combat
        narrative = NarrativeEngine(game_state)
        combat = combat_system.start_combat([enemy])
        
        description = narrative.describe_combat_start([enemy])
        description += "\n\n" + combat_system.get_combat_status()
        
        return {**state, "response": description, "game_state": game_state}
    
    elif action == "combat":
        if not game_state.combat_state or not game_state.combat_state.is_active:
            return {**state, "response": "You're not in combat."}
        
        combat = game_state.combat_state
        current = combat.current_combatant
        
        if not current:
            return {**state, "response": "Combat error."}
        
        # Player's turn
        if current.character.character_id.startswith("player"):
            player = current.character
            
            # Find target
            target_combatant = None
            for c in combat.combatants:
                if target.lower() in c.character.name.lower() and c.character.character_id != player.character_id:
                    target_combatant = c
                    break
            
            if not target_combatant:
                # Attack first enemy
                for c in combat.combatants:
                    if not c.character.character_id.startswith("player"):
                        target_combatant = c
                        break
            
            if not target_combatant:
                return {**state, "response": "No valid target."}
            
            # Perform attack
            weapon = player.inventory.main_hand
            action_result = combat_system.attack(player, target_combatant.character, weapon)
            
            response = action_result.description
            
            # Check if target defeated
            if not target_combatant.character.is_alive:
                combat.remove_combatant(target_combatant.character.character_id)
                
                # Check for combat end
                enemies_remaining = [c for c in combat.combatants if not c.character.character_id.startswith("player")]
                if not enemies_remaining:
                    response += "\n\n" + combat_system.end_combat(victory=True)
                    return {**state, "response": response, "game_state": game_state}
            
            # Advance turn
            combat.next_turn()
            
            # Process enemy turns
            while combat.current_combatant and not combat.current_combatant.character.character_id.startswith("player"):
                enemy = combat.current_combatant.character
                
                # Enemy attacks random player
                target_player = random.choice(game_state.players)
                enemy_action = combat_system.monster_attack(enemy, target_player)
                response += f"\n\n{enemy_action.description}"
                
                # Check for player defeat
                if not target_player.is_alive:
                    response += f"\n\n💀 {target_player.name} falls unconscious!"
                
                combat.next_turn()
            
            response += "\n\n" + combat_system.get_combat_status()
            
            return {**state, "response": response, "game_state": game_state}
        
        else:
            return {**state, "response": "It's not your turn!"}
    
    elif action == "combat_end_turn":
        if game_state.combat_state:
            game_state.combat_state.next_turn()
            return {**state, "response": combat_system.get_combat_status(), "game_state": game_state}
        return {**state, "response": "Not in combat."}
    
    return {**state, "response": "Combat action not recognized."}


def dialogue_handler(state: EngineState) -> EngineState:
    """Handle NPC dialogue."""
    game_state = state["game_state"]
    target = state["target"]
    player_input = state["player_input"]
    
    # Find NPC
    npc = None
    location = game_state.world_map.get_current_location()
    
    if location:
        for npc_id in location.npcs:
            npc_candidate = game_state.get_npc(npc_id)
            if npc_candidate and target.lower() in npc_candidate.name.lower():
                npc = npc_candidate
                break
    
    # Also check NPCS dict
    if not npc:
        for npc_id, npc_obj in NPCS.items():
            if target.lower() in npc_obj.name.lower():
                npc = npc_obj
                game_state.npcs[npc_id] = npc
                break
    
    if not npc:
        return {**state, "response": f"You don't see anyone named '{target}' here."}
    
    # Generate dialogue
    dialogue_system = DialogueSystem(game_state)
    
    # Extract what player wants to say
    say_patterns = ["say", "ask", "tell"]
    message = player_input
    for pattern in say_patterns:
        if pattern in player_input.lower():
            message = player_input.lower().split(pattern)[-1].strip()
            break
    
    response = dialogue_system.generate_npc_response(npc, message)
    
    game_state.add_event(
        EventType.DIALOGUE,
        f"Spoke with {npc.name}",
        actor_id=npc.character_id,
    )
    
    formatted_response = f"**{npc.name}:** {response}"
    
    return {**state, "response": formatted_response, "game_state": game_state}


def skill_handler(state: EngineState) -> EngineState:
    """Handle skill checks."""
    game_state = state["game_state"]
    action = state["target"]
    
    if not game_state.players:
        return {**state, "response": "No character to make the check."}
    
    player = game_state.players[0]
    dice_system = DiceSystem()
    narrative = NarrativeEngine(game_state)
    
    # Determine skill based on action
    skill = Skill.INVESTIGATION  # Default
    dc = 12  # Default DC
    
    action_lower = action.lower()
    
    if any(word in action_lower for word in ["search", "look", "examine", "investigate"]):
        skill = Skill.INVESTIGATION
        dc = 12
    elif any(word in action_lower for word in ["sneak", "hide", "stealth"]):
        skill = Skill.STEALTH
        dc = 13
    elif any(word in action_lower for word in ["persuade", "convince"]):
        skill = Skill.PERSUASION
        dc = 14
    elif any(word in action_lower for word in ["intimidate", "threaten"]):
        skill = Skill.INTIMIDATION
        dc = 14
    elif any(word in action_lower for word in ["deceive", "lie", "bluff"]):
        skill = Skill.DECEPTION
        dc = 14
    elif any(word in action_lower for word in ["perceive", "notice", "spot"]):
        skill = Skill.PERCEPTION
        dc = 12
    elif any(word in action_lower for word in ["climb", "jump", "athletic"]):
        skill = Skill.ATHLETICS
        dc = 13
    elif any(word in action_lower for word in ["arcana", "magic", "spell"]):
        skill = Skill.ARCANA
        dc = 14
    elif any(word in action_lower for word in ["history", "lore", "ancient"]):
        skill = Skill.HISTORY
        dc = 12
    elif any(word in action_lower for word in ["nature", "animal", "track"]):
        skill = Skill.NATURE
        dc = 12
    elif any(word in action_lower for word in ["religion", "divine", "god"]):
        skill = Skill.RELIGION
        dc = 12
    elif any(word in action_lower for word in ["medicine", "heal", "wound"]):
        skill = Skill.MEDICINE
        dc = 12
    elif any(word in action_lower for word in ["insight", "read", "motive"]):
        skill = Skill.INSIGHT
        dc = 13
    
    # Make the check
    roll, success, check_desc = dice_system.skill_check(player, skill, dc)
    
    # Generate narrative description
    description = narrative.describe_action_result(action, skill, roll, success, dc)
    
    # Add mechanical info
    response = f"**{skill.display_name} Check**\n🎲 {check_desc}\n\n{description}"
    
    # Special success effects
    if success:
        location = game_state.world_map.get_current_location()
        if skill == Skill.INVESTIGATION and location and location.secrets:
            secret = location.secrets[0]
            response += f"\n\n💡 *You discover: {secret}*"
    
    game_state.add_event(
        EventType.SKILL_CHECK,
        f"{skill.display_name} check: {'success' if success else 'failure'}",
        data={"skill": skill.name, "roll": roll.total, "dc": dc},
    )
    
    return {**state, "response": response, "game_state": game_state}


def inventory_handler(state: EngineState) -> EngineState:
    """Handle inventory actions."""
    game_state = state["game_state"]
    action = state["target"].lower()
    
    if not game_state.players:
        return {**state, "response": "No character loaded."}
    
    player = game_state.players[0]
    inv = player.inventory
    
    if "inventory" in action or not action:
        # Show inventory
        response = "🎒 **Inventory**\n\n"
        
        if inv.main_hand:
            response += f"**Main Hand:** {inv.main_hand.name}\n"
        if inv.off_hand:
            response += f"**Off Hand:** {inv.off_hand.name}\n"
        if inv.armor:
            response += f"**Armor:** {inv.armor.name}\n"
        
        response += f"\n**Gold:** {inv.gold} gp\n\n"
        
        if inv.items:
            response += "**Items:**\n"
            for item in inv.items:
                qty = f" (x{item.quantity})" if item.quantity > 1 else ""
                response += f"  • {item.name}{qty}\n"
        else:
            response += "*Your pack is empty.*"
        
        return {**state, "response": response}
    
    elif "use" in action:
        # Use an item
        item_name = action.replace("use", "").strip()
        
        for item in inv.items:
            if item_name.lower() in item.name.lower():
                if "healing" in item.name.lower():
                    # Healing potion
                    heal_roll = roll_dice("2d4+2")
                    healed = player.heal(heal_roll.total)
                    inv.items.remove(item)
                    response = f"🧪 You drink the {item.name}.\n\nHealed for {healed} HP! (Now {player.current_hp}/{player.max_hp})"
                else:
                    response = f"You use the {item.name}."
                
                return {**state, "response": response, "game_state": game_state}
        
        return {**state, "response": f"You don't have '{item_name}' in your inventory."}
    
    return {**state, "response": "What do you want to do with your inventory?"}


def narrative_handler(state: EngineState) -> EngineState:
    """Handle general narrative actions."""
    game_state = state["game_state"]
    action = state["player_input"]
    
    narrative = NarrativeEngine(game_state)
    
    # Use LLM to generate appropriate response
    if narrative.llm:
        context = game_state.to_context()
        location = game_state.world_map.get_current_location()
        
        prompt = f"""You are the Dungeon Master for a D&D game.

{context}

CURRENT LOCATION: {location.name if location else 'Unknown'}
LOCATION DESCRIPTION: {location.description if location else 'Unknown'}
FEATURES: {', '.join(location.features) if location and location.features else 'None'}
NPCs PRESENT: {', '.join(location.npcs) if location and location.npcs else 'None'}

The player says: "{action}"

Respond as the DM with what happens. Be descriptive but concise (2-4 sentences).
If the action requires a skill check, describe the attempt but don't roll dice.
If it's impossible, explain why narratively.
End with a subtle prompt for what happens next or ask what they do.

DM:"""

        try:
            response = narrative.llm.invoke(prompt)
            dm_response = response.content
        except:
            dm_response = "You consider your options carefully. What do you do?"
    else:
        dm_response = f"You attempt to {action}. What happens next is up to fate. What do you do?"
    
    return {**state, "response": dm_response, "game_state": game_state}


def quit_handler(state: EngineState) -> EngineState:
    """Handle quit/save."""
    game_state = state["game_state"]
    
    # Save game
    try:
        save_path = f"saves/{game_state.state_id}.json"
        game_state.save(save_path)
        response = f"💾 Game saved to {save_path}\n\nThank you for playing! May your dice roll true. 🎲"
    except Exception as e:
        response = f"Could not save game: {e}\n\nThank you for playing!"
    
    return {**state, "response": response, "continue_game": False}


def output_response(state: EngineState) -> EngineState:
    """Final node - formats and outputs response."""
    return state


# =============================================================================
# Build the Game Graph
# =============================================================================

def build_game_graph():
    """Build the LangGraph for the game engine."""
    
    workflow = StateGraph(EngineState)
    
    # Add nodes
    workflow.add_node("parse_input", parse_input)
    workflow.add_node("info_handler", info_handler)
    workflow.add_node("movement_handler", movement_handler)
    workflow.add_node("combat_handler", combat_handler)
    workflow.add_node("dialogue_handler", dialogue_handler)
    workflow.add_node("skill_handler", skill_handler)
    workflow.add_node("inventory_handler", inventory_handler)
    workflow.add_node("narrative_handler", narrative_handler)
    workflow.add_node("quit_handler", quit_handler)
    workflow.add_node("output", output_response)
    
    # Set entry point
    workflow.set_entry_point("parse_input")
    
    # Add conditional routing
    workflow.add_conditional_edges(
        "parse_input",
        route_event,
        {
            "info_handler": "info_handler",
            "movement_handler": "movement_handler",
            "combat_handler": "combat_handler",
            "dialogue_handler": "dialogue_handler",
            "skill_handler": "skill_handler",
            "inventory_handler": "inventory_handler",
            "narrative_handler": "narrative_handler",
            "quit_handler": "quit_handler",
        }
    )
    
    # All handlers go to output
    for handler in ["info_handler", "movement_handler", "combat_handler", 
                    "dialogue_handler", "skill_handler", "inventory_handler",
                    "narrative_handler", "quit_handler"]:
        workflow.add_edge(handler, "output")
    
    # Output goes to END
    workflow.add_edge("output", END)
    
    return workflow.compile()


# =============================================================================
# Main Game Master Class
# =============================================================================

class AIGameMaster:
    """
    AI Dungeon Master - runs tabletop RPG campaigns.
    
    Example:
        gm = AIGameMaster()
        gm.new_game("Adventurer Name", "Fighter")
        gm.play()
    """
    
    def __init__(self):
        self.graph = build_game_graph()
        self.state: Optional[GameState] = None
    
    def new_game(
        self,
        player_name: str,
        character_class: str = "Fighter",
        race: str = "Human",
    ) -> str:
        """Start a new game."""
        from .models import CharacterClass, Race
        
        # Parse class and race
        char_class = CharacterClass.FIGHTER
        for cc in CharacterClass:
            if cc.value.lower() == character_class.lower():
                char_class = cc
                break
        
        char_race = Race.HUMAN
        for r in Race:
            if r.value.lower() == race.lower():
                char_race = r
                break
        
        # Create campaign
        campaign = create_starter_campaign()
        
        # Create player character
        player = create_starter_character(player_name, char_class, char_race)
        
        # Create world map
        world_map = WorldMap(
            locations={loc_id: loc for loc_id, loc in LOCATIONS.items()},
            current_location_id="village_square",
        )
        
        # Create game state
        self.state = GameState(
            state_id=f"game-{uuid.uuid4().hex[:8]}",
            campaign=campaign,
            time=GameTime(),
            world_map=world_map,
            players=[player],
            npcs={npc_id: npc for npc_id, npc in NPCS.items()},
            quests=list(STARTER_QUESTS),
            created_at=datetime.now().isoformat(),
        )
        
        # Generate opening
        narrative = NarrativeEngine(self.state)
        location = world_map.get_current_location()
        
        opening = f"""
╔══════════════════════════════════════════════════════════════╗
║         🐉 {campaign.name.upper():^42} 🐉         ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  {player.name} the {char_race.value} {char_class.value}                              
║                                                              ║
╚══════════════════════════════════════════════════════════════╝

{campaign.description}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"""
        
        scene_desc = narrative.describe_scene(location, entering=True)
        opening += scene_desc
        
        return opening
    
    def process_input(self, player_input: str) -> tuple[str, bool]:
        """
        Process player input and return (response, continue_game).
        """
        if not self.state:
            return "No game in progress. Start with new_game().", False
        
        engine_state: EngineState = {
            "game_state": self.state,
            "player_input": player_input,
            "action_type": "",
            "target": "",
            "response": "",
            "continue_game": True,
        }
        
        result = self.graph.invoke(engine_state)
        
        # Update state
        self.state = result["game_state"]
        
        return result["response"], result["continue_game"]
    
    def play(self):
        """Run interactive game loop."""
        if not self.state:
            print("No game started. Use new_game() first.")
            return
        
        print("\n" + "="*60)
        print("Type 'help' for commands, 'quit' to save and exit.")
        print("="*60 + "\n")
        
        while True:
            try:
                player_input = input("\n🎲 What do you do? > ").strip()
                
                if not player_input:
                    continue
                
                response, continue_game = self.process_input(player_input)
                
                print("\n" + response)
                
                if not continue_game:
                    break
                    
            except KeyboardInterrupt:
                print("\n\nGame interrupted. Saving...")
                self.process_input("save")
                break
            except Exception as e:
                print(f"\nError: {e}")
    
    def save_game(self, filepath: str = None):
        """Save the current game."""
        if not self.state:
            return "No game to save."
        
        if filepath is None:
            filepath = f"saves/{self.state.state_id}.json"
        
        self.state.save(filepath)
        return f"Game saved to {filepath}"
