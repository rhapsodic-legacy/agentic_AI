"""
AI Game Master - Core Data Models

Complete D&D 5e-inspired models for:
- Characters (Players and NPCs)
- Combat and actions
- World state and locations
- Quests and inventory
- Game events and history
"""

from typing import Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import random
import uuid
import json


# =============================================================================
# Core Enums
# =============================================================================

class AbilityScore(Enum):
    """D&D ability scores."""
    STRENGTH = "STR"
    DEXTERITY = "DEX"
    CONSTITUTION = "CON"
    INTELLIGENCE = "INT"
    WISDOM = "WIS"
    CHARISMA = "CHA"


class Skill(Enum):
    """D&D 5e skills."""
    # Strength
    ATHLETICS = ("Athletics", AbilityScore.STRENGTH)
    # Dexterity
    ACROBATICS = ("Acrobatics", AbilityScore.DEXTERITY)
    SLEIGHT_OF_HAND = ("Sleight of Hand", AbilityScore.DEXTERITY)
    STEALTH = ("Stealth", AbilityScore.DEXTERITY)
    # Intelligence
    ARCANA = ("Arcana", AbilityScore.INTELLIGENCE)
    HISTORY = ("History", AbilityScore.INTELLIGENCE)
    INVESTIGATION = ("Investigation", AbilityScore.INTELLIGENCE)
    NATURE = ("Nature", AbilityScore.INTELLIGENCE)
    RELIGION = ("Religion", AbilityScore.INTELLIGENCE)
    # Wisdom
    ANIMAL_HANDLING = ("Animal Handling", AbilityScore.WISDOM)
    INSIGHT = ("Insight", AbilityScore.WISDOM)
    MEDICINE = ("Medicine", AbilityScore.WISDOM)
    PERCEPTION = ("Perception", AbilityScore.WISDOM)
    SURVIVAL = ("Survival", AbilityScore.WISDOM)
    # Charisma
    DECEPTION = ("Deception", AbilityScore.CHARISMA)
    INTIMIDATION = ("Intimidation", AbilityScore.CHARISMA)
    PERFORMANCE = ("Performance", AbilityScore.CHARISMA)
    PERSUASION = ("Persuasion", AbilityScore.CHARISMA)
    
    def __init__(self, display_name: str, ability: AbilityScore):
        self.display_name = display_name
        self.ability = ability


class CharacterClass(Enum):
    """D&D character classes."""
    BARBARIAN = "Barbarian"
    BARD = "Bard"
    CLERIC = "Cleric"
    DRUID = "Druid"
    FIGHTER = "Fighter"
    MONK = "Monk"
    PALADIN = "Paladin"
    RANGER = "Ranger"
    ROGUE = "Rogue"
    SORCERER = "Sorcerer"
    WARLOCK = "Warlock"
    WIZARD = "Wizard"


class Race(Enum):
    """D&D races."""
    HUMAN = "Human"
    ELF = "Elf"
    DWARF = "Dwarf"
    HALFLING = "Halfling"
    GNOME = "Gnome"
    HALF_ELF = "Half-Elf"
    HALF_ORC = "Half-Orc"
    TIEFLING = "Tiefling"
    DRAGONBORN = "Dragonborn"


class DamageType(Enum):
    """Types of damage."""
    SLASHING = "slashing"
    PIERCING = "piercing"
    BLUDGEONING = "bludgeoning"
    FIRE = "fire"
    COLD = "cold"
    LIGHTNING = "lightning"
    THUNDER = "thunder"
    POISON = "poison"
    ACID = "acid"
    NECROTIC = "necrotic"
    RADIANT = "radiant"
    FORCE = "force"
    PSYCHIC = "psychic"


class ItemRarity(Enum):
    """Item rarity levels."""
    COMMON = "Common"
    UNCOMMON = "Uncommon"
    RARE = "Rare"
    VERY_RARE = "Very Rare"
    LEGENDARY = "Legendary"
    ARTIFACT = "Artifact"


class Condition(Enum):
    """Status conditions."""
    BLINDED = "Blinded"
    CHARMED = "Charmed"
    DEAFENED = "Deafened"
    FRIGHTENED = "Frightened"
    GRAPPLED = "Grappled"
    INCAPACITATED = "Incapacitated"
    INVISIBLE = "Invisible"
    PARALYZED = "Paralyzed"
    PETRIFIED = "Petrified"
    POISONED = "Poisoned"
    PRONE = "Prone"
    RESTRAINED = "Restrained"
    STUNNED = "Stunned"
    UNCONSCIOUS = "Unconscious"
    EXHAUSTION = "Exhaustion"


class ActionType(Enum):
    """Types of actions in combat."""
    ACTION = "Action"
    BONUS_ACTION = "Bonus Action"
    REACTION = "Reaction"
    MOVEMENT = "Movement"
    FREE_ACTION = "Free Action"


class QuestStatus(Enum):
    """Quest status."""
    NOT_STARTED = "Not Started"
    IN_PROGRESS = "In Progress"
    COMPLETED = "Completed"
    FAILED = "Failed"


class LocationType(Enum):
    """Types of locations."""
    TOWN = "Town"
    CITY = "City"
    VILLAGE = "Village"
    DUNGEON = "Dungeon"
    CAVE = "Cave"
    FOREST = "Forest"
    MOUNTAIN = "Mountain"
    CASTLE = "Castle"
    RUINS = "Ruins"
    TEMPLE = "Temple"
    TAVERN = "Tavern"
    SHOP = "Shop"
    WILDERNESS = "Wilderness"


class EventType(Enum):
    """Types of game events."""
    NARRATIVE = "narrative"
    COMBAT_START = "combat_start"
    COMBAT_END = "combat_end"
    COMBAT_ACTION = "combat_action"
    SKILL_CHECK = "skill_check"
    SAVING_THROW = "saving_throw"
    DIALOGUE = "dialogue"
    DISCOVERY = "discovery"
    QUEST_UPDATE = "quest_update"
    LEVEL_UP = "level_up"
    ITEM_ACQUIRED = "item_acquired"
    LOCATION_CHANGE = "location_change"
    REST = "rest"
    DEATH = "death"


# =============================================================================
# Dice System
# =============================================================================

@dataclass
class DiceRoll:
    """Result of a dice roll."""
    dice: str  # e.g., "1d20", "2d6+3"
    rolls: list[int]
    modifier: int = 0
    total: int = 0
    natural: int = 0  # For d20 rolls
    is_critical: bool = False
    is_fumble: bool = False
    advantage: bool = False
    disadvantage: bool = False
    
    def __str__(self) -> str:
        roll_str = f"[{', '.join(map(str, self.rolls))}]"
        if self.modifier > 0:
            return f"{roll_str} + {self.modifier} = {self.total}"
        elif self.modifier < 0:
            return f"{roll_str} - {abs(self.modifier)} = {self.total}"
        return f"{roll_str} = {self.total}"


def roll_dice(dice_str: str, advantage: bool = False, disadvantage: bool = False) -> DiceRoll:
    """
    Roll dice using standard notation (e.g., "1d20+5", "2d6", "4d6kh3").
    
    Supports:
    - Basic rolls: 1d20, 2d6, etc.
    - Modifiers: 1d20+5, 2d6-2
    - Advantage/disadvantage on d20s
    """
    import re
    
    # Parse dice notation
    match = re.match(r'(\d+)d(\d+)([+-]\d+)?', dice_str.lower().replace(' ', ''))
    if not match:
        return DiceRoll(dice=dice_str, rolls=[0], total=0)
    
    num_dice = int(match.group(1))
    die_size = int(match.group(2))
    modifier = int(match.group(3)) if match.group(3) else 0
    
    # Roll the dice
    rolls = [random.randint(1, die_size) for _ in range(num_dice)]
    
    # Handle advantage/disadvantage for d20
    if die_size == 20 and num_dice == 1:
        if advantage and not disadvantage:
            extra_roll = random.randint(1, 20)
            rolls = [max(rolls[0], extra_roll)]
        elif disadvantage and not advantage:
            extra_roll = random.randint(1, 20)
            rolls = [min(rolls[0], extra_roll)]
    
    total = sum(rolls) + modifier
    natural = rolls[0] if len(rolls) == 1 else sum(rolls)
    
    return DiceRoll(
        dice=dice_str,
        rolls=rolls,
        modifier=modifier,
        total=total,
        natural=natural,
        is_critical=die_size == 20 and natural == 20,
        is_fumble=die_size == 20 and natural == 1,
        advantage=advantage,
        disadvantage=disadvantage,
    )


def get_modifier(score: int) -> int:
    """Calculate ability modifier from score."""
    return (score - 10) // 2


# =============================================================================
# Items and Equipment
# =============================================================================

@dataclass
class Item:
    """A game item."""
    item_id: str
    name: str
    description: str = ""
    
    # Properties
    weight: float = 0.0  # in pounds
    value: int = 0  # in gold pieces
    rarity: ItemRarity = ItemRarity.COMMON
    
    # Flags
    magical: bool = False
    consumable: bool = False
    quantity: int = 1
    
    def to_dict(self) -> dict:
        return {
            "item_id": self.item_id,
            "name": self.name,
            "description": self.description,
            "rarity": self.rarity.value,
            "magical": self.magical,
            "quantity": self.quantity,
        }


@dataclass
class Weapon(Item):
    """A weapon item."""
    damage_dice: str = "1d6"
    damage_type: DamageType = DamageType.SLASHING
    properties: list[str] = field(default_factory=list)  # "finesse", "two-handed", etc.
    range_normal: int = 5  # feet
    range_long: int = 0
    
    # Bonus
    attack_bonus: int = 0
    damage_bonus: int = 0


@dataclass
class Armor(Item):
    """An armor item."""
    armor_class: int = 10
    armor_type: str = "light"  # "light", "medium", "heavy", "shield"
    strength_required: int = 0
    stealth_disadvantage: bool = False
    ac_bonus: int = 0  # For magical armor


@dataclass
class Inventory:
    """Character inventory."""
    items: list[Item] = field(default_factory=list)
    gold: int = 0
    silver: int = 0
    copper: int = 0
    
    # Equipped items
    main_hand: Optional[Weapon] = None
    off_hand: Optional[Item] = None  # Shield or weapon
    armor: Optional[Armor] = None
    
    def add_item(self, item: Item):
        # Check if stackable
        for existing in self.items:
            if existing.name == item.name and existing.consumable:
                existing.quantity += item.quantity
                return
        self.items.append(item)
    
    def remove_item(self, item_name: str) -> Optional[Item]:
        for i, item in enumerate(self.items):
            if item.name.lower() == item_name.lower():
                return self.items.pop(i)
        return None
    
    def get_total_weight(self) -> float:
        return sum(item.weight * item.quantity for item in self.items)
    
    def get_total_gold(self) -> float:
        return self.gold + self.silver / 10 + self.copper / 100


# =============================================================================
# Characters
# =============================================================================

@dataclass
class AbilityScores:
    """Character ability scores."""
    strength: int = 10
    dexterity: int = 10
    constitution: int = 10
    intelligence: int = 10
    wisdom: int = 10
    charisma: int = 10
    
    def get(self, ability: AbilityScore) -> int:
        return getattr(self, ability.name.lower())
    
    def get_modifier(self, ability: AbilityScore) -> int:
        return get_modifier(self.get(ability))
    
    def to_dict(self) -> dict:
        return {
            "STR": self.strength,
            "DEX": self.dexterity,
            "CON": self.constitution,
            "INT": self.intelligence,
            "WIS": self.wisdom,
            "CHA": self.charisma,
        }


@dataclass
class Character:
    """Base character (player or NPC)."""
    character_id: str
    name: str
    
    # Core stats
    level: int = 1
    experience: int = 0
    
    # Health
    max_hp: int = 10
    current_hp: int = 10
    temp_hp: int = 0
    
    # Defense
    armor_class: int = 10
    
    # Abilities
    abilities: AbilityScores = field(default_factory=AbilityScores)
    
    # Proficiencies
    proficiency_bonus: int = 2
    skill_proficiencies: list[Skill] = field(default_factory=list)
    saving_throw_proficiencies: list[AbilityScore] = field(default_factory=list)
    
    # Conditions
    conditions: list[Condition] = field(default_factory=list)
    
    # Inventory
    inventory: Inventory = field(default_factory=Inventory)
    
    # Speed
    speed: int = 30
    
    # Combat
    initiative: int = 0
    
    @property
    def is_alive(self) -> bool:
        return self.current_hp > 0
    
    @property
    def is_bloodied(self) -> bool:
        return self.current_hp <= self.max_hp // 2
    
    def get_skill_modifier(self, skill: Skill) -> int:
        """Get total modifier for a skill check."""
        ability_mod = self.abilities.get_modifier(skill.ability)
        prof = self.proficiency_bonus if skill in self.skill_proficiencies else 0
        return ability_mod + prof
    
    def get_saving_throw_modifier(self, ability: AbilityScore) -> int:
        """Get modifier for a saving throw."""
        ability_mod = self.abilities.get_modifier(ability)
        prof = self.proficiency_bonus if ability in self.saving_throw_proficiencies else 0
        return ability_mod + prof
    
    def take_damage(self, amount: int, damage_type: DamageType = None) -> int:
        """Take damage and return actual damage taken."""
        # Apply temp HP first
        if self.temp_hp > 0:
            absorbed = min(self.temp_hp, amount)
            self.temp_hp -= absorbed
            amount -= absorbed
        
        actual_damage = min(self.current_hp, amount)
        self.current_hp -= actual_damage
        return actual_damage
    
    def heal(self, amount: int) -> int:
        """Heal and return actual HP restored."""
        actual_heal = min(self.max_hp - self.current_hp, amount)
        self.current_hp += actual_heal
        return actual_heal
    
    def roll_initiative(self) -> DiceRoll:
        """Roll initiative."""
        roll = roll_dice("1d20")
        self.initiative = roll.total + self.abilities.get_modifier(AbilityScore.DEXTERITY)
        return roll


@dataclass
class PlayerCharacter(Character):
    """A player character."""
    # Class and race
    character_class: CharacterClass = CharacterClass.FIGHTER
    race: Race = Race.HUMAN
    
    # Background
    background: str = "Folk Hero"
    alignment: str = "Neutral Good"
    
    # Appearance
    appearance: str = ""
    personality: str = ""
    bonds: str = ""
    flaws: str = ""
    ideals: str = ""
    
    # Resources
    hit_dice: str = "1d10"
    hit_dice_remaining: int = 1
    
    # Spellcasting (if applicable)
    spell_slots: dict = field(default_factory=dict)
    spells_known: list[str] = field(default_factory=list)
    
    # Death saves
    death_save_successes: int = 0
    death_save_failures: int = 0
    
    def to_dict(self) -> dict:
        return {
            "character_id": self.character_id,
            "name": self.name,
            "class": self.character_class.value,
            "race": self.race.value,
            "level": self.level,
            "hp": f"{self.current_hp}/{self.max_hp}",
            "ac": self.armor_class,
            "abilities": self.abilities.to_dict(),
        }
    
    def to_character_sheet(self) -> str:
        """Generate a text character sheet."""
        ab = self.abilities
        sheet = f"""
╔══════════════════════════════════════════════════════════════╗
║  {self.name:^58}  ║
║  {self.race.value} {self.character_class.value} (Level {self.level})  ║
╠══════════════════════════════════════════════════════════════╣
║  HP: {self.current_hp}/{self.max_hp}  |  AC: {self.armor_class}  |  Speed: {self.speed} ft  |  Prof: +{self.proficiency_bonus}  ║
╠══════════════════════════════════════════════════════════════╣
║  ABILITY SCORES                                              ║
║  STR: {ab.strength:2} ({ab.get_modifier(AbilityScore.STRENGTH):+d})  DEX: {ab.dexterity:2} ({ab.get_modifier(AbilityScore.DEXTERITY):+d})  CON: {ab.constitution:2} ({ab.get_modifier(AbilityScore.CONSTITUTION):+d})  ║
║  INT: {ab.intelligence:2} ({ab.get_modifier(AbilityScore.INTELLIGENCE):+d})  WIS: {ab.wisdom:2} ({ab.get_modifier(AbilityScore.WISDOM):+d})  CHA: {ab.charisma:2} ({ab.get_modifier(AbilityScore.CHARISMA):+d})  ║
╠══════════════════════════════════════════════════════════════╣
║  SKILLS                                                      ║"""
        
        skills_str = ""
        for skill in self.skill_proficiencies[:6]:
            mod = self.get_skill_modifier(skill)
            skills_str += f"  • {skill.display_name}: {mod:+d}"
        
        sheet += f"\n║{skills_str:62}║"
        sheet += "\n╚══════════════════════════════════════════════════════════════╝"
        
        return sheet


@dataclass
class NPCPersonality:
    """NPC personality traits and behavior."""
    trait: str = ""  # Main personality trait
    ideal: str = ""  # What they believe in
    bond: str = ""   # What they're connected to
    flaw: str = ""   # Their weakness
    
    voice: str = ""  # How they speak (accent, vocabulary)
    mannerism: str = ""  # Physical habits
    
    disposition: int = 50  # 0-100, attitude toward party
    
    # Conversation memory
    topics_discussed: list[str] = field(default_factory=list)
    secrets_revealed: list[str] = field(default_factory=list)
    
    # Relationship with party
    relationship: str = "neutral"  # "friendly", "neutral", "hostile"


@dataclass
class NPC(Character):
    """A non-player character."""
    # Role
    role: str = "Commoner"  # "Merchant", "Guard", "Villain", etc.
    creature_type: str = "Humanoid"
    
    # Description
    description: str = ""
    appearance: str = ""
    
    # Personality
    personality: NPCPersonality = field(default_factory=NPCPersonality)
    
    # Location
    home_location: str = ""
    current_location: str = ""
    
    # Combat stats (for hostile NPCs)
    challenge_rating: float = 0.0
    attacks: list[dict] = field(default_factory=list)  # [{name, bonus, damage, type}]
    
    # Dialogue
    greeting: str = ""
    farewell: str = ""
    
    # Flags
    is_hostile: bool = False
    is_essential: bool = False  # Can't be killed
    is_merchant: bool = False
    
    # Merchant inventory (if applicable)
    shop_inventory: list[Item] = field(default_factory=list)
    
    def get_dialogue_context(self) -> str:
        """Get context for generating dialogue."""
        p = self.personality
        return f"""
Name: {self.name}
Role: {self.role}
Personality: {p.trait}
Speaking style: {p.voice}
Mannerism: {p.mannerism}
Disposition: {p.disposition}/100 ({'friendly' if p.disposition > 60 else 'neutral' if p.disposition > 40 else 'unfriendly'})
Topics discussed: {', '.join(p.topics_discussed[-5:]) if p.topics_discussed else 'None yet'}
"""


# =============================================================================
# Locations and World
# =============================================================================

@dataclass
class Location:
    """A location in the game world."""
    location_id: str
    name: str
    description: str
    
    # Type and properties
    location_type: LocationType = LocationType.WILDERNESS
    
    # Atmosphere
    atmosphere: str = ""  # Sounds, smells, lighting
    
    # Connections
    connections: dict[str, str] = field(default_factory=dict)  # {direction: location_id}
    
    # Contents
    npcs: list[str] = field(default_factory=list)  # NPC IDs present
    items: list[Item] = field(default_factory=list)  # Items that can be found
    
    # Flags
    is_safe: bool = True
    is_discovered: bool = False
    
    # Special features
    features: list[str] = field(default_factory=list)  # Interactive elements
    secrets: list[str] = field(default_factory=list)  # Hidden things
    
    def get_full_description(self) -> str:
        """Get complete location description."""
        desc = f"**{self.name}**\n\n{self.description}"
        
        if self.atmosphere:
            desc += f"\n\n{self.atmosphere}"
        
        if self.features:
            desc += "\n\nYou notice: " + ", ".join(self.features)
        
        if self.connections:
            exits = [f"{direction} to {loc}" for direction, loc in self.connections.items()]
            desc += f"\n\nExits: {', '.join(exits)}"
        
        return desc


@dataclass
class WorldMap:
    """The game world map."""
    locations: dict[str, Location] = field(default_factory=dict)
    current_location_id: str = ""
    
    def get_current_location(self) -> Optional[Location]:
        return self.locations.get(self.current_location_id)
    
    def move_to(self, location_id: str) -> bool:
        if location_id in self.locations:
            self.current_location_id = location_id
            self.locations[location_id].is_discovered = True
            return True
        return False


# =============================================================================
# Quests
# =============================================================================

@dataclass
class QuestObjective:
    """An objective within a quest."""
    objective_id: str
    description: str
    completed: bool = False
    optional: bool = False
    
    # Progress tracking
    current_progress: int = 0
    required_progress: int = 1


@dataclass
class Quest:
    """A quest or mission."""
    quest_id: str
    name: str
    description: str
    
    # Status
    status: QuestStatus = QuestStatus.NOT_STARTED
    
    # Objectives
    objectives: list[QuestObjective] = field(default_factory=list)
    
    # Rewards
    gold_reward: int = 0
    xp_reward: int = 0
    item_rewards: list[str] = field(default_factory=list)
    
    # Quest giver
    quest_giver: str = ""  # NPC ID or name
    
    # Location hints
    locations: list[str] = field(default_factory=list)
    
    # Prerequisites
    required_level: int = 1
    required_quests: list[str] = field(default_factory=list)
    
    @property
    def is_complete(self) -> bool:
        required_objectives = [o for o in self.objectives if not o.optional]
        return all(o.completed for o in required_objectives)
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "status": self.status.value,
            "objectives": [
                {"description": o.description, "completed": o.completed}
                for o in self.objectives
            ],
        }


# =============================================================================
# Combat
# =============================================================================

@dataclass
class CombatAction:
    """An action taken in combat."""
    actor_id: str
    actor_name: str
    action_type: ActionType
    
    # Action details
    action_name: str  # "Attack", "Cast Fireball", "Dodge", etc.
    target_id: str = ""
    target_name: str = ""
    
    # Results
    roll: Optional[DiceRoll] = None
    damage_roll: Optional[DiceRoll] = None
    damage_dealt: int = 0
    damage_type: DamageType = None
    
    # Outcome
    hit: bool = False
    critical: bool = False
    description: str = ""


@dataclass
class Combatant:
    """A participant in combat."""
    character: Character
    initiative: int = 0
    has_action: bool = True
    has_bonus_action: bool = True
    has_reaction: bool = True
    has_moved: bool = False
    
    # Concentration (for spells)
    concentrating_on: str = ""


@dataclass
class CombatState:
    """Current state of combat."""
    combat_id: str
    is_active: bool = True
    round_number: int = 1
    
    # Combatants
    combatants: list[Combatant] = field(default_factory=list)
    current_turn_index: int = 0
    
    # History
    action_log: list[CombatAction] = field(default_factory=list)
    
    @property
    def current_combatant(self) -> Optional[Combatant]:
        if 0 <= self.current_turn_index < len(self.combatants):
            return self.combatants[self.current_turn_index]
        return None
    
    @property
    def initiative_order(self) -> list[str]:
        return [c.character.name for c in self.combatants]
    
    def next_turn(self):
        """Advance to the next turn."""
        self.current_turn_index += 1
        if self.current_turn_index >= len(self.combatants):
            self.current_turn_index = 0
            self.round_number += 1
        
        # Reset action economy
        current = self.current_combatant
        if current:
            current.has_action = True
            current.has_bonus_action = True
            current.has_moved = False
    
    def remove_combatant(self, character_id: str):
        """Remove a defeated combatant."""
        self.combatants = [c for c in self.combatants if c.character.character_id != character_id]
        if self.current_turn_index >= len(self.combatants):
            self.current_turn_index = 0


# =============================================================================
# Game Events and History
# =============================================================================

@dataclass
class GameEvent:
    """A recorded game event."""
    event_id: str
    event_type: EventType
    timestamp: str
    
    # Content
    description: str
    
    # Related entities
    actor_id: str = ""
    target_id: str = ""
    location_id: str = ""
    
    # Associated data
    data: dict = field(default_factory=dict)
    
    # Importance (for narrative continuity)
    importance: int = 1  # 1-5, higher = more important


@dataclass
class GameTime:
    """In-game time tracking."""
    day: int = 1
    hour: int = 8
    minute: int = 0
    
    # Calendar
    month: str = "Hammer"  # Forgotten Realms calendar
    year: int = 1492  # DR (Dale Reckoning)
    
    def advance_minutes(self, minutes: int):
        self.minute += minutes
        while self.minute >= 60:
            self.minute -= 60
            self.hour += 1
        while self.hour >= 24:
            self.hour -= 24
            self.day += 1
    
    def advance_hours(self, hours: int):
        self.advance_minutes(hours * 60)
    
    @property
    def time_of_day(self) -> str:
        if 5 <= self.hour < 12:
            return "morning"
        elif 12 <= self.hour < 17:
            return "afternoon"
        elif 17 <= self.hour < 21:
            return "evening"
        else:
            return "night"
    
    def __str__(self) -> str:
        return f"Day {self.day}, {self.hour:02d}:{self.minute:02d} ({self.time_of_day})"


# =============================================================================
# Campaign and Game State
# =============================================================================

@dataclass
class Campaign:
    """A D&D campaign."""
    campaign_id: str
    name: str
    description: str
    
    # Setting
    setting: str = "Forgotten Realms"
    
    # Current state
    current_chapter: str = ""
    current_scene: str = ""
    
    # History and summary
    summary: str = ""  # Running summary of events
    major_events: list[str] = field(default_factory=list)


@dataclass
class GameState:
    """Complete game state."""
    state_id: str
    
    # Campaign
    campaign: Campaign
    
    # Time
    time: GameTime = field(default_factory=GameTime)
    
    # World
    world_map: WorldMap = field(default_factory=WorldMap)
    
    # Characters
    players: list[PlayerCharacter] = field(default_factory=list)
    npcs: dict[str, NPC] = field(default_factory=dict)
    
    # Active state
    current_scene: str = ""
    combat_state: Optional[CombatState] = None
    
    # Quests
    quests: list[Quest] = field(default_factory=list)
    
    # History
    events: list[GameEvent] = field(default_factory=list)
    
    # Session info
    session_number: int = 1
    created_at: str = ""
    last_played: str = ""
    
    def get_player(self, name: str) -> Optional[PlayerCharacter]:
        for player in self.players:
            if player.name.lower() == name.lower():
                return player
        return None
    
    def get_npc(self, name: str) -> Optional[NPC]:
        for npc_id, npc in self.npcs.items():
            if npc.name.lower() == name.lower():
                return npc
        return self.npcs.get(name)
    
    def add_event(self, event_type: EventType, description: str, **kwargs):
        """Add a game event to history."""
        event = GameEvent(
            event_id=f"evt-{uuid.uuid4().hex[:8]}",
            event_type=event_type,
            timestamp=datetime.now().isoformat(),
            description=description,
            **kwargs
        )
        self.events.append(event)
        
        # Keep only recent events to manage memory
        if len(self.events) > 100:
            # Keep important events and recent ones
            important = [e for e in self.events if e.importance >= 3]
            recent = self.events[-50:]
            self.events = list(set(important + recent))
    
    def get_active_quests(self) -> list[Quest]:
        return [q for q in self.quests if q.status == QuestStatus.IN_PROGRESS]
    
    def get_recent_events(self, n: int = 10) -> list[GameEvent]:
        return self.events[-n:] if self.events else []
    
    def to_context(self) -> str:
        """Generate context string for AI."""
        location = self.world_map.get_current_location()
        
        context = f"""
CURRENT GAME STATE:

Campaign: {self.campaign.name}
Time: {self.time}
Location: {location.name if location else 'Unknown'}

Players:
"""
        for player in self.players:
            context += f"- {player.name} ({player.race.value} {player.character_class.value} L{player.level}): {player.current_hp}/{player.max_hp} HP\n"
        
        if self.combat_state and self.combat_state.is_active:
            context += f"\nCOMBAT ACTIVE (Round {self.combat_state.round_number})\n"
            context += f"Turn: {self.combat_state.current_combatant.character.name if self.combat_state.current_combatant else 'None'}\n"
        
        active_quests = self.get_active_quests()
        if active_quests:
            context += "\nActive Quests:\n"
            for quest in active_quests[:3]:
                context += f"- {quest.name}\n"
        
        recent = self.get_recent_events(5)
        if recent:
            context += "\nRecent Events:\n"
            for event in recent:
                context += f"- {event.description[:80]}\n"
        
        return context
    
    def save(self, filepath: str):
        """Save game state to file."""
        # Convert to JSON-serializable format
        data = {
            "state_id": self.state_id,
            "campaign": {
                "campaign_id": self.campaign.campaign_id,
                "name": self.campaign.name,
                "description": self.campaign.description,
                "setting": self.campaign.setting,
                "summary": self.campaign.summary,
            },
            "time": {
                "day": self.time.day,
                "hour": self.time.hour,
                "minute": self.time.minute,
            },
            "session_number": self.session_number,
            "players": [p.to_dict() for p in self.players],
            "current_scene": self.current_scene,
            "last_played": datetime.now().isoformat(),
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
