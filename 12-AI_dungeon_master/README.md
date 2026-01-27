# 🎲 AI Dungeon Master

An **AI-powered Dungeon Master** that runs tabletop RPG campaigns with dynamic storytelling, NPC management, D&D 5e combat rules, and persistent world state.

![LangGraph](https://img.shields.io/badge/Framework-LangGraph-purple)
![Architecture](https://img.shields.io/badge/Architecture-Event_Driven-blue)
![D&D](https://img.shields.io/badge/Rules-D%26D_5e-red)
![Complexity](https://img.shields.io/badge/Complexity-⭐⭐⭐⭐-yellow)

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🐉 **Dynamic Narrative** | AI-generated atmospheric descriptions |
| ⚔️ **D&D 5e Combat** | Full initiative, attacks, damage, conditions |
| 👥 **NPC Personalities** | Unique voices, memories, and dispositions |
| 🗺️ **Persistent World** | Locations, NPCs, quests that remember you |
| 🎲 **Dice System** | All standard dice with advantage/disadvantage |
| 📜 **Quest Tracking** | Objectives, rewards, and progress |
| 🎒 **Inventory System** | Weapons, armor, potions, and loot |
| 💾 **Save/Load** | Persistent campaign state |

## 🏗️ Event-Driven Architecture

```
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
│             │       │             │       │             │
│ Story       │       │ Initiative  │       │ Dialogue    │
│ Description │       │ Actions     │       │ Personality │
│ Atmosphere  │       │ Damage      │       │ Memory      │
└─────────────┘       └─────────────┘       └─────────────┘
```

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt

# Set API key for enhanced AI narratives (optional)
export GOOGLE_API_KEY="your-key"
```

### Play in Terminal

```bash
# Start a new game (interactive character creation)
python main.py play

# Start with specific character
python main.py play --name "Thorin" --class Fighter --race Dwarf

# Quick start with defaults
python main.py quick

# Roll some dice
python main.py roll 2d6+3
python main.py roll 1d20
```

### Web Interface

```bash
# Start the web server
python main.py serve

# Open http://localhost:8000 in your browser
```

### Python API

```python
from game_master import AIGameMaster, quick_start

# Quick start
gm = quick_start("Gandalf", "Wizard")
gm.play()

# Or manual setup
gm = AIGameMaster()
print(gm.new_game("Thorin", "Fighter", "Dwarf"))

# Process actions
response, continue_game = gm.process_input("examine the runes")
print(response)

# Combat
response, _ = gm.process_input("attack the goblin")
```

## 📁 Project Structure

```
ai-game-master/
├── game_master/
│   ├── __init__.py           # Main exports
│   ├── engine/
│   │   └── __init__.py       # LangGraph game engine
│   ├── systems/
│   │   └── __init__.py       # Combat, dialogue, narrative
│   ├── models/
│   │   └── __init__.py       # D&D data models
│   └── data/
│       └── __init__.py       # NPCs, locations, items, monsters
├── frontend/
│   └── index.html            # React web interface
├── saves/                     # Saved games
├── api.py                     # FastAPI backend
├── main.py                    # Rich CLI
├── requirements.txt
└── README.md
```

## 🎮 Gameplay Commands

### Exploration
| Command | Example |
|---------|---------|
| Movement | `go north`, `enter tavern`, `leave` |
| Look | `look around`, `examine runes`, `search` |
| Interact | `talk to blacksmith`, `open chest` |

### Combat
| Command | Example |
|---------|---------|
| Attack | `attack goblin`, `hit the orc` |
| Defend | `dodge`, `disengage` |
| Items | `use healing potion` |
| Spells | `cast fireball` |

### Information
| Command | Example |
|---------|---------|
| Status | `status`, `hp`, `character` |
| Inventory | `inventory`, `equipment` |
| Quests | `quests`, `journal` |
| Map | `map`, `where am I` |

## ⚔️ D&D 5e Combat System

### Initiative
- All combatants roll 1d20 + DEX modifier
- Sorted highest to lowest
- Round-based turns

### Attack Resolution
```
1d20 + ability modifier + proficiency bonus vs target AC

Natural 20 = Critical Hit (double damage dice)
Natural 1  = Critical Miss
```

### Damage Types
Slashing, Piercing, Bludgeoning, Fire, Cold, Lightning, Thunder, Poison, Acid, Necrotic, Radiant, Force, Psychic

## 👥 NPCs

### Starter NPCs

| Name | Role | Location |
|------|------|----------|
| Thordak Ironforge | Blacksmith | Village Square |
| Elara Moonwhisper | Herbalist | Herb Shop |
| Marcus Brightblade | Guard Captain | Gate |
| Old Tom | Tavern Keeper | Prancing Pony |
| Whisper | Information Broker | Tavern (shadows) |

### NPC Features
- **Personalities**: Unique traits, ideals, bonds, flaws
- **Voice**: Distinct speaking styles and mannerisms
- **Memory**: Remember previous conversations
- **Disposition**: Changes based on interactions

## 🗺️ Starter Campaign: The Whispering Caverns

### Locations

| Location | Type | Features |
|----------|------|----------|
| Village Square | Safe | Fountain, notice board, shops |
| Prancing Pony Tavern | Safe | Inn, rumors, NPCs |
| Ironforge Smithy | Safe | Weapons, armor |
| Moonwhisper's Herbs | Safe | Potions, remedies |
| Forest Road | Dangerous | Random encounters |
| Cavern Entrance | Dangerous | Dungeon entrance |

### Quests

1. **The Lost Tomb of King Thrain**
   - Find the Whispering Caverns
   - Obtain the ancient dwarven medallion
   - Enter and explore the tomb

2. **The Missing Shipment**
   - Investigate the lost supplies
   - Track down the bandits
   - Recover the goods

## 🧪 Character Classes

| Class | Hit Die | Primary Stat | Playstyle |
|-------|---------|--------------|-----------|
| Fighter | d10 | STR | Martial combat master |
| Rogue | d8 | DEX | Stealth and precision |
| Wizard | d6 | INT | Arcane spellcaster |
| Cleric | d8 | WIS | Divine healer |
| Ranger | d10 | DEX/WIS | Wilderness hunter |
| Barbarian | d12 | STR | Raging warrior |

## 🎲 Dice System

```python
from game_master import roll_dice

# Standard rolls
roll_dice("1d20")      # Attack roll
roll_dice("2d6+3")     # Damage roll
roll_dice("4d6")       # Ability scores
roll_dice("1d100")     # Percentile

# With advantage/disadvantage
roll_dice("1d20", advantage=True)
roll_dice("1d20", disadvantage=True)
```

## 🌐 Web API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/game/new` | POST | Start new game |
| `/api/game/action` | POST | Process player action |
| `/api/game/{id}` | GET | Get game state |
| `/api/dice/roll` | POST | Roll dice |
| `/api/classes` | GET | List classes |
| `/api/races` | GET | List races |
| `/ws/{game_id}` | WS | Real-time gameplay |

## 💾 Save System

Games are automatically saved and can be loaded:

```python
gm = AIGameMaster()
gm.new_game("Hero", "Fighter")

# Save
gm.save_game("saves/my_campaign.json")

# Load (future feature)
gm.load_game("saves/my_campaign.json")
```

## 🎭 Sample Session

```
DM: You stand at the entrance of the Whispering Caverns. 
    The air is cold and damp, and you hear distant echoes 
    of dripping water. Your torchlight reveals ancient 
    dwarven runes carved into the stone archway.
    
    What do you do?

> examine the runes

DM: [Investigation Check: 14 + 3 = 17]
    
    Your keen eyes study the runes carefully. You recognize 
    this as Old Dwarven script, a warning: "Beyond lies the 
    tomb of King Thrain. Only the worthy may enter."
    
    You notice a small indentation in the center of the 
    archway, about the size of a coin.

> place the dwarven medallion in the indentation

DM: The medallion fits perfectly! A deep rumble shakes 
    the cavern as the stone archway begins to glow with 
    a soft blue light. The wall before you shimmers and 
    becomes transparent, revealing a passage beyond.
    
    [Quest Updated: "The Lost Tomb" - Entered the tomb]
    
    Do you proceed into the passage?
```

## ⚙️ Configuration

```python
# Environment variables
GOOGLE_API_KEY=your-gemini-key     # For enhanced narratives
ANTHROPIC_API_KEY=your-claude-key  # Alternative LLM
OPENAI_API_KEY=your-openai-key     # Alternative LLM
```

## 🔮 Future Enhancements

- [ ] Multiplayer support
- [ ] Custom campaign builder
- [ ] Character level progression
- [ ] Spell system with slots
- [ ] ASCII dungeon maps
- [ ] Voice input/output
- [ ] Character art generation

## 📝 License

MIT License

---

*May your dice roll true, adventurer!* 🎲⚔️🐉
