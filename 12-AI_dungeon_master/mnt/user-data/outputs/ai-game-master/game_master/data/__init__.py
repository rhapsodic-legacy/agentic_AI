"""
AI Game Master - Game Data

Pre-built content:
- Starter campaigns
- NPCs with personalities
- Locations and dungeons
- Items and treasure
- Monsters and encounters
- Loot tables
"""

from ..models import (
    Campaign, Location, LocationType, NPC, NPCPersonality,
    Item, Weapon, Armor, ItemRarity, DamageType,
    Quest, QuestObjective, QuestStatus,
    AbilityScores, AbilityScore, Skill, CharacterClass, Race,
    PlayerCharacter, Inventory
)
import uuid
import random


# =============================================================================
# Pre-built NPCs
# =============================================================================

NPCS = {
    "thordak": NPC(
        character_id="npc-thordak",
        name="Thordak Ironforge",
        role="Blacksmith",
        creature_type="Humanoid",
        level=5,
        max_hp=45,
        current_hp=45,
        armor_class=14,
        abilities=AbilityScores(strength=16, dexterity=10, constitution=14, intelligence=12, wisdom=10, charisma=8),
        description="A burly dwarf with soot-stained hands and a magnificent braided beard.",
        appearance="Muscular arms covered in old burn scars, wearing a leather apron. His eyes gleam with the passion of a master craftsman.",
        personality=NPCPersonality(
            trait="Gruff but fair, takes immense pride in his work",
            ideal="Quality craftsmanship is worth any price",
            bond="His forge has been in the family for seven generations",
            flaw="Refuses to compromise on quality, even when it costs him business",
            voice="Deep, gravelly voice with a thick dwarven accent. Uses forge metaphors.",
            mannerism="Constantly examines metal objects, testing their weight and balance",
            disposition=55,
        ),
        greeting="*looks up from the anvil* What can I forge for ye?",
        farewell="May yer blade stay sharp and yer armor hold true.",
        is_merchant=True,
        home_location="village_square",
    ),
    
    "elara": NPC(
        character_id="npc-elara",
        name="Elara Moonwhisper",
        role="Herbalist/Healer",
        creature_type="Humanoid",
        level=4,
        max_hp=28,
        current_hp=28,
        armor_class=11,
        abilities=AbilityScores(strength=8, dexterity=12, constitution=12, intelligence=14, wisdom=16, charisma=14),
        description="A half-elf woman with silver-streaked auburn hair and kind, knowing eyes.",
        appearance="Wears flowing green robes adorned with pressed flowers. A gentle aura of herbal scents surrounds her.",
        personality=NPCPersonality(
            trait="Gentle and nurturing, with hidden depths of wisdom",
            ideal="All life is precious and worth preserving",
            bond="Protects an ancient grove sacred to her elven ancestors",
            flaw="Sometimes too trusting, sees good in everyone",
            voice="Soft, melodic voice with occasional elvish phrases",
            mannerism="Touches plants as she walks by, as if greeting old friends",
            disposition=70,
        ),
        greeting="Welcome, traveler. You look weary from the road. How may I help you?",
        farewell="Walk gently upon the earth, and it shall support your journey.",
        is_merchant=True,
        home_location="herbalist_shop",
    ),
    
    "marcus": NPC(
        character_id="npc-marcus",
        name="Marcus Brightblade",
        role="Captain of the Guard",
        creature_type="Humanoid",
        level=6,
        max_hp=52,
        current_hp=52,
        armor_class=18,
        abilities=AbilityScores(strength=16, dexterity=14, constitution=14, intelligence=10, wisdom=12, charisma=14),
        description="A tall, disciplined human in polished armor bearing the town's crest.",
        appearance="Clean-shaven with a military haircut and piercing blue eyes. A scar runs across his left cheek.",
        personality=NPCPersonality(
            trait="Honorable and duty-bound, but not without warmth",
            ideal="Justice must be served, but tempered with mercy",
            bond="Swore an oath to protect Willowmere after it saved him as a child",
            flaw="His rigid sense of duty sometimes blinds him to shades of gray",
            voice="Clear, commanding voice. Speaks formally but not coldly.",
            mannerism="Stands at attention even when relaxed, hand often resting on sword pommel",
            disposition=50,
        ),
        greeting="Halt. State your business in Willowmere.",
        farewell="Keep out of trouble, and we'll have no quarrel.",
        home_location="guard_barracks",
        is_hostile=False,
        attacks=[
            {"name": "Longsword", "bonus": 6, "damage": "1d8+3", "type": DamageType.SLASHING},
        ],
    ),
    
    "whisper": NPC(
        character_id="npc-whisper",
        name="Whisper",
        role="Information Broker",
        creature_type="Humanoid",
        level=5,
        max_hp=32,
        current_hp=32,
        armor_class=14,
        abilities=AbilityScores(strength=10, dexterity=16, constitution=12, intelligence=14, wisdom=14, charisma=16),
        description="A hooded figure whose face is always in shadow. Gender and race unclear.",
        appearance="Wears dark clothing that seems to absorb light. Only glinting eyes visible beneath the hood.",
        personality=NPCPersonality(
            trait="Mysterious and calculating, never reveals more than necessary",
            ideal="Information is the most valuable currency",
            bond="Owes a life debt to someone they've never revealed",
            flaw="Paranoid - always assumes someone is watching",
            voice="Soft whisper that somehow carries clearly. No identifiable accent.",
            mannerism="Never sits with their back to a door. Eyes constantly scanning.",
            disposition=40,
        ),
        greeting="*a soft voice from the shadows* You seek something. Everyone does.",
        farewell="Remember - you never saw me.",
        home_location="tavern",
        skill_proficiencies=[Skill.STEALTH, Skill.DECEPTION, Skill.INSIGHT, Skill.PERCEPTION],
    ),
    
    "old_tom": NPC(
        character_id="npc-old-tom",
        name="Old Tom",
        role="Tavern Keeper",
        creature_type="Humanoid",
        level=2,
        max_hp=18,
        current_hp=18,
        armor_class=10,
        abilities=AbilityScores(strength=10, dexterity=10, constitution=12, intelligence=12, wisdom=14, charisma=14),
        description="A portly, balding human with a perpetual smile and flour-dusted apron.",
        appearance="Rosy cheeks, laugh lines around his eyes, and surprisingly strong arms from years of barrel-lifting.",
        personality=NPCPersonality(
            trait="Friendly gossip who knows everyone's business",
            ideal="A good meal and warm hearth can solve most problems",
            bond="The Prancing Pony tavern has been his family's for three generations",
            flaw="Can't keep a secret to save his life",
            voice="Warm, booming voice that carries across the room. Laughs often.",
            mannerism="Wipes down the bar constantly, even when it's clean",
            disposition=75,
        ),
        greeting="Welcome, welcome! Come in from the cold! What'll it be?",
        farewell="Don't be a stranger now! Safe travels!",
        home_location="tavern",
        is_merchant=True,
    ),
}


# =============================================================================
# Locations
# =============================================================================

LOCATIONS = {
    "village_square": Location(
        location_id="village_square",
        name="Willowmere Village Square",
        description="The heart of the small village of Willowmere. A weathered stone fountain stands in the center, depicting a willow tree with water trickling from its leaves. Cobblestone paths radiate outward to various shops and homes.",
        location_type=LocationType.VILLAGE,
        atmosphere="The sound of merchants hawking their wares mingles with children's laughter. The smell of fresh bread wafts from a nearby bakery.",
        connections={
            "north": "tavern",
            "east": "blacksmith",
            "south": "village_gate",
            "west": "herbalist_shop",
        },
        features=["stone fountain", "notice board", "merchant stalls"],
        is_safe=True,
        is_discovered=True,
    ),
    
    "tavern": Location(
        location_id="tavern",
        name="The Prancing Pony Tavern",
        description="A cozy two-story establishment with dark wooden beams and a roaring fireplace. The common room is filled with worn but comfortable furniture, and the walls are decorated with hunting trophies and old maps.",
        location_type=LocationType.TAVERN,
        atmosphere="The warmth of the fire contrasts with the cold outside. Mugs clink, dice roll, and conversation fills the air. A bard in the corner plays a soft melody.",
        connections={
            "south": "village_square",
            "up": "tavern_rooms",
        },
        npcs=["npc-old-tom", "npc-whisper"],
        features=["fireplace", "bar counter", "notice board", "stairs to rooms"],
        is_safe=True,
    ),
    
    "blacksmith": Location(
        location_id="blacksmith",
        name="Ironforge Smithy",
        description="Heat radiates from the open forge as the rhythmic clang of hammer on anvil fills the air. Weapons and tools of all kinds line the walls, each bearing the distinctive Ironforge mark.",
        location_type=LocationType.SHOP,
        atmosphere="The air shimmers with heat. Sparks fly with each hammer strike. The smell of hot metal and coal smoke permeates everything.",
        connections={
            "west": "village_square",
        },
        npcs=["npc-thordak"],
        features=["forge", "anvil", "weapon racks", "armor stands"],
        is_safe=True,
    ),
    
    "herbalist_shop": Location(
        location_id="herbalist_shop",
        name="Moonwhisper's Herbs & Remedies",
        description="A small cottage overflowing with plants. Dried herbs hang from every rafter, potted plants crowd every surface, and mysterious bottles line the shelves. Soft light filters through stained glass windows.",
        location_type=LocationType.SHOP,
        atmosphere="A hundred different scents blend together - lavender, mint, something floral, something earthy. Wind chimes tinkle softly despite no breeze.",
        connections={
            "east": "village_square",
        },
        npcs=["npc-elara"],
        features=["herb shelves", "potion display", "garden door"],
        is_safe=True,
    ),
    
    "village_gate": Location(
        location_id="village_gate",
        name="Southern Gate of Willowmere",
        description="Heavy wooden gates reinforced with iron bands stand open during the day. Two guards in tabards bearing the village crest stand watch. Beyond lies the road south into the wilderness.",
        location_type=LocationType.VILLAGE,
        atmosphere="The guards chat idly, occasionally scanning the treeline. Merchants and travelers pass through, their carts creaking.",
        connections={
            "north": "village_square",
            "south": "forest_road",
        },
        npcs=["npc-marcus"],
        features=["wooden gates", "guard post", "road south"],
        is_safe=True,
    ),
    
    "forest_road": Location(
        location_id="forest_road",
        name="The Old Forest Road",
        description="A well-worn dirt path winds through ancient trees whose branches form a canopy overhead. Dappled sunlight creates shifting patterns on the ground. The road stretches south toward the rumored location of the Whispering Caverns.",
        location_type=LocationType.FOREST,
        atmosphere="Birds sing in the branches above. Leaves rustle in a gentle breeze. Occasionally, something larger moves in the undergrowth.",
        connections={
            "north": "village_gate",
            "south": "cavern_entrance",
            "east": "forest_clearing",
        },
        features=["ancient trees", "worn path", "milestone marker"],
        is_safe=False,
    ),
    
    "cavern_entrance": Location(
        location_id="cavern_entrance",
        name="Entrance to the Whispering Caverns",
        description="A yawning cave mouth set into a rocky hillside. Ancient dwarven runes are carved into a stone archway framing the entrance. Cold, damp air flows outward, carrying distant echoes of dripping water.",
        location_type=LocationType.CAVE,
        atmosphere="The temperature drops noticeably. Your torchlight barely penetrates the darkness within. You hear faint whispers that might be wind... or something else.",
        connections={
            "north": "forest_road",
            "in": "cavern_entrance_hall",
        },
        features=["dwarven runes", "stone archway", "mysterious indentation"],
        secrets=["The indentation accepts a dwarven medallion"],
        is_safe=False,
    ),
    
    "cavern_entrance_hall": Location(
        location_id="cavern_entrance_hall",
        name="Cavern Entrance Hall",
        description="A vast natural chamber opens before you. Stalactites hang from the ceiling like stone teeth. Multiple passages lead deeper into the darkness. Old mining equipment lies abandoned against the walls.",
        location_type=LocationType.DUNGEON,
        atmosphere="Water drips steadily somewhere in the darkness. Your footsteps echo endlessly. The air smells of earth and age.",
        connections={
            "out": "cavern_entrance",
            "east": "cavern_corridor",
            "west": "collapsed_tunnel",
        },
        features=["abandoned mining cart", "rusted pickaxes", "ancient torch sconces"],
        is_safe=False,
    ),
}


# =============================================================================
# Items and Equipment
# =============================================================================

WEAPONS = {
    "longsword": Weapon(
        item_id="wpn-longsword",
        name="Longsword",
        description="A versatile blade favored by warriors across the realms.",
        weight=3.0,
        value=15,
        damage_dice="1d8",
        damage_type=DamageType.SLASHING,
        properties=["versatile"],
    ),
    "shortbow": Weapon(
        item_id="wpn-shortbow",
        name="Shortbow",
        description="A compact bow suitable for hunting and combat.",
        weight=2.0,
        value=25,
        damage_dice="1d6",
        damage_type=DamageType.PIERCING,
        properties=["ammunition", "two-handed"],
        range_normal=80,
        range_long=320,
    ),
    "dagger": Weapon(
        item_id="wpn-dagger",
        name="Dagger",
        description="A simple blade, easily concealed.",
        weight=1.0,
        value=2,
        damage_dice="1d4",
        damage_type=DamageType.PIERCING,
        properties=["finesse", "light", "thrown"],
        range_normal=20,
        range_long=60,
    ),
    "greataxe": Weapon(
        item_id="wpn-greataxe",
        name="Greataxe",
        description="A massive two-handed axe favored by barbarians.",
        weight=7.0,
        value=30,
        damage_dice="1d12",
        damage_type=DamageType.SLASHING,
        properties=["heavy", "two-handed"],
    ),
    "staff": Weapon(
        item_id="wpn-staff",
        name="Quarterstaff",
        description="A simple but effective wooden staff.",
        weight=4.0,
        value=2,
        damage_dice="1d6",
        damage_type=DamageType.BLUDGEONING,
        properties=["versatile"],
    ),
}

MAGIC_WEAPONS = {
    "flametongue": Weapon(
        item_id="wpn-flametongue",
        name="Flametongue Longsword",
        description="This sword's blade bursts into flame when you speak its command word. The flames shed bright light in a 40-foot radius.",
        weight=3.0,
        value=5000,
        rarity=ItemRarity.RARE,
        magical=True,
        damage_dice="1d8+2d6",
        damage_type=DamageType.SLASHING,
        properties=["versatile"],
        attack_bonus=0,
        damage_bonus=0,
    ),
    "dwarven_medallion": Item(
        item_id="item-dwarven-medallion",
        name="Ancient Dwarven Medallion",
        description="A heavy bronze medallion bearing the seal of the ancient dwarven kingdom. It hums faintly when near dwarven architecture.",
        weight=0.5,
        value=100,
        rarity=ItemRarity.UNCOMMON,
        magical=True,
    ),
}

ARMORS = {
    "leather": Armor(
        item_id="arm-leather",
        name="Leather Armor",
        description="Light armor made of cured leather.",
        weight=10.0,
        value=10,
        armor_class=11,
        armor_type="light",
    ),
    "chain_mail": Armor(
        item_id="arm-chainmail",
        name="Chain Mail",
        description="Heavy armor made of interlocking metal rings.",
        weight=55.0,
        value=75,
        armor_class=16,
        armor_type="heavy",
        strength_required=13,
        stealth_disadvantage=True,
    ),
    "shield": Armor(
        item_id="arm-shield",
        name="Shield",
        description="A wooden or metal shield.",
        weight=6.0,
        value=10,
        armor_class=2,
        armor_type="shield",
    ),
}

CONSUMABLES = {
    "healing_potion": Item(
        item_id="cons-healing",
        name="Potion of Healing",
        description="A red liquid that shimmers when agitated. Heals 2d4+2 hit points.",
        weight=0.5,
        value=50,
        consumable=True,
    ),
    "greater_healing_potion": Item(
        item_id="cons-greater-healing",
        name="Potion of Greater Healing",
        description="A crimson potion that glows faintly. Heals 4d4+4 hit points.",
        weight=0.5,
        value=150,
        rarity=ItemRarity.UNCOMMON,
        consumable=True,
    ),
    "antidote": Item(
        item_id="cons-antidote",
        name="Antidote",
        description="A vial containing a murky liquid that neutralizes poison.",
        weight=0.1,
        value=25,
        consumable=True,
    ),
    "torch": Item(
        item_id="cons-torch",
        name="Torch",
        description="A wooden torch that burns for 1 hour, providing bright light in a 20-foot radius.",
        weight=1.0,
        value=0.01,
        consumable=True,
    ),
    "rations": Item(
        item_id="cons-rations",
        name="Rations (1 day)",
        description="Dried food suitable for travel.",
        weight=2.0,
        value=0.5,
        consumable=True,
    ),
}


# =============================================================================
# Loot Tables
# =============================================================================

LOOT_TABLES = {
    "common_enemy": {
        "gold_range": (1, 10),
        "items": [
            (0.3, "cons-healing"),
            (0.5, "wpn-dagger"),
            (0.2, "cons-torch"),
        ],
    },
    "dungeon_chest": {
        "gold_range": (20, 100),
        "items": [
            (0.5, "cons-healing"),
            (0.3, "cons-greater-healing"),
            (0.2, "item-dwarven-medallion"),
            (0.4, "wpn-longsword"),
        ],
    },
    "boss_loot": {
        "gold_range": (100, 500),
        "items": [
            (0.8, "cons-greater-healing"),
            (0.3, "wpn-flametongue"),
            (1.0, "item-dwarven-medallion"),
        ],
    },
}


def generate_loot(table_name: str) -> tuple[int, list[Item]]:
    """Generate loot from a loot table."""
    if table_name not in LOOT_TABLES:
        return 0, []
    
    table = LOOT_TABLES[table_name]
    
    # Gold
    gold = random.randint(*table["gold_range"])
    
    # Items
    items = []
    all_items = {**WEAPONS, **MAGIC_WEAPONS, **ARMORS, **CONSUMABLES}
    
    for chance, item_id in table["items"]:
        if random.random() < chance:
            if item_id in all_items:
                items.append(all_items[item_id])
    
    return gold, items


# =============================================================================
# Monsters
# =============================================================================

MONSTERS = {
    "goblin": {
        "name": "Goblin",
        "creature_type": "Humanoid",
        "challenge_rating": 0.25,
        "hp": 7,
        "ac": 15,
        "abilities": AbilityScores(strength=8, dexterity=14, constitution=10, intelligence=10, wisdom=8, charisma=8),
        "speed": 30,
        "attacks": [
            {"name": "Scimitar", "bonus": 4, "damage": "1d6+2", "type": DamageType.SLASHING},
            {"name": "Shortbow", "bonus": 4, "damage": "1d6+2", "type": DamageType.PIERCING},
        ],
        "description": "A small, ugly humanoid with pointed ears and sharp teeth.",
        "loot_table": "common_enemy",
    },
    "skeleton": {
        "name": "Skeleton",
        "creature_type": "Undead",
        "challenge_rating": 0.25,
        "hp": 13,
        "ac": 13,
        "abilities": AbilityScores(strength=10, dexterity=14, constitution=15, intelligence=6, wisdom=8, charisma=5),
        "speed": 30,
        "attacks": [
            {"name": "Shortsword", "bonus": 4, "damage": "1d6+2", "type": DamageType.PIERCING},
            {"name": "Shortbow", "bonus": 4, "damage": "1d6+2", "type": DamageType.PIERCING},
        ],
        "vulnerabilities": [DamageType.BLUDGEONING],
        "immunities": [DamageType.POISON],
        "description": "Animated bones held together by dark magic, its empty eye sockets glow with unholy light.",
        "loot_table": "common_enemy",
    },
    "giant_spider": {
        "name": "Giant Spider",
        "creature_type": "Beast",
        "challenge_rating": 1,
        "hp": 26,
        "ac": 14,
        "abilities": AbilityScores(strength=14, dexterity=16, constitution=12, intelligence=2, wisdom=11, charisma=4),
        "speed": 30,
        "attacks": [
            {"name": "Bite", "bonus": 5, "damage": "1d8+3", "type": DamageType.PIERCING},
        ],
        "description": "A horse-sized spider with glistening fangs dripping venom.",
        "loot_table": "common_enemy",
    },
    "cave_troll": {
        "name": "Cave Troll",
        "creature_type": "Giant",
        "challenge_rating": 5,
        "hp": 84,
        "ac": 15,
        "abilities": AbilityScores(strength=18, dexterity=13, constitution=20, intelligence=7, wisdom=9, charisma=7),
        "speed": 30,
        "attacks": [
            {"name": "Claw", "bonus": 7, "damage": "2d6+4", "type": DamageType.SLASHING},
            {"name": "Bite", "bonus": 7, "damage": "1d8+4", "type": DamageType.PIERCING},
        ],
        "regeneration": 10,
        "vulnerabilities": [DamageType.FIRE, DamageType.ACID],
        "description": "A massive, regenerating horror with mottled gray skin and long claws.",
        "loot_table": "boss_loot",
    },
}


def create_monster(monster_type: str) -> NPC:
    """Create a monster NPC from template."""
    if monster_type not in MONSTERS:
        return None
    
    template = MONSTERS[monster_type]
    
    return NPC(
        character_id=f"monster-{uuid.uuid4().hex[:8]}",
        name=template["name"],
        role="Monster",
        creature_type=template["creature_type"],
        level=max(1, int(template["challenge_rating"] * 2)),
        max_hp=template["hp"],
        current_hp=template["hp"],
        armor_class=template["ac"],
        abilities=template["abilities"],
        speed=template.get("speed", 30),
        description=template["description"],
        challenge_rating=template["challenge_rating"],
        attacks=template["attacks"],
        is_hostile=True,
    )


# =============================================================================
# Quests
# =============================================================================

STARTER_QUESTS = [
    Quest(
        quest_id="quest-lost-tomb",
        name="The Lost Tomb of King Thrain",
        description="Legends speak of an ancient dwarven king buried in the Whispering Caverns with his greatest treasures. Find the tomb and discover what secrets it holds.",
        status=QuestStatus.NOT_STARTED,
        objectives=[
            QuestObjective(
                objective_id="obj-1",
                description="Find the entrance to the Whispering Caverns",
            ),
            QuestObjective(
                objective_id="obj-2",
                description="Obtain the ancient dwarven medallion",
            ),
            QuestObjective(
                objective_id="obj-3",
                description="Enter the tomb of King Thrain",
            ),
            QuestObjective(
                objective_id="obj-4",
                description="Discover the secret of the tomb",
            ),
        ],
        gold_reward=500,
        xp_reward=300,
        quest_giver="Old Tom",
        locations=["cavern_entrance", "cavern_entrance_hall"],
    ),
    Quest(
        quest_id="quest-missing-shipment",
        name="The Missing Shipment",
        description="A shipment of supplies never arrived from the south. Thordak is worried - investigate what happened.",
        status=QuestStatus.NOT_STARTED,
        objectives=[
            QuestObjective(
                objective_id="obj-1",
                description="Talk to Thordak about the missing shipment",
            ),
            QuestObjective(
                objective_id="obj-2",
                description="Search the forest road for clues",
            ),
            QuestObjective(
                objective_id="obj-3",
                description="Deal with the bandits",
            ),
        ],
        gold_reward=150,
        xp_reward=150,
        quest_giver="Thordak",
        locations=["forest_road"],
    ),
]


# =============================================================================
# Starter Campaign
# =============================================================================

def create_starter_campaign() -> Campaign:
    """Create the starter campaign."""
    return Campaign(
        campaign_id=f"campaign-{uuid.uuid4().hex[:8]}",
        name="The Whispering Caverns",
        description="""
Welcome to Willowmere, a peaceful village on the edge of civilization. 
Beyond the safety of its walls, ancient ruins and dark caves hold secrets 
waiting to be discovered. Rumors speak of a lost dwarven tomb filled with 
treasure—and danger.

Your adventure begins in the village square, where opportunity and peril 
await in equal measure.
""",
        setting="Forgotten Realms",
        current_chapter="Chapter 1: The Village",
        current_scene="You arrive in Willowmere as the sun begins to set.",
    )


def create_starter_character(
    name: str,
    character_class: CharacterClass = CharacterClass.FIGHTER,
    race: Race = Race.HUMAN,
) -> PlayerCharacter:
    """Create a starter player character."""
    
    # Class-based stats
    class_stats = {
        CharacterClass.FIGHTER: AbilityScores(16, 14, 14, 10, 12, 8),
        CharacterClass.ROGUE: AbilityScores(10, 16, 12, 14, 12, 14),
        CharacterClass.WIZARD: AbilityScores(8, 14, 12, 16, 14, 10),
        CharacterClass.CLERIC: AbilityScores(14, 10, 14, 10, 16, 12),
        CharacterClass.RANGER: AbilityScores(12, 16, 14, 10, 14, 10),
        CharacterClass.BARBARIAN: AbilityScores(16, 14, 16, 8, 12, 8),
    }
    
    # Class-based HP
    class_hp = {
        CharacterClass.FIGHTER: 12,
        CharacterClass.ROGUE: 10,
        CharacterClass.WIZARD: 8,
        CharacterClass.CLERIC: 10,
        CharacterClass.RANGER: 12,
        CharacterClass.BARBARIAN: 14,
    }
    
    abilities = class_stats.get(character_class, AbilityScores())
    base_hp = class_hp.get(character_class, 10)
    max_hp = base_hp + abilities.get_modifier(AbilityScore.CONSTITUTION)
    
    character = PlayerCharacter(
        character_id=f"player-{uuid.uuid4().hex[:8]}",
        name=name,
        character_class=character_class,
        race=race,
        level=1,
        max_hp=max_hp,
        current_hp=max_hp,
        abilities=abilities,
        proficiency_bonus=2,
    )
    
    # Add starting equipment
    if character_class == CharacterClass.FIGHTER:
        character.inventory.main_hand = WEAPONS["longsword"]
        character.inventory.armor = ARMORS["chain_mail"]
        character.inventory.off_hand = ARMORS["shield"]
        character.armor_class = 18
        character.skill_proficiencies = [Skill.ATHLETICS, Skill.INTIMIDATION]
        character.saving_throw_proficiencies = [AbilityScore.STRENGTH, AbilityScore.CONSTITUTION]
    elif character_class == CharacterClass.ROGUE:
        character.inventory.main_hand = WEAPONS["shortbow"]
        character.inventory.armor = ARMORS["leather"]
        character.armor_class = 14
        character.skill_proficiencies = [Skill.STEALTH, Skill.ACROBATICS, Skill.PERCEPTION, Skill.DECEPTION]
        character.saving_throw_proficiencies = [AbilityScore.DEXTERITY, AbilityScore.INTELLIGENCE]
    elif character_class == CharacterClass.WIZARD:
        character.inventory.main_hand = WEAPONS["staff"]
        character.armor_class = 12
        character.skill_proficiencies = [Skill.ARCANA, Skill.INVESTIGATION]
        character.saving_throw_proficiencies = [AbilityScore.INTELLIGENCE, AbilityScore.WISDOM]
        character.spells_known = ["Fire Bolt", "Mage Hand", "Magic Missile", "Shield"]
    elif character_class == CharacterClass.CLERIC:
        character.inventory.main_hand = WEAPONS["staff"]
        character.inventory.armor = ARMORS["chain_mail"]
        character.armor_class = 16
        character.skill_proficiencies = [Skill.MEDICINE, Skill.RELIGION]
        character.saving_throw_proficiencies = [AbilityScore.WISDOM, AbilityScore.CHARISMA]
        character.spells_known = ["Sacred Flame", "Guidance", "Cure Wounds", "Bless"]
    
    # Add basic supplies
    character.inventory.add_item(CONSUMABLES["healing_potion"])
    character.inventory.add_item(CONSUMABLES["torch"])
    character.inventory.gold = 15
    
    return character
