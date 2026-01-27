"""
AI Game Master - Game Systems

Core systems for:
- Combat resolution (D&D 5e rules)
- Skill checks and saving throws
- NPC dialogue and personality
- Narrative generation
- Loot and rewards
"""

from typing import Optional
import random
import os

from ..models import (
    GameState, Character, PlayerCharacter, NPC, Combatant, CombatState, CombatAction,
    DiceRoll, roll_dice, get_modifier,
    AbilityScore, Skill, DamageType, ActionType, Condition, EventType,
    Item, Weapon, Quest, QuestStatus, Location
)
from ..data import MONSTERS, create_monster, generate_loot, NPCS


# =============================================================================
# Dice and Skill Check System
# =============================================================================

class DiceSystem:
    """Handles all dice rolling and checks."""
    
    @staticmethod
    def roll(dice_str: str, advantage: bool = False, disadvantage: bool = False) -> DiceRoll:
        """Roll dice with standard notation."""
        return roll_dice(dice_str, advantage, disadvantage)
    
    @staticmethod
    def skill_check(
        character: Character,
        skill: Skill,
        dc: int,
        advantage: bool = False,
        disadvantage: bool = False,
    ) -> tuple[DiceRoll, bool, str]:
        """
        Make a skill check.
        
        Returns: (roll, success, description)
        """
        modifier = character.get_skill_modifier(skill)
        roll = roll_dice("1d20", advantage, disadvantage)
        total = roll.total + modifier
        success = total >= dc
        
        # Generate description
        if roll.is_critical:
            desc = f"Critical Success! Natural 20!"
        elif roll.is_fumble:
            desc = f"Critical Failure! Natural 1!"
        elif success:
            desc = f"Success! ({roll.natural} + {modifier} = {total} vs DC {dc})"
        else:
            desc = f"Failure. ({roll.natural} + {modifier} = {total} vs DC {dc})"
        
        return roll, success, desc
    
    @staticmethod
    def saving_throw(
        character: Character,
        ability: AbilityScore,
        dc: int,
        advantage: bool = False,
        disadvantage: bool = False,
    ) -> tuple[DiceRoll, bool, str]:
        """
        Make a saving throw.
        
        Returns: (roll, success, description)
        """
        modifier = character.get_saving_throw_modifier(ability)
        roll = roll_dice("1d20", advantage, disadvantage)
        total = roll.total + modifier
        success = total >= dc
        
        if roll.is_critical:
            desc = f"Natural 20! Automatic success!"
            success = True
        elif roll.is_fumble:
            desc = f"Natural 1! Automatic failure!"
            success = False
        elif success:
            desc = f"Save successful! ({roll.natural} + {modifier} = {total} vs DC {dc})"
        else:
            desc = f"Save failed! ({roll.natural} + {modifier} = {total} vs DC {dc})"
        
        return roll, success, desc
    
    @staticmethod
    def contest(
        actor: Character,
        actor_skill: Skill,
        target: Character,
        target_skill: Skill,
    ) -> tuple[DiceRoll, DiceRoll, bool, str]:
        """
        Make a contested skill check.
        
        Returns: (actor_roll, target_roll, actor_wins, description)
        """
        actor_mod = actor.get_skill_modifier(actor_skill)
        target_mod = target.get_skill_modifier(target_skill)
        
        actor_roll = roll_dice("1d20")
        target_roll = roll_dice("1d20")
        
        actor_total = actor_roll.total + actor_mod
        target_total = target_roll.total + target_mod
        
        actor_wins = actor_total > target_total  # Ties go to defender
        
        winner = actor.name if actor_wins else target.name
        desc = f"{actor.name} ({actor_total}) vs {target.name} ({target_total}): {winner} wins!"
        
        return actor_roll, target_roll, actor_wins, desc


# =============================================================================
# Combat System
# =============================================================================

class CombatSystem:
    """Handles D&D 5e combat mechanics."""
    
    def __init__(self, state: GameState):
        self.state = state
        self.dice = DiceSystem()
    
    def start_combat(self, enemies: list[NPC]) -> CombatState:
        """Initialize combat with enemy NPCs."""
        import uuid
        
        combat = CombatState(
            combat_id=f"combat-{uuid.uuid4().hex[:8]}",
            is_active=True,
            round_number=1,
        )
        
        # Add players
        for player in self.state.players:
            roll = player.roll_initiative()
            combat.combatants.append(Combatant(
                character=player,
                initiative=player.initiative,
            ))
        
        # Add enemies
        for enemy in enemies:
            roll = enemy.roll_initiative()
            combat.combatants.append(Combatant(
                character=enemy,
                initiative=enemy.initiative,
            ))
        
        # Sort by initiative (highest first)
        combat.combatants.sort(key=lambda c: c.initiative, reverse=True)
        
        self.state.combat_state = combat
        self.state.add_event(
            EventType.COMBAT_START,
            f"Combat begins! Initiative order: {', '.join(combat.initiative_order)}",
            importance=3,
        )
        
        return combat
    
    def get_attack_roll(
        self,
        attacker: Character,
        weapon: Weapon = None,
        advantage: bool = False,
        disadvantage: bool = False,
    ) -> tuple[DiceRoll, int]:
        """
        Make an attack roll.
        
        Returns: (roll, total_attack)
        """
        # Determine ability modifier
        if weapon and "finesse" in weapon.properties:
            # Use higher of STR or DEX
            str_mod = attacker.abilities.get_modifier(AbilityScore.STRENGTH)
            dex_mod = attacker.abilities.get_modifier(AbilityScore.DEXTERITY)
            ability_mod = max(str_mod, dex_mod)
        elif weapon and weapon.range_long > 0:
            # Ranged weapon uses DEX
            ability_mod = attacker.abilities.get_modifier(AbilityScore.DEXTERITY)
        else:
            # Melee uses STR
            ability_mod = attacker.abilities.get_modifier(AbilityScore.STRENGTH)
        
        # Add proficiency bonus (assuming proficiency)
        prof_bonus = attacker.proficiency_bonus
        
        # Weapon bonus
        weapon_bonus = weapon.attack_bonus if weapon else 0
        
        roll = roll_dice("1d20", advantage, disadvantage)
        total = roll.total + ability_mod + prof_bonus + weapon_bonus
        
        return roll, total
    
    def get_damage_roll(
        self,
        attacker: Character,
        weapon: Weapon,
        critical: bool = False,
    ) -> tuple[DiceRoll, int, DamageType]:
        """
        Roll damage.
        
        Returns: (roll, total_damage, damage_type)
        """
        # Double dice on crit
        dice_str = weapon.damage_dice
        if critical:
            # Parse and double the dice
            import re
            match = re.match(r'(\d+)d(\d+)', dice_str)
            if match:
                num = int(match.group(1)) * 2
                die = match.group(2)
                dice_str = f"{num}d{die}"
        
        # Determine ability modifier
        if "finesse" in weapon.properties:
            str_mod = attacker.abilities.get_modifier(AbilityScore.STRENGTH)
            dex_mod = attacker.abilities.get_modifier(AbilityScore.DEXTERITY)
            ability_mod = max(str_mod, dex_mod)
        elif weapon.range_long > 0:
            ability_mod = attacker.abilities.get_modifier(AbilityScore.DEXTERITY)
        else:
            ability_mod = attacker.abilities.get_modifier(AbilityScore.STRENGTH)
        
        roll = roll_dice(dice_str)
        total = roll.total + ability_mod + weapon.damage_bonus
        
        return roll, max(1, total), weapon.damage_type
    
    def attack(
        self,
        attacker: Character,
        target: Character,
        weapon: Weapon = None,
        advantage: bool = False,
        disadvantage: bool = False,
    ) -> CombatAction:
        """
        Perform an attack.
        
        Returns: CombatAction with results
        """
        # Default weapon
        if weapon is None:
            weapon = Weapon(
                item_id="unarmed",
                name="Unarmed Strike",
                damage_dice="1d4",
                damage_type=DamageType.BLUDGEONING,
            )
        
        # Attack roll
        attack_roll, attack_total = self.get_attack_roll(
            attacker, weapon, advantage, disadvantage
        )
        
        # Determine hit
        critical = attack_roll.is_critical
        fumble = attack_roll.is_fumble
        
        if fumble:
            hit = False
        elif critical:
            hit = True
        else:
            hit = attack_total >= target.armor_class
        
        # Damage if hit
        damage_dealt = 0
        damage_roll = None
        damage_type = None
        
        if hit:
            damage_roll, damage_dealt, damage_type = self.get_damage_roll(
                attacker, weapon, critical
            )
            actual_damage = target.take_damage(damage_dealt, damage_type)
        
        # Generate description
        if fumble:
            desc = f"{attacker.name} swings wildly and misses completely!"
        elif critical:
            desc = f"CRITICAL HIT! {attacker.name} strikes {target.name} with {weapon.name} for {damage_dealt} {damage_type.value} damage!"
        elif hit:
            desc = f"{attacker.name} hits {target.name} with {weapon.name} for {damage_dealt} {damage_type.value} damage!"
        else:
            desc = f"{attacker.name}'s attack misses {target.name}. (Rolled {attack_total} vs AC {target.armor_class})"
        
        if hit and not target.is_alive:
            desc += f" {target.name} falls!"
        elif hit and target.is_bloodied:
            desc += f" {target.name} is bloodied!"
        
        action = CombatAction(
            actor_id=attacker.character_id,
            actor_name=attacker.name,
            action_type=ActionType.ACTION,
            action_name=f"Attack with {weapon.name}",
            target_id=target.character_id,
            target_name=target.name,
            roll=attack_roll,
            damage_roll=damage_roll,
            damage_dealt=damage_dealt,
            damage_type=damage_type,
            hit=hit,
            critical=critical,
            description=desc,
        )
        
        # Log action
        if self.state.combat_state:
            self.state.combat_state.action_log.append(action)
        
        self.state.add_event(
            EventType.COMBAT_ACTION,
            desc,
            actor_id=attacker.character_id,
            target_id=target.character_id,
        )
        
        return action
    
    def monster_attack(self, monster: NPC, target: Character) -> CombatAction:
        """Monster makes an attack using its stat block."""
        if not monster.attacks:
            return self.attack(monster, target)
        
        # Choose a random attack
        attack_info = random.choice(monster.attacks)
        
        # Attack roll
        roll = roll_dice("1d20")
        attack_total = roll.total + attack_info["bonus"]
        
        critical = roll.is_critical
        fumble = roll.is_fumble
        
        if fumble:
            hit = False
        elif critical:
            hit = True
        else:
            hit = attack_total >= target.armor_class
        
        # Damage
        damage_dealt = 0
        damage_roll = None
        
        if hit:
            damage_str = attack_info["damage"]
            if critical:
                # Double the dice
                import re
                match = re.match(r'(\d+)d(\d+)', damage_str)
                if match:
                    num = int(match.group(1)) * 2
                    die = match.group(2)
                    mod_match = re.search(r'[+-]\d+', damage_str)
                    mod = mod_match.group(0) if mod_match else ""
                    damage_str = f"{num}d{die}{mod}"
            
            damage_roll = roll_dice(damage_str)
            damage_dealt = damage_roll.total
            target.take_damage(damage_dealt, attack_info.get("type"))
        
        # Description
        if fumble:
            desc = f"{monster.name}'s {attack_info['name']} attack misses wildly!"
        elif critical:
            desc = f"CRITICAL! {monster.name}'s {attack_info['name']} strikes {target.name} for {damage_dealt} damage!"
        elif hit:
            desc = f"{monster.name} hits {target.name} with {attack_info['name']} for {damage_dealt} damage!"
        else:
            desc = f"{monster.name}'s {attack_info['name']} misses {target.name}."
        
        if hit and not target.is_alive:
            desc += f" {target.name} falls unconscious!"
        elif hit and target.is_bloodied:
            desc += f" {target.name} looks badly hurt!"
        
        action = CombatAction(
            actor_id=monster.character_id,
            actor_name=monster.name,
            action_type=ActionType.ACTION,
            action_name=attack_info["name"],
            target_id=target.character_id,
            target_name=target.name,
            roll=roll,
            damage_roll=damage_roll,
            damage_dealt=damage_dealt,
            hit=hit,
            critical=critical,
            description=desc,
        )
        
        if self.state.combat_state:
            self.state.combat_state.action_log.append(action)
        
        return action
    
    def end_combat(self, victory: bool = True) -> str:
        """End combat and generate summary."""
        if not self.state.combat_state:
            return "No combat to end."
        
        combat = self.state.combat_state
        combat.is_active = False
        
        # Calculate results
        total_damage_dealt = sum(
            a.damage_dealt for a in combat.action_log 
            if a.actor_id.startswith("player")
        )
        
        if victory:
            # Award XP and loot
            xp_earned = 50 * combat.round_number  # Simplified XP
            
            desc = f"""
Combat ends in VICTORY after {combat.round_number} rounds!

Total damage dealt: {total_damage_dealt}
XP earned: {xp_earned}
"""
            self.state.add_event(
                EventType.COMBAT_END,
                f"Victory! Combat lasted {combat.round_number} rounds.",
                importance=3,
            )
        else:
            desc = f"The party has been defeated after {combat.round_number} rounds..."
            self.state.add_event(
                EventType.COMBAT_END,
                "Defeat in combat.",
                importance=4,
            )
        
        self.state.combat_state = None
        return desc
    
    def get_combat_status(self) -> str:
        """Get current combat status."""
        if not self.state.combat_state:
            return "Not in combat."
        
        combat = self.state.combat_state
        status = f"""
⚔️ COMBAT - Round {combat.round_number}

Initiative Order:
"""
        for i, combatant in enumerate(combat.combatants):
            char = combatant.character
            hp_bar = "█" * (char.current_hp * 10 // char.max_hp) + "░" * (10 - char.current_hp * 10 // char.max_hp)
            current = "→ " if i == combat.current_turn_index else "  "
            status += f"{current}{char.name}: [{hp_bar}] {char.current_hp}/{char.max_hp} HP (Init: {combatant.initiative})\n"
        
        if combat.current_combatant:
            status += f"\nCurrent Turn: {combat.current_combatant.character.name}"
        
        return status


# =============================================================================
# NPC Dialogue System
# =============================================================================

class DialogueSystem:
    """Handles NPC conversations and personality."""
    
    def __init__(self, state: GameState):
        self.state = state
        self.llm = self._get_llm()
    
    def _get_llm(self):
        """Get LLM for dialogue generation."""
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                temperature=0.8,
            )
        except:
            return None
    
    def generate_npc_response(
        self,
        npc: NPC,
        player_message: str,
        context: str = "",
    ) -> str:
        """Generate an NPC's response to player dialogue."""
        
        # Build prompt
        prompt = f"""You are playing the role of {npc.name}, an NPC in a D&D game.

CHARACTER INFO:
{npc.get_dialogue_context()}

DESCRIPTION: {npc.description}
APPEARANCE: {npc.appearance}

CONVERSATION RULES:
- Stay completely in character
- Use the speaking style and mannerisms described
- React based on your disposition toward the party
- Remember topics previously discussed
- Keep responses concise (2-4 sentences usually)
- Don't break character or mention you're an AI

{context}

The player says: "{player_message}"

Respond as {npc.name}:"""

        if self.llm:
            try:
                response = self.llm.invoke(prompt)
                npc_response = response.content
            except:
                npc_response = self._generate_fallback_response(npc, player_message)
        else:
            npc_response = self._generate_fallback_response(npc, player_message)
        
        # Update NPC memory
        npc.personality.topics_discussed.append(player_message[:50])
        
        return npc_response
    
    def _generate_fallback_response(self, npc: NPC, message: str) -> str:
        """Generate a simple response without LLM."""
        message_lower = message.lower()
        
        # Check for common interactions
        if any(word in message_lower for word in ["hello", "hi", "greetings", "hey"]):
            return npc.greeting or f"{npc.name} nods in greeting."
        
        if any(word in message_lower for word in ["bye", "goodbye", "farewell", "leave"]):
            return npc.farewell or f"{npc.name} waves goodbye."
        
        if any(word in message_lower for word in ["buy", "sell", "shop", "wares"]) and npc.is_merchant:
            return f"\"Of course! Let me show you what I have.\" {npc.name} gestures to their wares."
        
        if any(word in message_lower for word in ["quest", "work", "job", "help"]):
            return f"{npc.name} considers for a moment. \"There might be something you could help with...\""
        
        # Generic responses based on disposition
        if npc.personality.disposition > 60:
            responses = [
                f"{npc.name} smiles warmly. \"Happy to help, friend.\"",
                f"\"Interesting,\" {npc.name} says thoughtfully.",
                f"{npc.name} nods with interest.",
            ]
        elif npc.personality.disposition < 40:
            responses = [
                f"{npc.name} eyes you suspiciously.",
                f"\"What do you want?\" {npc.name} asks curtly.",
                f"{npc.name} seems uninterested.",
            ]
        else:
            responses = [
                f"{npc.name} considers your words.",
                f"\"I see,\" {npc.name} responds neutrally.",
                f"{npc.name} nods slowly.",
            ]
        
        return random.choice(responses)
    
    def adjust_disposition(self, npc: NPC, amount: int, reason: str = ""):
        """Adjust an NPC's disposition toward the party."""
        old_disposition = npc.personality.disposition
        npc.personality.disposition = max(0, min(100, npc.personality.disposition + amount))
        
        # Update relationship status
        if npc.personality.disposition > 70:
            npc.personality.relationship = "friendly"
        elif npc.personality.disposition < 30:
            npc.personality.relationship = "hostile"
        else:
            npc.personality.relationship = "neutral"


# =============================================================================
# Narrative Engine
# =============================================================================

class NarrativeEngine:
    """Generates immersive narrative descriptions."""
    
    def __init__(self, state: GameState):
        self.state = state
        self.llm = self._get_llm()
    
    def _get_llm(self):
        """Get LLM for narrative generation."""
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                temperature=0.9,
            )
        except:
            return None
    
    def describe_scene(self, location: Location, entering: bool = True) -> str:
        """Generate a scene description."""
        
        if self.llm:
            time_of_day = self.state.time.time_of_day
            
            prompt = f"""You are the Dungeon Master for a D&D game. Describe this location to the players.

LOCATION: {location.name}
TYPE: {location.location_type.value}
BASE DESCRIPTION: {location.description}
ATMOSPHERE: {location.atmosphere}
TIME: {time_of_day}

FEATURES TO INCLUDE: {', '.join(location.features) if location.features else 'None specific'}

Write an immersive, atmospheric description (3-5 sentences). 
{"The players are entering this location for the first time." if entering else "The players are already here."}
Use sensory details (sight, sound, smell) appropriate to the {time_of_day}.
End with a subtle prompt for player action.

Description:"""

            try:
                response = self.llm.invoke(prompt)
                return response.content
            except:
                pass
        
        # Fallback description
        desc = f"**{location.name}**\n\n{location.description}"
        if location.atmosphere:
            desc += f"\n\n{location.atmosphere}"
        
        if location.features:
            desc += f"\n\nYou notice: {', '.join(location.features)}."
        
        if location.connections:
            exits = [f"{direction}" for direction in location.connections.keys()]
            desc += f"\n\n*Exits: {', '.join(exits)}*"
        
        desc += "\n\nWhat do you do?"
        
        return desc
    
    def describe_combat_start(self, enemies: list[NPC]) -> str:
        """Generate combat encounter description."""
        
        enemy_names = [e.name for e in enemies]
        enemy_descriptions = [e.description for e in enemies]
        
        if self.llm:
            prompt = f"""You are the Dungeon Master. Combat is about to begin!

ENEMIES: {', '.join(enemy_names)}
DESCRIPTIONS: {' '.join(enemy_descriptions)}
LOCATION: {self.state.world_map.get_current_location().name if self.state.world_map.get_current_location() else 'Unknown'}

Write an exciting, tense description of the enemies appearing/attacking (2-3 sentences).
Be dramatic but concise. End by announcing combat has begun.

Description:"""

            try:
                response = self.llm.invoke(prompt)
                return response.content
            except:
                pass
        
        # Fallback
        if len(enemies) == 1:
            return f"A {enemies[0].name} appears! {enemies[0].description}\n\n⚔️ **ROLL FOR INITIATIVE!**"
        else:
            return f"You are ambushed by {', '.join(enemy_names)}!\n\n⚔️ **ROLL FOR INITIATIVE!**"
    
    def describe_action_result(
        self,
        action: str,
        skill: Skill,
        roll: DiceRoll,
        success: bool,
        dc: int,
    ) -> str:
        """Generate description of a skill check result."""
        
        if self.llm:
            prompt = f"""You are the Dungeon Master. A player just attempted an action.

ACTION: {action}
SKILL: {skill.display_name}
ROLL: {roll.natural} + modifier = {roll.total} vs DC {dc}
RESULT: {"SUCCESS" if success else "FAILURE"}
{"CRITICAL SUCCESS (Natural 20)!" if roll.is_critical else ""}
{"CRITICAL FAILURE (Natural 1)!" if roll.is_fumble else ""}

Write a brief, dramatic description of what happens (2-3 sentences).
Make the outcome match the roll - spectacular for crits, disastrous for fumbles.

Description:"""

            try:
                response = self.llm.invoke(prompt)
                return response.content
            except:
                pass
        
        # Fallback descriptions
        if roll.is_critical:
            return f"Against all odds, you succeed spectacularly! (Natural 20!)"
        elif roll.is_fumble:
            return f"Things go terribly wrong... (Natural 1!)"
        elif success:
            return f"You succeed! ({roll.total} vs DC {dc})"
        else:
            return f"You fail. ({roll.total} vs DC {dc})"
    
    def generate_campaign_summary(self) -> str:
        """Generate a summary of the campaign so far."""
        
        recent_events = self.state.get_recent_events(10)
        event_descriptions = [e.description for e in recent_events]
        
        active_quests = self.state.get_active_quests()
        quest_names = [q.name for q in active_quests]
        
        if self.llm:
            prompt = f"""Summarize the recent events of this D&D campaign.

CAMPAIGN: {self.state.campaign.name}
RECENT EVENTS:
{chr(10).join(f'- {e}' for e in event_descriptions)}

ACTIVE QUESTS: {', '.join(quest_names) if quest_names else 'None'}

Write a brief narrative summary (2-3 sentences) of what has happened recently.

Summary:"""

            try:
                response = self.llm.invoke(prompt)
                return response.content
            except:
                pass
        
        # Fallback
        if event_descriptions:
            return f"Recent events: {'. '.join(event_descriptions[-3:])}"
        return "The adventure is just beginning..."
