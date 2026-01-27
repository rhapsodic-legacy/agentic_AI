"""
🎲 AI Game Master - FastAPI Backend

REST API for web-based D&D gameplay.
"""

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional, Dict
from pathlib import Path
from datetime import datetime
import asyncio
import json


app = FastAPI(
    title="AI Dungeon Master API",
    description="🎲 AI-Powered D&D Game Master",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Models
# =============================================================================

class NewGameRequest(BaseModel):
    player_name: str
    character_class: str = "Fighter"
    race: str = "Human"


class ActionRequest(BaseModel):
    game_id: str
    action: str


class DiceRollRequest(BaseModel):
    dice: str = "1d20"


# =============================================================================
# Game State Management
# =============================================================================

class GameManager:
    def __init__(self):
        self.games: Dict[str, 'game_master.AIGameMaster'] = {}
        self.game_history: Dict[str, list] = {}
    
    def create_game(self, player_name: str, character_class: str, race: str) -> tuple[str, str]:
        from game_master import AIGameMaster
        
        gm = AIGameMaster()
        opening = gm.new_game(player_name, character_class, race)
        
        game_id = gm.state.state_id
        self.games[game_id] = gm
        self.game_history[game_id] = [{"role": "dm", "content": opening}]
        
        return game_id, opening
    
    def process_action(self, game_id: str, action: str) -> tuple[str, bool, dict]:
        if game_id not in self.games:
            raise ValueError("Game not found")
        
        gm = self.games[game_id]
        response, continue_game = gm.process_input(action)
        
        # Track history
        self.game_history[game_id].append({"role": "player", "content": action})
        self.game_history[game_id].append({"role": "dm", "content": response})
        
        # Get game state info
        state_info = {
            "in_combat": gm.state.combat_state is not None and gm.state.combat_state.is_active,
            "location": gm.state.world_map.get_current_location().name if gm.state.world_map.get_current_location() else "Unknown",
            "player_hp": f"{gm.state.players[0].current_hp}/{gm.state.players[0].max_hp}" if gm.state.players else "0/0",
        }
        
        return response, continue_game, state_info
    
    def get_game_state(self, game_id: str) -> dict:
        if game_id not in self.games:
            raise ValueError("Game not found")
        
        gm = self.games[game_id]
        state = gm.state
        
        return {
            "game_id": game_id,
            "campaign": state.campaign.name,
            "time": str(state.time),
            "location": state.world_map.get_current_location().name if state.world_map.get_current_location() else "Unknown",
            "players": [p.to_dict() for p in state.players],
            "in_combat": state.combat_state is not None and state.combat_state.is_active,
            "active_quests": [q.name for q in state.get_active_quests()],
        }


game_manager = GameManager()


# =============================================================================
# Routes
# =============================================================================

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the frontend."""
    html_path = Path(__file__).parent / "frontend" / "index.html"
    if html_path.exists():
        return html_path.read_text()
    return """
    <html>
        <head><title>AI Dungeon Master</title></head>
        <body style="background: #1a1a2e; color: white; font-family: sans-serif; padding: 40px; text-align: center;">
            <h1>🎲 AI Dungeon Master 🐉</h1>
            <p>Visit <a href="/docs" style="color: #4ecdc4;">/docs</a> for API documentation</p>
        </body>
    </html>
    """


@app.get("/api/status")
async def get_status():
    """Get API status."""
    return {
        "status": "ready",
        "version": "1.0.0",
        "active_games": len(game_manager.games),
    }


@app.post("/api/game/new")
async def new_game(request: NewGameRequest):
    """Start a new game."""
    try:
        game_id, opening = game_manager.create_game(
            request.player_name,
            request.character_class,
            request.race,
        )
        
        return {
            "game_id": game_id,
            "opening": opening,
            "character": {
                "name": request.player_name,
                "class": request.character_class,
                "race": request.race,
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/game/action")
async def game_action(request: ActionRequest):
    """Process a player action."""
    try:
        response, continue_game, state_info = game_manager.process_action(
            request.game_id,
            request.action,
        )
        
        return {
            "response": response,
            "continue_game": continue_game,
            "state": state_info,
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/game/{game_id}")
async def get_game(game_id: str):
    """Get game state."""
    try:
        state = game_manager.get_game_state(game_id)
        history = game_manager.game_history.get(game_id, [])
        
        return {
            "state": state,
            "history": history[-20:],  # Last 20 messages
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/api/game/{game_id}/history")
async def get_history(game_id: str):
    """Get game history."""
    if game_id not in game_manager.game_history:
        raise HTTPException(status_code=404, detail="Game not found")
    
    return {"history": game_manager.game_history[game_id]}


@app.post("/api/dice/roll")
async def roll_dice(request: DiceRollRequest):
    """Roll dice."""
    from game_master import roll_dice as do_roll
    
    result = do_roll(request.dice)
    
    return {
        "dice": request.dice,
        "rolls": result.rolls,
        "modifier": result.modifier,
        "total": result.total,
        "natural": result.natural,
        "critical": result.is_critical,
        "fumble": result.is_fumble,
    }


@app.get("/api/classes")
async def get_classes():
    """Get available character classes."""
    return {
        "classes": [
            {"name": "Fighter", "description": "Masters of martial combat, skilled with weapons and armor."},
            {"name": "Rogue", "description": "Stealthy and cunning, experts at subterfuge and precision strikes."},
            {"name": "Wizard", "description": "Scholarly magic-users who command arcane powers."},
            {"name": "Cleric", "description": "Divine spellcasters who channel the power of their deity."},
            {"name": "Ranger", "description": "Skilled hunters and trackers at home in the wilderness."},
            {"name": "Barbarian", "description": "Fierce warriors who channel primal rage in battle."},
        ]
    }


@app.get("/api/races")
async def get_races():
    """Get available races."""
    return {
        "races": [
            {"name": "Human", "description": "Versatile and ambitious, humans are the most common race."},
            {"name": "Elf", "description": "Graceful and long-lived, with a deep connection to magic."},
            {"name": "Dwarf", "description": "Stout and hardy, known for their craftsmanship and resilience."},
            {"name": "Halfling", "description": "Small but brave, known for their luck and stealth."},
            {"name": "Half-Orc", "description": "Powerful warriors with orcish heritage."},
            {"name": "Tiefling", "description": "Descended from fiends, marked by their infernal heritage."},
        ]
    }


# =============================================================================
# WebSocket for real-time gameplay
# =============================================================================

@app.websocket("/ws/{game_id}")
async def websocket_endpoint(websocket: WebSocket, game_id: str):
    """WebSocket for real-time game updates."""
    await websocket.accept()
    
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") == "action":
                action = message.get("action", "")
                response, continue_game, state_info = game_manager.process_action(game_id, action)
                
                await websocket.send_json({
                    "type": "response",
                    "content": response,
                    "continue_game": continue_game,
                    "state": state_info,
                })
            
            elif message.get("type") == "roll":
                from game_master import roll_dice as do_roll
                dice = message.get("dice", "1d20")
                result = do_roll(dice)
                
                await websocket.send_json({
                    "type": "roll",
                    "dice": dice,
                    "total": result.total,
                    "critical": result.is_critical,
                    "fumble": result.is_fumble,
                })
    
    except WebSocketDisconnect:
        pass


# =============================================================================
# Run
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
