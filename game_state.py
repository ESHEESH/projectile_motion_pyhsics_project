# game_state.py
# Turn / round / game state machine. No drawing, no physics, no input.

from enum import Enum, auto
import time


class Phase(Enum):
    MENU        = auto()   # difficulty select screen
    PLAYER_READY = auto()  # player must show both fists before aiming
    PLAYER_AIM  = auto()   # player is aiming (hand input active)
    PLAYER_SHOT = auto()   # arrow in flight (player's)
    AI_THINK    = auto()   # brief pause before AI shoots
    AI_SHOT     = auto()   # arrow in flight (AI's)
    HIT_PAUSE   = auto()   # short freeze after a hit
    ROUND_OVER  = auto()   # win / lose screen


class GameState:
    def __init__(self):
        self.phase         = Phase.MENU
        self.difficulty    = "normal"   # "easy" | "normal" | "hard"
        self.round_num     = 1
        self.arrow         = None       # active Arrow object (or None)
        self.phase_start   = time.time()
        self.message       = ""         # displayed centre-screen
        self.last_hit_who  = None       # "player" | "ai" | None

        # Shot snapshot (set when player fires, read by renderer)
        self.shot_angle    = 0.0
        self.shot_power    = 0.0

    # ── transitions ───────────────────────────────────────────────────
    def go(self, phase: Phase, message: str = ""):
        self.phase       = phase
        self.phase_start = time.time()
        self.message     = message

    def phase_elapsed(self) -> float:
        return time.time() - self.phase_start

    # ── helpers ───────────────────────────────────────────────────────
    def set_difficulty(self, d: str):
        self.difficulty = d

    def next_round(self):
        self.round_num += 1
        self.arrow      = None
        self.message    = ""
        self.go(Phase.PLAYER_READY)

    def record_shot(self, angle: float, power: float):
        self.shot_angle = angle
        self.shot_power = power
