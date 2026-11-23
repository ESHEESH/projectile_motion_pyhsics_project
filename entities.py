# entities.py
# Player and AI entity definitions. No drawing, no physics, no input.

import math
import random
from physics import solve_angle

# ── Stickman body proportions (pixels) ────────────────────────────────
HEAD_R      = 22
TORSO_H     = 55
ARM_L       = 38
LEG_H       = 48
BODY_W      = 20     # half-width of hitbox
BODY_H      = HEAD_R * 2 + TORSO_H + LEG_H   # ~147 px

MAX_HP      = 10
MAX_SPEED   = 900.0  # px/s at power=1.0


class Player:
    """Human-controlled archer (left side, faces right)."""
    facing  = 1          # shoots right
    color   = (60, 200, 60)

    def __init__(self, x: float, ground_y: float):
        self.x        = x
        self.ground_y = ground_y
        self.foot_y   = ground_y
        self.hp       = MAX_HP
        self.alive    = True

    @property
    def cx(self) -> float:
        return self.x

    @property
    def cy(self) -> float:
        """Centre of mass (mid-torso)."""
        return self.foot_y - LEG_H - TORSO_H // 2

    @property
    def bow_origin(self):
        """Where the arrow launches from (roughly hand height)."""
        return (self.x + self.facing * (BODY_W + 10),
                self.foot_y - LEG_H - TORSO_H * 0.65)

    def take_hit(self):
        self.hp -= 1
        if self.hp <= 0:
            self.alive = False

    def hitbox(self):
        """Returns (cx, cy, half_w, half_h)."""
        return (self.cx, self.cy, BODY_W, BODY_H // 2)


class AIPlayer:
    """AI-controlled archer (right side, faces left)."""
    facing = -1
    color  = (220, 60, 60)

    def __init__(self, x: float, ground_y: float, difficulty: str = "normal"):
        self.x          = x
        self.ground_y   = ground_y
        self.foot_y     = ground_y
        self.hp         = MAX_HP
        self.alive      = True
        self.difficulty = difficulty.lower()

    @property
    def cx(self) -> float:
        return self.x

    @property
    def cy(self) -> float:
        return self.foot_y - LEG_H - TORSO_H // 2

    @property
    def bow_origin(self):
        return (self.x + self.facing * (BODY_W + 10),
                self.foot_y - LEG_H - TORSO_H * 0.65)

    def take_hit(self):
        self.hp -= 1
        if self.hp <= 0:
            self.alive = False

    def hitbox(self):
        return (self.cx, self.cy, BODY_W, BODY_H // 2)

    def compute_shot(self, target_cx: float, target_cy: float) -> tuple:
        """
        Returns (angle_deg, power) based on difficulty.
        Easy   : big random spread
        Normal : medium spread
        Hard   : solves exact trajectory + tiny spread
        """
        ox, oy = self.bow_origin

        if self.difficulty == "easy":
            power  = random.uniform(0.45, 0.85)
            exact  = solve_angle(ox, oy, target_cx, target_cy, power, MAX_SPEED, self.facing)
            angle  = exact + random.uniform(-28, 28)
            power += random.uniform(-0.25, 0.25)

        elif self.difficulty == "normal":
            power  = random.uniform(0.55, 0.90)
            exact  = solve_angle(ox, oy, target_cx, target_cy, power, MAX_SPEED, self.facing)
            angle  = exact + random.uniform(-12, 12)
            power += random.uniform(-0.12, 0.12)

        else:   # hard
            power = random.uniform(0.70, 0.95)
            exact = solve_angle(ox, oy, target_cx, target_cy, power, MAX_SPEED, self.facing)
            angle = exact + random.uniform(-3, 3)
            power += random.uniform(-0.04, 0.04)

        angle = max(2.0,  min(85.0, angle))
        power = max(0.15, min(1.0,  power))
        return angle, power
