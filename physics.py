# physics.py
# Projectile motion engine — no rendering, no input, no game state.

import math
from dataclasses import dataclass, field
from typing import List, Tuple

GRAVITY = 500.0        # pixels per second squared (downward)
SCALE   = 1.0          # px per metre if needed later

@dataclass
class Arrow:
    x:     float
    y:     float
    vx:    float
    vy:    float
    owner: str          # "player" or "ai"
    alive: bool = True
    trail: List[Tuple[int,int]] = field(default_factory=list)
    TRAIL_LEN: int = 18

    def update(self, dt: float) -> None:
        """Advance physics one timestep dt (seconds)."""
        self.vy += GRAVITY * dt
        self.x  += self.vx * dt
        self.y  += self.vy * dt
        self.trail.append((int(self.x), int(self.y)))
        if len(self.trail) > self.TRAIL_LEN:
            self.trail.pop(0)

    def out_of_bounds(self, w: int, h: int) -> bool:
        return self.x < -100 or self.x > w + 100 or self.y > h + 200 or self.y < -300

    def hits_rect(self, cx: float, cy: float, hw: float, hh: float) -> bool:
        """cx,cy = centre of target rect, hw/hh = half-width/half-height."""
        return abs(self.x - cx) < hw and abs(self.y - cy) < hh


def launch(ox: float, oy: float, angle_deg: float, power: float,
           max_speed: float, facing: int) -> Arrow:
    """
    Create an Arrow at (ox,oy).
    angle_deg : degrees above horizontal (0 = flat, 90 = straight up)
    power     : 0.0 – 1.0
    facing    : +1 = shoots right, -1 = shoots left
    max_speed : pixels per second at power=1.0
    """
    rad   = math.radians(angle_deg)
    speed = power * max_speed
    vx    = facing * speed * math.cos(rad)
    vy    = -speed * math.sin(rad)      # negative = upward in screen coords
    owner = "player" if facing == 1 else "ai"
    return Arrow(x=ox, y=oy, vx=vx, vy=vy, owner=owner)


def solve_angle(ox: float, oy: float, tx: float, ty: float,
                power: float, max_speed: float, facing: int) -> float:
    """
    Return the LOW angle (degrees) needed to hit (tx,ty) from (ox,oy)
    with the given power, using the quadratic formula for projectile range.
    Returns the best available angle (clamped 5–85°) if no exact solution.
    facing: +1 or -1
    """
    speed = power * max_speed
    dx    = (tx - ox) * facing          # positive = forward
    dy    = oy - ty                     # positive = target is below launch

    if speed < 1:
        return 45.0

    # Standard formula: tan(θ) from quadratic
    # dy = dx*tan(θ) - g*dx²/(2v²cos²θ)  →  rearranges to:
    # (g*dx²/2v²)tan²θ - dx·tanθ + (dy + g*dx²/2v²) = 0
    g  = GRAVITY
    v2 = speed * speed
    a_coef = (g * dx * dx) / (2 * v2)
    b_coef = -dx
    c_coef = dy + a_coef

    disc = b_coef**2 - 4 * a_coef * c_coef
    if disc < 0:
        # No real solution — return 45° as best-effort
        return 45.0

    sqrt_d = math.sqrt(disc)
    tan1   = (-b_coef + sqrt_d) / (2 * a_coef)
    tan2   = (-b_coef - sqrt_d) / (2 * a_coef)

    angles = []
    for t in (tan1, tan2):
        a = math.degrees(math.atan(t))
        if 2 <= a <= 88:
            angles.append(a)

    if not angles:
        return 45.0
    # Prefer the lower (flatter) trajectory
    return min(angles)
