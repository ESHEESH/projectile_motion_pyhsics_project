# renderer.py
# All drawing: scene, stickmen, arrows, HUD, menus. No game logic, no input.

import cv2
import numpy as np
import math
from typing import Optional

from entities import HEAD_R, TORSO_H, ARM_L, LEG_H, MAX_HP
from physics import Arrow

# ── Canvas ────────────────────────────────────────────────────────────
GW, GH = 1920, 1080   # game resolution (1080p)

# ── Palette ───────────────────────────────────────────────────────────
SKY_TOP    = (210, 160, 100)   # BGR dawn sky top
SKY_BOT    = (100,  80, 200)   # BGR dawn sky bottom
GROUND_COL = (40,  110,  40)   # dark green ground
GROUND_H   = 160               # px from bottom
GROUND_Y   = GH - GROUND_H

COL_PLAYER    = (60,  210, 60)
COL_AI        = (60,  60, 220)
COL_ARROW_P   = (0,   220, 255)
COL_ARROW_AI  = (50,  50, 255)
COL_PULL_LINE = (255,  0, 255)   # magenta
COL_WHITE     = (255, 255, 255)
COL_BLACK     = (0,   0,   0)
COL_GOLD      = (0,   200, 255)
COL_HIT       = (0,   80, 255)
COL_HP_FULL   = (50,  220, 50)
COL_HP_EMPTY  = (60,  60,  60)
COL_UI_BG     = (20,  20,  20)


# ── Gradient helpers ──────────────────────────────────────────────────
def _make_sky(h, w):
    sky = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        t   = y / h
        bgr = tuple(int(SKY_TOP[c] + t * (SKY_BOT[c] - SKY_TOP[c])) for c in range(3))
        sky[y, :] = bgr
    return sky

_SKY_CACHE = None

def _sky(canvas):
    global _SKY_CACHE
    h, w = canvas.shape[:2]
    if _SKY_CACHE is None or _SKY_CACHE.shape[:2] != (h, w):
        _SKY_CACHE = _make_sky(h, w)
    canvas[:] = _SKY_CACHE


# ── Scene ─────────────────────────────────────────────────────────────
def draw_scene(canvas: np.ndarray):
    """Draw sky gradient + ground."""
    _sky(canvas)
    ground_y = canvas.shape[0] - GROUND_H
    cv2.rectangle(canvas, (0, ground_y), (canvas.shape[1], canvas.shape[0]),
                  GROUND_COL, -1)
    # Ground edge highlight
    cv2.line(canvas, (0, ground_y), (canvas.shape[1], ground_y),
             (80, 160, 80), 3)


# ── Stickman ──────────────────────────────────────────────────────────
def draw_stickman(canvas: np.ndarray, entity, aiming: bool = False,
                  angle_deg: float = 0.0, power: float = 0.0,
                  pull_screen: Optional[tuple] = None):
    """
    Draw a stickman at entity's foot position.
    entity must have: .foot_y, .cx, .facing, .color
    If aiming=True, draws the bow string pull with the magenta line.
    """
    col    = entity.color
    fx     = int(entity.cx)
    fy     = int(entity.foot_y)
    facing = entity.facing

    # Feet
    lf = (fx - 10, fy)
    rf = (fx + 10, fy)
    # Knee
    lk = (fx - 8,  fy - LEG_H // 2)
    rk = (fx + 8,  fy - LEG_H // 2)
    # Hip
    hip = (fx, fy - LEG_H)
    # Shoulder
    sh  = (fx, fy - LEG_H - TORSO_H)
    # Head centre
    hc  = (fx, fy - LEG_H - TORSO_H - HEAD_R)

    # Legs
    cv2.line(canvas, lf, lk, col, 4)
    cv2.line(canvas, lk, hip, col, 4)
    cv2.line(canvas, rf, rk, col, 4)
    cv2.line(canvas, rk, hip, col, 4)
    # Torso
    cv2.line(canvas, hip, sh, col, 5)
    # Head
    cv2.circle(canvas, hc, HEAD_R, col, -1)
    cv2.circle(canvas, hc, HEAD_R, COL_BLACK, 2)

    # Arms — bow hand is the hand facing the enemy
    bow_elbow  = (sh[0] + facing * 20, sh[1] + 10)
    bow_hand   = (sh[0] + facing * (20 + ARM_L), sh[1] + 5)

    # Pull arm angle
    if aiming:
        rad         = math.radians(angle_deg)
        pull_dx     = -facing * ARM_L * 0.9 * math.cos(rad)
        pull_dy     = -ARM_L * 0.5
        pull_elbow  = (int(sh[0] - facing * 12), int(sh[1] + 8))
        pull_hand   = (int(sh[0] + pull_dx), int(sh[1] + pull_dy))
    else:
        pull_elbow = (sh[0] - facing * 20, sh[1] + 10)
        pull_hand  = (sh[0] - facing * (20 + ARM_L // 2), sh[1] + 15)

    cv2.line(canvas, sh, bow_elbow,  col, 4)
    cv2.line(canvas, bow_elbow, bow_hand, col, 4)
    cv2.line(canvas, sh, pull_elbow, col, 4)
    cv2.line(canvas, pull_elbow, pull_hand, col, 4)

    # Bow arc
    bow_len = 36
    perp_angle = math.atan2(
        bow_hand[1] - pull_hand[1],
        bow_hand[0] - pull_hand[0]
    ) + math.pi / 2
    b1 = (int(bow_hand[0] + bow_len * math.cos(perp_angle)),
          int(bow_hand[1] + bow_len * math.sin(perp_angle)))
    b2 = (int(bow_hand[0] - bow_len * math.cos(perp_angle)),
          int(bow_hand[1] - bow_len * math.sin(perp_angle)))
    pts = []
    for t in np.linspace(0, 1, 16):
        mx = int(b1[0] + t * (b2[0] - b1[0]))
        my = int(b1[1] + t * (b2[1] - b1[1]))
        bulge_angle = math.atan2(bow_hand[1] - pull_hand[1],
                                  bow_hand[0] - pull_hand[0])
        bx = int(mx + 16 * math.sin(t * math.pi) * math.cos(bulge_angle))
        by = int(my + 16 * math.sin(t * math.pi) * math.sin(bulge_angle))
        pts.append([bx, by])
    pts = np.array(pts, dtype=np.int32)
    cv2.polylines(canvas, [pts], False, (200, 160, 40), 3)

    # String
    cv2.line(canvas, b1, pull_hand, (220, 220, 220), 1)
    cv2.line(canvas, b2, pull_hand, (220, 220, 220), 1)

    # Magenta pull indicator when aiming
    if aiming and pull_screen:
        bow_screen = (int(entity.bow_origin[0]), int(entity.bow_origin[1]))
        mid = ((bow_screen[0] + pull_screen[0]) // 2,
               (bow_screen[1] + pull_screen[1]) // 2)
        cv2.line(canvas, bow_screen, pull_screen, COL_PULL_LINE, 3)
        cv2.circle(canvas, bow_screen,  12, COL_PULL_LINE, -1)
        cv2.circle(canvas, mid,          8, COL_PULL_LINE, -1)
        cv2.circle(canvas, pull_screen, 12, COL_PULL_LINE, -1)


# ── Arrow ─────────────────────────────────────────────────────────────
def draw_arrow(canvas: np.ndarray, arrow: Arrow):
    col = COL_ARROW_P if arrow.owner == "player" else COL_ARROW_AI
    # Trail
    for i, pt in enumerate(arrow.trail):
        alpha = max(30, int(200 * i / max(len(arrow.trail), 1)))
        cv2.circle(canvas, pt, 3, col, -1)
    # Shaft + head
    ax, ay = int(arrow.x), int(arrow.y)
    if len(arrow.trail) >= 2:
        cv2.line(canvas, arrow.trail[-3] if len(arrow.trail) >= 3
                 else arrow.trail[0], (ax, ay), col, 3)
    cv2.circle(canvas, (ax, ay), 6, col, -1)


# ── HUD ───────────────────────────────────────────────────────────────
def draw_hud(canvas: np.ndarray, player_hp: int, ai_hp: int,
             round_num: int, difficulty: str):
    h, w = canvas.shape[:2]
    bar_w = 300
    pip_w = bar_w // MAX_HP - 4

    # Player HP bar (top left)
    for i in range(MAX_HP):
        col = COL_HP_FULL if i < player_hp else COL_HP_EMPTY
        x1 = 20 + i * (pip_w + 4)
        cv2.rectangle(canvas, (x1, 20), (x1 + pip_w, 46), col, -1)
        cv2.rectangle(canvas, (x1, 20), (x1 + pip_w, 46), COL_WHITE, 1)
    cv2.putText(canvas, "YOU", (20, 68),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, COL_PLAYER, 2)

    # AI HP bar (top right)
    for i in range(MAX_HP):
        col = COL_HP_FULL if i < ai_hp else COL_HP_EMPTY
        x1 = w - 20 - (i + 1) * (pip_w + 4)
        cv2.rectangle(canvas, (x1, 20), (x1 + pip_w, 46), col, -1)
        cv2.rectangle(canvas, (x1, 20), (x1 + pip_w, 46), COL_WHITE, 1)
    cv2.putText(canvas, "AI", (w - 70, 68),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, COL_AI, 2)

    # Round + difficulty (top centre)
    label = f"Round {round_num}   [{difficulty.upper()}]"
    (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.putText(canvas, label, (w // 2 - tw // 2, 44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, COL_WHITE, 2)


def draw_aim_hud(canvas: np.ndarray, angle_deg: float, power: float,
                 entity_cx: float, entity_cy: float):
    """Power % and angle ° shown in a bubble near the player, like the reference."""
    px = int(entity_cx) - 60
    py = int(entity_cy) - 100

    # Background bubble
    cv2.rectangle(canvas, (px - 10, py - 36), (px + 160, py + 10),
                  COL_UI_BG, -1)
    cv2.rectangle(canvas, (px - 10, py - 36), (px + 160, py + 10),
                  COL_WHITE, 1)

    cv2.putText(canvas, f"{int(power*100)}%  POWER",
                (px, py - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.65, COL_GOLD, 2)
    cv2.putText(canvas, f"{int(angle_deg)}\u00b0  ANGLE",
                (px, py + 8),  cv2.FONT_HERSHEY_SIMPLEX, 0.65, COL_WHITE, 1)

    # Min-pull threshold marker below power text
    bar_x, bar_y = px, py + 18
    cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + 150, bar_y + 12), COL_UI_BG, -1)
    cv2.rectangle(canvas, (bar_x, bar_y),
                  (bar_x + int(150 * power), bar_y + 12), COL_PULL_LINE, -1)
    cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + 150, bar_y + 12), COL_WHITE, 1)
    # Threshold tick at 18%
    tx = bar_x + int(150 * 0.18)
    cv2.line(canvas, (tx, bar_y - 3), (tx, bar_y + 15), (0, 255, 0), 2)


def draw_message(canvas: np.ndarray, msg: str, sub: str = ""):
    """Large centred message with optional subtitle."""
    h, w = canvas.shape[:2]
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, canvas, 0.55, 0, canvas)

    (tw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_DUPLEX, 2.5, 4)
    cv2.putText(canvas, msg, (w // 2 - tw // 2, h // 2),
                cv2.FONT_HERSHEY_DUPLEX, 2.5, COL_GOLD, 4)
    if sub:
        (sw, _), _ = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        cv2.putText(canvas, sub, (w // 2 - sw // 2, h // 2 + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, COL_WHITE, 2)


def draw_menu(canvas: np.ndarray, gesture_count: int = 0,
              hold_progress: float = 0.0):
    """
    Difficulty select screen.
    gesture_count : currently detected finger count (0 = none)
    hold_progress : 0.0-1.0, how long the gesture has been held
    """
    h, w = canvas.shape[:2]
    draw_scene(canvas)

    # Title
    title = "BOW MASTER"
    (tw, _), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_DUPLEX, 3.0, 5)
    cv2.putText(canvas, title, (w // 2 - tw // 2, h // 4),
                cv2.FONT_HERSHEY_DUPLEX, 3.0, COL_GOLD, 5)

    # Three gesture cards
    options = [
        (1, "☝",  "EASY",   (50, 200, 50)),
        (2, "✌",  "NORMAL", (50, 200, 220)),
        (3, "👌", "HARD",   (50, 50, 220)),
    ]
    card_w, card_h = 280, 220
    gap            = 60
    total_w        = len(options) * card_w + (len(options) - 1) * gap
    start_x        = w // 2 - total_w // 2
    card_y         = h // 2 - card_h // 2 + 30

    for i, (num, symbol, label, col) in enumerate(options):
        cx = start_x + i * (card_w + gap)
        is_selected = (gesture_count == num)

        # Card background — highlight if selected
        bg_col  = (40, 40, 60) if not is_selected else tuple(max(0, c - 60) for c in col)
        border  = col if is_selected else (120, 120, 120)
        thick   = 4   if is_selected else 1

        cv2.rectangle(canvas, (cx, card_y), (cx + card_w, card_y + card_h),
                      bg_col, -1)
        cv2.rectangle(canvas, (cx, card_y), (cx + card_w, card_y + card_h),
                      border, thick)

        # Finger count number (large)
        (nw, nh), _ = cv2.getTextSize(str(num), cv2.FONT_HERSHEY_DUPLEX, 4.0, 6)
        cv2.putText(canvas, str(num),
                    (cx + card_w // 2 - nw // 2, card_y + 100),
                    cv2.FONT_HERSHEY_DUPLEX, 4.0, col, 6)

        # Difficulty label
        (lw2, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        cv2.putText(canvas, label,
                    (cx + card_w // 2 - lw2 // 2, card_y + card_h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, col, 2)

        # Countdown arc around card centre when this card is selected
        if is_selected and hold_progress > 0:
            arc_cx = cx + card_w // 2
            arc_cy = card_y + card_h // 2
            angle  = int(360 * hold_progress)
            # Grey background ring
            cv2.ellipse(canvas, (arc_cx, arc_cy), (card_w // 2 - 8, card_h // 2 - 8),
                        -90, 0, 360, (60, 60, 60), 5)
            # Coloured progress arc
            cv2.ellipse(canvas, (arc_cx, arc_cy), (card_w // 2 - 8, card_h // 2 - 8),
                        -90, 0, angle, col, 6)

    # Instruction subtitle
    sub = "Hold  1 / 2 / 3  fingers  for  1.5 s  to  select"
    (sw, _), _ = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2)
    cv2.putText(canvas, sub, (w // 2 - sw // 2, h - 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, COL_WHITE, 2)

    # Live finger count feedback (bottom centre)
    if gesture_count in (1, 2, 3):
        names = {1: "EASY", 2: "NORMAL", 3: "HARD"}
        feedback = f"Detected: {gesture_count} finger{'s' if gesture_count > 1 else ''}  →  {names[gesture_count]}"
        col_fb   = [c[3] for c in options if c[0] == gesture_count][0]
        (fw2, _), _ = cv2.getTextSize(feedback, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        cv2.putText(canvas, feedback, (w // 2 - fw2 // 2, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, col_fb, 2)
    elif gesture_count == 0:
        sub2 = "Show your hand to the camera"
        (sw2, _), _ = cv2.getTextSize(sub2, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 1)
        cv2.putText(canvas, sub2, (w // 2 - sw2 // 2, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (160, 160, 160), 1)


def draw_hit_flash(canvas: np.ndarray, cx: int, cy: int):
    cv2.circle(canvas, (cx, cy), 55, COL_HIT, 4)
    cv2.circle(canvas, (cx, cy), 30, (0, 180, 255), -1)


# ── Camera pane ───────────────────────────────────────────────────────
CAM_W, CAM_H = 480, 360   # size of the camera overlay window

def draw_camera_pane(frame_bgr, hs, phase_label: str = "",
                     ready: bool = False) -> np.ndarray:
    """
    Build the camera pane image:
    - Resizes the raw camera frame to CAM_W × CAM_H
    - Draws the magenta line between left and right wrist
    - Labels left = BOW hand, right = STRING hand
    - Shows power bar + angle value
    - Shows phase label and ready indicator
    """
    if frame_bgr is None:
        pane = np.zeros((CAM_H, CAM_W, 3), dtype=np.uint8)
    else:
        pane = cv2.resize(frame_bgr, (CAM_W, CAM_H))

    fh, fw = pane.shape[:2]

    # Scale wrist pixel coords to pane size
    orig_h = frame_bgr.shape[0] if frame_bgr is not None else CAM_H
    orig_w = frame_bgr.shape[1] if frame_bgr is not None else CAM_W
    sx = fw / orig_w
    sy = fh / orig_h

    lp = rp = None
    if hs.left_wrist_px:
        lp = (int(hs.left_wrist_px[0] * sx), int(hs.left_wrist_px[1] * sy))
    if hs.right_wrist_px:
        rp = (int(hs.right_wrist_px[0] * sx), int(hs.right_wrist_px[1] * sy))

    # Magenta line + dots
    if lp and rp:
        mid = ((lp[0] + rp[0]) // 2, (lp[1] + rp[1]) // 2)
        cv2.line(pane, lp, rp, (255, 0, 255), 3)
        cv2.circle(pane, lp,  12, (255, 0, 255), -1)   # bow hand anchor
        cv2.circle(pane, mid,  7, (255, 0, 255), -1)   # midpoint
        cv2.circle(pane, rp,  12, (255, 0, 255), -1)   # string hand tip

        # Labels
        cv2.putText(pane, "BOW",    (lp[0] - 20, lp[1] - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.putText(pane, "STRING", (rp[0] - 28, rp[1] - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        # Power bar at bottom of pane
        bar_x, bar_y, bar_w, bar_h2 = 10, fh - 36, fw - 20, 14
        cv2.rectangle(pane, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h2),
                      (40, 40, 40), -1)
        filled = int(bar_w * hs.draw_power)
        bar_col = (255, 0, 255) if hs.draw_power >= 0.18 else (100, 100, 100)
        if filled > 0:
            cv2.rectangle(pane, (bar_x, bar_y), (bar_x + filled, bar_y + bar_h2),
                          bar_col, -1)
        cv2.rectangle(pane, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h2),
                      (200, 200, 200), 1)
        # Threshold tick
        tx = bar_x + int(bar_w * 0.18)
        cv2.line(pane, (tx, bar_y - 3), (tx, bar_y + bar_h2 + 3), (0, 255, 0), 2)
        cv2.putText(pane, f"PWR {int(hs.draw_power * 100)}%",
                    (bar_x, bar_y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1)

        # Angle indicator (right side)
        cv2.putText(pane, f"ANG {int(hs.angle_deg)}\u00b0",
                    (fw - 110, bar_y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1)

    # Both-fists ready indicator (top banner)
    if ready:
        banner_col = (0, 180, 0)
        banner_txt = "READY  -  Open palm to FIRE"
    else:
        banner_col = (0, 60, 180)
        banner_txt = "Make BOTH FISTS to ready up"

    cv2.rectangle(pane, (0, 0), (fw, 28), banner_col, -1)
    cv2.putText(pane, banner_txt, (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    # Phase label (top right)
    if phase_label:
        (tw, _), _ = cv2.getTextSize(phase_label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.putText(pane, phase_label, (fw - tw - 6, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 50), 1)

    # Border
    cv2.rectangle(pane, (0, 0), (fw - 1, fh - 1), (200, 200, 200), 2)

    return pane
