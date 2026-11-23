# main.py
# Entry point: game loop + turn state machine.

import cv2
import numpy as np
import time

from game_state    import GameState, Phase
from entities      import Player, AIPlayer, BODY_W, BODY_H, MAX_HP
from physics       import launch
from renderer      import (GW, GH, GROUND_Y, draw_scene, draw_stickman,
                            draw_arrow, draw_hud, draw_aim_hud,
                            draw_message, draw_menu, draw_hit_flash,
                            draw_camera_pane)
from input_handler import InputHandler

# ── Tunables ──────────────────────────────────────────────────────────
MAX_SPEED         = 900.0
MIN_PULL_FIRE     = 0.18    # minimum draw_power to allow firing
FIRE_COOLDOWN     = 0.7
AI_THINK_TIME     = 1.4
HIT_PAUSE_TIME    = 0.7
GESTURE_HOLD_TIME = 1.5     # hold finger gesture this long to pick difficulty

# ── Layout ────────────────────────────────────────────────────────────
PLAYER_X = 220
AI_X     = GW - 220

# ── Helpers ───────────────────────────────────────────────────────────
def _hint(canvas, text, w, h, col=(200, 200, 200)):
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    cv2.putText(canvas, text, (w // 2 - tw // 2, h - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2)

def letterbox(frame: np.ndarray, tw: int, th: int) -> np.ndarray:
    fh, fw  = frame.shape[:2]
    scale   = min(tw / fw, th / fh)
    nw, nh  = int(fw * scale), int(fh * scale)
    out     = np.zeros((th, tw, 3), dtype=np.uint8)
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    ox, oy  = (tw - nw) // 2, (th - nh) // 2
    out[oy:oy+nh, ox:ox+nw] = resized
    return out

# ── Windows ───────────────────────────────────────────────────────────
WIN_GAME = "Bow Master"
WIN_CAM  = "Camera"

cv2.namedWindow(WIN_GAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WIN_GAME, 1280, 720)
cv2.setWindowProperty(WIN_GAME, cv2.WND_PROP_TOPMOST, 1)

cv2.namedWindow(WIN_CAM, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WIN_CAM, 480, 360)
# Position camera pane to the right of the game window
cv2.moveWindow(WIN_CAM, 1300, 100)

is_fullscreen = False

# ── Game objects ──────────────────────────────────────────────────────
gs     = GameState()
inp    = InputHandler(camera_index=0, rotate_cw=True)
canvas = np.zeros((GH, GW, 3), dtype=np.uint8)

player  = Player(PLAYER_X, GROUND_Y)
ai      = None

last_fire      = 0.0
hit_target     = None
player_ready   = False   # True once both fists shown this turn

_gesture_held  = 0
_gesture_since = 0.0

pTime = time.time()

print("Bow Master  |  Hold 1/2/3 fingers to pick difficulty")
print("             Make BOTH FISTS on your turn to ready up")
print("             F = fullscreen  |  Q = quit  |  R = restart")

while True:
    now   = time.time()
    dt    = min(now - pTime, 0.05)
    pTime = now

    hs = inp.read()

    # ── keyboard ──────────────────────────────────────────────────────
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('r'):
        gs             = GameState()
        player         = Player(PLAYER_X, GROUND_Y)
        ai             = None
        last_fire      = 0.0
        hit_target     = None
        player_ready   = False
        _gesture_held  = 0
        _gesture_since = 0.0
    if key == ord('f'):
        is_fullscreen = not is_fullscreen
        prop = cv2.WINDOW_FULLSCREEN if is_fullscreen else cv2.WINDOW_NORMAL
        cv2.setWindowProperty(WIN_GAME, cv2.WND_PROP_FULLSCREEN, prop)
        if not is_fullscreen:
            cv2.resizeWindow(WIN_GAME, 1280, 720)

    # ── gesture hold: difficulty select ───────────────────────────────
    if gs.phase in (Phase.MENU, Phase.ROUND_OVER):
        fc = hs.finger_count
        if fc in (1, 2, 3):
            if fc != _gesture_held:
                _gesture_held  = fc
                _gesture_since = now
            elif now - _gesture_since >= GESTURE_HOLD_TIME:
                diff_map = {1: 'easy', 2: 'normal', 3: 'hard'}
                gs.set_difficulty(diff_map[fc])
                ai           = AIPlayer(AI_X, GROUND_Y, gs.difficulty)
                player       = Player(PLAYER_X, GROUND_Y)
                player_ready = False
                gs.go(Phase.PLAYER_READY)
                last_fire      = 0.0
                hit_target     = None
                _gesture_held  = 0
                _gesture_since = 0.0
        else:
            _gesture_held  = 0
            _gesture_since = 0.0

    # ══════════════════════════════════════════════════════════════════
    #  PHASE LOGIC
    # ══════════════════════════════════════════════════════════════════

    # ── MENU ──────────────────────────────────────────────────────────
    if gs.phase == Phase.MENU:
        hold_prog = 0.0
        if _gesture_held in (1, 2, 3):
            hold_prog = min((now - _gesture_since) / GESTURE_HOLD_TIME, 1.0)
        draw_menu(canvas, gesture_count=hs.finger_count, hold_progress=hold_prog)

    # ── PLAYER READY ──────────────────────────────────────────────────
    elif gs.phase == Phase.PLAYER_READY:
        draw_scene(canvas)
        draw_stickman(canvas, player)
        draw_stickman(canvas, ai)
        draw_hud(canvas, player.hp, ai.hp, gs.round_num, gs.difficulty)
        _hint(canvas,
              "Make BOTH FISTS to ready up  ✊✊",
              GW, GH, col=(255, 200, 50))

        if hs.both_fists and hs.both_visible:
            player_ready = True
            gs.go(Phase.PLAYER_AIM)

    # ── PLAYER AIM ────────────────────────────────────────────────────
    elif gs.phase == Phase.PLAYER_AIM:
        draw_scene(canvas)
        draw_stickman(canvas, player, aiming=True,
                      angle_deg=hs.angle_deg, power=hs.draw_power)
        draw_stickman(canvas, ai)
        draw_hud(canvas, player.hp, ai.hp, gs.round_num, gs.difficulty)
        draw_aim_hud(canvas, hs.angle_deg, hs.draw_power, player.cx, player.cy)

        # Fire: open right palm + enough pull + cooldown
        if (hs.both_visible
                and hs.right_palm_open
                and hs.draw_power >= MIN_PULL_FIRE
                and now - last_fire >= FIRE_COOLDOWN):
            arrow = launch(player.bow_origin[0], player.bow_origin[1],
                           hs.angle_deg, hs.draw_power, MAX_SPEED, player.facing)
            gs.arrow = arrow
            gs.record_shot(hs.angle_deg, hs.draw_power)
            last_fire    = now
            player_ready = False
            gs.go(Phase.PLAYER_SHOT)

        if not hs.both_visible:
            _hint(canvas, "Show BOTH hands to aim", GW, GH)
        elif hs.draw_power < MIN_PULL_FIRE:
            _hint(canvas, "Move hands further apart to charge", GW, GH)
        else:
            _hint(canvas, "Open right palm to FIRE!", GW, GH, col=(0, 255, 180))

    # ── PLAYER SHOT ───────────────────────────────────────────────────
    elif gs.phase == Phase.PLAYER_SHOT:
        gs.arrow.update(dt)
        draw_scene(canvas)
        draw_stickman(canvas, player)
        draw_stickman(canvas, ai)
        draw_arrow(canvas, gs.arrow)
        draw_hud(canvas, player.hp, ai.hp, gs.round_num, gs.difficulty)

        if gs.arrow.hits_rect(ai.cx, ai.cy, BODY_W + 10, BODY_H // 2 + 10):
            ai.take_hit()
            hit_target      = (int(ai.cx), int(ai.cy))
            gs.arrow        = None
            gs.last_hit_who = "ai"
            gs.go(Phase.HIT_PAUSE)
        elif gs.arrow.out_of_bounds(GW, GH):
            gs.arrow = None
            gs.go(Phase.AI_THINK) if ai.alive else gs.go(Phase.ROUND_OVER, "YOU WIN!")

    # ── AI THINK ──────────────────────────────────────────────────────
    elif gs.phase == Phase.AI_THINK:
        draw_scene(canvas)
        draw_stickman(canvas, player)
        draw_stickman(canvas, ai)
        draw_hud(canvas, player.hp, ai.hp, gs.round_num, gs.difficulty)
        _hint(canvas, "AI is aiming...", GW, GH, col=(255, 100, 100))

        if gs.phase_elapsed() >= AI_THINK_TIME:
            angle, power = ai.compute_shot(player.cx, player.cy)
            gs.arrow      = launch(ai.bow_origin[0], ai.bow_origin[1],
                                   angle, power, MAX_SPEED, ai.facing)
            gs.go(Phase.AI_SHOT)

    # ── AI SHOT ───────────────────────────────────────────────────────
    elif gs.phase == Phase.AI_SHOT:
        gs.arrow.update(dt)
        draw_scene(canvas)
        draw_stickman(canvas, player)
        draw_stickman(canvas, ai)
        draw_arrow(canvas, gs.arrow)
        draw_hud(canvas, player.hp, ai.hp, gs.round_num, gs.difficulty)

        if gs.arrow.hits_rect(player.cx, player.cy, BODY_W + 10, BODY_H // 2 + 10):
            player.take_hit()
            hit_target      = (int(player.cx), int(player.cy))
            gs.arrow        = None
            gs.last_hit_who = "player"
            gs.go(Phase.HIT_PAUSE)
        elif gs.arrow.out_of_bounds(GW, GH):
            gs.arrow = None
            if player.alive:
                player_ready = False
                gs.go(Phase.PLAYER_READY)
            else:
                gs.go(Phase.ROUND_OVER, "YOU LOSE!")

    # ── HIT PAUSE ─────────────────────────────────────────────────────
    elif gs.phase == Phase.HIT_PAUSE:
        draw_scene(canvas)
        draw_stickman(canvas, player)
        draw_stickman(canvas, ai)
        draw_hud(canvas, player.hp, ai.hp, gs.round_num, gs.difficulty)
        if hit_target:
            draw_hit_flash(canvas, *hit_target)

        if gs.phase_elapsed() >= HIT_PAUSE_TIME:
            hit_target = None
            if not ai.alive:
                gs.go(Phase.ROUND_OVER, "YOU WIN!")
            elif not player.alive:
                gs.go(Phase.ROUND_OVER, "YOU LOSE!")
            elif gs.last_hit_who == "ai":
                gs.go(Phase.AI_THINK)
            else:
                player_ready = False
                gs.go(Phase.PLAYER_READY)

    # ── ROUND OVER ────────────────────────────────────────────────────
    elif gs.phase == Phase.ROUND_OVER:
        hold_prog = 0.0
        if _gesture_held in (1, 2, 3):
            hold_prog = min((now - _gesture_since) / GESTURE_HOLD_TIME, 1.0)
        draw_scene(canvas)
        if ai:
            draw_stickman(canvas, player)
            draw_stickman(canvas, ai)
            draw_hud(canvas, player.hp, ai.hp, gs.round_num, gs.difficulty)
        draw_message(canvas, gs.message, "Hold 1 / 2 / 3 fingers to play again")
        draw_menu(canvas, gesture_count=hs.finger_count, hold_progress=hold_prog)

    # ── FPS ───────────────────────────────────────────────────────────
    fps = 1 / max(dt, 0.001)
    cv2.putText(canvas, f"{int(fps)} FPS", (GW - 110, GH - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (140, 140, 140), 1)

    # ── Game display ──────────────────────────────────────────────────
    if is_fullscreen:
        rect = cv2.getWindowImageRect(WIN_GAME)
        sw, sh = rect[2], rect[3]
        display = letterbox(canvas, sw, sh) if sw > 0 and sh > 0 else canvas
    else:
        display = canvas
    cv2.imshow(WIN_GAME, display)

    # ── Camera pane ───────────────────────────────────────────────────
    phase_label = gs.phase.name.replace("_", " ")
    cam_pane = draw_camera_pane(
        hs.frame_bgr,
        hs,
        phase_label=phase_label,
        ready=player_ready
    )
    cv2.imshow(WIN_CAM, cam_pane)

inp.release()
cv2.destroyAllWindows()
