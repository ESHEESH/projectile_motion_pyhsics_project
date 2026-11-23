import cv2
import mediapipe as mp
import numpy as np
import time
import os
import tempfile
import math
import random

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib_cache"))

# ─────────────────────────────────────────────
#  CAMERA SETUP
# ─────────────────────────────────────────────
if os.name == "nt":
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
else:
    cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

# ─────────────────────────────────────────────
#  MEDIAPIPE SETUP
# ─────────────────────────────────────────────
if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
    mpHands = mp.solutions.hands
    hands   = mpHands.Hands(max_num_hands=2,
                             min_detection_confidence=0.7,
                             min_tracking_confidence=0.6)
    mpDraw             = mp.solutions.drawing_utils
    mpHands_CONNECTIONS = mpHands.HAND_CONNECTIONS
    use_tasks_api      = False
else:
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    model_path = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
    with open(model_path, "rb") as f:
        model_buffer = f.read()

    base_options = mp_python.BaseOptions(model_asset_buffer=model_buffer)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6
    )
    detector = vision.HandLandmarker.create_from_options(options)
    use_tasks_api = True

    HAND_CONNECTIONS = [
        (0,1),(1,2),(2,3),(3,4),
        (0,5),(5,6),(6,7),(7,8),
        (5,9),(9,10),(10,11),(11,12),
        (9,13),(13,14),(14,15),(15,16),
        (13,17),(17,18),(18,19),(19,20),
        (0,17)
    ]

    class CompatibilityHand:
        def __init__(self, landmarks):
            self.landmark = landmarks

    class CompatibilityDraw:
        @staticmethod
        def draw_landmarks(image, hand_lms, connections=None):
            h, w, _ = image.shape
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lms.landmark]
            if connections:
                for s, e in connections:
                    if s < len(pts) and e < len(pts):
                        cv2.line(image, pts[s], pts[e], (0, 255, 0), 2)
            for pt in pts:
                cv2.circle(image, pt, 4, (0, 0, 255), -1)

    mpDraw             = CompatibilityDraw()
    mpHands_CONNECTIONS = HAND_CONNECTIONS

# ─────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────
W, H          = 640, 480
GRAVITY       = 0.25          # pixels / frame²
MAX_POWER     = 180           # px distance → full power
MAX_SPEED     = 28            # pixels / frame at full draw
ARROW_R       = 5             # arrow head radius
BLOCK_W, BLOCK_H = 50, 50
PLAYER_HP     = 5
AI_SHOOT_INTERVAL = 2.5       # seconds between AI shots
PALM_OPEN_THRESH  = 0.08      # finger-spread threshold for open palm

# colours
COL_BG_OVERLAY = (0, 0, 0)
COL_PLAYER     = (50, 200, 50)
COL_AI         = (50, 50, 220)
COL_ARROW_P    = (0, 255, 255)
COL_ARROW_AI   = (0, 100, 255)
COL_BOW        = (200, 180, 50)
COL_STRING     = (220, 220, 220)
COL_UI         = (255, 255, 255)
COL_HIT        = (0, 0, 255)
COL_POWER_BAR  = (0, 200, 255)

# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
def dist(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def normalize(v):
    l = math.hypot(v[0], v[1])
    return (v[0]/l, v[1]/l) if l > 0 else (1, 0)

def lerp(a, b, t):
    return a + (b - a) * t

def is_palm_open(lms, w, h):
    """Return True when all 4 fingers are extended (open palm)."""
    tips  = [8, 12, 16, 20]
    bases = [6, 10, 14, 18]
    open_count = 0
    for tip, base in zip(tips, bases):
        ty = lms.landmark[tip].y * h
        by = lms.landmark[base].y * h
        if ty < by:          # tip above base → finger up
            open_count += 1
    return open_count >= 3

def get_wrist(lms, w, h):
    return (int(lms.landmark[0].x * w), int(lms.landmark[0].y * h))

def get_index_mcp(lms, w, h):
    """Landmark 5 = index MCP — good proxy for 'knuckle centre'."""
    return (int(lms.landmark[5].x * w), int(lms.landmark[5].y * h))

def draw_bow(canvas, bow_pos, pull_pos, color=COL_BOW):
    """Draw a simple bow arc from bow_pos aimed toward pull_pos."""
    dx = bow_pos[0] - pull_pos[0]
    dy = bow_pos[1] - pull_pos[1]
    angle = math.atan2(dy, dx)          # bow faces away from pull

    bow_len = 40
    perp = angle + math.pi / 2

    p1 = (int(bow_pos[0] + bow_len * math.cos(perp)),
          int(bow_pos[1] + bow_len * math.sin(perp)))
    p2 = (int(bow_pos[0] - bow_len * math.cos(perp)),
          int(bow_pos[1] - bow_len * math.sin(perp)))

    # arc as a polyline approximation
    pts = []
    for t in np.linspace(0, 1, 15):
        mid = (int(lerp(p1[0], p2[0], t)), int(lerp(p1[1], p2[1], t)))
        bulge = 18 * math.sin(t * math.pi)
        bx = int(mid[0] + bulge * math.cos(angle))
        by = int(mid[1] + bulge * math.sin(angle))
        pts.append([bx, by])
    pts = np.array(pts, dtype=np.int32)
    cv2.polylines(canvas, [pts], False, color, 3)

    # string
    cv2.line(canvas, p1, pull_pos, COL_STRING, 1)
    cv2.line(canvas, p2, pull_pos, COL_STRING, 1)

    return p1, p2   # bow tips (for arrow nock point)

def draw_block(canvas, cx, cy, w, h, color, label="", hp=1, max_hp=1):
    x1, y1 = cx - w//2, cy - h//2
    x2, y2 = cx + w//2, cy + h//2
    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, -1)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), (255,255,255), 1)
    if label:
        cv2.putText(canvas, label, (x1+4, y1+18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    # hp bar
    bar_w = int((hp / max_hp) * w)
    cv2.rectangle(canvas, (x1, y2+3), (x1+bar_w, y2+8), (0,220,0), -1)
    cv2.rectangle(canvas, (x1, y2+3), (x2, y2+8), (100,100,100), 1)

# ─────────────────────────────────────────────
#  ARROW CLASS
# ─────────────────────────────────────────────
class Arrow:
    def __init__(self, x, y, vx, vy, owner):
        self.x, self.y   = float(x), float(y)
        self.vx, self.vy = float(vx), float(vy)
        self.owner       = owner   # "player" or "ai"
        self.alive       = True
        self.trail       = []

    def update(self):
        self.vy += GRAVITY
        self.x  += self.vx
        self.y  += self.vy
        self.trail.append((int(self.x), int(self.y)))
        if len(self.trail) > 8:
            self.trail.pop(0)
        if self.x < -50 or self.x > W+50 or self.y > H+50 or self.y < -50:
            self.alive = False

    def draw(self, canvas):
        col = COL_ARROW_P if self.owner == "player" else COL_ARROW_AI
        # trail
        for i, pt in enumerate(self.trail):
            alpha = int(80 * (i / max(len(self.trail), 1)))
            cv2.circle(canvas, pt, 2, col, -1)
        # head
        cv2.circle(canvas, (int(self.x), int(self.y)), ARROW_R, col, -1)
        # shaft direction
        if len(self.trail) >= 2:
            cv2.line(canvas, self.trail[0], (int(self.x), int(self.y)), col, 2)

    def hits(self, cx, cy, bw, bh):
        return (cx - bw//2 < self.x < cx + bw//2 and
                cy - bh//2 < self.y < cy + bh//2)

# ─────────────────────────────────────────────
#  AI CONTROLLER
# ─────────────────────────────────────────────
class AI:
    def __init__(self):
        self.cx        = W - 80          # AI block centre x
        self.cy        = H // 2
        self.hp        = PLAYER_HP
        self.last_shot = time.time() - AI_SHOOT_INTERVAL
        self.vy        = 0               # vertical drift
        self.target_y  = self.cy

    def update(self, player_cx, player_cy):
        # Drift vertically toward random target
        if abs(self.cy - self.target_y) < 5:
            self.target_y = random.randint(80, H - 80)
        self.cy += int(math.copysign(2, self.target_y - self.cy))
        self.cx = W - 80

    def should_shoot(self):
        return time.time() - self.last_shot >= AI_SHOOT_INTERVAL

    def shoot(self, player_cx, player_cy):
        self.last_shot = time.time()
        # Aim with slight random spread
        dx = player_cx - self.cx
        dy = player_cy - self.cy
        # compensate for gravity: simple iterative approximation
        speed = MAX_SPEED * 0.65
        d     = max(math.hypot(dx, dy), 1)
        t_est = d / max(speed, 1)
        # lead target for gravity
        aim_y = dy - 0.5 * GRAVITY * t_est * t_est
        nx, ny = normalize((dx, aim_y))
        spread = random.uniform(-0.08, 0.08)
        vx = -speed * nx + spread * speed   # AI shoots LEFT
        vy = -speed * ny + spread * speed
        return Arrow(self.cx, self.cy, vx, vy, "ai")

    def draw(self, canvas):
        draw_block(canvas, self.cx, self.cy,
                   BLOCK_W, BLOCK_H, COL_AI, "AI",
                   self.hp, PLAYER_HP)

# ─────────────────────────────────────────────
#  GAME STATE
# ─────────────────────────────────────────────
class Game:
    def __init__(self):
        self.reset()

    def reset(self):
        self.player_hp   = PLAYER_HP
        self.player_cx   = 80
        self.player_cy   = H // 2
        self.ai          = AI()
        self.arrows      = []
        self.hit_flashes = []   # [(x,y,t)]
        self.state       = "playing"   # "playing" | "win" | "lose"
        self.last_fire   = 0
        self.fire_cooldown = 0.6       # seconds

    def add_flash(self, x, y):
        self.hit_flashes.append([x, y, time.time()])

    def update_flashes(self):
        now = time.time()
        self.hit_flashes = [f for f in self.hit_flashes if now - f[2] < 0.3]

game = Game()

# ─────────────────────────────────────────────
#  FPS
# ─────────────────────────────────────────────
pTime = time.time()

# ─────────────────────────────────────────────
#  MAIN LOOP
# ─────────────────────────────────────────────
print("Bow Master – Hand Tracking. Press 'q' to quit, 'r' to restart, 'f' to toggle fullscreen.")
cv2.namedWindow("Bow Master", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Bow Master", W, H)
cv2.setWindowProperty("Bow Master", cv2.WND_PROP_TOPMOST, 1)
is_fullscreen = False

def letterbox(frame, target_w, target_h):
    """Fit frame into target dimensions without stretching, black bars fill the rest."""
    fh, fw = frame.shape[:2]
    scale  = min(target_w / fw, target_h / fh)
    nw, nh = int(fw * scale), int(fh * scale)
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    out = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x_off = (target_w - nw) // 2
    y_off = (target_h - nh) // 2
    out[y_off:y_off+nh, x_off:x_off+nw] = resized
    return out

while True:
    success, raw = cap.read()
    if not success or raw is None:
        continue

    # ── Rotation fix: rotate 90° clockwise to correct sideways camera ──
    raw = cv2.rotate(raw, cv2.ROTATE_90_CLOCKWISE)
    # After rotation the frame is 480×640; resize to our working size
    raw = cv2.resize(raw, (W, H))
    img = cv2.flip(raw, 1)

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # ── Hand detection ─────────────────────────────────────────────────
    if not use_tasks_api:
        results           = hands.process(imgRGB)
        hand_lms_list     = results.multi_hand_landmarks  if results.multi_hand_landmarks  else []
        hand_class_list   = results.multi_handedness      if results.multi_handedness      else []
    else:
        mp_image          = mp.Image(image_format=mp.ImageFormat.SRGB, data=imgRGB)
        det               = detector.detect(mp_image)
        hand_lms_list     = ([CompatibilityHand(lm) for lm in det.hand_landmarks]
                              if det and det.hand_landmarks else [])
        hand_class_list   = []   # tasks API handedness handled differently

    # ── Classify left / right hand ──────────────────────────────────────
    left_hand  = None
    right_hand = None

    if use_tasks_api and det and det.handedness:
        for i, hnd in enumerate(det.handedness):
            label = hnd[0].category_name   # "Left" or "Right"
            if label == "Left"  and i < len(hand_lms_list):
                left_hand  = hand_lms_list[i]
            if label == "Right" and i < len(hand_lms_list):
                right_hand = hand_lms_list[i]
    elif hand_class_list:
        for i, hc in enumerate(hand_class_list):
            label = hc.classification[0].label
            if label == "Left"  and i < len(hand_lms_list):
                left_hand  = hand_lms_list[i]
            if label == "Right" and i < len(hand_lms_list):
                right_hand = hand_lms_list[i]
    else:
        # Fallback: assign by x position (leftmost = left hand)
        if len(hand_lms_list) == 1:
            left_hand = hand_lms_list[0]
        elif len(hand_lms_list) >= 2:
            x0 = hand_lms_list[0].landmark[0].x
            x1 = hand_lms_list[1].landmark[0].x
            if x0 < x1:
                left_hand, right_hand = hand_lms_list[0], hand_lms_list[1]
            else:
                left_hand, right_hand = hand_lms_list[1], hand_lms_list[0]

    # ── Game canvas: semi-transparent overlay on camera ─────────────────
    canvas = img.copy()
    overlay = img.copy()

    # Dark tint on the right half (AI territory)
    cv2.rectangle(overlay, (W//2, 0), (W, H), (0, 0, 40), -1)
    cv2.addWeighted(overlay, 0.25, canvas, 0.75, 0, canvas)

    # Centre dividing line
    cv2.line(canvas, (W//2, 0), (W//2, H), (100, 100, 100), 1)

    if game.state == "playing":

        # ── Player block position tracks left-hand wrist ────────────────
        if left_hand:
            wx, wy = get_wrist(left_hand, W, H)
            # Clamp player to left half
            game.player_cx = max(BLOCK_W//2+5, min(W//2 - BLOCK_W//2 - 5, wx))
            game.player_cy = max(BLOCK_H//2+5, min(H - BLOCK_H//2 - 5, wy))

        # ── Bow / draw mechanics ────────────────────────────────────────
        bow_pos  = (game.player_cx, game.player_cy)
        draw_power = 0
        fire_arrow = False

        MIN_PULL_POWER = 0.18   # must pull at least 18% of MAX_POWER to fire

        if left_hand and right_hand:
            pull_pos    = get_wrist(right_hand, W, H)
            draw_dist   = dist(bow_pos, pull_pos)
            draw_power  = min(draw_dist / MAX_POWER, 1.0)

            # ── Pull indicator: magenta line + dots bow→pull ────────────
            COL_PULL = (255, 0, 255)   # magenta / pink
            dot_r    = 10
            # Line from bow hand to pull hand
            cv2.line(canvas, bow_pos, pull_pos, COL_PULL, 3)
            # Dot at bow hand (anchor / base)
            cv2.circle(canvas, bow_pos, dot_r, COL_PULL, -1)
            # Dot at midpoint
            mid = ((bow_pos[0]+pull_pos[0])//2, (bow_pos[1]+pull_pos[1])//2)
            cv2.circle(canvas, mid, dot_r - 3, COL_PULL, -1)
            # Dot at pull hand (tip)
            cv2.circle(canvas, pull_pos, dot_r, COL_PULL, -1)

            # Draw bow on canvas
            draw_bow(canvas, bow_pos, pull_pos)

            # Power bar above player block
            bar_x = game.player_cx - 25
            bar_y = game.player_cy - BLOCK_H//2 - 22
            cv2.rectangle(canvas, (bar_x, bar_y), (bar_x+50, bar_y+10), (60,60,60), -1)
            if draw_power >= MIN_PULL_POWER:
                bar_col = COL_POWER_BAR
            else:
                bar_col = (100, 100, 100)   # grey = not enough pull yet
            cv2.rectangle(canvas, (bar_x, bar_y),
                          (bar_x + int(50*draw_power), bar_y+10), bar_col, -1)
            cv2.rectangle(canvas, (bar_x, bar_y), (bar_x+50, bar_y+10), COL_UI, 1)
            # Min-pull threshold marker on bar
            thresh_x = bar_x + int(50 * MIN_PULL_POWER)
            cv2.line(canvas, (thresh_x, bar_y-2), (thresh_x, bar_y+12), (0,255,0), 2)

            # Detect open-palm on RIGHT hand → fire
            if is_palm_open(right_hand, W, H):
                now = time.time()
                if now - game.last_fire >= game.fire_cooldown and draw_power >= MIN_PULL_POWER:
                    fire_arrow = True
                    game.last_fire = now

        if fire_arrow:
            pull_pos = get_wrist(right_hand, W, H)
            dx = bow_pos[0] - pull_pos[0]
            dy = bow_pos[1] - pull_pos[1]
            nx, ny = normalize((dx, dy))
            speed  = draw_power * MAX_SPEED
            game.arrows.append(Arrow(bow_pos[0], bow_pos[1],
                                     nx * speed, ny * speed, "player"))

        # ── AI update & shoot ──────────────────────────────────────────
        game.ai.update(game.player_cx, game.player_cy)
        if game.ai.should_shoot():
            game.arrows.append(game.ai.shoot(game.player_cx, game.player_cy))

        # ── Arrow physics & collision ──────────────────────────────────
        for arrow in game.arrows:
            arrow.update()
            if not arrow.alive:
                continue
            if arrow.owner == "player":
                if arrow.hits(game.ai.cx, game.ai.cy, BLOCK_W, BLOCK_H):
                    game.ai.hp -= 1
                    arrow.alive = False
                    game.add_flash(game.ai.cx, game.ai.cy)
            else:
                if arrow.hits(game.player_cx, game.player_cy, BLOCK_W, BLOCK_H):
                    game.player_hp -= 1
                    arrow.alive = False
                    game.add_flash(game.player_cx, game.player_cy)

        game.arrows = [a for a in game.arrows if a.alive]

        # ── Draw everything ────────────────────────────────────────────
        for arrow in game.arrows:
            arrow.draw(canvas)

        # Hit flashes
        game.update_flashes()
        for fx, fy, ft in game.hit_flashes:
            cv2.circle(canvas, (fx, fy), 30, COL_HIT, 3)

        # Player block
        draw_block(canvas, game.player_cx, game.player_cy,
                   BLOCK_W, BLOCK_H, COL_PLAYER, "P1",
                   game.player_hp, PLAYER_HP)

        # AI block
        game.ai.draw(canvas)

        # Draw hand skeletons (small, for reference)
        for hnd in hand_lms_list:
            mpDraw.draw_landmarks(canvas, hnd, mpHands_CONNECTIONS)

        # ── HUD ────────────────────────────────────────────────────────
        # Player HP
        for i in range(PLAYER_HP):
            col = (0,220,0) if i < game.player_hp else (60,60,60)
            cv2.rectangle(canvas, (10 + i*22, 10), (28 + i*22, 28), col, -1)
        cv2.putText(canvas, "P1", (10, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL_PLAYER, 1)

        # AI HP
        for i in range(PLAYER_HP):
            col = (0,80,220) if i < game.ai.hp else (60,60,60)
            cv2.rectangle(canvas, (W-30 - i*22, 10), (W-12 - i*22, 28), col, -1)
        cv2.putText(canvas, "AI", (W-35, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL_AI, 1)

        # Win / lose check
        if game.ai.hp <= 0:
            game.state = "win"
        if game.player_hp <= 0:
            game.state = "lose"

        # Instruction hint
        hint = ""
        if not left_hand and not right_hand:
            hint = "Show both hands to play"
        elif not right_hand:
            hint = "Raise right hand to draw"
        elif left_hand and right_hand:
            if draw_power >= MIN_PULL_POWER:
                hint = "Open palm to FIRE!"
            else:
                hint = "Pull right hand further to charge"
        cv2.putText(canvas, hint, (W//2 - 150, H-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1)

    else:
        # ── End screen ────────────────────────────────────────────────
        msg   = "YOU WIN!" if game.state == "win" else "YOU LOSE!"
        col   = (0, 255, 100) if game.state == "win" else (0, 60, 255)
        cv2.putText(canvas, msg,
                    (W//2 - 90, H//2 - 10),
                    cv2.FONT_HERSHEY_DUPLEX, 1.6, col, 3)
        cv2.putText(canvas, "Press R to restart",
                    (W//2 - 100, H//2 + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COL_UI, 2)

    # ── FPS ────────────────────────────────────────────────────────────
    cTime = time.time()
    fps   = 1 / max(cTime - pTime, 0.001)
    pTime = cTime
    cv2.putText(canvas, f"{int(fps)} FPS", (W-90, H-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160,160,160), 1)

    # ── Display: letterbox when fullscreen so it never stretches ──────
    if is_fullscreen:
        screen_w = cv2.getWindowImageRect("Bow Master")[2]
        screen_h = cv2.getWindowImageRect("Bow Master")[3]
        if screen_w > 0 and screen_h > 0:
            display = letterbox(canvas, screen_w, screen_h)
        else:
            display = canvas
    else:
        display = canvas

    cv2.imshow("Bow Master", display)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('r'):
        game.reset()
    if key == ord('f'):
        is_fullscreen = not is_fullscreen
        if is_fullscreen:
            cv2.setWindowProperty("Bow Master", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.setWindowProperty("Bow Master", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Bow Master", W, H)

cap.release()
cv2.destroyAllWindows()
