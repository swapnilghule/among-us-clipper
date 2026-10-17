"""
Among Us player colors: detect a crewmate's color from their avatar.

Used by chat.py (reads the avatar next to each chat bubble) and by the
editor (draws the color name, e.g. "MAROON", instead of the player's name).
"""

import cv2
import numpy as np

# name: body color (RGB)
PALETTE = {
    "red": (198, 17, 17),
    "blue": (19, 46, 210),
    "green": (17, 128, 45),
    "pink": (238, 84, 187),
    "orange": (240, 125, 13),
    "yellow": (246, 246, 87),
    "black": (63, 71, 78),
    "white": (215, 225, 241),
    "purple": (107, 47, 188),
    "brown": (113, 73, 30),
    "cyan": (56, 255, 221),
    "lime": (80, 240, 57),
    "maroon": (95, 29, 46),
    "rose": (236, 192, 211),
    "banana": (240, 231, 168),
    "gray": (117, 133, 147),
    "tan": (145, 136, 119),
    "coral": (215, 100, 100),
}
VISOR = np.array([149, 202, 220])
SHADE = 0.66  # the darker side of a crewmate's body


def hex_color(name):
    r, g, b = PALETTE.get(name, (255, 255, 255))
    return f"#{r:02x}{g:02x}{b:02x}"


def label(name):
    return name.upper() if name else ""


def detect_color(frame_bgr, box):
    """Color name of the crewmate inside box = (x1, y1, x2, y2), or None.
    Looks at the lower part of the avatar (the body), skipping the black
    outline, the light-blue visor and the white bubble background."""
    x1, y1, x2, y2 = (int(v) for v in box)
    h, w = frame_bgr.shape[:2]
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    if x2 - x1 < 6 or y2 - y1 < 6:
        return None
    # Skip the top (hats), keep head sides + body (outfits often cover the legs)
    body = frame_bgr[y1 + int((y2 - y1) * 0.25):y2 - int((y2 - y1) * 0.1), x1:x2]
    px = cv2.cvtColor(body, cv2.COLOR_BGR2RGB).reshape(-1, 3).astype(int)
    if len(px) == 0:
        return None
    keep = (px.max(axis=1) > 45)                       # not the black outline
    keep &= (px.min(axis=1) < 245)                     # not the white bubble
    keep &= (np.abs(px - VISOR).sum(axis=1) > 70)      # not the visor
    px = px[keep]
    if len(px) < 25:
        return None

    # Most common color (coarse bins), then its average
    bins = (px // 32)
    keys = bins[:, 0] * 64 + bins[:, 1] * 8 + bins[:, 2]
    vals, counts = np.unique(keys, return_counts=True)
    top = vals[np.argmax(counts)]
    color = px[keys == top].mean(axis=0)

    best, best_d = None, 1e9
    for name, rgb in PALETTE.items():
        rgb = np.array(rgb)
        d = min(np.linalg.norm(color - rgb), np.linalg.norm(color - rgb * SHADE))
        if d < best_d:
            best, best_d = name, d
    return best if best_d < 110 else None
