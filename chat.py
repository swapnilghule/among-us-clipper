"""
Step 2 of the Among Us clipper: read the chat from each meeting.

Needs scan.py to have run first (uses storage/scans/<video>.json).

What it does:
  - Goes to each meeting found by scan.py
  - Looks at one frame per second while the chat window is open
  - Finds the chat bubbles using "SKIP VOTE" and the "Send" button as anchors
  - Ignores faded names behind the chat (only text on white bubbles counts)
  - Reads: player name + message (left side), your own messages (right side),
    votes ("X has voted"), and "X left the game"
  - Joins frames together into one clean conversation, without repeats
  - Saves to storage/chats/<video>.json and prints the chat log
  - Saves a debug picture per meeting in storage/debug (green = text it read)

Usage (from ~/amongus-clipper with the venv active):
  python chat.py                      # newest scanned video
  python chat.py path/to/video.mp4    # a specific video
"""

import difflib
import json
import re
import sys
from collections import Counter
from pathlib import Path

import cv2
from PIL import Image
from ocrmac import ocrmac

import colors

BASE = Path(__file__).resolve().parent
RAW = BASE / "storage" / "raw"
SCANS = BASE / "storage" / "scans"
CHATS = BASE / "storage" / "chats"
DEBUG = BASE / "storage" / "debug"

SAMPLE_EVERY = 1.0   # seconds between frames inside a meeting
PADDING = 3          # extra seconds before/after each meeting

# Layout, measured from your screenshots. All values are fractions of W,
# the meeting panel width (worked out from SKIP VOTE and the Send button).
REGION_LEFT, REGION_RIGHT = 0.08, 0.78      # chat panel, horizontally
REGION_TOP, REGION_BOTTOM = -0.70, -0.225   # chat panel, above SKIP VOTE
NAME_COL = 0.171        # left edge of other players' names/messages
ME_COL = 0.668          # right edge of your own names/messages
COL_TOL = 0.012         # how far a line may be from its column
BUBBLE_GAP = 0.045      # bigger vertical gap than this = new bubble

JUNK = re.compile(r"^(\d+\s*/\s*100|send)$", re.I)


# ---------- reading a frame ----------

def ocr_lines(frame):
    h, w = frame.shape[:2]
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    lines = []
    for text, conf, (x, y, bw, bh) in ocrmac.OCR(img, recognition_level="accurate").recognize():
        text = text.strip()
        if not text:
            continue
        # Apple's OCR measures from the bottom-left; convert to top-left pixels
        lines.append({
            "text": text,
            "left": x * w,
            "right": (x + bw) * w,
            "top": (1 - y - bh) * h,
            "bottom": (1 - y) * h,
        })
    return lines


def find(lines, pattern):
    rx = re.compile(pattern, re.I)
    return next((l for l in lines if rx.search(l["text"])), None)


SEND_RATIO = 0.7476  # (right of "Send" - left of SKIP VOTE) / W


def get_anchor(lines):
    """Return (x0, y0, W) from SKIP VOTE + the Send button, or None if the
    chat window isn't open in a meeting. The timer text changes during a
    meeting, so it isn't used."""
    skip = find(lines, r"skip\s*vote")
    send = find(lines, r"^send$")
    if not skip or not send or send["right"] <= skip["right"]:
        return None
    W = (send["right"] - skip["left"]) / SEND_RATIO
    return skip["left"], skip["top"], W


def on_bubble(gray, line, side):
    """True if the text sits on a white chat bubble (not faded text behind it).
    Checks a small strip next to the text on the given side(s)."""
    h = line["bottom"] - line["top"]
    y1, y2 = int(line["top"] + h * 0.2), int(line["bottom"] - h * 0.2)
    strips = []
    if side in ("right", "both"):
        x1 = int(line["right"] + 6)
        strips.append((x1, x1 + int(h * 0.6)))
    if side in ("left", "both"):
        x2 = int(line["left"] - 4)
        strips.append((x2 - int(h * 0.4), x2))
    for x1, x2 in strips:
        patch = gray[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
        if patch.size and patch.mean() > 210:  # white bubble ~230-255, faded ~165
            return True
    return False


def chat_region(anchor):
    x0, y0, W = anchor
    return (x0 + REGION_LEFT * W, y0 + REGION_TOP * W,
            x0 + REGION_RIGHT * W, y0 + REGION_BOTTOM * W)


def group_bubbles(col, is_me, region_top, W):
    col = sorted(col, key=lambda l: l["top"])
    groups, current = [], []
    for l in col:
        if current and l["top"] - current[-1]["top"] > BUBBLE_GAP * W:
            groups.append(current)
            current = []
        current.append(l)
    if current:
        groups.append(current)

    bubbles = []
    for g in groups:
        if len(g) < 2:
            continue  # name or message missing (bubble cut off)
        if g[0]["top"] < region_top + 0.03 * W:
            continue  # bubble cut off at the top of the chat window
        bubbles.append({
            "top": g[0]["top"],
            "name": g[0]["text"],
            "text": " ".join(l["text"] for l in g[1:]),
            "is_me": is_me,
            "lines": g,
        })
    return bubbles


def _box(lines):
    return [min(l["left"] for l in lines), min(l["top"] for l in lines),
            max(l["right"] for l in lines), max(l["bottom"] for l in lines)]


def add_geometry(frame, b, anchor):
    """Where the bubble is on screen (so the editor can zoom on it later) and
    the speaker's color (read from their avatar)."""
    x0, y0, W = anchor
    name_l, msg_ls = b["lines"][0], b["lines"][1:]
    name_box, text_box = _box([name_l]), _box(msg_ls)
    top, bottom = name_box[1] - 0.012 * W, text_box[3] + 0.012 * W
    if not b["is_me"]:
        name_x = x0 + NAME_COL * W
        avatar = [name_x - 0.085 * W, name_box[1] - 0.01 * W, name_x - 0.018 * W, bottom]
        bubble = [avatar[0] - 0.01 * W, top, max(name_box[2], text_box[2]) + 0.025 * W, bottom]
    else:
        me_x = x0 + ME_COL * W
        avatar = [me_x + 0.005 * W, name_box[1] - 0.01 * W, me_x + 0.075 * W, bottom]
        bubble = [min(name_box[0], text_box[0]) - 0.025 * W, top, avatar[2] + 0.01 * W, bottom]
    b["geo"] = {k: [round(v) for v in box] for k, box in
                (("bubble", bubble), ("name", name_box), ("text", text_box), ("avatar", avatar))}
    b["color"] = colors.detect_color(frame, avatar)
    del b["lines"]


def parse_frame(frame, lines, anchor):
    """Return (messages, votes, left_game, lines_used) seen in one frame."""
    x0, y0, W = anchor
    left, top, right, bottom = chat_region(anchor)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # A line counts if it STARTS inside the chat panel, so a message at the
    # very bottom edge isn't lost.
    inside = [
        l for l in lines
        if l["left"] >= left and l["right"] <= right
        and l["top"] >= top and l["top"] <= bottom
        and not JUNK.match(l["text"])
    ]

    votes = [l["text"] for l in inside if "has voted" in l["text"].lower()]
    left_game = [l["text"] for l in lines if "left the game" in l["text"].lower()]

    name_x = x0 + NAME_COL * W
    me_x = x0 + ME_COL * W
    others = [l for l in inside if abs(l["left"] - name_x) < COL_TOL * W
              and "has voted" not in l["text"].lower()
              and on_bubble(gray, l, "both")]
    mine = [l for l in inside if abs(l["right"] - me_x) < COL_TOL * W
            and l not in others and "has voted" not in l["text"].lower()
            and on_bubble(gray, l, "left")]

    bubbles = group_bubbles(others, False, top, W) + group_bubbles(mine, True, top, W)
    for b in bubbles:
        add_geometry(frame, b, anchor)
    bubbles.sort(key=lambda b: b["top"])
    return bubbles, votes, left_game, others + mine


# ---------- joining frames together ----------

def norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def same(a, b):
    sa, sb = norm(a["name"] + "|" + a["text"]), norm(b["name"] + "|" + b["text"])
    return sa == sb or difflib.SequenceMatcher(None, sa, sb).ratio() > 0.85


def merge(history, new):
    """Add messages from a new frame without repeating ones already seen."""
    if not new:
        return
    for k in range(min(len(history), len(new)), 0, -1):
        if all(same(history[-k + i], new[i]) for i in range(k)):
            history.extend(new[k:])
            return
    recent = history[-20:]
    if recent and all(any(same(m, r) for r in recent) for m in new):
        return  # all already seen (e.g. chat scrolled up)
    history.extend(new)


def add_unique(lst, items, t=None):
    """Add new texts once. With t, items are stored as {"text", "t"}."""
    for it in items:
        existing = [x["text"] if isinstance(x, dict) else x for x in lst]
        if not any(norm(it) == norm(x) for x in existing):
            lst.append({"text": it, "t": round(t, 1)} if t is not None else it)


# ---------- main ----------

def fmt(t):
    return f"{int(t // 60):02d}:{int(t % 60):02d}"


def save_debug(frame, anchor, used, path):
    img = frame.copy()
    l, t, r, b = (int(v) for v in chat_region(anchor))
    cv2.rectangle(img, (l, t), (r, b), (255, 120, 0), 3)
    for ln in used:
        cv2.rectangle(img, (int(ln["left"]), int(ln["top"])),
                      (int(ln["right"]), int(ln["bottom"])), (0, 200, 0), 2)
    cv2.imwrite(str(path), img)


def pick_video():
    if len(sys.argv) > 1:
        return Path(sys.argv[1])
    scans = sorted(SCANS.glob("*.json"), key=lambda p: p.stat().st_mtime)
    if not scans:
        sys.exit("No scans found. Run scan.py first.")
    name = json.loads(scans[-1].read_text())["video"]
    return RAW / name


def main():
    video = pick_video()
    scan_file = SCANS / f"{video.stem}.json"
    if not scan_file.exists():
        sys.exit(f"No scan for {video.name}. Run: python scan.py \"{video}\"")
    scan = json.loads(scan_file.read_text())

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        sys.exit(f"Could not open {video}")
    DEBUG.mkdir(parents=True, exist_ok=True)

    # Among Us keeps old messages in the chat when a new meeting opens.
    # So there is ONE history for the whole game: a message belongs to the
    # meeting where it was first seen, and repeats in later meetings are dropped.
    history = []
    results = []
    for m in scan["meetings"]:
        start = max(0, m["start_seconds"] - PADDING)
        end = m["end_seconds"] + PADDING
        votes, left_game = [], []
        debug_saved = False
        print(f"Reading meeting {m['number']} ({m['start']} - {m['end']})...")

        t = start
        while t <= end:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, frame = cap.read()
            if not ok:
                break
            lines = ocr_lines(frame)
            anchor = get_anchor(lines)
            if anchor:  # chat window is open in a meeting
                bubbles, v, lg, used = parse_frame(frame, lines, anchor)
                for b in bubbles:
                    b["t"] = round(t, 1)  # second this message was first seen
                    b["meeting"] = m["number"]
                merge(history, bubbles)
                add_unique(votes, v, t)
                add_unique(left_game, lg, t)
                if not debug_saved and bubbles:
                    save_debug(frame, anchor, used,
                               DEBUG / f"{video.stem}_meeting{m['number']}.png")
                    debug_saved = True
            t += SAMPLE_EVERY

        mine = [b for b in history if b["meeting"] == m["number"]]
        me_names = Counter(b["name"] for b in mine if b["is_me"])
        results.append({
            "number": m["number"],
            "start": m["start"],
            "end": m["end"],
            "start_seconds": m["start_seconds"],
            "end_seconds": m["end_seconds"],
            "me": me_names.most_common(1)[0][0] if me_names else None,
            "messages": [{"t": b["t"], "time": fmt(b["t"]), "name": b["name"],
                          "text": b["text"], "is_me": b["is_me"],
                          "color": b.get("color"), "geo": b.get("geo")}
                         for b in mine],
            "votes": votes,
            "left_game": left_game,
        })

    cap.release()

    out = {
        "video": video.name,
        "role": scan.get("role"),
        "ejections": scan.get("ejections", []),
        "meetings": results,
    }
    CHATS.mkdir(parents=True, exist_ok=True)
    out_file = CHATS / f"{video.stem}.json"
    out_file.write_text(json.dumps(out, indent=2, ensure_ascii=False))

    print()
    for r in results:
        print(f"===== Meeting {r['number']} ({r['start']} - {r['end']}) =====")
        if not r["messages"]:
            print("  (no chat read — was the chat window open?)")
        for msg in r["messages"]:
            who = f"{msg['name']} (you)" if msg["is_me"] else msg["name"]
            col = f" [{msg['color']}]" if msg.get("color") else ""
            print(f"  {msg['time']}  {who}{col}: {msg['text']}")
        for v in r["votes"]:
            print(f"  {fmt(v['t'])}  [vote] {v['text']}")
        for lg in r["left_game"]:
            print(f"  {fmt(lg['t'])}  [left] {lg['text']}")
        print()
    print(f"Saved: {out_file}")
    print(f"Debug pictures: {DEBUG}")


if __name__ == "__main__":
    main()
