"""
Gameplay moments: find funny things that happen OUTSIDE meetings.

Needs scan.py to have run first (to know where meetings are, and your role).
Needs a vision AI in Ollama, to look at screenshots:
    ollama pull qwen2.5vl:7b

What it does:
  1. Looks at the gameplay between meetings (2 frames a second) and finds
     sudden big screen changes. A kill animation is a huge one.
  2. Shows the vision AI those moments and asks what they are.
  3. For every kill, it looks at the seconds BEFORE: what were you doing?
     ("petting my pet", "doing the card swipe") and who was the killer (color).
  4. Saves storage/moments/<video>.json. story.py and planner.py pick it up.

As crewmate: finds when YOU get killed.  As impostor: finds YOUR kills.

Usage (from ~/amongus-clipper with the venv active):
  python moments.py                      # newest scanned video
  python moments.py path/to/video.mp4    # a specific video
"""

import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np

import colors

BASE = Path(__file__).resolve().parent
RAW = BASE / "storage" / "raw"
SCANS = BASE / "storage" / "scans"
MOMENTS = BASE / "storage" / "moments"
SHOTS = BASE / "storage" / "moments_frames"

VISION_MODELS = ["qwen2.5vl:7b", "qwen2.5vl:3b", "llama3.2-vision:11b",
                 "llama3.2-vision", "llava:13b", "llava:7b", "llava"]
SAMPLE_FPS = 2           # frames looked at per second of gameplay
SPIKE = 26               # how big a screen change must be (0-255)
MAX_CANDIDATES = 30      # most moments shown to the vision AI
MIN_GAP = 4              # seconds between two candidates
CONTEXT = (7.0, 4.0, 1.5)  # seconds before a kill to look at

SCENES = ["kill_animation", "body_reported", "emergency_meeting", "task_window",
          "map", "sabotage_alarm", "vent", "walking_around", "meeting_or_voting",
          "role_reveal", "ejection", "victory_or_defeat", "other"]


def fmt(t):
    return f"{int(t // 60):02d}:{int(t % 60):02d}"


# ---------- 1. find big screen changes ----------

def skip_windows(scan):
    """Times to ignore: meetings (+ ejection after), role reveal, end screen."""
    win = [(m["start_seconds"] - 6, m["end_seconds"] + 15) for m in scan["meetings"]]
    role = scan.get("role")
    if role:
        m, s = role["time"].split(":")
        t = int(m) * 60 + int(s)
        win.append((t - 6, t + 6))
    end = scan.get("end")
    if end:
        win.append((end["seconds"] - 5, 10 ** 9))
    return win


def find_candidates(video, scan):
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    step = max(1, round(fps / SAMPLE_FPS))
    skips = skip_windows(scan)
    prev, i, spikes = None, 0, []
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
    while True:
        ok = cap.grab()
        if not ok:
            break
        if i % step == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            t = i / fps
            small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (96, 54)).astype(int)
            if prev is not None:
                diff = float(np.abs(small - prev).mean())
                if diff > SPIKE and not any(a <= t <= b for a, b in skips):
                    spikes.append((t, diff))
            prev = small
            if int(t) % 30 == 0:
                print(f"\r  looking at gameplay {fmt(t)} / {fmt(total)}", end="", flush=True)
        i += 1
    cap.release()
    print()

    chosen = []
    for t, d in sorted(spikes, key=lambda x: -x[1]):
        if all(abs(t - c) >= MIN_GAP for c in chosen):
            chosen.append(t)
        if len(chosen) >= MAX_CANDIDATES:
            break
    return sorted(chosen)


def save_frame(video, t, path):
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0, t) * 1000)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    h, w = frame.shape[:2]
    if w > 1280:  # smaller pictures = faster vision AI
        frame = cv2.resize(frame, (1280, int(h * 1280 / w)))
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return path


# ---------- 2. ask the vision AI ----------

def pick_vision_model(ollama):
    try:
        listing = ollama.list()
    except Exception:
        sys.exit("Can't reach Ollama. Start the Ollama app or run: ollama serve")
    models = listing.get("models", []) if isinstance(listing, dict) else getattr(listing, "models", [])
    have = {(m.get("model") or m.get("name")) if isinstance(m, dict)
            else (getattr(m, "model", None) or getattr(m, "name", None)) for m in models}
    model = next((m for m in VISION_MODELS if m in have), None)
    if not model:
        sys.exit("No vision AI found. Install one with:  ollama pull qwen2.5vl:7b")
    return model


def ask_vision(ollama, model, prompt, images):
    try:
        resp = ollama.chat(model=model, format="json", options={"temperature": 0.1},
                           messages=[{"role": "user", "content": prompt,
                                      "images": [str(p) for p in images]}])
        data = json.loads(resp["message"]["content"])
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


CLASSIFY = f"""These are 2 screenshots, one second apart, from the game Among Us.
What is happening on screen? Pick ONE scene:
{", ".join(SCENES)}

"kill_animation" = the full-screen cutscene where one crewmate kills another
(a knife, a gun, a bite, the victim's body falling).
If it is a kill_animation, also say the color of the KILLER and of the VICTIM.
Colors: {", ".join(colors.PALETTE)}.

Reply ONLY with JSON:
{{"scene": "...", "killer_color": "<color or null>", "victim_color": "<color or null>",
 "description": "<what you see, max 12 words>"}}"""

DOING = """These are screenshots from the game Among Us, a few seconds apart, just
BEFORE the player got killed. The player is the crewmate in the CENTER of the screen.
What was the player doing? Answer in a short phrase starting with a verb in -ing form,
max 8 words. Examples: "petting my pet", "doing the card swipe task",
"standing alone in electrical", "walking down an empty hallway", "fixing wires".
Reply ONLY with JSON: {"doing": "..."}"""


def clean_color(c):
    c = str(c or "").strip().lower()
    return c if c in colors.PALETTE else None


def clean_doing(s):
    s = re.sub(r"[^a-zA-Z0-9 '\-]", "", str(s or "")).strip().lower()
    s = re.sub(r"^(the player is|player is|i am|i was|he is|they are)\s+", "", s)
    return " ".join(s.split()[:8]) or "minding my own business"


# ---------- main ----------

def main():
    import ollama

    if len(sys.argv) > 1:
        video = Path(sys.argv[1])
    else:
        scans = sorted(SCANS.glob("*.json"), key=lambda p: p.stat().st_mtime)
        if not scans:
            sys.exit("No scans found. Run scan.py first.")
        video = RAW / json.loads(scans[-1].read_text())["video"]
    scan_f = SCANS / f"{video.stem}.json"
    if not scan_f.exists():
        sys.exit(f"No scan for {video.name}. Run: python scan.py \"{video}\"")
    scan = json.loads(scan_f.read_text())
    role = (scan.get("role") or {}).get("role", "crewmate")

    model = pick_vision_model(ollama)
    print(f"Using vision AI: {model}")
    print(f"Finding big screen changes in the gameplay of {video.name}...")
    candidates = find_candidates(video, scan)
    print(f"{len(candidates)} moments to check")

    shots = SHOTS / video.stem
    events = []
    for n, t in enumerate(candidates, 1):
        a = save_frame(video, t + 0.5, shots / f"c{n}_a.jpg")
        b = save_frame(video, t + 1.5, shots / f"c{n}_b.jpg")
        if not a or not b:
            continue
        info = ask_vision(ollama, model, CLASSIFY, [a, b])
        scene = info.get("scene", "other")
        print(f"  {fmt(t)}  {scene:<18} {str(info.get('description', ''))[:60]}")
        if scene != "kill_animation":
            continue
        if role != "impostor" and any(e["type"] == "death" for e in events):
            continue  # a crewmate only dies once; ignore look-alikes after that

        ctx = [save_frame(video, t - s, shots / f"c{n}_before{s}.jpg") for s in CONTEXT]
        doing = clean_doing(ask_vision(ollama, model, DOING, [p for p in ctx if p]).get("doing"))
        ev = {
            "type": "my_kill" if role == "impostor" else "death",
            "t": round(t, 1), "time": fmt(t),
            "killer_color": clean_color(info.get("killer_color")),
            "victim_color": clean_color(info.get("victim_color")),
            "doing": doing,
            "description": str(info.get("description", "")),
            "frames": [str(a), str(b)],
        }
        events.append(ev)
        who = ev["killer_color"] or "someone"
        print(f"     -> {'you killed someone' if ev['type'] == 'my_kill' else f'{who} killed you'}"
              f" while you were {doing}")

    MOMENTS.mkdir(parents=True, exist_ok=True)
    out = MOMENTS / f"{video.stem}.json"
    out.write_text(json.dumps({"video": video.name, "role": role, "model": model,
                               "events": events}, indent=2, ensure_ascii=False))
    print(f"\n{len(events)} gameplay moment(s) found. Saved: {out}")
    print(f"Screenshots it looked at: {shots}")


if __name__ == "__main__":
    main()
