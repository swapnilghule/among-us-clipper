"""
Step 1 of the Among Us clipper: scan a recording and find what happened.

What it does:
  - Looks at one frame every few seconds
  - Reads all on-screen text with Apple's built-in OCR
  - Detects: your role, meetings (start/end), ejection results,
    and the Victory/Defeat screen at the end (with the names shown on it)
  - Saves a summary + rough chat text per meeting to storage/scans/
  - Saves one sample image per meeting to storage/frames/ (to check the chat box)

Usage (from ~/amongus-clipper with the venv active):
  python scan.py                      # scans the newest video in storage/raw
  python scan.py path/to/video.mp4    # scans a specific video
"""

import json
import sys
from pathlib import Path

import cv2
from PIL import Image
from ocrmac import ocrmac

BASE = Path(__file__).resolve().parent
RAW = BASE / "storage" / "raw"
SCANS = BASE / "storage" / "scans"
FRAMES = BASE / "storage" / "frames"

STEP_SECONDS = 2  # how often to look at a frame

MEETING_WORDS = ["who is the impostor", "skip vote", "skipped voting", "voting ends", "proceeding in"]
EJECT_WORDS = ["was not the impostor", "was the impostor", "was not an impostor",
               "was an impostor", "no one was ejected"]
REPORT_WORDS = ["dead body reported", "emergency meeting"]


def newest_video():
    videos = sorted(
        [p for p in RAW.glob("*") if p.suffix.lower() in (".mp4", ".mov", ".mkv")],
        key=lambda p: p.stat().st_mtime,
    )
    if not videos:
        sys.exit(f"No videos found in {RAW}")
    return videos[-1]


def read_text(frame):
    """Return list of (text, confidence, box) for one frame."""
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return ocrmac.OCR(img, recognition_level="accurate").recognize()


def has_any(lines, words):
    joined = " | ".join(lines).lower()
    return any(w in joined for w in words)


def fmt(t):
    return f"{int(t // 60):02d}:{int(t % 60):02d}"


def scan(video):
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        sys.exit(f"Could not open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
    print(f"Scanning {video.name} ({fmt(duration)} long)...")

    role = None
    ejections = []
    reports = []
    meetings = []
    current = None  # meeting in progress
    end = None      # Victory / Defeat screen

    t = 0.0
    while t < duration:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if not ok:
            break

        results = read_text(frame)
        lines = [r[0].strip() for r in results if r[0].strip()]
        lower = [l.lower() for l in lines]

        # Role screen: a big line that is exactly "Crewmate" or "Impostor"
        if role is None:
            if "crewmate" in lower:
                role = {"role": "crewmate", "time": fmt(t)}
            elif "impostor" in lower:
                role = {"role": "impostor", "time": fmt(t)}

        # End screen: a line that is exactly "Victory" or "Defeat".
        # Keep every text seen on it (the winners' names show there).
        result = next((l for l in lower if l in ("victory", "defeat")), None)
        if result:
            if end is None:
                end = {"result": result, "time": fmt(t), "seconds": t, "texts": []}
                FRAMES.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(FRAMES / f"{video.stem}_end.png"), frame)
            for l in lines:
                if l.lower() not in ("victory", "defeat") and l not in end["texts"]:
                    end["texts"].append(l)

        if has_any(lines, REPORT_WORDS):
            if not reports or t - reports[-1]["seconds"] > 10:
                reports.append({"time": fmt(t), "seconds": t})

        for l in lines:
            if any(w in l.lower() for w in EJECT_WORDS):
                if not ejections or ejections[-1]["text"] != l:
                    ejections.append({"time": fmt(t), "seconds": t, "text": l})

        if has_any(lines, MEETING_WORDS):
            if current is None:
                current = {"start": t, "texts": [], "sample_saved": False}
            for l in lines:
                if l not in current["texts"]:
                    current["texts"].append(l)
            if not current["sample_saved"] and t - current["start"] >= 6:
                FRAMES.mkdir(parents=True, exist_ok=True)
                path = FRAMES / f"{video.stem}_meeting{len(meetings) + 1}.png"
                cv2.imwrite(str(path), frame)
                current["sample"] = str(path)
                current["sample_saved"] = True
            current["end"] = t
        elif current is not None:
            meetings.append(current)
            current = None

        print(f"\r  {fmt(t)} / {fmt(duration)}", end="", flush=True)
        t += STEP_SECONDS

    if current is not None:
        meetings.append(current)
    cap.release()
    print()

    summary = {
        "video": video.name,
        "duration": fmt(duration),
        "role": role,
        "reports": [r["time"] for r in reports],
        "meetings": [
            {
                "number": i + 1,
                "start": fmt(m["start"]),
                "end": fmt(m["end"]),
                "start_seconds": m["start"],
                "end_seconds": m["end"],
                "sample_image": m.get("sample"),
                "all_text_seen": m["texts"],
            }
            for i, m in enumerate(meetings)
        ],
        "ejections": [{"time": e["time"], "seconds": e["seconds"], "text": e["text"]}
                      for e in ejections],
        "end": end,
    }

    SCANS.mkdir(parents=True, exist_ok=True)
    out = SCANS / f"{video.stem}.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print(f"\nRole: {role['role'] if role else 'not found'}")
    print(f"Meetings found: {len(meetings)}")
    for m in summary["meetings"]:
        print(f"  #{m['number']}  {m['start']} - {m['end']}")
    print(f"Ejections: {len(ejections)}")
    for e in summary["ejections"]:
        print(f"  {e['time']}  {e['text']}")
    print(f"End screen: {end['result'] + ' at ' + end['time'] if end else 'not found'}")
    print(f"\nSaved: {out}")
    if meetings:
        print(f"Sample meeting images: {FRAMES}")


if __name__ == "__main__":
    video = Path(sys.argv[1]) if len(sys.argv) > 1 else newest_video()
    scan(video)
