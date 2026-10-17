"""
Step 4 of the Among Us clipper: cut the funny meetings into vertical clips.

Needs judge.py to have run first (uses storage/judged/<video>.json).

What it does, for every meeting the AI marked as a clip:
  - Cuts from just before the meeting to just after the ejection screen
  - Crops to the chat area of the game so the text is big enough on a phone
  - Makes it vertical 1080x1920 with a blurred background
  - Puts the AI's title at the top and the caption at the bottom
  - Saves the MP4 + a .txt with title, caption and hashtags to storage/clips/

Usage (from ~/amongus-clipper with the venv active):
  python cut.py                      # newest judged video
  python cut.py path/to/video.mp4    # a specific video
  python cut.py --all                # also cut meetings below the score threshold
"""

import json
import re
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import cv2
import numpy as np

BASE = Path(__file__).resolve().parent
RAW = BASE / "storage" / "raw"
CHATS = BASE / "storage" / "chats"
JUDGED = BASE / "storage" / "judged"
CLIPS = BASE / "storage" / "clips"
TMP = BASE / "storage" / "tmp"

BEFORE = 3           # seconds to keep before the meeting starts
AFTER_EJECT = 4      # seconds to keep after the ejection text appears
EJECT_WINDOW = 40    # how long after a meeting to look for the ejection
NO_EJECT_TAIL = 6    # if no ejection found, keep this long after the meeting

# Part of the game window to keep (fractions of its width). This covers the
# chat panel and the centre of the screen, so chat text stays readable.
CROP_X_START, CROP_X_WIDTH = 0.12, 0.65

FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/Library/Fonts/Arial Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
]

HASHTAGS = "#amongus #amongusfunny #amongusclips #gaming #shorts"


def to_seconds(mmss):
    m, s = mmss.split(":")
    return int(m) * 60 + int(s)


def find_ejection_time(meeting, ejections):
    for e in ejections:
        t = to_seconds(e["time"])
        if meeting["end_seconds"] <= t <= meeting["end_seconds"] + EJECT_WINDOW:
            return t
    return None


def game_window(video, t):
    """Find the game window inside the recording (skips black borders)."""
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cols = np.where((gray > 12).mean(axis=0) > 0.3)[0]
    rows = np.where((gray > 12).mean(axis=1) > 0.3)[0]
    if len(cols) == 0 or len(rows) == 0:
        h, w = gray.shape
        return 0, 0, w, h
    return int(cols[0]), int(rows[0]), int(cols[-1] - cols[0]), int(rows[-1] - rows[0])


def even(n):
    return int(n) // 2 * 2


def has_drawtext():
    out = subprocess.run(["ffmpeg", "-hide_banner", "-filters"],
                         capture_output=True, text=True).stdout
    return " drawtext " in out


def clean_text(s):
    # FFmpeg's text can't draw emoji, so drop them
    return re.sub(r"[^\u0000-￿]|[☀-➿️]", "", s).strip()


def text_filters(lines, font, size, y_start, line_gap, from_bottom=False):
    """One drawtext per line so each line is centred."""
    filters = []
    for i, line in enumerate(lines):
        f = TMP / f"line_{abs(hash((line, y_start, i)))}.txt"
        f.write_text(line)
        y = f"h-{y_start}+{i * line_gap}" if from_bottom else f"{y_start + i * line_gap}"
        filters.append(
            f"drawtext=fontfile='{font}':textfile='{f}':fontsize={size}:"
            f"fontcolor=white:borderw={max(3, size // 12)}:bordercolor=black:"
            f"x=(w-text_w)/2:y={y}"
        )
    return filters


def slug(s):
    s = re.sub(r"[^a-zA-Z0-9]+", "-", clean_text(s).lower()).strip("-")
    return s[:40] or "clip"


def cut_clip(video, start, end, crop, title, caption, out, font):
    cx, cy, cw, ch = crop
    chain = [
        f"[0:v]crop={cw}:{ch}:{cx}:{cy},split[a][b]",
        "[a]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,"
        "gblur=sigma=30,eq=brightness=-0.15[bg]",
        "[b]scale=1080:-2[fg]",
        "[bg][fg]overlay=0:(H-h)/2[v0]",
    ]
    if font:
        title_lines = textwrap.wrap(clean_text(title), 20)[:3]
        cap_lines = textwrap.wrap(clean_text(caption), 32)[:3]
        texts = (text_filters(title_lines, font, 74, 150, 88)
                 + text_filters(cap_lines, font, 46, 330, 60, from_bottom=True))
        chain.append("[v0]" + ",".join(texts) + "[v]" if texts else "[v0]null[v]")
    else:
        chain.append("[v0]null[v]")

    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-ss", f"{start:.2f}", "-t", f"{end - start:.2f}", "-i", str(video),
        "-filter_complex", ";".join(chain),
        "-map", "[v]", "-map", "0:a?",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-r", "30",
        "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart",
        str(out),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("   FFmpeg error:", result.stderr.strip()[-500:])
        return False
    return True


def pick_judged():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if args:
        f = JUDGED / f"{Path(args[0]).stem}.json"
        if not f.exists():
            sys.exit(f"No judged file for that video. Run: python judge.py \"{args[0]}\"")
        return f
    files = sorted(JUDGED.glob("*.json"), key=lambda p: p.stat().st_mtime)
    if not files:
        sys.exit("No judged files found. Run judge.py first.")
    return files[-1]


def main():
    if not shutil.which("ffmpeg"):
        sys.exit("FFmpeg not found. Install it with: brew install ffmpeg")
    cut_all = "--all" in sys.argv

    judged_file = pick_judged()
    judged = json.loads(judged_file.read_text())
    video = RAW / judged["video"]
    if not video.exists():
        sys.exit(f"Video not found: {video}")

    chat_file = CHATS / judged_file.name
    ejections = json.loads(chat_file.read_text()).get("ejections", []) if chat_file.exists() else []

    font = next((f for f in FONT_CANDIDATES if Path(f).exists()), None)
    if font and not has_drawtext():
        print("Note: your FFmpeg can't draw text, so clips will have no title/caption.")
        font = None

    CLIPS.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)

    todo = [m for m in judged["meetings"] if m.get("make_clip") or (cut_all and m["score"] > 0)]
    if not todo:
        print("No meetings marked as clips. Use --all to cut every meeting anyway.")
        return

    made = 0
    for m in sorted(todo, key=lambda x: x["score"], reverse=True):
        eject_t = find_ejection_time(m, ejections)
        start = max(0, m["start_seconds"] - BEFORE)
        end = (eject_t + AFTER_EJECT) if eject_t else (m["end_seconds"] + NO_EJECT_TAIL)

        win = game_window(video, m["start_seconds"] + 5)
        if not win:
            print(f"Meeting {m['number']}: couldn't read the video, skipped")
            continue
        wx, wy, ww, wh = win
        crop = (even(wx + CROP_X_START * ww), even(wy),
                even(CROP_X_WIDTH * ww), even(wh))

        stem = f"{video.stem}_m{m['number']}_{slug(m.get('title', ''))}"
        out = CLIPS / f"{stem}.mp4"
        print(f"Meeting {m['number']} (score {m['score']}): cutting "
              f"{end - start:.0f}s clip...")

        if cut_clip(video, start, end, crop, m.get("title", ""),
                    m.get("caption", ""), out, font):
            info = (f"Title: {m.get('title', '')}\n"
                    f"Caption: {m.get('caption', '')}\n\n"
                    f"{m.get('caption', '')}\n{HASHTAGS}\n\n"
                    f"Score: {m['score']}/10\n"
                    f"Why: {m.get('reason', '')}\n"
                    f"Moment: {m.get('funny_moment', '')}\n"
                    f"Result: {m.get('ejection') or 'unknown'}\n")
            out.with_suffix(".txt").write_text(info)
            print(f"   Saved: {out.name}")
            made += 1

    shutil.rmtree(TMP, ignore_errors=True)
    print(f"\n{made} clip(s) saved to {CLIPS}")
    print("Watch them, post the good ones, then move posted files to storage/posted.")


if __name__ == "__main__":
    main()
