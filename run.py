"""
One command for the whole Among Us editor.

Runs every step in order on your newest recording in storage/raw:
  scan -> moments -> chat -> story -> planner -> render

Usage (from ~/amongus-clipper with the venv active):
  python run.py                          # newest recording
  python run.py "storage/raw/my game.mp4"  # a specific recording
  python run.py --from story             # start from a later step (reuses earlier results)
  python run.py --no-ai                  # captions from templates only (faster)

Steps:
  scan     find meetings, role, ejections, end screen      (slow: a few minutes)
  moments  find kills between meetings (needs the vision AI; skipped if missing)
  chat     read every chat bubble, color and position      (slow: a few minutes)
  story    who said what, who was the impostor, colors
  planner  find the funny moments and write the captions
  render   build the finished videos in storage/edits/
"""

import subprocess
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
RAW = BASE / "storage" / "raw"
LIBRARY = BASE / "storage" / "memes" / "library.json"

STEPS = ["scan", "moments", "chat", "story", "planner", "render"]
OPTIONAL = {"moments"}  # if this one fails, keep going


def newest_video():
    videos = sorted([p for p in RAW.glob("*") if p.suffix.lower() in (".mp4", ".mov", ".mkv")],
                    key=lambda p: p.stat().st_mtime)
    if not videos:
        sys.exit(f"No recordings found in {RAW}")
    return videos[-1]


def main():
    args = sys.argv[1:]
    start = "scan"
    if "--from" in args:
        i = args.index("--from")
        if i + 1 >= len(args) or args[i + 1] not in STEPS:
            sys.exit(f"Use: python run.py --from <step>   steps: {', '.join(STEPS)}")
        start = args[i + 1]
        del args[i:i + 2]
    no_ai = "--no-ai" in args
    paths = [a for a in args if not a.startswith("--")]
    video = Path(paths[0]).resolve() if paths else newest_video()
    if not video.exists():
        sys.exit(f"Recording not found: {video}")

    print(f"=== Among Us editor: {video.name} ===")
    if not LIBRARY.exists():
        print("\n--- memes (first time setup) ---")
        subprocess.run([sys.executable, str(BASE / "memes.py")], check=True)

    total = time.time()
    for step in STEPS[STEPS.index(start):]:
        print(f"\n--- {step} ---", flush=True)
        cmd = [sys.executable, str(BASE / f"{step}.py"), str(video)]
        if step == "planner" and no_ai:
            cmd.append("--no-ai")
        t = time.time()
        result = subprocess.run(cmd)
        if result.returncode != 0:
            if step in OPTIONAL:
                print(f"(skipping {step}; the rest still works without it)")
                continue
            sys.exit(f"\nStopped at '{step}'. Fix the problem above, then continue with:\n"
                     f"  python run.py --from {step}")
        print(f"({step} took {time.time() - t:.0f}s)")

    print(f"\n=== Done in {(time.time() - total) / 60:.1f} min. "
          f"Your edits are in storage/edits/ ===")


if __name__ == "__main__":
    main()
