"""
Step 3 of the Among Us clipper: ask the local AI which meetings are funny.

Needs chat.py to have run first (uses storage/chats/<video>.json).
Needs Ollama running (`ollama serve`) with the model pulled (`ollama pull llama3.1:8b`).

What it does:
  - Builds a short story for each meeting: your role, the chat, the votes,
    who left, and the ejection result right after
  - Asks the AI to score how funny it is (1-10), explain why, and suggest a
    clip title and caption
  - Saves to storage/judged/<video>.json and prints meetings from funniest down

Usage (from ~/amongus-clipper with the venv active):
  python judge.py                      # newest chat file
  python judge.py path/to/video.mp4    # a specific video
"""

import json
import sys
from pathlib import Path

import ollama

BASE = Path(__file__).resolve().parent
CHATS = BASE / "storage" / "chats"
JUDGED = BASE / "storage" / "judged"

# The first of these that you have installed is used (check with: ollama list)
PREFERRED_MODELS = ["qwen2.5:14b", "qwen2.5:7b", "llama3.1:8b", "qwen2.5-coder:7b"]
MODEL = None  # picked automatically in main()
MIN_MESSAGES = 3       # meetings with less chat than this are skipped
CLIP_THRESHOLD = 7     # score needed to become a clip (used by the next step)
EJECT_WINDOW = 40      # seconds after a meeting to look for its ejection result

SYSTEM_PROMPT = """You are the editor of a funny Among Us clips account on Instagram Reels and YouTube Shorts.
You read the text chat from one emergency meeting and decide if it would make a funny short clip.

Things that ARE funny:
- Several players claiming the same thing at once (e.g. everyone says "I'm engineer")
- Total confusion: lots of "where?", "what?", "who?", nobody knows anything
- Terrible or obviously fake excuses, or the impostor lying with full confidence
- Random, off-topic or weird messages in the middle of a serious accusation
- Players accusing each other in circles, chaos, panic
- Voting out an innocent player, especially after a confident accusation
- Trolling, self-reports, someone leaving the game mid-meeting
- A short punchline message that flips the mood

Things that are NOT funny:
- A calm, normal discussion where people share locations and vote sensibly
- Very little chat
- Only vote notifications

Be strict. Most meetings are a 3-5. Only truly clip-worthy chaos gets 7+.

Reply ONLY with JSON in exactly this shape:
{
  "score": <integer 1-10>,
  "funny_moment": "<the specific messages that make it funny, or empty>",
  "reason": "<one sentence on why it is or isn't funny>",
  "title": "<clip title, max 8 words, can use 1 emoji>",
  "caption": "<one-line caption for the post>"
}"""


def to_seconds(mmss):
    m, s = mmss.split(":")
    return int(m) * 60 + int(s)


def ejection_after(meeting, ejections):
    for e in ejections:
        t = to_seconds(e["time"])
        if meeting["end_seconds"] <= t <= meeting["end_seconds"] + EJECT_WINDOW:
            return e["text"]
    return None


def build_story(meeting, role, ejection):
    lines = []
    if role:
        lines.append(f"My role this game: {role['role']}")
    if meeting.get("me"):
        lines.append(f"My player name: {meeting['me']}")
    lines.append("")
    lines.append("Meeting chat (in order):")
    for m in meeting["messages"]:
        who = f"{m['name']} (me)" if m["is_me"] else m["name"]
        lines.append(f"{who}: {m['text']}")
    if meeting["votes"]:
        lines.append("")
        lines.append(f"Votes cast: {len(meeting['votes'])}")
    for lg in meeting["left_game"]:
        lines.append(f"Event: {lg['text'] if isinstance(lg, dict) else lg}")
    lines.append("")
    lines.append(f"Result after voting: {ejection or 'unknown'}")
    return "\n".join(lines)


def ask_ai(story):
    resp = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": story},
        ],
        format="json",
        options={"temperature": 0.3},
    )
    try:
        data = json.loads(resp["message"]["content"])
    except (json.JSONDecodeError, KeyError, TypeError):
        return {"score": 0, "funny_moment": "", "reason": "AI reply was not valid JSON",
                "title": "", "caption": ""}
    try:
        data["score"] = max(1, min(10, int(data.get("score", 0))))
    except (TypeError, ValueError):
        data["score"] = 0
    for k in ("funny_moment", "reason", "title", "caption"):
        data[k] = str(data.get(k, "") or "")
    return data


def pick_chat_file():
    if len(sys.argv) > 1:
        f = CHATS / f"{Path(sys.argv[1]).stem}.json"
        if not f.exists():
            sys.exit(f"No chat file for that video. Run: python chat.py \"{sys.argv[1]}\"")
        return f
    files = sorted(CHATS.glob("*.json"), key=lambda p: p.stat().st_mtime)
    if not files:
        sys.exit("No chat files found. Run chat.py first.")
    return files[-1]


def main():
    chat_file = pick_chat_file()
    data = json.loads(chat_file.read_text())
    role = data.get("role")
    ejections = data.get("ejections", [])

    global MODEL
    try:
        listing = ollama.list()
    except Exception:
        sys.exit("Can't reach Ollama. Start it in another Terminal window with: ollama serve")
    models = listing.get("models", []) if isinstance(listing, dict) else getattr(listing, "models", [])
    installed = {(m.get("model") or m.get("name")) if isinstance(m, dict)
                 else (getattr(m, "model", None) or getattr(m, "name", None)) for m in models}
    MODEL = next((m for m in PREFERRED_MODELS if m in installed), None)
    if not MODEL:
        sys.exit("No suitable model found. Install one with: ollama pull qwen2.5:7b")
    print(f"Using AI model: {MODEL}")

    results = []
    for m in data["meetings"]:
        ejection = ejection_after(m, ejections)
        base = {
            "number": m["number"],
            "start": m["start"],
            "end": m["end"],
            "start_seconds": m["start_seconds"],
            "end_seconds": m["end_seconds"],
            "ejection": ejection,
            "message_count": len(m["messages"]),
        }
        if len(m["messages"]) < MIN_MESSAGES:
            print(f"Meeting {m['number']}: skipped (only {len(m['messages'])} messages)")
            results.append({**base, "score": 0, "funny_moment": "",
                            "reason": "Too little chat", "title": "", "caption": ""})
            continue

        print(f"Meeting {m['number']}: asking the AI...")
        verdict = ask_ai(build_story(m, role, ejection))
        results.append({**base, **verdict})

    for r in results:
        r["make_clip"] = r["score"] >= CLIP_THRESHOLD

    JUDGED.mkdir(parents=True, exist_ok=True)
    out_file = JUDGED / chat_file.name
    out_file.write_text(json.dumps(
        {"video": data["video"], "role": role, "meetings": results},
        indent=2, ensure_ascii=False))

    print()
    for r in sorted(results, key=lambda x: x["score"], reverse=True):
        mark = "CLIP" if r["make_clip"] else "skip"
        print(f"[{mark}] Meeting {r['number']} ({r['start']} - {r['end']})  score {r['score']}/10")
        if r["title"]:
            print(f"   Title:   {r['title']}")
        if r["funny_moment"]:
            print(f"   Moment:  {r['funny_moment']}")
        print(f"   Why:     {r['reason']}")
        if r["ejection"]:
            print(f"   Result:  {r['ejection']}")
        print()

    clips = sum(r["make_clip"] for r in results)
    print(f"{clips} of {len(results)} meetings scored {CLIP_THRESHOLD}+ and will become clips.")
    print(f"Saved: {out_file}")


if __name__ == "__main__":
    main()
