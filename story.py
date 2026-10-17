"""
Story step: combine everything into one clean "game story" with hindsight.

Needs scan.py and chat.py to have run first.

What it does:
  - Cleans player names (OCR typos like "Quizily" -> "Quizfly")
  - Works out who you are and your role
  - Works out who was really an impostor, from ejections
    ("X was An Impostor.") and the Victory/Defeat screen
  - Builds one timeline: every message with its exact time, every vote,
    every ejection, each speaker marked IMPOSTOR / crew / unknown
  - Saves storage/stories/<video>.json (the meme editor builds on this)
    and prints the timeline so you can check it

Usage (from ~/amongus-clipper with the venv active):
  python story.py                      # newest video
  python story.py path/to/video.mp4    # a specific video
"""

import difflib
import json
import re
import sys
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parent
SCANS = BASE / "storage" / "scans"
CHATS = BASE / "storage" / "chats"
STORIES = BASE / "storage" / "stories"
COLOR_FIXES = BASE / "storage" / "colors.json"
MOMENTS = BASE / "storage" / "moments"

EJECT_WINDOW = 40  # seconds after a meeting to look for its ejection
EJECT_RX = re.compile(r"^(.+?)\s+was\s+(not\s+)?(an|the)\s+impostor", re.I)
VOTE_RX = re.compile(r"^(.+?)\s+has\s+voted", re.I)
NAME_MATCH = 0.75  # how similar two spellings must be to count as the same name
END_AFTER_EJECT = 30  # end screen this soon after an impostor ejection = voted win
REMAINING_RX = re.compile(r"(\d+)\s+remaining", re.I)


def fmt(t):
    return f"{int(t // 60):02d}:{int(t % 60):02d}"


def to_seconds(mmss):
    m, s = mmss.split(":")
    return int(m) * 60 + int(s)


def key(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def similar(a, b):
    ka, kb = key(a), key(b)
    return bool(ka) and bool(kb) and (
        ka == kb or difflib.SequenceMatcher(None, ka, kb).ratio() >= NAME_MATCH)


# ---------- names ----------

class Roster:
    """Groups OCR spellings of the same player; the most common spelling wins."""

    def __init__(self):
        self.groups = []  # list of Counter(spelling -> count)

    def add(self, name, weight=1):
        name = name.strip()
        if not key(name):
            return
        for g in self.groups:
            if any(similar(name, s) for s in g):
                g[name] += weight
                return
        self.groups.append(Counter({name: weight}))

    def canon(self, name):
        for g in self.groups:
            if any(similar(name, s) for s in g):
                return g.most_common(1)[0][0]
        return name.strip()

    def find_in(self, text):
        """Return a roster name that appears in a line of text, if any."""
        for g in self.groups:
            best = g.most_common(1)[0][0]
            if similar(text, best):
                return best
        return None

    def names(self):
        return sorted(g.most_common(1)[0][0] for g in self.groups)


# ---------- player colors ----------

def assign_colors(meetings, roster):
    """Majority color per player from their chat avatars. Every color is used
    once (two players can't share a color in Among Us). Your fixes in
    storage/colors.json always win."""
    votes = {}
    for mt in meetings:
        for m in mt["messages"]:
            if m.get("color"):
                n = roster.canon(m["name"])
                votes.setdefault(n, Counter())[m["color"]] += 1

    fixes = {}
    if COLOR_FIXES.exists():
        try:
            fixes = {roster.canon(k): v.lower() for k, v in
                     json.loads(COLOR_FIXES.read_text()).items()}
        except (json.JSONDecodeError, AttributeError):
            print(f"Note: couldn't read {COLOR_FIXES.name}, ignoring it")

    result = dict(fixes)
    taken = set(fixes.values())
    options = sorted(((cnt, n, c) for n, ctr in votes.items() if n not in fixes
                      for c, cnt in ctr.items()), reverse=True)
    for cnt, n, c in options:
        if n not in result and c not in taken:
            result[n] = c
            taken.add(c)
    return result


# ---------- gameplay moments (from moments.py) ----------

def add_gameplay(stem, color_of, ejections, truth, label):
    f = MOMENTS / f"{stem}.json"
    if not f.exists():
        return []
    by_color = {c: n for n, c in color_of.items()}
    out = []
    for e in json.loads(f.read_text()).get("events", []):
        killer = by_color.get(e.get("killer_color"))
        victim = by_color.get(e.get("victim_color"))
        caught = next((ej for ej in ejections if killer and ej["name"] == killer
                       and ej["t"] > e["t"]), None)
        out.append({
            **{k: v for k, v in e.items() if k != "frames"},
            "killer_label": label(killer) if killer else (
                e["killer_color"].title() if e.get("killer_color") else "Someone"),
            "victim_label": label(victim) if victim else (
                e["victim_color"].title() if e.get("victim_color") else "someone"),
            "killer_truth": truth(killer) if killer else "impostor",
            "killer_caught": fmt(caught["t"]) if caught else None,
            "killer_caught_t": caught["t"] if caught else None,
        })
    return out


# ---------- leftover chat from earlier meetings ----------

def fix_leftover_chat(meetings, ejections, roster, me):
    """Among Us keeps old chat when a new meeting opens. Two safety rules:
    1. A player ejected before this meeting can't talk in it, so their
       message (and everything shown above it) is leftover from the meeting
       before -> moved back there.
    2. "N remaining" can only go down within one meeting. If it jumps up,
       the votes before the jump are leftovers -> dropped."""
    for i, mt in enumerate(meetings):
        if i == 0:
            continue
        dead = {roster.canon(e["name"]) for e in ejections
                if e["name"] and e["t"] < mt["start_seconds"]}
        dead.discard(me)
        msgs = mt["messages"]
        last_dead = max((j for j, m in enumerate(msgs)
                         if roster.canon(m["name"]) in dead), default=None)
        if last_dead is not None:
            moved = msgs[:last_dead + 1]
            mt["messages"] = msgs[last_dead + 1:]
            prev = meetings[i - 1]["messages"]
            for m in moved:
                if not any(key(m["name"] + m["text"]) == key(p["name"] + p["text"])
                           for p in prev):
                    prev.append({**m, "late_seen": True})

    for mt in meetings:
        votes = [v if isinstance(v, dict) else {"text": v, "t": None} for v in mt["votes"]]
        keep_from, last = 0, None
        for j, v in enumerate(votes):
            r = REMAINING_RX.search(v["text"])
            if not r:
                continue
            n = int(r.group(1))
            if last is not None and n > last:
                keep_from = j
            last = n
        mt["votes"] = votes[keep_from:]
        for lg in mt["left_game"]:
            if isinstance(lg, dict):
                lg["text"] = lg["text"].lstrip("-–— ").strip()


# ---------- main ----------

def pick_stem():
    if len(sys.argv) > 1:
        return Path(sys.argv[1]).stem
    files = sorted(CHATS.glob("*.json"), key=lambda p: p.stat().st_mtime)
    if not files:
        sys.exit("No chat files found. Run scan.py and chat.py first.")
    return files[-1].stem


def main():
    stem = pick_stem()
    scan_f, chat_f = SCANS / f"{stem}.json", CHATS / f"{stem}.json"
    for f, step in ((scan_f, "scan.py"), (chat_f, "chat.py")):
        if not f.exists():
            sys.exit(f"Missing {f.name}. Run {step} first.")
    scan = json.loads(scan_f.read_text())
    chat = json.loads(chat_f.read_text())

    if any("t" not in m for mt in chat["meetings"] for m in mt["messages"]):
        sys.exit("This chat file has no message times. Re-run the new chat.py first.")

    # 1. Collect every spelling of every name
    roster = Roster()
    me_count = Counter()
    for mt in chat["meetings"]:
        for m in mt["messages"]:
            roster.add(m["name"], 2)  # chat names are the most reliable
            if m["is_me"]:
                me_count[m["name"]] += 1
        for v in mt["votes"]:
            mv = VOTE_RX.match(v["text"] if isinstance(v, dict) else v)
            if mv:
                roster.add(mv.group(1))

    ejections = []
    for e in scan.get("ejections", []):
        t = e.get("seconds", to_seconds(e["time"]))
        me = EJECT_RX.match(e["text"])
        if me:
            roster.add(me.group(1))
            ejections.append({"t": t, "text": e["text"], "name": me.group(1),
                              "was_impostor": me.group(2) is None})
        elif "no one was ejected" in e["text"].lower():
            ejections.append({"t": t, "text": e["text"], "name": None,
                              "was_impostor": None})
    for e in ejections:
        if e["name"]:
            e["name"] = roster.canon(e["name"])

    me = roster.canon(me_count.most_common(1)[0][0]) if me_count else None
    role = (scan.get("role") or {}).get("role")

    fix_leftover_chat(chat["meetings"], ejections, roster, me)

    # 2. Work out the truth: who was an impostor
    impostors, crew = set(), set()
    for e in ejections:
        if e["name"]:
            (impostors if e["was_impostor"] else crew).add(e["name"])
    if me and role:
        (impostors if role == "impostor" else crew).add(me)

    end = scan.get("end")
    winners = []
    if end:
        for text in end["texts"]:
            n = roster.find_in(text)
            if n and n not in winners:
                winners.append(n)
        my_team_won = end["result"] == "victory"
        if role:
            winners_are_impostors = (role == "impostor") == my_team_won
            (impostors if winners_are_impostors else crew).update(winners)

        # Crew won right after an impostor was voted out = all impostors are
        # gone, so everyone not already known as impostor was crew.
        last_ej = max((e for e in ejections if e["t"] <= end["seconds"]),
                      key=lambda e: e["t"], default=None)
        crew_won = (role == "crewmate") == my_team_won if role else None
        if (crew_won and last_ej and last_ej["was_impostor"]
                and end["seconds"] - last_ej["t"] <= END_AFTER_EJECT):
            crew.update(n for n in roster.names() if n not in impostors)

    def truth(name):
        if name in impostors:
            return "impostor"
        if name in crew:
            return "crew"
        return "unknown"

    # 2b. One color per player: majority vote over all their chat bubbles,
    # each color used once, then your fixes from storage/colors.json
    color_of = assign_colors(chat["meetings"], roster)

    def label(name, is_me=False):
        if is_me or (me and name == me):
            return "Me"
        c = color_of.get(name)
        return c.title() if c else "Someone"

    # 3. Build the timeline for each meeting
    meetings = []
    for mt in chat["meetings"]:
        msgs = []
        for m in mt["messages"]:
            name = roster.canon(m["name"])
            msgs.append({"t": m["t"], "time": m["time"], "name": name,
                         "text": m["text"], "is_me": m["is_me"], "truth": truth(name),
                         "color": color_of.get(name), "label": label(name, m["is_me"]),
                         "geo": m.get("geo"), "late_seen": m.get("late_seen", False)})
        votes = []
        for v in mt["votes"]:
            text = v["text"] if isinstance(v, dict) else v
            mv = VOTE_RX.match(text)
            votes.append({"t": v.get("t") if isinstance(v, dict) else None,
                          "voter": roster.canon(mv.group(1)) if mv else None,
                          "text": text})
        left = [{"t": l.get("t") if isinstance(l, dict) else None,
                 "text": l["text"] if isinstance(l, dict) else l}
                for l in mt["left_game"]]
        ejection = next((e for e in ejections
                         if mt["end_seconds"] <= e["t"] <= mt["end_seconds"] + EJECT_WINDOW), None)
        meetings.append({
            "number": mt["number"],
            "start_seconds": mt["start_seconds"],
            "end_seconds": mt["end_seconds"],
            "start": fmt(mt["start_seconds"]),
            "end": fmt(mt["end_seconds"]),
            "messages": msgs,
            "votes": votes,
            "left_game": left,
            "ejection": ejection and {**ejection, "time": fmt(ejection["t"]),
                                      "truth": truth(ejection["name"]) if ejection["name"] else None,
                                      "color": color_of.get(ejection["name"]),
                                      "label": label(ejection["name"]) if ejection["name"] else None},
        })

    story = {
        "video": chat["video"],
        "me": me,
        "role": role,
        "players": [{"name": n, "truth": truth(n), "is_me": n == me,
                     "color": color_of.get(n), "label": label(n)} for n in roster.names()],
        "impostors": sorted(impostors),
        "end": end and {"result": end["result"], "t": end["seconds"],
                        "time": end["time"], "winners": winners},
        "meetings": meetings,
    }

    # Gameplay moments from moments.py (optional): who killed whom, with hindsight
    story["gameplay"] = add_gameplay(stem, color_of, ejections, truth, label)

    STORIES.mkdir(parents=True, exist_ok=True)
    out = STORIES / f"{stem}.json"
    out.write_text(json.dumps(story, indent=2, ensure_ascii=False))

    # 4. Print the story
    tag = {"impostor": " [IMPOSTOR]", "crew": "", "unknown": " [?]"}
    print(f"GAME STORY  {chat['video']}")
    print(f"You: {me or 'unknown'}  |  Role: {role or 'unknown'}  |  "
          f"Result: {end['result'] if end else 'end screen not found'}")
    print(f"Impostors: {', '.join(label(n) for n in story['impostors']) or 'unknown'}")
    print("Players (name -> color):")
    for n in roster.names():
        c = color_of.get(n)
        print(f"  {n:<14} -> {c or '?? (not detected)'}{'  (you)' if n == me else ''}")
    print(f"  Wrong color? Fix it in {COLOR_FIXES.relative_to(BASE)}, e.g. "
          '{"TAPPU XD": "red"}')
    for mt in meetings:
        print(f"\n===== Meeting {mt['number']} ({mt['start']} - {mt['end']}) =====")
        events = ([(m["t"], f"{m['label']}"
                    f"{tag[m['truth']]}: {m['text']}"
                    f"{'   (moved back from next meeting)' if m['late_seen'] else ''}")
                   for m in mt["messages"]]
                  + [(v["t"], f"[vote] {v['text']}") for v in mt["votes"] if v["t"] is not None]
                  + [(l["t"], f"[left] {l['text']}") for l in mt["left_game"] if l["t"] is not None])
        for t, line in sorted(events, key=lambda x: x[0]):
            print(f"  {fmt(t)}  {line}")
        if mt["ejection"]:
            ej = mt["ejection"]
            who = f"  ({ej['label']})" if ej.get("label") else ""
            print(f"  {ej['time']}  >>> {ej['text']}{who}")
    if story["gameplay"]:
        print("\n===== Gameplay moments =====")
        for g in story["gameplay"]:
            if g["type"] == "death":
                caught = (f" -> {g['killer_label']} got ejected at {g['killer_caught']}"
                          if g.get("killer_caught") else "")
                print(f"  {g['time']}  {g['killer_label']} killed you while you were "
                      f"{g['doing']}{caught}")
            else:
                print(f"  {g['time']}  You killed {g['victim_label']} ({g['doing']})")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
