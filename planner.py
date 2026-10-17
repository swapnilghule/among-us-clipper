"""
Step 3 of the meme editor: plan the funny edit for each meeting.

Needs story.py and memes.py to have run first.
Needs Ollama with a model (best: `ollama pull qwen2.5:14b`).

How it works (two jobs, split on purpose):
  1. CODE finds the funny moments, using hindsight it knows for sure:
       - an impostor denying / giving an alibi          -> "lying" meme
       - a chain of "me too" / "i am also eng" claims   -> "copycat" meme
       - the message right after that chain              -> punchline "dead" meme
       - a burst of "what?" "where?" "who?"              -> "confused" meme
       - a random off-topic message                      -> "awkward" meme
       - someone naming the real impostor                -> "detective" meme
       - the ejection result                             -> ending meme
  2. The AI only WRITES the captions for those moments (top + bottom text),
     plus the title and post caption. If its caption is weak or just repeats
     the chat, a built-in template is used instead.

The hook at the start always comes from the real ejection result.

Saves storage/plans/<video>.json and prints each plan to review.

Usage (from ~/amongus-clipper with the venv active):
  python planner.py                      # newest story
  python planner.py path/to/video.mp4    # a specific video
  python planner.py --all                # plan every meeting, even unfunny ones
  python planner.py --no-ai              # templates only (fast, no Ollama needed)
"""

import json
import re
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
STORIES = BASE / "storage" / "stories"
PLANS = BASE / "storage" / "plans"
LIBRARY = BASE / "storage" / "memes" / "library.json"

PREFERRED_MODELS = ["qwen2.5:14b", "qwen2.5:7b", "llama3.1:8b", "qwen2.5-coder:7b"]
MAX_MEMES = 3          # meme cut-aways per clip (plus the ending meme)
MIN_SPACING = 2.5      # seconds between two moments (except a punchline)
MIN_STRENGTH = 4       # total moment strength needed to make an edit
READ_DELAY = 1.0       # seconds after a message appears before the meme
MEME_SECONDS = 1.6     # how long a meme card stays on screen
ZOOM_SECONDS = 1.0     # zoom on the message just before its meme
BEFORE = 3             # clip starts this long before the first event
AFTER_EJECT = 4        # clip ends this long after the ejection text
NO_EJECT_TAIL = 6

# ---------- word lists for spotting moments ----------

QUESTION_RX = re.compile(r"^\s*(what+|wat|wht|where+|wher|who+|why+|wh|huh+|how|\?+)\s*\?*\s*$", re.I)
COPY_RX = re.compile(r"\b(too|also|same|me2|\+1)\b", re.I)
DENIAL_RX = re.compile(r"\b(not me|wasn'?t me|wasnt me|its not|i was|i saw|trust|"
                       r"someone|somebody|clear|safe|i did|doing|my task|im \w+)\b", re.I)
GAME_WORDS = {
    "vent", "vented", "sus", "imposter", "impostor", "body", "kill", "killed", "dead",
    "saw", "task", "tasks", "eng", "engineer", "enginee", "electrical", "elec", "admin",
    "medbay", "med", "nav", "o2", "reactor", "lights", "comms", "cams", "security", "sec",
    "vote", "voted", "skip", "report", "reported", "where", "who", "self", "follow",
    "following", "clear", "scan", "card", "wires", "shields", "weapons", "storage",
    "cafeteria", "cafe", "lab", "specimen", "red", "blue", "green", "yellow", "black",
    "white", "pink", "purple", "orange", "cyan", "lime", "brown", "viper", "doctor",
    "scientist", "guardian", "shapeshifter", "tracker", "noisemaker", "phantom",
}
STOP = {"i", "im", "i'm", "a", "an", "the", "is", "am", "are", "was", "to", "me", "you",
        "u", "he", "she", "it", "its", "it's", "and", "or", "of", "in", "on", "at", "bro"}

STRENGTH = {"lying": 3, "copycat": 3, "punchline": 3, "awkward": 2, "detective": 2,
            "confused": 1}
MOOD_FOR = {"lying": "lying", "copycat": "copycat", "punchline": "dead",
            "awkward": "awkward", "detective": "detective", "confused": "confused"}


def fmt(t):
    return f"{int(t // 60):02d}:{int(t % 60):02d}"


def toks(s):
    return re.findall(r"[a-z0-9+]+", str(s or "").lower().replace("'", "").replace("’", ""))


def short_quote(text, n=8):
    w = str(text).split()
    return " ".join(w[:n]) + ("..." if len(w) > n else "")


# ---------- events ----------

def meeting_events(mt):
    """Numbered events. Messages only seen later (leftovers moved back) have no
    reliable on-screen time, so they're left out."""
    events = []
    for m in mt["messages"]:
        if m.get("late_seen"):
            continue
        events.append({"t": m["t"], "kind": "message", "name": m["name"],
                       "label": m.get("label") or m["name"], "color": m.get("color"),
                       "geo": m.get("geo"), "is_me": m["is_me"], "truth": m["truth"],
                       "text": m["text"]})
    for v in mt["votes"]:
        if v.get("t") is not None:
            events.append({"t": v["t"], "kind": "vote", "text": v["text"]})
    for l in mt["left_game"]:
        if l.get("t") is not None:
            events.append({"t": l["t"], "kind": "left", "text": l["text"]})
    events.sort(key=lambda e: e["t"])
    for i, e in enumerate(events, 1):
        e["n"] = i
    return events


# ---------- 1. code finds the moments ----------

def find_moments(events, impostors, labels):
    msgs = [e for e in events if e["kind"] == "message"]
    found = []

    def add(kind, ev, **extra):
        if any(m["event"] == ev["n"] for m in found):
            return  # one moment per message
        found.append({"kind": kind, "event": ev["n"], "t": ev["t"], "name": ev.get("name"),
                      "who": ev.get("label") or ev.get("name"),
                      "text": ev["text"], "mood": MOOD_FOR[kind],
                      "strength": STRENGTH[kind], **extra})

    # Copycat chain: a claim followed by 2+ "me too" / "also" / same-word replies
    i = 0
    while i < len(msgs):
        first = msgs[i]
        key = {w for w in toks(first["text"]) if w not in STOP and len(w) >= 3}
        chain = [first]
        j = i + 1
        while j < len(msgs):
            m = msgs[j]
            words = set(toks(m["text"]))
            if COPY_RX.search(m["text"]) or (key and words & key) or \
                    any(w.startswith(k[:3]) for w in words for k in key if len(k) >= 3):
                chain.append(m)
                j += 1
            else:
                break
        if len(chain) >= 3:
            add("copycat", chain[-1], claim=first["text"], chain_from=first["n"],
                count=len(chain))
            if j < len(msgs) and msgs[j]["t"] - chain[-1]["t"] <= 8:
                add("punchline", msgs[j], after_chain=True)
            i = j
        else:
            i += 1

    # Impostor denial / alibi (we know they're lying), first one per impostor
    seen_imp = set()
    for m in msgs:
        if m["truth"] == "impostor" and m["name"] not in seen_imp and len(toks(m["text"])) >= 2:
            seen_imp.add(m["name"])
            add("lying", m, denial=bool(DENIAL_RX.search(m["text"])))
            if not DENIAL_RX.search(m["text"]):
                found[-1]["strength"] = 2

    # Someone names a real impostor
    for m in msgs:
        if m["truth"] == "impostor":
            continue
        words = set(toks(m["text"]))
        for imp in impostors:
            itoks = [w for w in toks(imp) if len(w) >= 3]
            if itoks and all(w in words or any(x.startswith(w[:4]) for x in words) for w in itoks):
                add("detective", m, accused=imp, accused_who=labels.get(imp, "the impostor"))
                break

    # Random off-topic message in a murder meeting
    for m in msgs:
        words = toks(m["text"])
        if len(words) >= 5 and not set(words) & GAME_WORDS:
            add("awkward", m)

    # A burst of "what?" "where?" "who?"
    qs = [m for m in msgs if QUESTION_RX.match(m["text"])]
    for k in range(1, len(qs)):
        if qs[k]["t"] - qs[k - 1]["t"] <= 8 and sum(
                1 for q in qs if 0 <= qs[k]["t"] - q["t"] <= 8) >= 3:
            add("confused", qs[k], burst=[q["text"] for q in qs
                                          if 0 <= qs[k]["t"] - q["t"] <= 8])
            break

    # Strongest first; skip moments too close to a stronger one (a punchline
    # right after its copycat chain is the exception: that's the joke)
    found.sort(key=lambda m: (-m["strength"], m["t"]))
    chosen = []
    for m in found:
        if len(chosen) >= MAX_MEMES:
            break
        if not m.get("after_chain") and any(abs(m["t"] - c["t"]) < MIN_SPACING for c in chosen):
            continue
        chosen.append(m)
    chosen.sort(key=lambda m: m["t"])
    for label, m in zip("ABCDEFG", chosen):
        m["label"] = label
    return chosen


def ending_moment(mt, impostors, labels):
    ej = mt.get("ejection")
    if ej and ej.get("name"):
        if ej.get("truth") == "impostor":
            return {"kind": "caught", "mood": "victory", "name": ej["name"],
                    "who": labels.get(ej["name"], "Someone"), "strength": 2}
        if ej.get("truth") == "crew":
            return {"kind": "innocent_out", "mood": "clown", "name": ej["name"],
                    "who": labels.get(ej["name"], "Someone"), "strength": 4}
    imp_here = sorted({m["name"] for m in mt["messages"] if m["truth"] == "impostor"})
    if ej and ej.get("name") is None and imp_here:
        return {"kind": "skipped_with_impostor", "mood": "facepalm",
                "name": imp_here[0], "who": labels.get(imp_here[0], "Someone"), "strength": 2}
    return None


# ---------- templates (used when the AI's caption is weak) ----------

def template(m):
    q = short_quote(m["text"])
    k = m["kind"]
    if k == "lying":
        return f"Me watching {m['who']}", f"say '{q}'"
    if k == "copycat":
        return f"When one guy says '{short_quote(m['claim'], 4)}'", "and suddenly the whole lobby is"
    if k == "punchline":
        return f"{m['who']}:", f"'{q}'"
    if k == "awkward":
        return f"{m['who']} in the middle of a murder case:", f"'{q}'"
    if k == "detective":
        return f"{m['who']} calling out {m['accused_who']}", "Everyone: ignores"
    if k == "confused":
        return "Me trying to follow", "this meeting"
    return "", ""


def ending_template(e, mt):
    if e["kind"] == "caught":
        return f"{e['who']} thinking he got away", "Crew: hold up"
    if e["kind"] == "innocent_out":
        return f"Us voting out {e['who']}", "who was innocent"
    if e["kind"] == "skipped_with_impostor":
        return "Crew skipping the vote", f"with {e['who']} right there"
    return "", ""


def describe_moment(m):
    k = m["kind"]
    if k == "lying":
        return (f"{m['who']} is secretly the IMPOSTOR and says '{m['text']}'. "
                f"We know it's a lie.")
    if k == "copycat":
        return (f"{m['count']} players in a row copy the same claim, starting with "
                f"'{m['claim']}'. This is the last one: {m['who']}: '{m['text']}'.")
    if k == "punchline":
        return f"Right after the copycat chain, {m['who']} says '{m['text']}'. The punchline."
    if k == "awkward":
        return f"In the middle of a murder meeting, {m['who']} randomly says '{m['text']}'."
    if k == "detective":
        return (f"{m['who']} says '{m['text']}', correctly calling out {m['accused_who']}, "
                f"the real impostor.")
    if k == "confused":
        return f"Total confusion, people only typing: {', '.join(repr(x) for x in m['burst'])}."
    return ""


# ---------- 2. the AI writes the captions ----------

SYSTEM = """You write meme captions for a funny Among Us channel on Reels / YouTube Shorts.
The channel owner is "me". Write captions from my point of view ("Me watching...",
"Me when...", "When...") or as a reaction to what someone said.

You get the funny moments that were already found in a meeting. For EACH moment
write a classic meme: TOP text and BOTTOM text (max 7 words each).

Rules:
- ADD a joke. Never just repeat what someone said with "X says:".
  BAD:  "Usedwrench says:" / "'I'm engineer'"
  GOOD: "When one guy says he's engineer" / "and suddenly the whole lobby is"
  GOOD: "Me watching Halfwebbed" / "say 'not me' while being the impostor"
- Call players ONLY by their color (Red, Maroon, Cyan...). Quote their words in 'single quotes'.
- Only use facts given to you. Never invent events.
- The ENDING meme reacts to how this meeting ended.
- Score 1-10 how funny the whole clip is. Be honest: most are 5-7.

Reply ONLY with JSON:
{"title": "<max 8 words>", "caption": "<one-line post caption>", "score": <1-10>,
 "memes": [{"moment": "A", "top": "...", "bottom": "..."}],
 "ending": {"top": "...", "bottom": "..."}}"""

EXAMPLE_IN = """Moments:
[A] Red is secretly the IMPOSTOR and says 'i was in admin doing card'. We know it's a lie.
[B] 4 players in a row copy the same claim, starting with 'i did card'. This is the last one: Blue: 'me too'.
Ending: Red got voted out and WAS the impostor."""

EXAMPLE_OUT = {
    "title": "Red really said 'doing card' 💀", "score": 7,
    "caption": "He really thought the card swipe alibi would work",
    "memes": [
        {"moment": "A", "top": "Me watching Red", "bottom": "describe a task he never did"},
        {"moment": "B", "top": "Admin when 4 people", "bottom": "swiped the same card"},
    ],
    "ending": {"top": "Red thinking 'doing card' works", "bottom": "Crew: nah"},
}


def pick_model(ollama):
    try:
        listing = ollama.list()
    except Exception:
        sys.exit("Can't reach Ollama. Start the Ollama app, or use --no-ai")
    models = listing.get("models", []) if isinstance(listing, dict) else getattr(listing, "models", [])
    installed = {(m.get("model") or m.get("name")) if isinstance(m, dict)
                 else (getattr(m, "model", None) or getattr(m, "name", None)) for m in models}
    model = next((m for m in PREFERRED_MODELS if m in installed), None)
    if not model:
        sys.exit("No suitable model found. Install one with: ollama pull qwen2.5:14b")
    return model


def ask_ai(ollama, model, story, moments, ending):
    labels = {p["name"]: p.get("label") or p["name"] for p in story.get("players", [])}
    lines = [f"Me: my role is {story.get('role')}. Players are called by their COLOR. "
             f"Impostors this game: {', '.join(labels.get(n, n) for n in story['impostors']) or 'unknown'}.",
             "Moments:"]
    lines += [f"[{m['label']}] {describe_moment(m)}" for m in moments]
    if ending:
        lines.append({
            "caught": f"Ending: {ending['who']} got voted out and WAS the impostor.",
            "innocent_out": f"Ending: {ending['who']} got voted out but was INNOCENT.",
            "skipped_with_impostor": f"Ending: everyone skipped the vote while "
                                     f"{ending['who']} (the impostor) was right there.",
        }[ending["kind"]])
    else:
        lines.append("Ending: none (leave ending empty).")
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": EXAMPLE_IN},
        {"role": "assistant", "content": json.dumps(EXAMPLE_OUT, ensure_ascii=False)},
        {"role": "user", "content": "\n".join(lines)},
    ]
    for temp in (0.7, 0.4):
        try:
            resp = ollama.chat(model=model, messages=messages, format="json",
                               options={"temperature": temp})
            data = json.loads(resp["message"]["content"])
            if isinstance(data, dict):
                return data
        except Exception:
            continue
    return {}


# ---------- checks ----------

FILLER = {"says", "said", "say", "claims", "asks", "asked", "types", "typed", "the",
          "a", "and", "then", "just", "me", "when"}


def is_repeat(top, bottom, m):
    meme = toks(f"{top} {bottom}")
    said = set(toks(m["text"])) | set(toks(m.get("who") or ""))
    return len([w for w in meme if w not in said and w not in FILLER]) < 2


def words(s, n=8):
    return " ".join(str(s or "").split()[:n])


QUOTE_RX = re.compile(r"['\"‘’“”]([^'\"‘’“”]{2,})['\"‘’“”]")


def quotes_match(top, bottom, said):
    """Anything in 'quotes' must be words someone actually said."""
    said_toks = set(toks(said))

    def was_said(w):  # tolerate typos: "engineer" matches "enginee"
        return w in said_toks or (len(w) >= 4 and any(
            len(s) >= 4 and (s.startswith(w[:5]) or w.startswith(s[:5])) for s in said_toks))

    for q in QUOTE_RX.findall(f"{top} {bottom}"):
        qt = toks(q)
        if qt and sum(was_said(w) for w in qt) / len(qt) < 0.75:
            return False  # made-up quote
    return True


def has_quote(top, bottom, text):
    """The caption includes most of the real message."""
    want = set(toks(text))
    got = set(toks(f"{top} {bottom}"))
    return bool(want) and len(want & got) / len(want) >= 0.6


def good_caption(top, bottom, m, later_ejected, said="", must_name=None):
    if not top or not bottom:
        return False
    if m and m["kind"] not in ("punchline", "awkward") and is_repeat(top, bottom, m):
        return False
    if m and m["kind"] in ("punchline", "awkward") and not has_quote(top, bottom, m["text"]):
        return False  # for these, the real quote IS the joke
    if not quotes_match(top, bottom, said or (m["text"] if m else "")):
        return False
    if must_name and not set(toks(must_name)) <= set(toks(f"{top} {bottom}")):
        return False
    caption_words = set(toks(f"{top} {bottom}"))
    if any(set(toks(n)) and set(toks(n)) <= caption_words for n in later_ejected):
        return False  # talks about someone who's only caught later
    return True


# ---------- building the plan ----------

def build_plan(story, mt, events, moments, ending, ai, later_ejected, plan_all):
    ai_memes = {str(x.get("moment", "")).strip().upper(): x
                for x in (ai.get("memes") or []) if isinstance(x, dict)}
    all_said = " ".join(e["text"] for e in events if e["kind"] == "message")
    # Each moment = zoom on the message, then the meme. Moments are in time
    # order and never overlap: the next zoom starts after the previous meme.
    beats, last_meme_t = [], -1.0
    for m in moments:
        a = ai_memes.get(m["label"], {})
        top, bottom = words(a.get("top")), words(a.get("bottom"))
        source = "ai"
        said = m["text"] + " " + str(m.get("claim", ""))
        if not good_caption(top, bottom, m, later_ejected, said=said):
            top, bottom = template(m)
            source = "template"
        zoom_t = max(m["t"], last_meme_t + 0.1)
        meme_t = max(m["t"] + READ_DELAY, zoom_t + 0.5)
        beats.append({"t": round(zoom_t, 1), "time": fmt(zoom_t), "after_event": m["event"],
                      "effect": "zoom", "duration": ZOOM_SECONDS})
        beats.append({"t": round(meme_t, 1), "time": fmt(meme_t), "after_event": m["event"],
                      "effect": "meme", "mood": m["mood"], "kind": m["kind"],
                      "top": top, "bottom": bottom, "caption_by": source,
                      "duration": MEME_SECONDS})
        last_meme_t = meme_t

    end_meme = None
    if ending:
        a = ai.get("ending") if isinstance(ai.get("ending"), dict) else {}
        top, bottom = words(a.get("top")), words(a.get("bottom"))
        source = "ai"
        if not good_caption(top, bottom, None, later_ejected, said=all_said,
                            must_name=ending.get("who") if ending.get("who") != "Someone" else None):
            top, bottom = ending_template(ending, mt)
            source = "template"
        end_meme = {"mood": ending["mood"], "kind": ending["kind"], "top": top,
                    "bottom": bottom, "caption_by": source, "duration": 2.0}

    # Hook: the reveal, said with the COLOR (never the player's name)
    ej = mt.get("ejection")
    hook = None
    if ej and ej.get("name"):
        who = ej.get("label") or "Someone"
        if who == "Someone":
            text = ("The impostor got caught" if ej.get("truth") == "impostor"
                    else "Wrong person ejected")
        else:
            text = f"{who} was {'The' if ej.get('was_impostor') else 'not The'} Impostor"
        hook = {"text": text, "reveal_t": ej["t"]}

    first_t = min([m["t"] for m in moments] + [mt["start_seconds"]])
    start = max(0, min(mt["start_seconds"], first_t) - BEFORE)
    end = ej["t"] + AFTER_EJECT if ej else mt["end_seconds"] + NO_EJECT_TAIL

    # Every chat message in the clip, in order: the renderer zooms on each
    # bubble and reads it out loud
    dialogue = [{"event": e["n"], "t": e["t"], "label": e["label"], "color": e["color"],
                 "is_me": e["is_me"], "text": e["text"], "geo": e["geo"]}
                for e in events if e["kind"] == "message" and e.get("geo")
                and start <= e["t"] <= end]

    strength = sum(m["strength"] for m in moments) + (ending["strength"] if ending else 0)
    try:
        ai_score = max(1, min(10, int(ai.get("score"))))
    except (TypeError, ValueError):
        ai_score = None
    score = min(10, round(strength * 0.8)) if ai_score is None else \
        round((min(10, strength * 0.8) + ai_score) / 2)

    names = [m["who"] for m in moments if m.get("who")]
    title = words(ai.get("title"), 10) if ai.get("title") else ""
    if not title or not good_caption(title, "x x", None, later_ejected, said=all_said):
        title = {"copycat": "Suddenly everyone is the same role",
                 "lying": f"{names[0] if names else 'He'} really said that",
                 }.get(moments[0]["kind"] if moments else "", "This meeting was chaos")
    caption = str(ai.get("caption") or "").strip() or title

    return {
        "meeting": mt["number"],
        "score": score,
        "strength": strength,
        "moments": [{k: v for k, v in m.items() if k != "strength"} for m in moments],
        "title": title,
        "hook": hook,
        "clip_start": round(start, 1),
        "clip_end": round(end, 1),
        "results_start": round(max(start, mt["end_seconds"] - 2), 1),
        "dialogue": dialogue,
        "beats": beats,
        "ending": end_meme,
        "caption": caption,
        "to_render": plan_all or strength >= MIN_STRENGTH,
        "approved": False,
        "events": events,
    }


# ---------- gameplay moments (kills between meetings) ----------

GAMEPLAY_LEADUP = 8      # seconds of gameplay before the kill
GAMEPLAY_AFTER = 3.5     # seconds after the kill
GAMEPLAY_MOODS = {"shock", "dead", "facepalm", "clown", "laughing", "nervous", "victory"}

GAMEPLAY_SYSTEM = """You write captions for a funny Among Us channel (Reels / Shorts).
Players are called ONLY by their color. Write from my point of view ("Me...").
Given what happened, write:
- "title": max 8 words
- "intro": a caption shown BEFORE the kill, setting up how peaceful/innocent it was,
  max 8 words, starting with "Me" (e.g. "Me just petting my pet")
- "meme": a classic meme shown right AFTER the kill: "mood" (one of: shock, dead,
  facepalm, clown, laughing, nervous), "top" and "bottom" (max 7 words each)
- "caption": one-line post caption
Only use the facts given. Never invent what anyone said.
Reply ONLY with JSON: {"title": "...", "intro": "...", "meme": {"mood": "...",
"top": "...", "bottom": "..."}, "caption": "..."}"""


def ask_ai_gameplay(ollama, model, story, g):
    if g["type"] == "death":
        facts = (f"I was {g['doing']}. Out of nowhere, {g['killer_label']} killed me.")
        if g.get("killer_caught"):
            facts += f" Later, {g['killer_label']} got voted out as the impostor."
    else:
        facts = (f"I am the impostor. I was {g['doing']}, then I killed "
                 f"{g['victim_label']}.")
    try:
        resp = ollama.chat(model=model, format="json", options={"temperature": 0.7},
                           messages=[{"role": "system", "content": GAMEPLAY_SYSTEM},
                                     {"role": "user", "content": facts}])
        data = json.loads(resp["message"]["content"])
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def gameplay_plan(i, g, ai):
    killer, doing = g["killer_label"], g["doing"]
    death = g["type"] == "death"
    # Templates (used when the AI's answer is missing or weak)
    t_title = f"I was just {doing} 💀" if death else f"Sorry {g['victim_label']} 🔪"
    t_intro = f"Me just {doing}" if death else f"Me pretending to be {doing}"
    doing_you = re.sub(r"\bmy\b", "your", doing)
    t_meme = (("dead", f"When you're just {doing_you}", f"and {killer} shows up") if death
              else ("nervous", f"Me after getting {g['victim_label']}", "act natural"))

    meme = ai.get("meme") if isinstance(ai.get("meme"), dict) else {}
    mood = meme.get("mood") if meme.get("mood") in GAMEPLAY_MOODS else None
    top, bottom = words(meme.get("top")), words(meme.get("bottom"))
    meme_ok = bool(mood and top and bottom and quotes_match(top, bottom, ""))
    intro = words(ai.get("intro"), 9)
    title = words(ai.get("title"), 10)
    by = "ai" if meme_ok else "template"
    if not meme_ok:
        mood, top, bottom = t_meme

    ending = None
    if death and g.get("killer_caught_t"):
        ending = {"mood": "victory", "top": f"{killer} after killing me",
                  "bottom": "getting voted out", "caption_by": "template", "duration": 2.0}

    t = g["t"]
    return {
        "kind": "gameplay",
        "meeting": f"g{i}",
        "type": g["type"],
        "score": 7 if death else 6,
        "strength": 6,
        "title": title if title and quotes_match(title, "", "") else t_title,
        "intro_text": intro if intro and quotes_match(intro, "", "") else t_intro,
        "clip_start": round(max(0, t - GAMEPLAY_LEADUP), 1),
        "kill_t": t,
        "clip_end": round(t + GAMEPLAY_AFTER, 1),
        "meme": {"mood": mood, "top": top, "bottom": bottom, "caption_by": by,
                 "duration": 1.8},
        "ending": ending,
        "caption": str(ai.get("caption") or "").strip() or t_title,
        "to_render": True,
        "approved": False,
    }


def print_gameplay_plan(p):
    print(f"\n[EDIT] Gameplay {p['meeting']}  ({p['type']})  "
          f"{fmt(p['clip_start'])}-{fmt(p['clip_end'])}")
    print(f"  Title:   {p['title']}")
    print(f"  {fmt(p['clip_start'])}    CAPTION  {p['intro_text']}")
    print(f"  {fmt(p['kill_t'])}    SLOW-MO + ZOOM on the kill")
    m = p["meme"]
    tag = "" if m["caption_by"] == "ai" else "  (template)"
    print(f"           MEME [{m['mood']}]  {m['top'].upper()} / {m['bottom'].upper()}{tag}")
    if p["ending"]:
        e = p["ending"]
        print(f"  end      ENDING [{e['mood']}]  {e['top'].upper()} / {e['bottom'].upper()}")
    print(f"  Caption: {p['caption']}")


def print_plan(p):
    if p.get("kind") == "gameplay":
        return print_gameplay_plan(p)
    mark = "EDIT" if p["to_render"] else "skip"
    print(f"\n[{mark}] Meeting {p['meeting']}  score {p['score']}/10  "
          f"{fmt(p['clip_start'])}-{fmt(p['clip_end'])}")
    if not p["to_render"]:
        return
    print(f"  Title:   {p['title']}")
    if p["hook"]:
        print(f"  0:00     HOOK: '{p['hook']['text']}' -> 'a few moments earlier...'")
    events = {e["n"]: e for e in p["events"]}
    for b in p["beats"]:
        ev = events[b["after_event"]]
        if b["effect"] == "zoom":
            print(f"  {b['time']}    ZOOM  {ev.get('label', '')}: \"{ev['text']}\"")
        else:
            tag = "" if b["caption_by"] == "ai" else "  (template)"
            print(f"  {b['time']}    MEME [{b['mood']}]  {b['top'].upper()} / "
                  f"{b['bottom'].upper()}{tag}")
    if p["ending"]:
        e = p["ending"]
        tag = "" if e["caption_by"] == "ai" else "  (template)"
        print(f"  end      ENDING [{e['mood']}]  {e['top'].upper()} / {e['bottom'].upper()}{tag}")
    print(f"  Caption: {p['caption']}")


def pick_story():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if args:
        f = STORIES / f"{Path(args[0]).stem}.json"
        if not f.exists():
            sys.exit(f"No story for that video. Run: python story.py \"{args[0]}\"")
        return f
    files = sorted(STORIES.glob("*.json"), key=lambda p: p.stat().st_mtime)
    if not files:
        sys.exit("No stories found. Run story.py first.")
    return files[-1]


def main():
    story_f = pick_story()
    story = json.loads(story_f.read_text())
    if not LIBRARY.exists():
        sys.exit("No meme library yet. Run: python memes.py")
    plan_all = "--all" in sys.argv
    use_ai = "--no-ai" not in sys.argv

    ollama, model = None, None
    if use_ai:
        import ollama as _ollama
        ollama = _ollama
        model = pick_model(ollama)
        print(f"Using AI model: {model}")

    plans = []
    for mt in story["meetings"]:
        events = meeting_events(mt)
        labels = {p["name"]: p.get("label") or p["name"] for p in story.get("players", [])}
        moments = find_moments(events, story["impostors"], labels)
        ending = ending_moment(mt, story["impostors"], labels)
        later_ejected = [m["ejection"].get("label") or m["ejection"]["name"] for m in story["meetings"]
                         if m["start_seconds"] > mt["start_seconds"]
                         and m.get("ejection") and m["ejection"].get("name")]
        if not moments and not ending:
            print(f"Meeting {mt['number']}: no funny moments found")
            continue
        found = ", ".join(f"{m['kind']} ({m['who']})" for m in moments) or "ending only"
        print(f"Meeting {mt['number']}: found {found}")
        ai = ask_ai(ollama, model, story, moments, ending) if use_ai else {}
        plans.append(build_plan(story, mt, events, moments, ending, ai,
                                later_ejected, plan_all))

    # Gameplay moments (kills between meetings, from moments.py)
    for i, g in enumerate(story.get("gameplay", []), 1):
        print(f"Gameplay {i}: {g['type']} at {g['time']} ({g['doing']})")
        ai = ask_ai_gameplay(ollama, model, story, g) if use_ai else {}
        plans.append(gameplay_plan(i, g, ai))

    PLANS.mkdir(parents=True, exist_ok=True)
    out = PLANS / story_f.name
    out.write_text(json.dumps({"video": story["video"], "model": model, "plans": plans},
                              indent=2, ensure_ascii=False))

    for p in sorted(plans, key=lambda x: -x["strength"]):
        print_plan(p)
    n = sum(p["to_render"] for p in plans)
    print(f"\n{n} edit(s) planned. Saved: {out}")


if __name__ == "__main__":
    main()
