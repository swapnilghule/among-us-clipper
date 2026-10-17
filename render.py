"""
Step 4 of the meme editor: turn each edit plan into a finished vertical video.

Needs planner.py to have run first (uses storage/plans/<video>.json).

The video, in order:
  1. HOOK       the ejection screen first ("Maroon was The Impostor") + sting
  2. "A few moments earlier..." + whoosh
  3. DIALOGUE   every chat message, one by one: the bubble zoomed in big, the
                player's name replaced by their COLOR, the line read out loud
                in that color's voice, and a big subtitle (for muted viewers)
  4. MEMES      meme cards cut in right after the message they react to + boom
  5. RESULTS    the real vote results and ejection, with game audio
  6. ENDING     the ending meme + boom

Output: storage/edits/<video>_m<N>_<title>.mp4 + a .txt with the post caption.

Usage (from ~/amongus-clipper with the venv active):
  python render.py                 # every planned edit of the newest video
  python render.py --meeting 3     # just one meeting
  python render.py --all           # also meetings the planner skipped
"""

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

import colors
import memes
import voices

BASE = Path(__file__).resolve().parent
RAW = BASE / "storage" / "raw"
PLANS = BASE / "storage" / "plans"
EDITS = BASE / "storage" / "edits"
WORK = BASE / "storage" / "tmp_render"

W, H, FPS = 1080, 1920, 30
HOOK_BEFORE, HOOK_AFTER = 0.3, 2.2   # seconds around the ejection text
EARLIER_SECONDS = 1.0
VOICE_DELAY = 0.15                   # pop first, then the voice
SHOT_TAIL = 0.35                     # pause after each line is spoken
MIN_SHOT = 1.1
BUBBLE_MAX_W, BUBBLE_MAX_H, BUBBLE_MAX_SCALE = 1000, 560, 4.0
BUBBLE_Y = 880                       # centre of the zoomed bubble
SUBTITLE_Y = 1420
VOLUME = {"pop": 0.5, "voice": 1.0, "boom": 0.8, "whoosh": 0.6, "sting": 0.7}
HASHTAGS = "#amongus #amongusfunny #amongusmemes #gaming #shorts"

ENC = ["-r", str(FPS), "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
       "-pix_fmt", "yuv420p", "-c:a", "aac", "-ar", "44100", "-ac", "2", "-b:a", "160k"]


# ---------- small helpers ----------

def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip()[-800:])


def even(n):
    return int(n) // 2 * 2


def no_emoji(s):
    return re.sub(r"[^\u0000-￿]|[☀-➿️‍]", "", str(s or "")).strip()


def slug(s):
    return re.sub(r"[^a-z0-9]+", "-", no_emoji(s).lower()).strip("-")[:40] or "edit"


def has_audio(video):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a",
                        "-show_entries", "stream=index", "-of", "csv=p=0", str(video)],
                       capture_output=True, text=True)
    return bool(r.stdout.strip())


def read_frame(video, t):
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def game_window(frame):
    """The game window inside the recording, without black borders."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cols = np.where((gray > 12).mean(axis=0) > 0.3)[0]
    rows = np.where((gray > 12).mean(axis=1) > 0.3)[0]
    if len(cols) == 0 or len(rows) == 0:
        h, w = gray.shape
        return 0, 0, w, h
    return int(cols[0]), int(rows[0]), int(cols[-1] - cols[0]), int(rows[-1] - rows[0])


def to_pil(frame):
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


# ---------- images (drawn with Pillow) ----------

def text_overlay(path, top_text=None, low_text=None):
    """Transparent 1080x1920 image with a title near the top and/or a big
    line lower down."""
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    if top_text:
        memes._draw_block(d, no_emoji(top_text), 215, 80, max_lines=2)
    if low_text:
        memes._draw_block(d, no_emoji(low_text), 1650, 96, max_lines=2)
    img.save(path)
    return path


def plain_card(path, text, color="#111111"):
    img = memes._background(color)
    memes._draw_block(ImageDraw.Draw(img), text, H // 2, 110, max_lines=3)
    img.save(path)
    return path


def meme_card(path, mood, top, bottom):
    m = memes.get_meme(mood, top, bottom, path)
    if not m["needs_caption"]:
        return m["path"]
    # One of your own meme images: put it on a card with the captions
    emoji, color, *_ = memes.MOODS.get(mood, memes.MOODS["laughing"])
    card = memes._background(color)
    pic = Image.open(m["path"]).convert("RGBA")
    pic.thumbnail((W - 120, 900))
    card.paste(pic, ((W - pic.width) // 2, 1040 - pic.height // 2), pic)
    d = ImageDraw.Draw(card)
    memes._draw_block(d, top, 430, 104)
    memes._draw_block(d, bottom, 1580, 92)
    card.save(path)
    return path


def dialogue_card(path, frame, msg, title):
    """The chat bubble zoomed in big on a blurred game background, with the
    player's name covered and replaced by their color."""
    wx, wy, ww, wh = game_window(frame)
    game = to_pil(frame).crop((wx, wy, wx + ww, wy + wh))

    # Background: the game, blurred and darkened, filling the screen
    scale = max(W / game.width, H / game.height)
    bg = game.resize((int(game.width * scale) + 1, int(game.height * scale) + 1))
    left, top = (bg.width - W) // 2, (bg.height - H) // 2
    bg = bg.crop((left, top, left + W, top + H)).filter(ImageFilter.GaussianBlur(22))
    bg = Image.eval(bg, lambda v: int(v * 0.45)).convert("RGBA")

    # The bubble, zoomed
    g = msg["geo"]
    bx1, by1, bx2, by2 = g["bubble"]
    bubble = to_pil(frame).crop((bx1, by1, bx2, by2))
    s = min(BUBBLE_MAX_W / bubble.width, BUBBLE_MAX_H / bubble.height, BUBBLE_MAX_SCALE)
    bubble = bubble.resize((int(bubble.width * s), int(bubble.height * s)), Image.LANCZOS)
    d = ImageDraw.Draw(bubble)

    # Cover the player's name with the bubble's own color, write the COLOR instead
    nx1, ny1, nx2, ny2 = g["name"]
    nx1, ny1, nx2, ny2 = ((nx1 - bx1) * s - 4, (ny1 - by1) * s - 3,
                          (nx2 - bx1) * s + 4, (ny2 - by1) * s + 3)
    sample = bubble.getpixel((int(min(bubble.width - 1, nx2 + 6)), int((ny1 + ny2) / 2))) \
        if not msg["is_me"] else bubble.getpixel((int(max(0, nx1 - 6)), int((ny1 + ny2) / 2)))
    d.rectangle([nx1, ny1, nx2, ny2], fill=sample)
    word = msg["label"].upper()
    font = memes._text_font(max(40, int((ny2 - ny1) * 0.95)))
    tw = d.textlength(word, font=font)
    tx = nx2 - tw if msg["is_me"] else nx1 + 4
    ty = max(0, ny2 - font.size * 1.05)
    d.text((tx, ty), word, font=font,
           fill=colors.hex_color(msg["color"]) if msg.get("color") else "#333333",
           stroke_width=max(2, font.size // 12), stroke_fill="black")

    # Shadow + paste
    x, y = (W - bubble.width) // 2, BUBBLE_Y - bubble.height // 2
    shadow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(shadow).rounded_rectangle(
        [x + 10, y + 16, x + bubble.width + 10, y + bubble.height + 16], 28, fill=(0, 0, 0, 150))
    bg = Image.alpha_composite(bg, shadow.filter(ImageFilter.GaussianBlur(16)))
    bg.paste(bubble, (x, y))

    # Title on top, big subtitle below
    d = ImageDraw.Draw(bg)
    if title:
        memes._draw_block(d, no_emoji(title), 215, 80, max_lines=2)
    memes._draw_block(d, no_emoji(msg["text"]), SUBTITLE_Y, 96, max_lines=3)
    bg.convert("RGB").save(path)
    return path


# ---------- rendering pieces ----------

class Renderer:
    def __init__(self, video, work):
        self.video, self.work = video, work
        self.audio = has_audio(video)
        self.n = 0
        self.pieces = []

    def _out(self):
        self.n += 1
        return self.work / f"piece_{self.n:03d}.mp4"

    @staticmethod
    def _mix(sounds, first_input, dur, base=None):
        """Audio graph: optional base track + sounds at offsets, no volume drop."""
        parts, labels = [], []
        if base:
            parts.append(f"{base}aresample=44100,aformat=channel_layouts=stereo[b0]")
            labels.append("[b0]")
        for i, (path, offset, vol) in enumerate(sounds):
            ms = int(offset * 1000)
            parts.append(f"[{first_input + i}:a]aresample=44100,aformat=channel_layouts=stereo,"
                         f"volume={vol},adelay={ms}|{ms}[s{i}]")
            labels.append(f"[s{i}]")
        if not labels:
            return None
        parts.append(f"{''.join(labels)}amix=inputs={len(labels)}:duration=longest:"
                     f"normalize=0,apad,atrim=0:{dur:.3f}[aout]")
        return ";".join(parts)

    def footage(self, start, end, window, overlay, sounds=(), speed=1.0, zoom=1.0):
        """Real gameplay, the whole game window, vertical, overlay on top.
        speed < 1 = slow motion; zoom > 1 = punch in on the centre."""
        src = end - start
        if src < 0.1:
            return
        dur = src / speed
        cx, cy, cw, ch = (even(v) for v in window)
        if zoom > 1:
            zw, zh = even(cw / zoom), even(ch / zoom)
            cx, cy, cw, ch = even(cx + (cw - zw) / 2), even(cy + (ch - zh) / 2), zw, zh
        slow = f"setpts=PTS/{speed}," if speed != 1 else ""
        video_f = (f"[0:v]{slow}crop={cw}:{ch}:{cx}:{cy},split[a][b];"
                   f"[a]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},"
                   "gblur=sigma=30,eq=brightness=-0.15[bg];"
                   f"[b]scale={W}:-2[fg];[bg][fg]overlay=0:(H-h)/2[v0];"
                   "[v0][1:v]overlay=0:0,format=yuv420p[v]")
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
               "-ss", f"{start:.3f}", "-t", f"{src:.3f}", "-i", str(self.video),
               "-loop", "1", "-t", f"{dur:.3f}", "-i", str(overlay)]
        nxt = 2
        if not self.audio:
            cmd += ["-f", "lavfi", "-t", f"{dur:.3f}", "-i", "anullsrc=r=44100:cl=stereo"]
            base, nxt = "[2:a]", 3
        else:
            base = f"[0:a]atempo={max(0.5, speed)}," if speed != 1 else "[0:a]"
        for path, _, _ in sounds:
            cmd += ["-i", str(path)]
        audio_f = self._mix(list(sounds), nxt, dur, base=base)
        out = self._out()
        run(cmd + ["-filter_complex", f"{video_f};{audio_f}", "-map", "[v]",
                   "-map", "[aout]", "-t", f"{dur:.3f}"] + ENC + [str(out)])
        self.pieces.append(out)

    def card(self, image, dur, sounds=()):
        """A still image with a slow push-in, plus sounds (voice, effects)."""
        out = self._out()
        grow = f"(1+0.05*t/{dur:.2f})"
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
               "-loop", "1", "-t", f"{dur:.3f}", "-i", str(image),
               "-f", "lavfi", "-t", f"{dur:.3f}", "-i", "anullsrc=r=44100:cl=stereo"]
        for path, _, _ in sounds:
            cmd += ["-i", str(path)]
        audio_f = self._mix(list(sounds), 2, dur, base="[1:a]")
        video_f = (f"[0:v]scale=w='trunc({W}*{grow}/2)*2':h=-2:eval=frame,"
                   f"crop={W}:{H},format=yuv420p[v]")
        run(cmd + ["-filter_complex", f"{video_f};{audio_f}", "-map", "[v]",
                   "-map", "[aout]", "-t", f"{dur:.3f}"] + ENC + [str(out)])
        self.pieces.append(out)

    def join(self, out_path):
        lst = self.work / "pieces.txt"
        lst.write_text("".join(f"file '{p.as_posix()}'\n" for p in self.pieces))
        run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "concat",
             "-safe", "0", "-i", str(lst), "-c", "copy", "-movflags", "+faststart",
             str(out_path)])


# ---------- building one edit ----------

def render_plan(video, plan, out_path):
    work = WORK / f"m{plan['meeting']}"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    r = Renderer(video, work)
    sfx = {n: voices.sfx(n, work) for n in ("pop", "boom", "whoosh", "sting")}
    title_png = text_overlay(work / "title.png", top_text=plan["title"])

    first = read_frame(video, plan["clip_start"] + 1)
    if first is None:
        raise RuntimeError("couldn't read the video")
    window = game_window(first)

    # 1-2. Hook: the reveal first, then "a few moments earlier"
    hook = plan.get("hook")
    if hook:
        hook_png = text_overlay(work / "hook.png", low_text=hook["text"])
        r.footage(max(0, hook["reveal_t"] - HOOK_BEFORE), hook["reveal_t"] + HOOK_AFTER,
                  window, hook_png, sounds=[(sfx["sting"], 0.2, VOLUME["sting"])])
        r.card(plain_card(work / "earlier.png", "A few moments earlier..."), EARLIER_SECONDS,
               sounds=[(sfx["whoosh"], 0, VOLUME["whoosh"])])

    # Which dialogue line each meme follows
    dialogue = plan.get("dialogue") or []
    memes_after = {}
    for b in plan["beats"]:
        if b["effect"] != "meme":
            continue
        idx = next((i for i, d in enumerate(dialogue) if d["event"] == b["after_event"]), None)
        if idx is None:
            before = [i for i, d in enumerate(dialogue) if d["t"] <= b["t"]]
            idx = before[-1] if before else -1
        memes_after.setdefault(idx, []).append(b)

    def play_memes(idx):
        for k, mb in enumerate(memes_after.get(idx, [])):
            img = meme_card(work / f"meme_{idx}_{k}.png", mb["mood"], mb["top"], mb["bottom"])
            r.card(img, mb["duration"], sounds=[(sfx["boom"], 0, VOLUME["boom"])])

    # 3-4. Dialogue, one bubble at a time, with memes in between
    play_memes(-1)
    for i, msg in enumerate(dialogue):
        frame = read_frame(video, msg["t"])
        if frame is None:
            continue
        img = dialogue_card(work / f"line_{i}.png", frame, msg, plan["title"])
        wav = work / f"voice_{i}.wav"
        secs = voices.speak(msg["text"], msg.get("color"), wav, is_me=msg["is_me"])
        dur = max(MIN_SHOT, VOICE_DELAY + secs + SHOT_TAIL)
        r.card(img, dur, sounds=[(sfx["pop"], 0, VOLUME["pop"]),
                                 (wav, VOICE_DELAY, VOLUME["voice"])])
        play_memes(i)

    # 5. The real vote results and ejection
    res_start = plan.get("results_start", plan["clip_start"])
    if not dialogue:
        res_start = plan["clip_start"]  # old plan without bubbles: show the meeting
    r.footage(res_start, plan["clip_end"], window, title_png)

    # 6. Ending meme
    e = plan.get("ending")
    if e:
        r.card(meme_card(work / "ending.png", e["mood"], e["top"], e["bottom"]),
               e.get("duration", 2.0), sounds=[(sfx["boom"], 0, VOLUME["boom"])])

    r.join(out_path)
    shutil.rmtree(work, ignore_errors=True)


SLOWMO_BEFORE, SLOWMO_AFTER, SLOWMO_SPEED, SLOWMO_ZOOM = 0.4, 1.6, 0.5, 1.3


def render_gameplay(video, plan, out_path):
    """Lead-up with a caption -> slow motion + zoom on the kill -> meme -> aftermath."""
    work = WORK / f"{plan['meeting']}"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    r = Renderer(video, work)
    sfx = {n: voices.sfx(n, work) for n in ("boom", "sting")}

    first = read_frame(video, plan["clip_start"] + 0.5)
    if first is None:
        raise RuntimeError("couldn't read the video")
    window = game_window(first)
    k = plan["kill_t"]

    intro_png = text_overlay(work / "intro.png", top_text=plan["title"],
                             low_text=plan["intro_text"])
    title_png = text_overlay(work / "title.png", top_text=plan["title"])

    r.footage(plan["clip_start"], k - SLOWMO_BEFORE, window, intro_png)
    r.footage(k - SLOWMO_BEFORE, k + SLOWMO_AFTER, window, title_png,
              speed=SLOWMO_SPEED, zoom=SLOWMO_ZOOM, sounds=[(sfx["sting"], 0.1, VOLUME["sting"])])
    m = plan["meme"]
    r.card(meme_card(work / "meme.png", m["mood"], m["top"], m["bottom"]), m["duration"],
           sounds=[(sfx["boom"], 0, VOLUME["boom"])])
    r.footage(k + SLOWMO_AFTER, plan["clip_end"], window, title_png)
    e = plan.get("ending")
    if e:
        r.card(meme_card(work / "ending.png", e["mood"], e["top"], e["bottom"]),
               e.get("duration", 2.0), sounds=[(sfx["boom"], 0, VOLUME["boom"])])
    r.join(out_path)
    shutil.rmtree(work, ignore_errors=True)


def pick_plans():
    files = sorted(PLANS.glob("*.json"), key=lambda p: p.stat().st_mtime)
    argv = sys.argv[1:]
    skip = {argv.index("--meeting") + 1} if "--meeting" in argv else set()
    args = [a for i, a in enumerate(argv) if not a.startswith("--") and i not in skip]
    if args:
        f = PLANS / f"{Path(args[0]).stem}.json"
        if not f.exists():
            sys.exit(f"No plan for that video. Run: python planner.py \"{args[0]}\"")
        return f
    if not files:
        sys.exit("No plans found. Run planner.py first.")
    return files[-1]


def main():
    if not shutil.which("ffmpeg"):
        sys.exit("FFmpeg not found. Install it with: brew install ffmpeg")
    plans_f = pick_plans()
    data = json.loads(plans_f.read_text())
    video = RAW / data["video"]
    if not video.exists():
        sys.exit(f"Video not found: {video}")

    only = None
    if "--meeting" in sys.argv:
        try:
            only = sys.argv[sys.argv.index("--meeting") + 1]
        except IndexError:
            sys.exit("Use: python render.py --meeting 3   (or g1 for a gameplay moment)")
    if only is not None:
        todo = [p for p in data["plans"] if str(p["meeting"]) == only]
    else:
        todo = [p for p in data["plans"] if p.get("to_render") or "--all" in sys.argv]
    if not todo:
        sys.exit("Nothing to render. Use --all, or check the meeting number.")
    if not any(p.get("dialogue") or p.get("kind") == "gameplay" for p in todo):
        print("Note: these plans have no chat bubble positions, so there's no dialogue zoom.\n"
              "      Re-run chat.py, story.py and planner.py to get it.")

    EDITS.mkdir(parents=True, exist_ok=True)
    made = 0
    for p in todo:
        name = f"{video.stem}_m{p['meeting']}_{slug(p['title'])}"
        out = EDITS / f"{name}.mp4"
        try:
            if p.get("kind") == "gameplay":
                print(f"Gameplay {p['meeting']}: rendering '{p['title']}'...")
                render_gameplay(video, p, out)
            else:
                print(f"Meeting {p['meeting']}: rendering '{p['title']}' "
                      f"({len(p.get('dialogue') or [])} lines of dialogue)...")
                render_plan(video, p, out)
        except RuntimeError as e:
            print(f"   Failed: {e}")
            continue
        out.with_suffix(".txt").write_text(f"{p['title']}\n\n{p['caption']}\n{HASHTAGS}\n")
        print(f"   Saved: {out.name}")
        made += 1
    shutil.rmtree(WORK, ignore_errors=True)
    print(f"\n{made} edit(s) saved to {EDITS}")


if __name__ == "__main__":
    main()
