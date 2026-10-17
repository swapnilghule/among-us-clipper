"""
Step 2 of the meme editor: the meme library.

Each meme is a MOOD (suspicious, lying, confused, ...) with an emoji, a color
and a description that tells the AI when to use it. The AI picks a mood and
writes the captions; this file turns that into a meme card.

Meme cards are full-screen vertical images (1080x1920), classic meme style:
  TOP TEXT      e.g. "Me watching Halfwebbed"
  big emoji
  BOTTOM TEXT   e.g. "say 'not me someone vent'"

Your own memes (later): drop images into storage/memes/<mood>/ (for example
storage/memes/lying/my_meme.png). The library picks them up automatically.

Usage (from ~/amongus-clipper with the venv active):
  python memes.py          # builds the library + preview cards for every mood
"""

import json
import random
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

BASE = Path(__file__).resolve().parent
MEMES = BASE / "storage" / "memes"
PREVIEW = MEMES / "preview"
LIBRARY = MEMES / "library.json"

W, H = 1080, 1920
IMAGE_TYPES = {".png", ".jpg", ".jpeg", ".webp", ".gif"}

# id: (emoji, background color, when to use it, example top, example bottom)
MOODS = {
    "suspicious": ("🤨", "#6C5CE7",
                   "Someone makes a weak claim or acts sus",
                   "Me reading Green's alibi", "for the third time"),
    "lying": ("🤥", "#E17055",
              "We know (with hindsight) this player is lying, usually the impostor defending themselves",
              "Me watching Halfwebbed", "say 'not me someone vent'"),
    "confused": ("😵", "#0984E3",
                 "Nobody knows what's going on: where? what? who?",
                 "The whole crew", "trying to find the body"),
    "shock": ("😱", "#D63031",
              "A big reveal or a twist nobody expected",
              "When it turns out", "the quiet one was the impostor"),
    "facepalm": ("🤦", "#636E72",
                 "Someone says something dumb or the crew makes an obvious mistake",
                 "Crew skipping the vote", "with the impostor right there"),
    "victory": ("😎", "#00B894",
                "The impostor gets caught, justice served",
                "Bhavin thinking he got away", "Crew: hold up"),
    "clown": ("🤡", "#FD79A8",
              "A confident accusation that turns out wrong, or an innocent gets ejected",
              "Us voting out Blue", "'was not The Impostor'"),
    "awkward": ("😬", "#FDCB6E",
                "A weird, random or off-topic message in the middle of the meeting",
                "Mere Papa in the middle of a murder case:", "'do u want to come to my home'"),
    "laughing": ("😂", "#F39C12",
                 "Pure chaos, everyone being silly",
                 "This lobby", "has zero brain cells"),
    "nervous": ("😰", "#00CEC9",
                "The impostor under pressure, sweating through an excuse",
                "The impostor when someone says", "'I saw you near electrical'"),
    "dead": ("💀", "#2D3436",
             "Something so absurd it kills us",
             "'how many'", "I can't breathe"),
    "popcorn": ("🍿", "#A29BFE",
                "Players arguing and you just watch the drama",
                "Me watching them", "accuse each other"),
    "detective": ("🔍", "#0652DD",
                  "Someone figured it out correctly but nobody listened",
                  "Quizily: 'green imposter'", "Everyone: ignores"),
    "copycat": ("🦜", "#20BF6B",
                "Everyone repeating the same claim ('me too', 'I'm also engineer')",
                "When one guy says he's engineer", "and suddenly the whole lobby is"),
}

TEXT_FONTS = [
    "/System/Library/Fonts/Supplemental/Impact.ttf",
    "/System/Library/Fonts/Supplemental/Arial Black.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/usr/share/fonts/opentype/inter/Inter-Black.otf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
]
EMOJI_FONTS = [
    ("/System/Library/Fonts/Apple Color Emoji.ttc", 160),
    ("/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf", 109),
]


# ---------- drawing helpers ----------

def _first_font(paths):
    return next((p for p in paths if Path(p).exists()), None)


def _text_font(size):
    path = _first_font(TEXT_FONTS)
    return ImageFont.truetype(path, size) if path else ImageFont.load_default()


def _emoji_image(emoji, size):
    """Render an emoji as a square RGBA image, or None if no emoji font."""
    for path, native in EMOJI_FONTS:
        if not Path(path).exists():
            continue
        try:
            font = ImageFont.truetype(path, native)
            canvas = Image.new("RGBA", (native * 2, native * 2), (0, 0, 0, 0))
            ImageDraw.Draw(canvas).text((native // 2, native // 2), emoji,
                                        font=font, embedded_color=True)
            box = canvas.getbbox()
            if not box:
                continue
            img = canvas.crop(box)
            scale = size / max(img.size)
            return img.resize((int(img.width * scale), int(img.height * scale)),
                              Image.LANCZOS)
        except Exception:
            continue
    return None


def _hex(c):
    c = c.lstrip("#")
    return tuple(int(c[i:i + 2], 16) for i in (0, 2, 4))


def _background(color):
    r, g, b = _hex(color)
    bg = Image.new("RGB", (W, H))
    px = ImageDraw.Draw(bg)
    for y in range(H):  # vertical gradient: lighter top, darker bottom
        k = 1.15 - 0.55 * (y / H)
        px.line([(0, y), (W, y)], fill=(min(255, int(r * k)), min(255, int(g * k)),
                                         min(255, int(b * k))))
    glow = Image.new("L", (W, H), 0)
    ImageDraw.Draw(glow).ellipse([W * 0.1, H * 0.3, W * 0.9, H * 0.72], fill=90)
    glow = glow.filter(ImageFilter.GaussianBlur(120))
    return Image.composite(Image.new("RGB", (W, H), (255, 255, 255)), bg, glow)


def _fit_text(draw, text, max_lines, box_w, start_size):
    """Wrap and shrink text until it fits. Returns (font, lines)."""
    size = start_size
    while size >= 44:
        font = _text_font(size)
        chars = max(8, int(box_w / (size * 0.55)))
        lines = textwrap.wrap(text, chars)
        widest = max((draw.textlength(l, font=font) for l in lines), default=0)
        if len(lines) <= max_lines and widest <= box_w:
            return font, lines
        size -= 6
    font = _text_font(44)
    return font, textwrap.wrap(text, int(box_w / 24))[:max_lines]


def _draw_block(draw, text, y_center, start_size, max_lines=4):
    if not text:
        return
    text = text.upper()
    font, lines = _fit_text(draw, text, max_lines, W - 120, start_size)
    line_h = int(font.size * 1.12)
    y = y_center - line_h * len(lines) // 2
    stroke = max(4, font.size // 11)
    for line in lines:
        w = draw.textlength(line, font=font)
        draw.text(((W - w) / 2, y), line, font=font, fill="white",
                  stroke_width=stroke, stroke_fill="black")
        y += line_h


# ---------- public ----------

def make_card(mood, top, bottom, out_path):
    """Make a 1080x1920 meme card for a mood and save it. Returns the path."""
    emoji, color, *_ = MOODS.get(mood, MOODS["laughing"])
    img = _background(color)
    draw = ImageDraw.Draw(img)

    _draw_block(draw, top, 430, 104)

    em = _emoji_image(emoji, 560)
    if em:
        shadow = Image.new("RGBA", img.size, (0, 0, 0, 0))
        sx, sy = (W - em.width) // 2, 1040 - em.height // 2
        shadow.paste((0, 0, 0, 110), (sx + 18, sy + 28), em)
        shadow = shadow.filter(ImageFilter.GaussianBlur(18))
        img = Image.alpha_composite(img.convert("RGBA"), shadow)
        img.paste(em, (sx, sy), em)
        img = img.convert("RGB")
        draw = ImageDraw.Draw(img)
    else:
        _draw_block(draw, mood, 1040, 150, 1)  # no emoji font: show the mood word

    _draw_block(draw, bottom, 1580, 92)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    return out_path


def your_memes(mood):
    folder = MEMES / mood
    if not folder.exists():
        return []
    return sorted(str(p) for p in folder.iterdir() if p.suffix.lower() in IMAGE_TYPES)


def get_meme(mood, top, bottom, out_path, use_your_memes=True):
    """What the editor calls. Uses one of your own memes for this mood if you
    added any, otherwise generates a card.
    Returns {"type": "card"|"file", "path": ..., "needs_caption": bool}."""
    files = your_memes(mood) if use_your_memes else []
    if files:
        return {"type": "file", "path": random.choice(files), "needs_caption": True}
    return {"type": "card", "path": str(make_card(mood, top, bottom, out_path)),
            "needs_caption": False}


def build_library():
    """Write library.json: the menu of moods the AI chooses from."""
    lib = {"moods": []}
    for mood, (emoji, color, when, ex_top, ex_bottom) in MOODS.items():
        (MEMES / mood).mkdir(parents=True, exist_ok=True)
        lib["moods"].append({
            "id": mood, "emoji": emoji, "color": color, "use_when": when,
            "example": {"top": ex_top, "bottom": ex_bottom},
            "your_memes": your_memes(mood),
        })
    LIBRARY.write_text(json.dumps(lib, indent=2, ensure_ascii=False))
    return lib


def main():
    lib = build_library()
    PREVIEW.mkdir(parents=True, exist_ok=True)
    thumbs = []
    for m in lib["moods"]:
        p = make_card(m["id"], m["example"]["top"], m["example"]["bottom"],
                      PREVIEW / f"{m['id']}.png")
        thumbs.append(Image.open(p).resize((270, 480), Image.LANCZOS))
        extra = f"  (+{len(m['your_memes'])} of your memes)" if m["your_memes"] else ""
        print(f"  {m['emoji']}  {m['id']:<11} {m['use_when']}{extra}")

    cols = 7
    rows = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 270 + (cols + 1) * 12, rows * 480 + (rows + 1) * 12), "#111")
    for i, t in enumerate(thumbs):
        sheet.paste(t, (12 + (i % cols) * 282, 12 + (i // cols) * 492))
    sheet.save(PREVIEW / "all_moods.png")

    if not _first_font([p for p, _ in EMOJI_FONTS]):
        print("\nNote: no emoji font found, so cards show the mood word instead.")
    print(f"\n{len(lib['moods'])} moods ready.")
    print(f"Preview of every card: {PREVIEW / 'all_moods.png'}")
    print(f"Add your own memes later in: {MEMES}/<mood>/")


if __name__ == "__main__":
    main()
