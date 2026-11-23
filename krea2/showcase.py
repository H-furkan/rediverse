#!/usr/bin/env python
"""Seed-matched Turbo vs ReDiverse figures (CPU only; PIL).

  python -m krea2.showcase --root <out>/showcase --rows student,rediverse_a4 --prompts prompts/krea2_showcase.txt \
      --ids 0-47 --seeds 8 --out <out>/showcase/figures
Writes one figure per prompt (pair_pNNN.jpg: a row per set, a column per seed, same initial noise down each column) and
composite.jpg (a few prompts, fewer seeds) for a model card.
"""
from __future__ import annotations

import argparse
import os

from PIL import Image, ImageDraw, ImageFont

LABEL = {"student": "Krea 2 Turbo", "rediverse_a4": "Krea 2 ReDiverse", "rediverse_a2": "Krea 2 ReDiverse (alpha 2)",
         "teacher": "Krea 2 Raw (teacher)"}


def font(size):
    for f in ("/usr/share/fonts/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if os.path.exists(f):
            return ImageFont.truetype(f, size)
    try:
        return ImageFont.load_default(size=size)    # Pillow >= 10.1: scalable built-in font
    except TypeError:
        return ImageFont.load_default()


def pair(root, rows, idx, text, seeds, cell, label_w=0, title=True):
    head = 34 if title else 0
    W, H = label_w + cell * seeds, head + cell * len(rows) + 6 * (len(rows) - 1)
    g = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(g)
    if title:
        d.text((8, 7), text if len(text) < 140 else text[:137] + "...", fill=(20, 20, 20), font=font(18))
    for r, s in enumerate(rows):
        y = head + r * (cell + 6)
        if label_w:
            d.text((8, y + cell // 2 - 10), LABEL.get(s, s), fill=(20, 20, 20), font=font(16))
        for k in range(seeds):
            f = os.path.join(root, s, "images", f"{idx:03d}_s{k}.png")
            if os.path.exists(f):
                g.paste(Image.open(f).convert("RGB").resize((cell, cell), Image.LANCZOS), (label_w + k * cell, y))
    return g


def readme_figure(root, idx, text, out, font_dir=None, seeds=6, model="Krea 2", left="Krea 2 Turbo",
                  left_set="student", right_set="rediverse_a4"):
    """The README's figure layout (900x750): model badge and prompt, a grey left panel and a blue ReDiverse panel of
    2x3 seed-matched samples."""
    def f(name, size):
        if font_dir and os.path.exists(os.path.join(font_dir, name)):
            return ImageFont.truetype(os.path.join(font_dir, name), size)
        return font(size)
    W, H = 900, 750
    g = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(g)
    bold, reg = f("Cantarell-Bold.otf", 20), f("Cantarell-Regular.otf", 20)
    bw = d.textlength(model, font=bold) + 28
    pw = d.textlength(f"\u201c{text}\u201d", font=reg)
    x0 = (W - (bw + 14 + pw)) / 2
    d.rounded_rectangle((x0, 22, x0 + bw, 54), radius=16, fill=(22, 27, 38))
    d.text((x0 + 14, 26), model, fill="white", font=bold)
    d.text((x0 + bw + 14, 26), f"\u201c{text}\u201d", fill=(55, 60, 70), font=reg)
    for k, (title, setname, head, edge) in enumerate(((left, left_set, (100, 116, 139), (205, 212, 222)),
                                                      ("ReDiverse", right_set, (37, 99, 235), (37, 99, 235)))):
        px = 15 + k * 447
        d.rounded_rectangle((px, 64, px + 425, 735), radius=12, outline=edge, width=2, fill="white")
        d.rounded_rectangle((px, 64, px + 425, 104), radius=12, fill=head)
        d.rectangle((px, 90, px + 425, 104), fill=head)
        tw = d.textlength(title, font=f("Cantarell-Bold.otf", 22))
        d.text((px + (425 - tw) / 2, 72), title, fill="white", font=f("Cantarell-Bold.otf", 22))
        for s in range(seeds):
            r, c = divmod(s, 2)
            x, y = px + 8 + c * 207, 112 + r * 207
            im = Image.open(os.path.join(root, setname, "images", f"{idx:03d}_s{s}.png")).convert("RGB").resize((201, 201), Image.LANCZOS)
            m = Image.new("L", (201, 201), 0)
            ImageDraw.Draw(m).rounded_rectangle((0, 0, 200, 200), radius=10, fill=255)
            g.paste(im, (x, y), m)
    g.save(out, quality=90)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", required=True)
    ap.add_argument("--rows", default="student,rediverse_a4")
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--ids", default="0-47")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--cell", type=int, default=256)
    ap.add_argument("--composite-ids", default="0,4,8,14,22,35")
    ap.add_argument("--composite-seeds", type=int, default=6)
    ap.add_argument("--out", required=True)
    ap.add_argument("--readme", default=None, help="comma list of prompt ids: README-style figures readme_pNNN.jpg only")
    ap.add_argument("--font-dir", default=None, help="folder with Cantarell-Bold.otf / Cantarell-Regular.otf (optional)")
    a = ap.parse_args()
    from rediverse.data import parse_ids
    texts = [l.strip() for l in open(a.prompts) if l.strip()]
    rows = a.rows.split(",")
    os.makedirs(a.out, exist_ok=True)
    if a.readme:
        for i in parse_ids(a.readme):
            readme_figure(a.root, i, texts[i], os.path.join(a.out, f"readme_p{i:03d}.jpg"), a.font_dir, right_set=rows[-1])
        print(f"README figures -> {a.out}")
        return
    for i in parse_ids(a.ids):
        pair(a.root, rows, i, texts[i], a.seeds, a.cell, label_w=190).save(os.path.join(a.out, f"pair_p{i:03d}.jpg"), quality=90)
    parts = [pair(a.root, rows, i, texts[i], a.composite_seeds, 200, label_w=170) for i in parse_ids(a.composite_ids)]
    comp = Image.new("RGB", (parts[0].width, sum(p.height for p in parts) + 14 * (len(parts) - 1)), "white")
    y = 0
    for p in parts:
        comp.paste(p, (0, y))
        y += p.height + 14
    comp.save(os.path.join(a.out, "composite.jpg"), quality=88)
    print(f"{len(parse_ids(a.ids))} pair figures + composite.jpg -> {a.out}")


if __name__ == "__main__":
    main()
