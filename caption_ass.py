"""CapCut-style caption generator: SRT -> styled ASS with word-by-word karaoke pop.

Usage:
    from caption_ass import make_caption_ass
    ass_path = make_caption_ass("sub.srt", "sub_cap.ass",
                                font_size_px=68, text_color="#FFFFFF",
                                hl_color="#FFD400", box=True, pos="lower",
                                karaoke=True, play_w=1280, play_h=720)
"""

import re


def _ass_color(hexcolor):
    """'#RRGGBB' or 'RRGGBB' -> ASS &HAABBGGRR. Alpha 00 = opaque."""
    h = hexcolor.strip().lstrip("#")
    if len(h) != 6:
        h = "FFFFFF"
    r, g, b = h[0:2], h[2:4], h[4:6]
    return "&H00" + b + g + r


def _parse_srt(srt_path):
    with open(srt_path, "r", encoding="utf-8-sig", errors="replace") as f:
        raw = f.read()
    # strip BOM / stray
    blocks = re.split(r"\n\s*\n", raw.strip())
    cues = []
    _ts = re.compile(
        r"(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,.](\d+)"
    )
    for blk in blocks:
        lines = [l for l in blk.splitlines() if l.strip()]
        if len(lines) < 2:
            continue
        m = _ts.search(lines[1] if not lines[0].isdigit() else "\n".join(lines[1:]))
        # find the timing line (contains -->)
        tl = next((l for l in lines if "-->" in l), None)
        if not tl:
            continue
        m = _ts.search(tl)
        if not m:
            continue
        def _sec(a, b, c, d):
            return int(a) * 3600 + int(b) * 60 + int(c) + int(d) / 1000.0
        start = _sec(m.group(1), m.group(2), m.group(3), m.group(4))
        end = _sec(m.group(5), m.group(6), m.group(7), m.group(8))
        # text = lines after the timing line
        idx = next((i for i, l in enumerate(lines) if "-->" in l), 1)
        text = " ".join(lines[idx + 1:]).strip()
        if not text:
            continue
        text = re.sub(r"<[^>]+>", "", text)  # strip inline tags
        if end <= start:
            end = start + 0.6
        cues.append((start, end, text))
    return cues


def _esc(text):
    return text.replace("{", "(").replace("}", ")")


def _style_line(name, size, color, box, pos, play_h, font="Arial Black",
                outline_px=None, shadow_px=None):
    al = {"lower": 2, "center": 5, "top": 8}[pos]
    if box:
        border = 3  # opaque box behind text
        outline = 2
        shadow = 1
        back = "&H90000000"  # ~56% black
    else:
        border = 1  # outline + shadow only
        outline = outline_px if outline_px is not None else 3
        shadow = shadow_px if shadow_px is not None else 2
        back = "&H80000000"
    m = max(30, int(play_h * 0.06)) if size <= 0 else size
    return (f"Style: {name},{font},{m},{_ass_color(color)},{_ass_color(color)},"
            f"&H00000000,{back},1,0,0,0,100,100,0,0,{border},{outline},{shadow},"
            f"{al},40,40,40,1")


def make_caption_ass(srt_path, ass_path, font_size_px=0, text_color="#FFFFFF",
                     hl_color="#FFD400", box=True, pos="lower", karaoke=True,
                     font="Arial Black", glow=False, glow_color="#FFD400",
                     glow_strength=6, play_w=1280, play_h=720, style="highlight"):
    cues = _parse_srt(srt_path)
    _s = (style or "highlight").lower()
    # Map the UI's 4 caption styles to render behavior.
    do_karaoke = _s in ("highlight", "pop")   # word-by-word highlight/pop
    force_box = (_s == "box")                  # opaque box behind the line
    pop_scale = (_s == "pop")                  # current word scales up briefly
    _box = bool(box) or force_box
    L = []
    L.append("[Script Info]")
    L.append("ScriptType: v4.00+")
    L.append(f"PlayResX: {play_w}")
    L.append(f"PlayResY: {play_h}")
    L.append("WrapStyle: 0")
    L.append("ScaledBorderAndShadow: yes")
    L.append("")
    L.append("[V4+ Styles]")
    L.append("Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
             "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
             "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
             "Alignment, MarginL, MarginR, MarginV, Encoding")
    if glow:
        # thin outline on the crisp text so the halo shows through
        L.append(_style_line("Cap", font_size_px, text_color, _box, pos, play_h,
                             font, outline_px=1, shadow_px=1))
        L.append(_style_line("CapHl", font_size_px, hl_color, _box, pos, play_h,
                             font, outline_px=1, shadow_px=1))
        # glow underlayer is BIGGER + blurred so it peeks out around the text
        _gsize = int((font_size_px if font_size_px > 0 else int(play_h * 0.06)) * 1.25)
        L.append(_style_line("CapGlow", _gsize, glow_color, _box, pos, play_h,
                             font, outline_px=0, shadow_px=0))
    else:
        L.append(_style_line("Cap", font_size_px, text_color, _box, pos, play_h, font))
        L.append(_style_line("CapHl", font_size_px, hl_color, _box, pos, play_h, font))
    L.append("")
    L.append("[Events]")
    L.append("Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, "
             "Effect, Text")

    def _ts(t):
        ms = int(round(t * 1000))
        hh, ms = divmod(ms, 3600000)
        mm, ms = divmod(ms, 60000)
        ss, ms = divmod(ms, 1000)
        return "%d:%02d:%02d.%02d" % (hh, mm, ss, ms // 10)

    layer = 0
    for (start, end, text) in cues:
        words = text.split()
        if not words:
            continue
        line = " ".join(_esc(w) for w in words)
        layer += 1
        if glow:
            # layer 0: big blurred colored underlayer (the glow halo)
            L.append(f"Dialogue: 0,{_ts(start)},{_ts(end)},CapGlow,,0,0,0,,"
                     f"{{\\blur{int(glow_strength)}}}{line}")
            # layer 1: crisp white caption on top
            L.append(f"Dialogue: 1,{_ts(start)},{_ts(end)},Cap,,0,0,0,,{line}")
        else:
            L.append(f"Dialogue: 0,{_ts(start)},{_ts(end)},Cap,,0,0,0,,{line}")
        if karaoke and do_karaoke and len(words) > 1:
            step = (end - start) / len(words)
            hl_layer = 2 if glow else 1
            for i, w in enumerate(words):
                ws = start + i * step
                we = start + (i + 1) * step
                _fx = "{\\fscx120\\fscy120}" if pop_scale else ""
                # only the currently-spoken word is highlighted (overlay on top)
                L.append(f"Dialogue: {hl_layer},{_ts(ws)},{_ts(we)},CapHl,,0,0,0,,"
                         f"{_fx}{_esc(w)}")
    with open(ass_path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    return ass_path


if __name__ == "__main__":
    import sys
    make_caption_ass(sys.argv[1], sys.argv[2])
    print("wrote", sys.argv[2])
