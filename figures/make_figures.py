#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Architecture figures for BoseNet (bosonic Psiformer NQS with per-layer
lambda-conditioning + fixed K0 Jastrow, 2D dipolar bosons on a torus).

This script is the *source of truth* for the diagrams: it emits clean,
self-contained vector SVGs on a shared grid, with one visual language:

    indigo-blue  = learned network / amplitude path
    amber        = lambda-conditioning path (the generative mechanism)
    muted teal   = fixed, non-learned physics (K0 Jastrow, Hamiltonian)
    grey mono    = tensor shapes  [N x d]

Outputs (written next to this file):
    bosenet_architecture.svg   full pipeline + physics/VMC panel (overview / talks)
    bosenet_dataflow.svg       compact single-column PRB main figure
    bosenet_block.svg          compact conditioned-attention-block detail (inset)

Render to PDF/PNG with:  rsvg-convert -f pdf -o x.pdf x.svg   (see build_all()).
"""
import html
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))

# ----- palette (print: dark ink on white plate) -----------------------------
INK        = "#16202b"   # cool near-black
MUTED      = "#5d6b79"   # secondary text
FAINT      = "#8a97a4"   # tensor-shape grey
HAIR       = "#cdd7e1"   # hairlines
PLATE      = "#ffffff"
PANEL      = "#eef3f8"   # generic learned block fill (very light cool)

BLUE       = "#2f56d6"; BLUE_FILL = "#e7ecfb"; BLUE_INK = "#20419e"
AMBER      = "#c9760a"; AMBER_FILL= "#fbead2"; AMBER_LN = "#d9820f"
TEAL       = "#0e8a76"; TEAL_FILL = "#dcf1ec"; TEAL_INK = "#0b6d5d"

# ----- fonts (fontconfig names + fallbacks; rsvg resolves) ------------------
SANS  = "'Nimbus Sans','Helvetica Neue','Arial','DejaVu Sans',sans-serif"
SERIF = "'Nimbus Roman No9 L','Nimbus Roman','Times New Roman','DejaVu Serif',serif"
MONO  = "'DejaVu Sans Mono','Menlo','Consolas',monospace"

# dims used throughout the labels
DIMS = dict(N="N", d="256", D="16", L="4", H="4", dk="64")


# ======================================================================
# tiny SVG math typesetter:  supports  _{..} ^{..} _x ^x  and
#   \u{..} upright run, \i{..} italic run, \b{..} bold run, \c{..} colored-ink
# default run uses the style/family/fill passed to Text().
# ======================================================================
def _esc(s):
    return html.escape(s, quote=True)


def _frag(s, size, style, weight, fill, family, dy=0.0):
    if s == "":
        return ""
    a = [f'font-size="{size:.2f}"']
    if dy:
        a.append(f'dy="{dy:.2f}"')
    a.append(f'font-style="{style}"')
    a.append(f'font-weight="{weight}"')
    a.append(f'fill="{fill}"')
    a.append(f'font-family="{family}"')
    return f'<tspan {" ".join(a)}>{_esc(s)}</tspan>'


def _typeset(s, size, style, weight, fill, family):
    out = []
    i, n = 0, len(s)
    pending = 0.0  # baseline reset to fold into next fragment

    def emit(txt, sz, st, wt, fl, fam, dy):
        nonlocal pending
        out.append(_frag(txt, sz, st, wt, fl, fam, dy + pending))
        pending = 0.0

    buf = ""

    def flush():
        nonlocal buf
        if buf:
            emit(buf, size, style, weight, fill, family, 0.0)
            buf = ""

    while i < n:
        c = s[i]
        if c in "_^":
            flush()
            i += 1
            if i < n and s[i] == "{":
                j = s.index("}", i)
                grp = s[i + 1:j]
                i = j + 1
            else:
                grp = s[i]; i += 1
            shift = 0.30 * size if c == "_" else -0.42 * size
            emit(grp, size * 0.72, style, weight, fill, family, shift)
            pending = -shift  # restore baseline on next fragment
        elif c == "\\" and i + 1 < n and s[i + 1] in "uibc" and i + 2 < n and s[i + 2] == "{":
            flush()
            kind = s[i + 1]
            j = s.index("}", i + 2)
            grp = s[i + 2 + 1 - 1 + 1:j] if False else s[i + 3:j]
            i = j + 1
            st = "normal" if kind in "ub" else "italic"
            wt = "700" if kind == "b" else weight
            fl = fill
            fam = family
            emit(grp, size, st, wt, fl, fam, 0.0)
        else:
            buf += c; i += 1
    flush()
    return "".join(out)


def Text(x, y, s, size=12, anchor="middle", fill=INK, family=SANS,
         style="normal", weight="400", ls=None):
    a = [f'x="{x:.1f}"', f'y="{y:.1f}"', f'text-anchor="{anchor}"']
    if ls is not None:
        a.append(f'letter-spacing="{ls}"')
    inner = _typeset(s, size, style, weight, fill, family)
    return f'<text {" ".join(a)}>{inner}</text>'


# ----- primitives -----------------------------------------------------------
def rrect(x, y, w, h, rx=7, fill=PLATE, stroke=HAIR, sw=1.4, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
            f'rx="{rx}" ry="{rx}" fill="{fill}" stroke="{stroke}" '
            f'stroke-width="{sw}"{d}/>')


def line(x1, y1, x2, y2, stroke=INK, sw=1.6, dash=None, marker="ink"):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    m = f' marker-end="url(#ar-{marker})"' if marker else ""
    return (f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{stroke}" stroke-width="{sw}" stroke-linecap="round"{d}{m}/>')


def poly(points, stroke=INK, sw=1.6, dash=None, marker="ink"):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    m = f' marker-end="url(#ar-{marker})"' if marker else ""
    pts = " ".join(f"{px:.1f},{py:.1f}" for px, py in points)
    return (f'<polyline points="{pts}" fill="none" stroke="{stroke}" '
            f'stroke-width="{sw}" stroke-linejoin="round" '
            f'stroke-linecap="round"{d}{m}/>')


def dot(x, y, r=2.6, fill=INK):
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{fill}"/>'


def oplus(x, y, r=8.5, stroke=TEAL, fill=PLATE, sw=1.8):
    return (f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{sw}"/>'
            f'<line x1="{x-r*0.55:.1f}" y1="{y:.1f}" x2="{x+r*0.55:.1f}" y2="{y:.1f}" '
            f'stroke="{stroke}" stroke-width="{sw}"/>'
            f'<line x1="{x:.1f}" y1="{y-r*0.55:.1f}" x2="{x:.1f}" y2="{y+r*0.55:.1f}" '
            f'stroke="{stroke}" stroke-width="{sw}"/>')


def shape_tag(x, y, txt, anchor="middle"):
    return Text(x, y, txt, size=9.5, anchor=anchor, fill=FAINT, family=MONO)


DEFS = f'''<defs>
  {"".join(
    f'<marker id="ar-{name}" viewBox="0 0 10 10" refX="8.5" refY="5" '
    f'markerWidth="7.5" markerHeight="7.5" orient="auto-start-reverse">'
    f'<path d="M0,0 L10,5 L0,10 L2.6,5 z" fill="{col}"/></marker>'
    for name, col in [("ink", INK), ("blue", BLUE), ("amber", AMBER_LN),
                      ("teal", TEAL), ("muted", MUTED)])}
</defs>'''


def svg(w, h, body, title):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
            f'width="{w}" height="{h}" font-family="{SANS}" '
            f'role="img" aria-label="{_esc(title)}">'
            f'<rect x="0" y="0" width="{w}" height="{h}" fill="{PLATE}"/>'
            f'{DEFS}{body}</svg>')


# helper: a titled box with an eyebrow label line + one or two content lines
def node(x, y, w, h, eyebrow, lines, fill=PANEL, stroke=HAIR, sw=1.4,
         ecol=INK, tcol=INK, efam=SANS, eweight="700", esize=11.5,
         lsize=10.5, lfam=SERIF, lstyle="italic", rx=7, dash=None,
         els=".2"):
    out = [rrect(x, y, w, h, rx=rx, fill=fill, stroke=stroke, sw=sw, dash=dash)]
    cx = x + w / 2
    if eyebrow:
        out.append(Text(cx, y + 15.5, eyebrow, size=esize, fill=ecol,
                        family=efam, weight=eweight, ls=els))
    ly = y + (30 if eyebrow else 18)
    for ln in lines:
        txt, st = (ln if isinstance(ln, tuple) else (ln, lstyle))
        out.append(Text(cx, ly, txt, size=lsize, fill=tcol, family=lfam,
                        style=st))
        ly += lsize + 4.5
    return "".join(out), (cx, y + h / 2)


# ======================================================================
# FIGURE 1 — compact single-column data-flow (PRB main figure)
# ======================================================================
def fig_dataflow():
    W, Hh = 388, 616
    b = []
    cx = 190          # main spine centre
    bw = 196          # main box width
    bx = cx - bw / 2  # 90

    # eyebrow
    b.append(Text(cx, 26, "BoseNet  —  conditioned bosonic Psiformer",
                  size=12.5, fill=INK, weight="700", ls=".3"))

    y = 40
    # 1. inputs
    n, _ = node(bx, y, bw, 40, "INPUTS",
                [("R = (r_1, …, r_N) ∈ 𝕋²,  spins σ,  coupling λ", "italic")],
                fill="#f6f8fb", stroke=HAIR, lsize=10)
    b.append(n); y_in = y + 40

    y = y_in + 26
    # 2. features + embed
    n, _ = node(bx, y, bw, 46, "PERIODIC FEATURES + EMBED",
                [("f_i = [ sin2πs_i , cos2πs_i ]  ,  s_i = A^{-1} r_i",
                  "italic"),
                 ("h_i^{(0)} = W_{emb} f_i", "italic")],
                fill=BLUE_FILL, stroke="#b9c6ef", ecol=BLUE_INK, sw=1.5)
    b.append(n); y_feat = y + 46

    # arrows in
    b.append(line(cx, y_in, cx, y_feat - 46 + 0.0, stroke=INK, sw=1.7))  # inputs->feat placeholder
    b[-1] = line(cx, y_in, cx, y, stroke=INK, sw=1.7)
    b.append(shape_tag(cx + 108, (y_in + y) / 2 + 3, "[N×d]", anchor="start"))

    # 3. lambda-embedding MLP (left tuck)  -> feeds injection row
    lam_x, lam_y, lam_w, lam_h = 8, 250, 92, 52
    n, _ = node(lam_x, lam_y, lam_w, lam_h, "λ-EMBED MLP",
                [("p_λ = ", "italic"), ("MLP_λ(λ)", "italic"),
                 ("∈ ℝ^{d}", "italic")],
                fill=AMBER_FILL, stroke="#e8c48c", ecol=AMBER, sw=1.5,
                esize=10.5, lsize=9.5)
    b.append(n)

    y = y_feat + 26
    # 4. conditioned block ×L  (three inner rows)
    blk_h = 118
    b.append(rrect(bx, y, bw, blk_h, rx=9, fill="#f4f7fb",
                   stroke="#aeb9c6", sw=1.6))
    b.append(Text(bx + 12, y + 16, "CONDITIONED BLOCK",
                  size=10.5, fill=INK, weight="700", anchor="start", ls=".2"))
    b.append(Text(bx + bw - 10, y + 16, "× L = 4", size=11, fill=BLUE_INK,
                  weight="700", anchor="end", family=SANS))
    # inner rows
    iw = bw - 24
    ix = bx + 12
    ry = y + 26
    # row a: MHSA
    b.append(rrect(ix, ry, iw, 26, rx=5, fill=BLUE_FILL, stroke="#c2cef0", sw=1.2))
    b.append(Text(ix + iw / 2, ry + 11.5, "Multi-head self-attention  +  residual",
                  size=9.6, fill=BLUE_INK, weight="600"))
    b.append(Text(ix + iw / 2, ry + 22, "softmax(Q K^{T} / √d_k) V   ( h=4, d_k=64 )",
                  size=8.8, fill=MUTED, family=SERIF, style="italic"))
    # row b: MLP
    ry2 = ry + 32
    b.append(rrect(ix, ry2, iw, 22, rx=5, fill=BLUE_FILL, stroke="#c2cef0", sw=1.2))
    b.append(Text(ix + iw / 2, ry2 + 14, "MLP  +  residual   ( + LayerNorm )",
                  size=9.6, fill=BLUE_INK, weight="600"))
    # row c: lambda injection
    ry3 = ry2 + 28
    b.append(rrect(ix, ry3, iw, 24, rx=5, fill=AMBER_FILL, stroke="#e8c48c", sw=1.4))
    b.append(oplus(ix + 15, ry3 + 12, r=7.5, stroke=AMBER))
    b.append(Text(ix + 30, ry3 + 15, "λ-inject:  H ← H + tanh(W_ℓ p_λ)",
                  size=9.5, fill=AMBER, weight="600", anchor="start",
                  family=SANS))
    y_blk = y + blk_h

    # feature->block
    b.append(line(cx, y_feat, cx, y, stroke=INK, sw=1.7))
    b.append(shape_tag(cx + 108, (y_feat + y) / 2 + 3, "[N×d]", anchor="start"))
    # p_lambda rail: from lambda box right edge to injection row
    b.append(poly([(lam_x + lam_w, lam_y + lam_h / 2), (ix - 6, lam_y + lam_h / 2),
                   (ix - 6, ry3 + 12), (ix - 1, ry3 + 12)],
                  stroke=AMBER_LN, sw=1.7, marker="amber"))

    y = y_blk + 24
    # 5. orbital head
    n, _ = node(bx, y, bw, 40, "ORBITAL HEAD",
                [("Φ = H^{(L)} W_{orb} ∈ ℝ^{N×D},  D = 16", "italic")],
                fill=BLUE_FILL, stroke="#b9c6ef", ecol=BLUE_INK, sw=1.5, lsize=10)
    b.append(n); y_orb = y + 40
    b.append(line(cx, y_blk, cx, y, stroke=INK, sw=1.7))
    b.append(shape_tag(cx + 108, (y_blk + y) / 2 + 3, "[N×d]", anchor="start"))

    y = y_orb + 24
    # 6. boson product head
    n, _ = node(bx, y, bw, 52, "BOSONIC PRODUCT HEAD",
                [("log|ψ| = log Σ_d exp Σ_i φ_d(r_i;λ)", "italic"),
                 ("sign = +1   (nodeless, symmetric)", "italic")],
                fill=BLUE_FILL, stroke="#b9c6ef", ecol=BLUE_INK, sw=1.6,
                lsize=10)
    b.append(n); y_bos = y + 52
    b.append(line(cx, y_orb, cx, y, stroke=INK, sw=1.7))
    b.append(shape_tag(cx + 108, (y_orb + y) / 2 + 3, "[N×D]", anchor="start"))

    # 7. K0 Jastrow (right tuck) into oplus
    jx, jy, jw, jh = 274, y_bos + 20, 86, 56
    n, _ = node(jx, jy, jw, jh, "FIXED K₀",
                [("+ Σ_{i<j}", "italic"), ("log K_0(2√(λ/r̃_{ij}))", "italic"),
                 ("(dipole cusp)", "italic")],
                fill=TEAL_FILL, stroke="#a7d8cd", ecol=TEAL_INK, sw=1.5,
                esize=10.5, lsize=9.2)
    b.append(n)

    # oplus join under boson head
    joy = y_bos + 22
    b.append(line(cx, y_bos, cx, joy - 8, stroke=BLUE, sw=1.8, marker=None))
    b.append(oplus(cx, joy, r=9, stroke=TEAL))
    # jastrow -> oplus
    b.append(poly([(jx, jy + jh / 2), (cx + 9, joy)], stroke=TEAL_LN if False else TEAL,
                  sw=1.7, marker="teal"))

    y = joy + 24
    # 8. output
    b.append(rrect(bx + 20, y, bw - 40, 38, rx=9, fill="#0f1b2e",
                   stroke="#0f1b2e", sw=1.4))
    b.append(Text(cx, y + 16, "OUTPUT  (log-amplitude)", size=9.5,
                  fill="#93a8d6", weight="700", ls=".2"))
    b.append(Text(cx, y + 31, "log ψ_θ(R; λ)", size=13, fill="#ffffff",
                  family=SERIF, style="italic", weight="600"))
    b.append(line(cx, joy + 9, cx, y, stroke=BLUE, sw=1.9, marker="blue"))
    y_out = y + 38

    # legend
    ly = y_out + 22
    b.append(Text(24, ly, "KEY", size=9, fill=MUTED, weight="700",
                  anchor="start", ls=".4"))
    for sx_, lab, col in [(54, "learned network", BLUE),
                          (182, "λ-conditioning", AMBER_LN),
                          (300, "fixed physics", TEAL)]:
        b.append(f'<rect x="{sx_}" y="{ly-8}" width="12" height="8.5" rx="2" '
                 f'fill="{col}"/>')
        b.append(Text(sx_ + 16, ly, lab, size=8.7, fill=INK, anchor="start"))

    return svg(W, Hh, "".join(b),
               "BoseNet conditioned bosonic Psiformer data-flow, inputs to log psi")


# teal line colour alias (used above)
TEAL_LN = TEAL


# ======================================================================
# FIGURE 2 — conditioned attention block detail (compact inset)
# ======================================================================
def fig_block():
    W, Hh = 566, 212
    b = []
    b.append(Text(20, 24, "Conditioned Psiformer block  (layer ℓ)",
                  size=12.5, fill=INK, weight="700", anchor="start", ls=".2"))

    cyr = 110        # main row centre-y
    # input token matrix
    ix = 20
    b.append(rrect(ix, cyr - 20, 66, 40, rx=6, fill="#f6f8fb", stroke=HAIR))
    b.append(Text(ix + 33, cyr - 2, "H^{(ℓ−1)}", size=11.5, fill=INK,
                  family=SERIF, style="italic"))
    b.append(shape_tag(ix + 33, cyr + 13, "N×d"))

    # MHSA
    ax = 112
    aw = 120
    b.append(rrect(ax, cyr - 26, aw, 52, rx=7, fill=BLUE_FILL, stroke="#b9c6ef", sw=1.6))
    b.append(Text(ax + aw / 2, cyr - 10, "Multi-head", size=11, fill=BLUE_INK,
                  weight="700"))
    b.append(Text(ax + aw / 2, cyr + 4, "self-attention", size=11, fill=BLUE_INK,
                  weight="700"))
    b.append(Text(ax + aw / 2, cyr + 18, "softmax(QK^{T}/√d_k)V", size=8.6,
                  fill=MUTED, family=SERIF, style="italic"))
    # residual add after attention
    a1 = ax + aw + 20
    b.append(oplus(a1, cyr, r=8.5, stroke=INK))
    # MLP
    mx = a1 + 22
    mw = 96
    b.append(rrect(mx, cyr - 26, mw, 52, rx=7, fill=BLUE_FILL, stroke="#b9c6ef", sw=1.6))
    b.append(Text(mx + mw / 2, cyr - 6, "MLP", size=12, fill=BLUE_INK, weight="700"))
    b.append(Text(mx + mw / 2, cyr + 10, "+ LayerNorm", size=9, fill=MUTED,
                  family=SANS))
    # residual add after MLP
    m1 = mx + mw + 20
    b.append(oplus(m1, cyr, r=8.5, stroke=INK))
    # lambda injection add
    inj = m1 + 42
    b.append(oplus(inj, cyr, r=10, stroke=AMBER))
    b.append(Text(inj, cyr - 20, "+ tanh(W_ℓ p_λ)", size=9, fill=AMBER,
                  family=SERIF, style="italic"))
    # output
    ox = inj + 30
    ow = 76
    b.append(rrect(ox, cyr - 20, ow, 40, rx=6, fill="#f6f8fb", stroke=HAIR))
    b.append(Text(ox + ow / 2, cyr - 2, "H^{(ℓ)}", size=11.5, fill=INK,
                  family=SERIF, style="italic"))
    b.append(shape_tag(ox + ow / 2, cyr + 13, "N×d"))

    # main flow lines
    b.append(line(ix + 66, cyr, ax, cyr, stroke=INK, sw=1.7))
    b.append(line(ax + aw, cyr, a1 - 8.5, cyr, stroke=INK, sw=1.7, marker=None))
    b.append(line(a1 + 8.5, cyr, mx, cyr, stroke=INK, sw=1.7))
    b.append(line(mx + mw, cyr, m1 - 8.5, cyr, stroke=INK, sw=1.7, marker=None))
    b.append(line(m1 + 8.5, cyr, inj - 10, cyr, stroke=INK, sw=1.7, marker=None))
    b.append(line(inj + 10, cyr, ox, cyr, stroke=BLUE, sw=1.8, marker="blue"))

    # residual skip arcs (over the top)
    yr = cyr - 42
    b.append(poly([(ax - 6, cyr - 14), (ax - 6, yr), (a1, yr), (a1, cyr - 8.5)],
                  stroke=MUTED, sw=1.4, dash="3 3", marker="muted"))
    b.append(Text((ax - 6 + a1) / 2, yr - 4, "residual", size=8.5, fill=MUTED))
    b.append(poly([(mx - 6, cyr - 14), (mx - 6, yr), (m1, yr), (m1, cyr - 8.5)],
                  stroke=MUTED, sw=1.4, dash="3 3", marker="muted"))
    b.append(Text((mx - 6 + m1) / 2, yr - 4, "residual", size=8.5, fill=MUTED))

    # lambda token feeding injection from below
    ly = cyr + 64
    lbw = 176
    b.append(rrect(inj - lbw / 2, ly - 17, lbw, 34, rx=7, fill=AMBER_FILL,
                   stroke="#e8c48c", sw=1.6))
    b.append(Text(inj, ly - 2, "p_λ = MLP_λ(λ)", size=11, fill=AMBER,
                  weight="600", family=SERIF, style="italic"))
    b.append(Text(inj, ly + 12, "shared token, injected every layer", size=8.6,
                  fill=MUTED))
    b.append(poly([(inj, ly - 17), (inj, cyr + 10)], stroke=AMBER_LN, sw=1.8,
                  marker="amber"))

    return svg(W, Hh, "".join(b),
               "Conditioned Psiformer block: self-attention, MLP, and per-layer lambda injection")


# ======================================================================
# FIGURE 3 — full architecture overview (landscape, with physics/VMC)
# ======================================================================
def fig_architecture():
    W, Hh = 1200, 792
    b = []
    b.append(Text(40, 34, "BoseNet", size=23, fill=INK, weight="700",
                  anchor="start", ls=".2"))
    b.append(Text(40, 54, "bosonic Psiformer neural quantum state with per-layer "
                  "λ-conditioning and a fixed K₀ Jastrow  ·  2D dipolar bosons on a torus",
                  size=12.5, fill=MUTED, anchor="start"))

    # legend (top-right)
    lx, ly = 858, 30
    b.append(rrect(lx, ly, 306, 58, rx=8, fill="#f7f9fc", stroke=HAIR, sw=1.2))
    items = [(BLUE, "learned network"), (AMBER_LN, "λ-conditioning"),
             (TEAL, "fixed physics"), (FAINT, "tensor shape")]
    px, py = lx + 16, ly + 22
    for i, (col, lab) in enumerate(items):
        cxp = px + (i % 2) * 150
        cyp = py + (i // 2) * 22
        if lab == "tensor shape":
            b.append(shape_tag(cxp + 6, cyp + 4, "N×d"))
        else:
            b.append(f'<rect x="{cxp}" y="{cyp-8}" width="15" height="9" rx="2" '
                     f'fill="{col}"/>')
        b.append(Text(cxp + 22, cyp + 4, lab, size=10.5, fill=INK, anchor="start"))

    midy = 300  # main pipeline centre-line

    # ---- Inputs (left) ----
    xi = 40
    b.append(Text(xi, 108, "INPUTS", size=11, fill=MUTED, weight="700",
                  anchor="start", ls=".3"))
    n, _ = node(xi, 122, 168, 44, "",
                [("R = (r_1,…,r_N) ∈ 𝕋²", "italic")],
                fill="#f6f8fb", stroke=HAIR, lsize=11)
    b.append(n)
    n, _ = node(xi, 178, 168, 34, "", [("spins  σ ∈ {↑,↓}^{N}", "italic")],
                fill="#f6f8fb", stroke=HAIR, lsize=11)
    b.append(n)
    n, _ = node(xi, 224, 168, 40, "", [("coupling  λ  (dipole length)", "italic")],
                fill=AMBER_FILL, stroke="#e8c48c", lsize=11, ecol=AMBER)
    b.append(n)
    b.append(Text(xi + 84, 252, "λ", size=13, fill=AMBER, family=SERIF,
                  style="italic", weight="700"))

    # ---- Features + embed ----
    xf = 250
    b.append(rrect(xf, 150, 150, 96, rx=8, fill=BLUE_FILL, stroke="#b9c6ef", sw=1.6))
    b.append(Text(xf + 75, 168, "PERIODIC", size=10.5, fill=BLUE_INK,
                  weight="700", ls=".2"))
    b.append(Text(xf + 75, 181, "FEATURES + EMBED", size=10.5, fill=BLUE_INK,
                  weight="700", ls=".2"))
    b.append(Text(xf + 75, 202, "f_i=[sin2πs_i,cos2πs_i]", size=9.6, fill=INK,
                  family=SERIF, style="italic"))
    b.append(Text(xf + 75, 216, "s_i = A^{-1} r_i", size=9.6, fill=INK,
                  family=SERIF, style="italic"))
    b.append(Text(xf + 75, 233, "h_i^{(0)} = W_{emb} f_i", size=10, fill=BLUE_INK,
                  family=SERIF, style="italic"))
    # arrows inputs->features
    b.append(line(xi + 168, 144, xf, 176, stroke=INK, sw=1.7))
    b.append(line(xi + 168, 195, xf, 195, stroke=INK, sw=1.7))

    # ---- lambda-embed MLP (top rail) ----
    xl = 470
    b.append(rrect(xl, 96, 190, 40, rx=8, fill=AMBER_FILL, stroke="#e8c48c", sw=1.6))
    b.append(Text(xl + 95, 112, "λ-EMBEDDING MLP", size=10.5, fill=AMBER,
                  weight="700", ls=".2"))
    b.append(Text(xl + 95, 128, "p_λ = MLP_λ(λ) ∈ ℝ^{d}", size=10, fill=INK,
                  family=SERIF, style="italic"))
    # lambda from input up to MLP
    b.append(poly([(xi + 84, 224), (xi + 84, 116), (xl, 116)], stroke=AMBER_LN,
                  sw=1.7, marker="amber"))

    # ---- Conditioned Psiformer stack (center) ----
    sx, sw_, sy, sh = 462, 300, 168, 250
    b.append(rrect(sx, sy, sw_, sh, rx=11, fill="#f3f6fb", stroke="#aeb9c6", sw=1.7))
    b.append(Text(sx + 14, sy + 20, "CONDITIONED PSIFORMER", size=11.5, fill=INK,
                  weight="700", anchor="start", ls=".2"))
    b.append(Text(sx + sw_ - 12, sy + 20, "× L = 4 layers", size=11.5,
                  fill=BLUE_INK, weight="700", anchor="end"))
    # inner block (one layer, expanded)
    iw = sw_ - 40
    ix = sx + 20
    r1 = sy + 36
    b.append(rrect(ix, r1, iw, 46, rx=7, fill=BLUE_FILL, stroke="#b9c6ef", sw=1.5))
    b.append(Text(ix + iw / 2, r1 + 19, "Multi-head self-attention  +  residual",
                  size=11, fill=BLUE_INK, weight="700"))
    b.append(Text(ix + iw / 2, r1 + 35, "Attn(Q,K,V) = softmax(Q K^{T}/√d_k) V   "
                  "( h = 4 heads, d_k = 64 )", size=9.2, fill=MUTED,
                  family=SERIF, style="italic"))
    r2 = r1 + 58
    b.append(rrect(ix, r2, iw, 38, rx=7, fill=BLUE_FILL, stroke="#b9c6ef", sw=1.5))
    b.append(Text(ix + iw / 2, r2 + 16, "MLP  +  residual   ( + LayerNorm )",
                  size=11, fill=BLUE_INK, weight="700"))
    b.append(Text(ix + iw / 2, r2 + 30, "position-free → permutation-equivariant "
                  "in the N particles", size=9, fill=MUTED))
    r3 = r2 + 50
    b.append(rrect(ix, r3, iw, 40, rx=7, fill=AMBER_FILL, stroke="#e8c48c", sw=1.6))
    b.append(oplus(ix + 20, r3 + 20, r=9, stroke=AMBER))
    b.append(Text(ix + 38, r3 + 17, "λ-injection  (per layer)", size=10.5,
                  fill=AMBER, weight="700", anchor="start"))
    b.append(Text(ix + 38, r3 + 32, "H ← H + tanh(W_ℓ p_λ)", size=10, fill=AMBER,
                  anchor="start", family=SERIF, style="italic"))
    # vertical flow within block
    b.append(line(ix + iw / 2, r1 + 46, ix + iw / 2, r2, stroke=INK, sw=1.5))
    b.append(line(ix + iw / 2, r2 + 38, ix + iw / 2, r3, stroke=INK, sw=1.5))
    # repeat glyph
    b.append(Text(sx + sw_ - 14, sy + sh - 12, "↻ shared weights reused each layer",
                  size=9, fill=MUTED, anchor="end", style="italic"))

    # p_lambda rail from MLP down into injection row (drops each layer)
    railx = sx + sw_ + 0
    b.append(poly([(xl + 190, 116), (railx + 22, 116), (railx + 22, r3 + 20),
                   (ix + iw, r3 + 20)], stroke=AMBER_LN, sw=1.8, marker="amber"))
    b.append(Text(railx + 40, (116 + r3) / 2, "p_λ", size=11.5, fill=AMBER,
                  family=SERIF, style="italic", weight="700", anchor="start"))
    b.append(Text(railx + 40, (116 + r3) / 2 + 15, "(each layer ℓ)", size=8.6,
                  fill=MUTED, anchor="start"))

    # features -> stack
    b.append(line(xf + 150, 197, sx, sy + 60, stroke=INK, sw=1.8, marker="ink"))
    b.append(shape_tag(xf + 150 + 26, 188, "[N×d]", anchor="start"))

    # ---- Heads (right of stack) ----
    hx = 800
    # orbital head
    b.append(rrect(hx, 168, 150, 62, rx=8, fill=BLUE_FILL, stroke="#b9c6ef", sw=1.6))
    b.append(Text(hx + 75, 186, "ORBITAL HEAD", size=10.5, fill=BLUE_INK,
                  weight="700", ls=".2"))
    b.append(Text(hx + 75, 205, "Φ = H^{(L)} W_{orb}", size=10.5, fill=INK,
                  family=SERIF, style="italic"))
    b.append(Text(hx + 75, 221, "∈ ℝ^{N×D},  D = 16", size=9.6, fill=MUTED,
                  family=SERIF, style="italic"))
    # boson product head
    b.append(rrect(hx, 250, 150, 78, rx=8, fill=BLUE_FILL, stroke="#a9b8ea", sw=1.8))
    b.append(Text(hx + 75, 268, "BOSONIC PRODUCT HEAD", size=10, fill=BLUE_INK,
                  weight="700", ls=".15"))
    b.append(Text(hx + 75, 288, "log|ψ| =", size=10.5, fill=INK, family=SERIF,
                  style="italic"))
    b.append(Text(hx + 75, 303, "log Σ_d exp Σ_i φ_d(r_i;λ)", size=10.5, fill=INK,
                  family=SERIF, style="italic"))
    b.append(Text(hx + 75, 320, "sign = +1  (nodeless)", size=9.2, fill=TEAL_INK,
                  weight="600"))
    # stack -> orbital -> boson
    b.append(line(sx + sw_, sy + 125, hx, 199, stroke=INK, sw=1.8, marker="ink"))
    b.append(shape_tag(sx + sw_ + 26, sy + 112, "[N×d]", anchor="start"))
    b.append(line(hx + 75, 230, hx + 75, 250, stroke=INK, sw=1.7))
    b.append(shape_tag(hx + 110, 244, "[N×D]", anchor="start"))

    # ---- K0 Jastrow ----
    jx = 800
    b.append(rrect(jx, 356, 150, 74, rx=8, fill=TEAL_FILL, stroke="#a7d8cd", sw=1.6))
    b.append(Text(jx + 75, 374, "FIXED K₀ JASTROW", size=10.5, fill=TEAL_INK,
                  weight="700", ls=".15"))
    b.append(Text(jx + 75, 393, "+ Σ_{i<j} log K_0(2√(λ/r̃_{ij}))", size=9.8,
                  fill=INK, family=SERIF, style="italic"))
    b.append(Text(jx + 75, 409, "exact 2D-dipole contact cusp;", size=9, fill=MUTED))
    b.append(Text(jx + 75, 422, "non-trainable, r̃ = periodic dist.", size=9,
                  fill=MUTED))

    # ---- oplus + output ----
    ojx = hx + 190
    ojy = 289
    b.append(oplus(ojx, ojy, r=11, stroke=TEAL))
    b.append(line(hx + 150, 289, ojx - 11, 289, stroke=BLUE, sw=1.9, marker=None))
    b.append(poly([(jx + 150, 393), (ojx, 393), (ojx, ojy + 11)], stroke=TEAL,
                  sw=1.8, marker="teal"))
    # output box
    b.append(rrect(ojx + 26, 262, 150, 54, rx=10, fill="#0f1b2e", stroke="#0f1b2e"))
    b.append(Text(ojx + 26 + 75, 282, "OUTPUT", size=10, fill="#93a8d6",
                  weight="700", ls=".3"))
    b.append(Text(ojx + 26 + 75, 302, "log ψ_θ(R; λ)", size=15, fill="#ffffff",
                  family=SERIF, style="italic", weight="600"))
    b.append(line(ojx + 11, 289, ojx + 26, 289, stroke=BLUE, sw=1.9, marker="blue"))

    # ================= PHYSICS / VMC bottom strip =================
    py = 560
    b.append(line(40, py - 16, 1160, py - 16, stroke=HAIR, sw=1.2, marker=None))
    b.append(Text(40, py + 2, "VARIATIONAL MONTE CARLO  (how the ansatz is trained)",
                  size=11, fill=MUTED, weight="700", anchor="start", ls=".3"))

    py2 = py + 18
    cards = [
        (40, "HAMILTONIAN (PBC)",
         ["Ĥ = −½ Σ_i ∇_i² + λ Σ_{i<j} (r_{ij}+a)^{-3}",
          "Ewald summation;  a = 0 (bare) or 0.1"], TEAL_INK, TEAL_FILL, "#a7d8cd"),
        (330, "LOCAL ENERGY",
         ["E_L(R) = Ĥ ψ_θ / ψ_θ",
          "T_L = −½(∇²log|ψ| + |∇log|ψ||²)"], INK, "#f6f8fb", HAIR),
        (620, "OPTIMISER",
         ["min_θ ⟨E_L⟩_{|ψ|²}  (VMC)",
          "KFAC natural gradient"], BLUE_INK, BLUE_FILL, "#b9c6ef"),
        (910, "GENERATIVE BATCH",
         ["walkers split over {λ_k} each step",
          "per-λ baseline + per-λ clip"], AMBER, AMBER_FILL, "#e8c48c"),
    ]
    cw = 270
    for cxp, title, lines, ecol, fillc, strokec in cards:
        b.append(rrect(cxp, py2, cw, 74, rx=8, fill=fillc, stroke=strokec, sw=1.4))
        b.append(Text(cxp + 14, py2 + 20, title, size=10.5, fill=ecol,
                      weight="700", anchor="start", ls=".15"))
        yy = py2 + 40
        for ln in lines:
            b.append(Text(cxp + 14, yy, ln, size=10, fill=INK, family=SERIF,
                          style="italic", anchor="start"))
            yy += 16
    # connect output down to E_L card
    b.append(poly([(ojx + 26 + 75, 316), (ojx + 26 + 75, py2 - 14),
                   (330 + cw / 2, py2 - 14), (330 + cw / 2, py2)],
                  stroke=INK, sw=1.6, dash="4 3", marker="ink"))
    b.append(Text(ojx + 26 + 90, 336, "ψ_θ", size=11, fill=INK, family=SERIF,
                  style="italic", anchor="start"))
    # chain arrows between cards
    for xA in (310, 600, 890):
        b.append(line(xA, py2 + 37, xA + 20, py2 + 37, stroke=MUTED, sw=1.5,
                      marker="muted"))

    return svg(W, Hh, "".join(b),
               "BoseNet full architecture: inputs, periodic features, conditioned "
               "Psiformer, bosonic product head, K0 Jastrow, output log psi, and "
               "the variational Monte Carlo training panel")


# ======================================================================
def build_all():
    figs = {
        "bosenet_dataflow.svg": fig_dataflow(),
        "bosenet_block.svg": fig_block(),
        "bosenet_architecture.svg": fig_architecture(),
    }
    for name, s in figs.items():
        p = os.path.join(HERE, name)
        with open(p, "w") as f:
            f.write(s)
        print("wrote", name, f"({len(s)} bytes)")
    return list(figs)


def render(names):
    for name in names:
        base = os.path.join(HERE, name[:-4])
        for fmt, extra in (("pdf", []), ("png", ["-z", "2.6"])):
            out = f"{base}.{fmt}"
            cmd = ["rsvg-convert", "-f", fmt, "-o", out] + extra + [os.path.join(HERE, name)]
            r = subprocess.run(cmd, capture_output=True, text=True)
            print(("OK " if r.returncode == 0 else "ERR ") + out,
                  r.stderr.strip()[:200])


if __name__ == "__main__":
    names = build_all()
    render(names)
