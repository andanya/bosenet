#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Assemble the standalone HTML viewer for the BoseNet figures.

Reads the three SVGs produced by make_figures.py, namespaces their internal
marker ids (so three inline SVGs coexist in one document), and writes
bosenet_architecture_viewer.html — a theme-aware page with white figure plates.
This HTML is also published as a Claude Artifact.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "bosenet_architecture_viewer.html")
MARKERS = ["ink", "blue", "amber", "teal", "muted"]


def load_svg(name, k):
    with open(os.path.join(HERE, name)) as f:
        s = f.read()
    # unique-ify marker ids per embedded figure
    for m in MARKERS:
        s = s.replace(f'id="ar-{m}"', f'id="f{k}-ar-{m}"')
        s = s.replace(f'url(#ar-{m})', f'url(#f{k}-ar-{m})')
    # let CSS size it
    s = s.replace('width="', 'data-w="', 1)
    s = s.replace('height="', 'data-h="', 1)
    return s


ARCH = load_svg("bosenet_architecture.svg", 1)
FLOW = load_svg("bosenet_dataflow.svg", 2)
BLOCK = load_svg("bosenet_block.svg", 3)

NOTATION = [
    ("R = (r₁,…,r_N)", "walker: N particle positions on the 2-torus 𝕋²"),
    ("λ", "interaction strength (dipole length) — the conditioning variable"),
    ("σ", "particle spins (spin-polarised here)"),
    ("sᵢ = A⁻¹rᵢ", "fractional coordinates; A = lattice matrix (columns = primitive vectors)"),
    ("H⁽ˡ⁾ ∈ ℝ^{N×d}", "token matrix after block ℓ; embedding dimension d = 256"),
    ("p_λ = MLP_λ(λ) ∈ ℝ^d", "shared λ-conditioning token"),
    ("L = 4", "number of conditioned Psiformer blocks"),
    ("h = 4, d_k = 64", "attention heads and per-head dimension (d = h·d_k)"),
    ("Φ = H⁽ᴸ⁾W_orb ∈ ℝ^{N×D}", "orbital logits; D = 16 orbitals per particle"),
    ("φ_d(rᵢ;λ)", "orbital d of particle i"),
    ("ψ_θ(R;λ)", "positive, permutation-symmetric bosonic amplitude"),
    ("K₀ , r̃ᵢⱼ", "modified Bessel function (fixed Jastrow) ; smooth periodic distance"),
    ("a", "interaction softening: 0 (bare 1/r³) or 0.1 (regularised 1/(r+a)³)"),
    ("Ĥ , E_L", "Hamiltonian and local energy E_L = Ĥψ_θ / ψ_θ"),
]


def notation_rows():
    out = []
    for sym, mean in NOTATION:
        out.append(
            f'<div class="nrow"><dt>{sym}</dt><dd>{mean}</dd></div>')
    return "\n".join(out)


LAMBDA_CARDS = [
    ("01", "Embedded", "amber",
     "A 2-layer MLP maps the scalar λ to a shared token "
     "<code>p_λ = MLP_λ(λ) ∈ ℝ^d</code>."),
    ("02", "Injected every layer", "amber",
     "After each self-attention + MLP block, "
     "<code>H ← H + tanh(W_ℓ p_λ)</code> — so λ modulates all L layers, "
     "not just the input."),
    ("03", "Enters the physics", "teal",
     "λ scales the Hamiltonian <code>Ĥ = −½Σ∇² + λΣ(r+a)⁻³</code> and sets the "
     "K₀ Jastrow cusp <code>K₀(2√(λ/r̃))</code>."),
]


def cards():
    out = []
    for n, t, tone, body in LAMBDA_CARDS:
        out.append(f'''<article class="card {tone}">
      <span class="cardnum">{n}</span>
      <h3>{t}</h3>
      <p>{body}</p>
    </article>''')
    return "\n    ".join(out)


DARK_TOKENS = """
    --ground:#0e1620; --surface:#16212d; --ink:#e8eef5; --muted:#94a3b3;
    --faint:#6d7d8d; --hair:#26333f; --line:#31414f;
    --blue:#7f98f2; --amber:#e7a24d; --teal:#48bea7;
    --plate:#ffffff; --plate-bd:#2a3743; --shadow:rgba(0,0,0,.55);
    --chip:#1b2836;
"""

HTML = f"""<title>BoseNet Architecture</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Serif:ital,wght@0,400;0,600;1,400&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
  :root {{
    --ground:#f6f8fc; --surface:#ffffff; --ink:#16202b; --muted:#57697a;
    --faint:#8a97a4; --hair:#e2e8f0; --line:#d9e1ea;
    --blue:#2f56d6; --amber:#c9760a; --teal:#0e8a76;
    --plate:#ffffff; --plate-bd:#e6ecf3; --shadow:rgba(20,32,43,.10);
    --chip:#eef3f9;
    --sans:'IBM Plex Sans',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
    --serif:'IBM Plex Serif',Georgia,'Times New Roman',serif;
    --mono:'IBM Plex Mono','DejaVu Sans Mono',ui-monospace,monospace;
  }}
  @media (prefers-color-scheme: dark) {{
    :root:not([data-theme="light"]) {{{DARK_TOKENS}}}
  }}
  :root[data-theme="dark"] {{{DARK_TOKENS}}}

  * {{ box-sizing:border-box; }}
  body {{
    margin:0; background:var(--ground); color:var(--ink);
    font-family:var(--sans); line-height:1.5;
    -webkit-font-smoothing:antialiased;
  }}
  .wrap {{ max-width:1080px; margin:0 auto; padding:clamp(20px,4vw,52px) clamp(16px,4vw,40px) 64px; }}

  /* hero */
  .eyebrow {{ font-family:var(--mono); font-size:12px; letter-spacing:.22em;
    text-transform:uppercase; color:var(--amber); margin:0 0 10px; }}
  h1 {{ font-family:var(--serif); font-weight:600; font-size:clamp(34px,6vw,56px);
    line-height:1.02; margin:0; letter-spacing:-.01em; text-wrap:balance; }}
  .lede {{ font-size:clamp(15px,2.2vw,18px); color:var(--muted); margin:14px 0 0;
    max-width:64ch; }}
  .lede b {{ color:var(--ink); font-weight:600; }}

  .legend {{ display:flex; flex-wrap:wrap; gap:8px 10px; margin:24px 0 0; }}
  .tag {{ display:inline-flex; align-items:center; gap:8px; font-size:13px;
    padding:6px 12px 6px 10px; border-radius:999px; background:var(--chip);
    border:1px solid var(--line); color:var(--ink); }}
  .tag i {{ width:11px; height:11px; border-radius:3px; display:inline-block; }}
  .sw-blue {{ background:var(--blue); }} .sw-amber {{ background:var(--amber); }}
  .sw-teal {{ background:var(--teal); }}

  /* sections */
  section {{ margin-top:clamp(40px,6vw,68px); }}
  .sec-head {{ display:flex; align-items:baseline; gap:12px; margin:0 0 4px; }}
  .sec-head h2 {{ font-family:var(--serif); font-weight:600;
    font-size:clamp(20px,3vw,26px); margin:0; letter-spacing:-.01em; }}
  .sec-head .idx {{ font-family:var(--mono); font-size:13px; color:var(--faint); }}
  .sec-sub {{ color:var(--muted); font-size:14.5px; margin:0 0 20px; max-width:66ch; }}

  /* figure plate */
  figure {{ margin:0; }}
  .plate {{ background:var(--plate); border:1px solid var(--plate-bd);
    border-radius:14px; padding:clamp(14px,2.5vw,26px);
    box-shadow:0 1px 2px var(--shadow), 0 14px 34px -22px var(--shadow); }}
  .plate.scroll {{ overflow-x:auto; }}
  .plate svg {{ display:block; width:100%; height:auto; }}
  .plate.scroll svg {{ min-width:760px; }}
  .compact {{ max-width:640px; margin-inline:auto; }}
  .compact.tall {{ max-width:432px; }}
  figcaption {{ font-size:13.5px; color:var(--muted); margin:14px 4px 0;
    max-width:74ch; line-height:1.55; }}
  figcaption b {{ color:var(--ink); font-weight:600; }}

  .duo {{ display:grid; grid-template-columns:1fr; gap:clamp(26px,4vw,40px); }}
  @media (min-width:860px) {{ .duo {{ grid-template-columns:minmax(0,0.9fr) minmax(0,1.1fr);
    align-items:start; }} }}

  /* lambda cards */
  .cards {{ display:grid; grid-template-columns:1fr; gap:14px; margin-top:6px; }}
  @media (min-width:720px) {{ .cards {{ grid-template-columns:repeat(3,1fr); }} }}
  .card {{ background:var(--surface); border:1px solid var(--line); border-radius:12px;
    padding:18px 18px 16px; position:relative; }}
  .card.amber {{ border-top:3px solid var(--amber); }}
  .card.teal {{ border-top:3px solid var(--teal); }}
  .cardnum {{ font-family:var(--mono); font-size:12px; color:var(--faint); }}
  .card h3 {{ font-family:var(--sans); font-weight:600; font-size:16px; margin:6px 0 8px; }}
  .card p {{ margin:0; font-size:13.5px; color:var(--muted); line-height:1.55; }}
  code {{ font-family:var(--mono); font-size:.86em; background:var(--chip);
    border:1px solid var(--line); border-radius:5px; padding:1px 5px; color:var(--ink);
    white-space:nowrap; }}

  /* notation */
  .notation {{ column-gap:36px; }}
  @media (min-width:760px) {{ .notation {{ columns:2; }} }}
  .nrow {{ break-inside:avoid; display:grid; grid-template-columns:minmax(120px,42%) 1fr;
    gap:14px; padding:9px 0; border-bottom:1px solid var(--hair); }}
  .nrow dt {{ font-family:var(--serif); font-style:italic; font-size:14px; color:var(--ink);
    margin:0; }}
  .nrow dd {{ margin:0; font-size:13px; color:var(--muted); line-height:1.45; }}

  footer {{ margin-top:56px; padding-top:22px; border-top:1px solid var(--hair);
    color:var(--faint); font-size:12.5px; font-family:var(--mono); line-height:1.7; }}
  footer b {{ color:var(--muted); font-weight:500; }}
</style>

<div class="wrap">
  <header>
    <p class="eyebrow">Neural quantum state · 2D dipolar bosons</p>
    <h1>BoseNet</h1>
    <p class="lede">A <b>bosonic Psiformer</b> wavefunction with <b>per-layer
      λ-conditioning</b> and a fixed <b>K₀ Jastrow</b>. One network represents
      ψ<sub>θ</sub>(R;&nbsp;λ) across a whole family of interaction strengths λ,
      then is queried at any λ to map the gas→crystal transition.</p>
    <div class="legend">
      <span class="tag"><i class="sw-blue"></i>learned network</span>
      <span class="tag"><i class="sw-amber"></i>λ-conditioning</span>
      <span class="tag"><i class="sw-teal"></i>fixed physics</span>
    </div>
  </header>

  <section>
    <div class="sec-head"><span class="idx">Fig. 1</span><h2>Full architecture</h2></div>
    <p class="sec-sub">Inputs to log-amplitude, with the variational Monte&nbsp;Carlo
      training loop underneath. The amber path is the generative mechanism; the teal
      factor is fixed, non-learned physics.</p>
    <figure>
      <div class="plate scroll">{ARCH}</div>
      <figcaption><b>Architecture of BoseNet.</b> A walker
        R&nbsp;=&nbsp;(r₁,…,r_N) on the 2D torus and the interaction strength λ are
        mapped to log&nbsp;ψ<sub>θ</sub>(R;λ). Periodic features are embedded into
        per-particle tokens and refined by L conditioned Psiformer blocks; the shared
        token p<sub>λ</sub>&nbsp;=&nbsp;MLP<sub>λ</sub>(λ) is injected after every
        block. A bosonic product head gives a symmetric, positive amplitude, to which
        a fixed K₀ Jastrow adds the exact 2D-dipole contact cusp. Lower panel: the VMC
        loop (Ewald Hamiltonian, KFAC natural gradient, generative batches over
        {{λ<sub>k</sub>}}).</figcaption>
    </figure>
  </section>

  <section>
    <div class="sec-head"><span class="idx">Fig. 2 · 3</span><h2>Data flow &amp; the conditioned block</h2></div>
    <p class="sec-sub">The compact, single-column figures for the paper: the end-to-end
      data flow (left) and a zoom on one conditioned attention block (right), where λ
      enters every layer.</p>
    <div class="duo">
      <figure>
        <div class="plate compact tall">{FLOW}</div>
        <figcaption><b>Data flow.</b> Inputs → log-amplitude. Colour encodes role;
          tensor shapes are annotated on the arrows (N particles, d&nbsp;=&nbsp;256,
          D&nbsp;=&nbsp;16).</figcaption>
      </figure>
      <figure>
        <div class="plate compact">{BLOCK}</div>
        <figcaption><b>One conditioned Psiformer block.</b> Self-attention and an MLP
          (each with a residual) act on the N particle tokens; the shared λ-token is
          injected additively at every layer,
          H&nbsp;←&nbsp;H&nbsp;+&nbsp;tanh(W<sub>ℓ</sub>&nbsp;p<sub>λ</sub>).</figcaption>
      </figure>
    </div>
  </section>

  <section>
    <div class="sec-head"><span class="idx">Mechanism</span><h2>How λ enters the network</h2></div>
    <p class="sec-sub">The conditioning variable reaches the amplitude in three
      distinct places — this is what makes a single set of weights represent
      ψ<sub>θ</sub>(·;λ) for every λ.</p>
    <div class="cards">
    {cards()}
    </div>
  </section>

  <section>
    <div class="sec-head"><span class="idx">Reference</span><h2>Notation</h2></div>
    <p class="sec-sub">Standard NQS / deep-learning conventions used throughout the
      figures.</p>
    <dl class="notation">
      {notation_rows()}
    </dl>
  </section>

  <footer>
    <b>Vector sources:</b> figures/bosenet_{{architecture,dataflow,block}}.svg → .pdf (LaTeX) + .png<br>
    <b>Regenerate:</b> python figures/make_figures.py · viewer: python figures/build_viewer.py
  </footer>
</div>
"""

with open(OUT, "w") as f:
    f.write(HTML)
print("wrote", OUT, f"({len(HTML)} bytes)")
