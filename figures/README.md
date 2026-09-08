# BoseNet architecture figures

Vector figures of the BoseNet ansatz — a **bosonic Psiformer** neural quantum
state with **per-layer λ-conditioning** and a **fixed K₀ Jastrow**, for 2D
dipolar bosons on a torus. `make_figures.py` is the source of truth; it emits the
SVGs and (via `rsvg-convert`) the PDF/PNG.

## Files

| figure | role | source | vector | raster |
|---|---|---|---|---|
| **architecture** | full pipeline + VMC training panel (overview / talks / poster) | `bosenet_architecture.svg` | `.pdf` | `.png` |
| **dataflow** | compact single-column PRB main figure (inputs → log ψ) | `bosenet_dataflow.svg` | `.pdf` | `.png` |
| **block** | conditioned-attention-block detail (Fig. 1 inset / 1b) | `bosenet_block.svg` | `.pdf` | `.png` |

`bosenet_architecture_viewer.html` is an easy-to-read viewer of all three (also
published as a Claude Artifact).

Regenerate everything:

```bash
python make_figures.py          # writes *.svg and renders *.pdf, *.png
```

The PDFs are true vector (fonts embedded by `rsvg-convert`) and drop straight
into a LaTeX build:

```latex
\begin{figure}[t]                      % single column
  \includegraphics[width=\columnwidth]{figures/bosenet_dataflow.pdf}
  \caption{\label{fig:bosenet} ...}    % caption text below
\end{figure}
```

Use `bosenet_architecture.pdf` in a `figure*` (double-column) environment.

## Role encoding (used in every figure)

| colour | meaning |
|---|---|
| **indigo-blue** | learned network / amplitude path |
| **amber** | λ-conditioning path (the generative mechanism) |
| **muted teal** | fixed, non-learned physics (K₀ Jastrow, Hamiltonian) |
| grey monospace | tensor shape, e.g. `[N×d]` |

## Notation (NQS / deep-learning conventions)

| symbol | meaning |
|---|---|
| $R=(\mathbf r_1,\dots,\mathbf r_N)$ | walker: $N$ particle positions on the 2-torus $\mathbb T^2$ |
| $\lambda$ | interaction strength (dipole length) — the **conditioning variable** |
| $\sigma$ | particle spins (spin-polarised here) |
| $\mathbf s_i = A^{-1}\mathbf r_i$ | fractional coordinates; $A$ = lattice matrix (columns = primitive vectors) |
| $\mathbf h_i^{(\ell)},\,H^{(\ell)}\!\in\mathbb R^{N\times d}$ | token / token matrix after layer $\ell$; embedding dim $d=256$ |
| $\mathbf p_\lambda = \mathrm{MLP}_\lambda(\lambda)\in\mathbb R^{d}$ | shared λ-conditioning token |
| $L=4$ | number of conditioned Psiformer blocks |
| $h=4,\ d_k=64$ | attention heads and per-head dim ($d = h\,d_k$) |
| $\Phi = H^{(L)}W_{\mathrm{orb}}\in\mathbb R^{N\times D}$ | orbital logits; $D=16$ |
| $\varphi_d(\mathbf r_i;\lambda)$ | orbital $d$ of particle $i$ |
| $\psi_\theta(R;\lambda)$ | positive, permutation-symmetric bosonic amplitude |
| $K_0$ | modified Bessel function (fixed Jastrow prefactor) |
| $\tilde r_{ij}$ | smooth periodic distance between $i$ and $j$ |
| $a$ | interaction softening: $0$ (bare $1/r^3$) or $0.1$ (regularised $1/(r+a)^3$) |
| $\hat H,\ E_L$ | Hamiltonian and local energy $E_L=\hat H\psi_\theta/\psi_\theta$ |

## Ready-to-paste captions

**Architecture (overview).**
> Architecture of BoseNet. A walker $R=(\mathbf r_1,\dots,\mathbf r_N)$ on the 2D
> torus and the interaction strength $\lambda$ are mapped to the log-amplitude
> $\log\psi_\theta(R;\lambda)$. Periodic features are embedded into per-particle
> tokens and processed by $L$ conditioned Psiformer blocks; a shared token
> $\mathbf p_\lambda=\mathrm{MLP}_\lambda(\lambda)$ is injected after every block
> (amber). A bosonic product head yields a symmetric, positive amplitude, to which
> a fixed $K_0$ Jastrow adds the exact 2D-dipole contact cusp (teal). The lower
> panel is the variational Monte Carlo loop: Ewald Hamiltonian, KFAC natural
> gradient, and generative batches split across $\{\lambda_k\}$.

**Dataflow (compact main).**
> Data flow of the BoseNet ansatz, from inputs to the log-amplitude. Colour
> encodes role: learned network (blue), λ-conditioning (amber), fixed physics
> (teal). Tensor shapes are annotated on the arrows ($N$ particles, embedding
> dimension $d=256$, $D=16$ orbitals per particle).

**Block (detail).**
> A single conditioned Psiformer block. Multi-head self-attention and an MLP (each
> with a residual connection) act on the $N$ particle tokens; the shared λ-token
> $\mathbf p_\lambda$ is injected additively at every layer,
> $H\leftarrow H+\tanh(W_\ell\,\mathbf p_\lambda)$ — the mechanism that conditions
> one network on the interaction strength.

## Design notes

* Typography (print): Nimbus Sans (labels) + Nimbus Roman *italic* (math) +
  DejaVu Mono (tensor shapes). The viewer HTML uses IBM Plex Sans/Serif/Mono.
* All geometry lives on a shared grid in `make_figures.py`; edit there, not in the
  SVG, then re-run to keep raster/vector in sync.
