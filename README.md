# Axisymmetric bubbles

Steady axisymmetric bubbles in an inviscid stream, computed from the sphere to thin bubble rings.

A bubble filled with vacuum moves at constant speed through an inviscid, incompressible fluid at
rest at infinity; the flow outside is irrotational, and surface tension balances the pressure at
the interface. After scaling, a single parameter remains, the Weber number We. This repository
continues two families of solutions that were constructed rigorously, each in its own limit, and
finds that they lie on one connected curve:

* the near-spherical bubbles of D. Meyer, L. Niebel and C. Seis, *Steady bubbles and drops in
  inviscid fluids*, Calc. Var. PDE 64 (2025), [doi:10.1007/s00526-025-03144-w](https://doi.org/10.1007/s00526-025-03144-w),
  [arXiv:2503.05503](https://arxiv.org/abs/2503.05503);
* the thin hollow vortex rings with surface tension of D. Meyer and C. Seis, *Steady ring-shaped
  vortex sheets*, J. Eur. Math. Soc. (2026), [doi:10.4171/JEMS/1757](https://doi.org/10.4171/JEMS/1757),
  [arXiv:2409.08220](https://arxiv.org/abs/2409.08220).

The page **[steady-bubbles.github.io/axisymmetric](https://steady-bubbles.github.io/axisymmetric/)**
shows the result: the bifurcation diagram, the computed bubbles and rings with their flow, and the
comparison with both theories. This repository holds the solver, the scripts that ran every
computation, the data, and the code that builds the page.

![Bifurcation diagram: the bubble branch from the sphere through the fold to the touching bubble, and the ring branch from the touching bubble to thin rings](docs/bifurcation_diagram.png)

*The bifurcation diagram (a screenshot of the page). Horizontal: the Weber number. Upper half:
bubbles, by the height z_pole of their pole; lower half: zero-flux rings, by their hole radius r_h
(both on log scales; the two meet at the touching bubble, z_pole = 0 = r_h). Dashed black: the
predictions of Meyer–Niebel–Seis near the sphere and of Meyer–Seis for thin rings. Diamonds:
rings from which non-axisymmetric rings branch off. The green circle is the page's selected
state, here the fold.*

**Notation.** We = ρV²R/σ, with ρ the density of the fluid, V the speed of the bubble, σ the
surface tension and R the radius of the sphere of the same volume; lengths are in units of R. On
the surface, H − (We/2)|∇φ|² = λ, where φ is the velocity potential, H the sum of the principal
curvatures (2 on the unit sphere) and λ the Bernoulli constant. The poles of a bubble are
stagnation points, so λ is the curvature at the pole, and λ = 0 marks the onset of dimples.
z_pole is the height of a bubble's pole above its equatorial plane (1 for the sphere, 0 when the
poles touch), χ = r_eq/z_pole its aspect ratio, and r_h the radius of a ring's hole. Rings can
carry a net flux through the hole, so they form a two-parameter family; the rings followed here
have zero flux, the only ones that can connect continuously to the touching bubble, which has no
hole.

## Results

Numbers from the high-resolution computation are marked (h), those from the first, lower-resolution
computation (1); see [Rerunning the computations](#rerunning-the-computations).

* **Bubbles.** From the sphere (We = 0) the bubbles flatten, pass a largest Weber number
  **We_max = 1.617652171892** (1; 13 digits, and an independent solver in 80-bit arithmetic gives
  the same), develop dimples at the poles from We = 1.562047 (1; λ = 0), stop being star-shaped at
  We = 1.230744 (1), and end when the two poles touch, at **We\* = 1.133089514170** (h).
* **Bubble rings.** The zero-flux rings start from the same touching bubble. Their Weber number
  has a largest value at hole radius r_h = 0.1162, We = 1.50120 (1), and as the hole opens they
  become thin rings; the computed branch runs to r_h ≈ 1200 (h).
* **Where they meet** (h). Bubbles approach the touching bubble along We − We\* ≈ 2.0364 z_pole,
  rings along We − We\* ≈ 2.0364 r_h log(1/r_h); the ring coefficient equals the bubble slope to
  3·10⁻⁷.
* **Near the sphere** (a sweep of 23 bubbles, `checks/check_mns.py`). The aspect ratio is χ = 1 + 0.2812499999985 We + …, against 9/32 from
  Meyer–Niebel–Seis, and λ = 2 − 0.7499999999986 We + …, against 3/4.
* **Thin rings** (h). The rings lie on the zero-flux curve of the Meyer–Seis family: their speed
  and Bernoulli constant match Theorem 1.3 of Meyer–Seis, and so does their flux constant, which
  is zero on these rings. The differences fall like
  ε² log²(1/ε) in the aspect ratio ε of the tube, to 10⁻⁷ at ring radius R ≈ 320; beyond R ≈ 550
  those of the speed and the flux constant level off at a few 10⁻⁸, the accuracy of We there, while
  that of the Bernoulli constant reaches 2·10⁻⁹ at R = 1199.
* **Bifurcations** (1). Within the axisymmetric class the second variation changes sign only at
  the two folds, so no other axisymmetric branch leaves this one, as far as the spectrum is
  resolved: for the thinnest rings, beyond r_h ≈ 340, its smallest eigenvalue (≈ −2.3/R²) drops
  below the accuracy of the computation. No non-axisymmetric branch leaves the bubbles (for any
  azimuthal wave number m). From the rings they do: an off-centre hole (m = 1) at r_h = 0.0777,
  and beaded rings with m = 2, …, 6 at r_h = 0.666, 1.017, 1.313, 1.579 and 1.824.

That the two families are connected is observed numerically; to prove it remains open. More
numbers, with how each was checked, are in [docs/numerics.md](docs/numerics.md).

## Contents

| path | what |
|---|---|
| `bubbles/` | the solver: boundary integral equations on Chebyshev panels, the bubble and ring states, Newton's method, continuation, and the second-variation spectra |
| `scripts/` | the computations (tables below) |
| `checks/` | the comparisons with Meyer–Niebel–Seis (`check_mns.py`) and Meyer–Seis (`check_ms.py`) |
| `verify/` | independent checks: a second bubble solver in 80-bit arithmetic (`independent_bubble.py`), a second ring solver, flow-field and Biot–Savart checks (see `verify/README.md`) |
| `tests/` | unit tests of the solver |
| `results/` | the computed data (table below) |
| `page/` | the page's template, text and references, and the scripts that build it and write `docs/numerics.md` and `docs/page_text.tex` |
| `docs/` | `numerics.md` (precision and checks in detail), `page_text.tex` (all prose of the page, for editing), the figure above |

## Running it

```
pip install -e ".[test]"
python -m pytest                       # the solver's tests, about 3 min
python page/build_page.py              # the page, from results/, into build/index.html (seconds)
```

The published page is a copy of `build/index.html` in
[steady-bubbles/steady-bubbles.github.io](https://github.com/steady-bubbles/steady-bubbles.github.io)
(`axisymmetric/index.html`). `python page/write_numerics.py` and `python page/export_text.py`
regenerate `docs/numerics.md` and `docs/page_text.tex`.

The checks: `python checks/check_ms.py` (about 1 min), `python checks/check_mns.py` (about 5 min on
4 cores), `python verify/independent_bubble.py` (see its docstring).

The scripts in `scripts/`, `checks/` and `page/` print their usage with `--help`.

### Rerunning the computations

The scripts continue from what their output already holds: `hires_rings.py` resumes unfinished
chains, `bubble_sectors.py`, `ring_sectors.py` and `crossing_rings.py` skip what is already
computed, `ring_qscan.py` extends scans whose directions stopped on the time limit, and
`ring_convergence.py` appends. With the committed results in place, the commands below therefore
do little; to recompute, give new output paths (for the ring chains a new folder, merged with
`merge_hires_rings.py --dir=`). Reruns agree with the committed files to round-off, not byte for
byte (`check_ms.py`, `merge_hires_rings.py` and `ring_asymptotics.py` reproduce theirs exactly,
and the page build up to its time stamp).

The times are rough, from the logs, for a 4-core machine. The bubble scripts (`run_branch*.py`,
`special_points.py`, `convergence.py`) run in one process; the others use `BUBBLES_NPROC` worker
processes (default 2 to 4, see each script).

**First computation** (`results/`, `results/rings/`; lower resolution, used for the stability
analysis, the branch points, the flux scans and the precision studies; the high-resolution ring
chains start from its rings):

| step | command | time |
|---|---|---|
| bubble branch, sphere to z_pole = 1.1·10⁻⁵ | `python scripts/run_branch.py results/branch.json 1e-5 0.05 0.125` | 8 min |
| fold, dimple onset, loss of star-shapedness | `python scripts/special_points.py` | 1 h |
| convergence in the resolution | `python scripts/convergence.py` | 5 min |
| bubble spectra | `python scripts/bubble_sectors.py results/bubble_sectors_all.json --every=1 --m=0,1` | 10 min |
| the first ring, r_h = 0.0392: a catenoidal neck glued into the touching bubble, then Newton | `python scripts/first_ring.py results/branch.json 0.03923612667872822 48 first_ring.npz` | 4 min |
| zero-flux rings, from the first ring or a stored one (`run.json:index`) | `python scripts/run_rings.py start out.json factor r_stop [M_max] [--neck] [--tail=1e-13] [--thin]` | 10 min – 4 h per run |
| ring spectra | `python scripts/ring_sectors.py results/rings/ring_sectors_all.json results/rings/rings_{neck3,neck2,neck,down2,down,up,thin,thin2,thin3,thin7}.json --sectors=0o,0eQ,0eG,1e,1o --bounds --nev=4` (bash) | 25 min |
| non-axisymmetric branch points | `python scripts/crossing_rings.py --m=2,3,4,5,6`, then `--m=1 --dQ=2e-4` | 40 min |
| flux scans at fixed hole radius, e.g. r_h = 0.069 | `python scripts/ring_qscan.py results/rings/rings_up.json 4 results/rings/sheet_rh0.069.json --M=64 --dQ0=1e-3 --tcap=25 --stall=15` (the options of each scan are in its `.meta.json`) | 1–2 h each |
| resolution of the rings | `python scripts/compare_precision.py`; `python scripts/ring_convergence.py results/rings/ring_convergence_thin.json results/rings/rings_thin2.json:3 results/rings/rings_thin7.json:2 results/rings/rings_thin7.json:4 --M=128,176,240` | 1 min; 3 h |

**High-resolution computation** (`results/hires/`; the page shows this):

| step | command | time |
|---|---|---|
| bubble branch, 549 states from the sphere to z_pole = 10⁻⁵ | `python scripts/run_branch_hires.py results/hires/branch.json` | 1.5 h |
| ring chains, each from a ring of the first computation (table in the script's docstring) | `python scripts/hires_rings.py results/hires/rings` | 0.3 to more than 2 h per chain |
| merged ring branch, 254 rings | `python scripts/merge_hires_rings.py --report` | seconds |
| bubble frames for the page | `python scripts/export_viz.py results/hires/branch.json results/hires/viz/bubble_viz.json 4 --step=2` | 10–30 min |
| ring frames for the page | `python scripts/export_rings_viz.py results/hires/rings_merged.json results/hires/viz/rings_viz.json --sheets=none` | 1–2 h |
| asymptotics at the touching bubble | `python scripts/ring_asymptotics.py results/hires/rings_merged.json --rh-max=0.0393 --branch=results/hires/branch.json --outdir=results/hires/asymptotics` | 10 min |

## Data

All files are JSON unless noted; each script's docstring describes its output.

| file | contents |
|---|---|
| `results/hires/branch.json` | 549 bubble states from the sphere to touching: shape coefficients, panels, We, λ and diagnostics (`branch.log`: the run's log up to state 142; the rest of the log was lost) |
| `results/hires/rings/` | the ring chains: `<chain>0_p<k>.json` (rings), `*_last.npz` (restart states), `*.log` (see below) |
| `results/hires/rings_merged.json` | the zero-flux ring branch, 254 rings; `source` names the chain of each |
| `results/hires/viz/` | the page's frames: shapes, surface speed, streamlines |
| `results/hires/asymptotics/` | the touching asymptotics (`asymptotics.md`) |
| `results/branch.json`, `results/rings/rings_*.json` | the first computation's bubbles and rings |
| `results/special_points.json`, `results/convergence.json` | fold, dimple onset and loss of star-shapedness at three resolutions; resolution study |
| `results/bubble_sectors_all.json`, `results/rings/ring_sectors_all.json` | second-variation spectra along both branches |
| `results/rings/crossing_rings.json` | the rings where non-axisymmetric branches leave |
| `results/rings/sheet_rh*.json` | flux scans: rings with flux through the hole at eight hole radii |
| `results/rings/precision.json`, `results/rings/ring_convergence_thin.json` | resolution studies of the rings |
| `results/checks/` | the outputs of `checks/` and `verify/independent_bubble.py` |
| `results/verify/` | the outputs of the other `verify/` scripts |

## Resolution and provenance

In the high-resolution computation the bubbles are resolved to a truncation indicator below
10⁻¹² and the rings to a Fourier tail below 10⁻¹³ (the first computation used a tail of 10⁻¹² for
rings with small holes and a residual indicator elsewhere, with 128 to 320 Fourier modes). Two
parameter maps are used for the rings: one adapted to the curvature, and one to the catenoidal
neck of rings with small holes; where both resolve a ring they agree to 5·10⁻¹³.

* The touch chain was stopped at r_h = 1.0·10⁻⁵ (a further part re-solved that ring but did not
  finish the next step, and was not continued), so the smallest ring of the merged branch,
  r_h = 8.6·10⁻⁶, comes from the first computation.
* Beyond r_h ≈ 130 the residual can only be driven to its round-off floor 3·10⁻¹⁵ r_o², which
  leaves a relative scatter of a few 10⁻⁹ (at most about 10⁻⁸) in the Weber numbers of the
  thinnest rings.
* Every ring descends from the first ring, r_h = 0.039236 (`results/rings/rings_up.json:0`, and
  at M = 240 `rings_up_hp.json:0`). It was made by gluing a catenoidal neck into the touching
  bubble and solving with Newton's method; the original driver was not kept, and
  `scripts/first_ring.py` does the same: refined by `run_rings.py` to M = 128, its ring agrees
  with the stored one to 10⁻¹² in We and λ.
* For several runs of the first computation (`results/rings/rings_*.json`) the exact command
  lines were not recorded; their logs give the starting ring, and the files record the
  resolution of every ring. `rings_thin7.json` starts from a ring at r_h = 336 that is not kept.
  `rings_down_part1.json` is the first part of `rings_down.json` (the two files are identical);
  some scripts read it under that name. The runs of `rings_neck`, `rings_thin2` and `rings_thin3`
  were stopped by hand, and their logs end with the tracebacks of the stopped worker processes.
* `results/convergence.json` is computed on `results/branch.json`; the three reference bubbles of
  `verify/independent_bubble.py` come from an earlier run of it, whose branch is not kept.
* The logs in `results/hires/` are copies taken while the runs went on: `branch.log` stops at
  state 142 of 549, the ring logs of most chains stop at an earlier ring than their JSON files,
  and most `*_main.log` files have no completion line. The JSON files hold every state and its
  resolution; the times above are therefore approximate.

## Author

[Lukas Niebel](https://lukasniebel.github.io), with the help of Claude Opus. There is no license
for now; please contact the author about reuse. If you use the data or the code, please cite the
two papers above and this repository.
