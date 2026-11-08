# Independent checks

## Bubbles

`independent_bubble.py` is a second solver for the bubble branch, written without
anything from `bubbles/`: a stretched radial-graph shape, the flow by fundamental
solutions (ring vortices), and Newton's method in 80-bit arithmetic. It
recomputes the largest Weber number, the dimple onset, states along the branch
and the small-We expansion, and gives We_max = 1.6176521718922 at six
resolutions that agree to 5·10⁻¹⁸, 7·10⁻¹⁴ above the boundary-integral value (about 1 minute on 4 cores; output
`results/checks/independent_bubble.json`).

## Zero-flux rings

These scripts re-check stored rings (`results/rings/rings_up.json`,
`rings_thin.json`) with their own quadrature, curvature and flow solves. The
solver's code (`bubbles.ring`) is used only to rebuild what is being checked:
the sheet strength γ and the node geometry in `check_velocity.py` and
`check_bernoulli.py`, and the shape sampled by `ring_geometry.py`, whose
centroid radius feeds `independent_solver.py` and `circularity_prediction.py`.
It is also used for the comparisons. `kress_nystrom.py` and
`independent_solver.py` solve the flow anew. The scripts were written to test
one surprise: the large rings are almost perfectly round in cross-section.
Run them from the repository root (`independent_solver.py`, the `test_*`
scripts and `thin_ring_expansion.py` from `verify/`). Outputs go to
`results/verify/`: JSON where the script writes one, otherwise its printed
output as `<script>.log`. Tags are `up:i` / `thin:i` (index into those
files).

| script | what it does independently | result |
|---|---|---|
| `check_velocity.py up:17 thin:7 [--hi]` | velocity of the stored vortex sheet + stream by direct 3D Biot–Savart quadrature (`biot_savart.py`): inside the tube, through the hole, circulation on loops | interior \|u\|/max\|γ\| 8·10⁻¹⁶ (r_h = 0.42) and 5·10⁻¹⁴ (r_h = 33); hole flux ≤ 3·10⁻¹⁴ relative to ∫\|u_z\|dA; circulation to 10⁻¹⁵ (`velocity_up17.json`, `velocity_thin7.json`, `check_velocity.log`) |
| `kress_nystrom.py up:17,up:28,thin:4,thin:7 [1024,2048]` | new flow solve on the stored shape: Nyström with Kress log-splitting, own Lamb kernel, spectral curve on the full closed cross-section | γ agrees with the solver to ≤ 7.8·10⁻¹⁰ (r_h = 0.42 … 33; `kress_up17_up28_thin4_thin7.json`) |
| `check_bernoulli.py up:17 up:28 thin:4 thin:7` | curvature from the points (Fourier and quintic spline), Bernoulli residual at 6001 points between nodes, volume two ways, force identity ∫H r dr = r_o + r_h, interior stream function | residual at the solver's own resolution level (10⁻⁸ … 7·10⁻¹¹ with the Fourier curvature; the spline curvature is cruder, 4·10⁻⁷ … 3·10⁻⁴), volume ≤ 10⁻¹², force identity to 1.1·10⁻¹³ (`bernoulli_up17_up28_thin4_thin7.json`) |
| `ring_geometry.py` | cross-section about its area centroid in polar form ρ(φ) = a₀(1 + Σ cₙ cos nφ); writes `geom.json` | c₂ ≈ 8.3·10⁻⁵ at r_h = 14.9, 9.7·10⁻⁶ at r_h = 32.8 |
| `independent_solver.py thin:7 12 160 0.7 [warm]` | a separate Newton solver: polar-Fourier shape, flow by fundamental solutions (`mfs.py`), started from a circle (or, with `warm`, from the measured shape) | reproduces We to ≤ 1.2·10⁻¹², and λ, Γ and c₁…c₈ to ≤ 5·10⁻¹³, on five rings r_h = 0.42 … 33 (`independent_solver.json`; up:17 64 600 0.6 warm, up:28 32 320 0.7 warm, up:35 20 240 0.7, thin:4 16 200 0.7, thin:7 12 160 0.7) |
| `circularity_prediction.py` | thin-ring prediction of c₂ from a, R, We only | c₂ = ε²·3(12μ²+8μ+5)/(16(9μ−1)), ε = a/R, μ = Λ − 5/2; 4·10⁻⁵ relative at r_h = 33 |
| `thin_ring_expansion.py` | mpmath check of the slender-torus expansion of the potential flow used above | d₂ = 1/8 to 2·10⁻⁹ at ρ = 10⁻⁴ (`thin_ring_expansion.log`) |
| `test_sphere_biot_savart.py` | the Biot–Savart quadrature on the exact flow past a sphere | 10⁻¹³ … 10⁻¹⁷ (`test_sphere_biot_savart.log`) |
| `test_torus_mfs.py` | MFS on a circular torus: boundary residual and source-count self-consistency; surface-speed modes against the thin-torus expansion | residual 10⁻¹⁵; expansion agrees to O(ε) (`test_torus_mfs.log`) |

Not covered: uniqueness, dynamic stability, and the physical model itself.
The flow equation for the sheet is of the first kind, and its conditioning
worsens for thin rings. The last digits therefore rest on two studies:
* the high-precision rerun `results/rings/rings_up_hp.json` (M = 240,
  r_h ≤ 5.2), which agrees with the standard run to 4·10⁻¹² in We (max
  4.1·10⁻¹²);
* re-solves at r_h = 111, 613 and 1116 with M up to 240 and doubled panel
  density (`scripts/ring_convergence.py`,
  `results/rings/ring_convergence_thin.json`). Their relative spreads in We
  are 7·10⁻¹¹, 1.6·10⁻⁸ and 8.8·10⁻⁸.

`kress_nystrom.py` also accepts the tags `thin2:i`, `thin7:i` and `hp:i`.
Its run on the rings at r_h = 111, 613 and 1116 is in
`results/verify/kress_thin23_thin72_thin74_up17.json`.
