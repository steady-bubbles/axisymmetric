# Numerics behind the page

The bubble frames, We*, the ring frames, the Meyer–Seis table and fits, and the touching
asymptotics come from the high-resolution computation (`results/hires/`). Rows marked *first
computation* are checks made on the first, lower-resolution runs (`results/`, `results/rings/`),
which the high-resolution chains start from. The checks are `checks/check_mns.py`,
`checks/check_ms.py` and `verify/independent_bubble.py`. This file is written by
`page/write_numerics.py`.

## Precision

| quantity | value | how it was checked |
|---|---|---|
| largest Weber number We_max (fold) | 1.6176521718921 | boundary integral method, three discretisations up to (M, N) = (80, 4032): spread 5·10^-14 |
| We_max, independent solver | 1.6176521718921727 | radial-graph shape, method of fundamental solutions, 80-bit arithmetic; 6 resolutions agree to 5·10^-18; 7·10^-14 above the first value, a steady relative offset of about 4·10^-14 along the branch whose source is not identified. Both codes give **1.617652171892** |
| dimple onset (λ = 0) | 1.5620469061383 | three discretisations agree to 2·10^-13; independent solver 1.562046906138189 (difference -9·10^-14) |
| loss of star-shapedness | 1.230744086430 | three discretisations agree to 4·10^-13 |
| touching, We* | 1.133089514170 | cubic fit in z_pole on the 147 states with z_pole < 10^-3 of the high-resolution branch (down to z_pole = 1.0·10^-5); windows z_pole < 3·10^-4 … 2·10^-3 (109–169 states) agree to 2·10^-13 |
| bubble branch, 8 points z_pole = 0.90 … 1.1·10^-5 (*first computation*) | We, λ | three discretisations (up to 4448 nodes): We to 4·10^-13, λ to 7·10^-14 |
| small-We sweep (23 bubbles, We = 10^-4 … 0.4) | χ, λ | two discretisations agree to 3·10^-15 and 7·10^-15 |
| rings, r_h = 0.03924 … 5.225 (36 rings, *first computation*) | We, λ, Γ | 128 against 240 Fourier modes: We to 4·10^-12, λ to 3·10^-12, Γ to 8·10^-12 |
| high-resolution rings, r_h = 0.014 … 0.039 (16 rings) | We, λ, Γ | the neck map (M = 160) and the curvature map (M = 240) agree to 5·10^-13 |
| thin rings, r_h = 111, 613, 1116 (*first computation*) | relative spread of We | 7.1·10^-11, 1.6·10^-8, 8.8·10^-8 (four re-solves: M = 128, 176, 240 and 240 with twice the panels); Γ to 8·10^-12, λ to 1.4·10^-9 |
| rings, independent checks (*first computation*) | | a second ring solver (polar Fourier shape, method of fundamental solutions) reproduces We to 1.2·10^-12; a Kress–Nyström flow solver reproduces the sheet strength to 8·10^-10; a direct 3D Biot–Savart quadrature finds zero velocity inside the tube to 5·10^-14 |
| volume of every ring (Pappus) | 2π r̄ A = 4π/3 | holds to 1.5·10^-10 |

Newton's method is iterated until the residual reaches its round-off floor, 3·10^-15 max(1, r_o²)
for the rings (cancellation in the vortex-sheet equation costs log10 r_o² digits); the
boundary-integral solver reproduces the exact flow past spheroids of aspect ratio 0.5 to 8 to
3·10^-15. The weakest point is therefore the far thin end, where that floor reaches 10^-9 at
r_o ≈ 600: on the high-resolution branch the Weber numbers of the rings beyond r_h ≈ 130 scatter
by a few 10^-9, at most about 10^-8 relative, about a smooth curve (in the first computation, by
10^-9–10^-7).

## Bifurcations within the axisymmetric class

Steady states are critical points of |∂B| − (We/2) m_zz at fixed volume. The lowest eigenvalues of
its second variation are followed in the two axisymmetric sectors: 0e (even under z ↦ −z, the class
in which the branch is computed) and 0o (odd: pear-shaped bubbles, tilted ring cross-sections).

| family | sector 0e | sector 0o | branches off |
|---|---|---|---|
| 136 bubbles, sphere to touching (*first computation*) | one sign change, between We = 1.61648 and 1.61762, at the fold | positive, smallest 1.9895 (at touching) | nothing |
| 105 zero-flux rings (*first computation*), r_h = 8.636e-06 … 1116 | one sign change, at the ring fold (We = 1.50120, r_h = 0.1162); the neck mode (≈ −2.3/R² on thin rings) stays negative where resolved, up to r_h ≈ 340, and is below the accuracy of the spectrum beyond | positive, smallest 1.9903 (at touching), growing like 3/a² | nothing in the resolved range |

At the touching bubble the count of negative 0e eigenvalues jumps from 1 to 2: the ring has the
neck mode, the bubble does not. The rings also have a second parameter, the flux through the hole;
the zero-flux rings are one line in a two-parameter sheet of axisymmetric rings, whose thin end is
the two-parameter family of Meyer and Seis.

## Asymptotics in detail

**Near the sphere** (Meyer–Niebel–Seis, Theorem 1.1 and Remark 1.2 with their parameter γ = 0):
d(θ) = −(3/32) We (3cos²θ − 1) + o(We), χ = 1 + (9/32) We + O(We²), hence λ = H_pole = 2 − (3/4) We.
Least-squares fits on 19 bubbles with We ≤ 0.1:

    χ = 1 + 0.2812499999985 We + 0.1072265632 We² + …
    λ = 2 − 0.7499999999986 We − 0.0562500004 We² + …

The We² coefficients agree with 549/5120 and −9/160 to 7·10^-10 (observed, not derived).

**Thin rings** (Meyer–Seis, Theorem 1.3, ρ = 0): W = (log(8/ε) − 1/2)/(4π) + πβ/2,
γ = 3 log(8/ε)/(8π) − 15/(16π) − πβ/4, ν = −1/(4π²) + β, each up to o(1). In our units
W = R/|Γ|, β = 2ε̄/(We Γ²), γ = −Q/(R|Γ|) = 0 on the zero-flux branch, ν = βε̄λ.
The informal Theorem 1.1, formula (1.19) of arXiv:2409.08220v2, writes the surface-tension term
of the speed as πR ε̄σ/(b_ext ρ), which is not a speed (apparently a misprint); Theorem 1.3 gives
π ε̄σ/(R b_ext ρ).

| R | ε | β | We | W computed | W Meyer–Seis | γ Meyer–Seis (computed 0) | ν − ν_MS |
|---|---|---|---|---|---|---|---|
| 5.38 | 3.688·10^-2 | 0.441383 | 3.62242·10^-2 | 1.0801433593 | 1.0816136427 | -3.0·10^-3 | -1.6·10^-3 |
| 33.65 | 2.360·10^-3 | 0.855482 | 6.24019·10^-4 | 1.9508248592 | 1.9508467119 | -3.9·10^-5 | -2.2·10^-5 |
| 115.93 | 3.690·10^-4 | 1.137446 | 3.61489·10^-5 | 2.5414142577 | 2.5414151924 | -1.6·10^-6 | -9.5·10^-7 |
| 347.32 | 7.117·10^-5 | 1.387580 | 2.77504·10^-6 | 3.0652965455 | 3.0652966011 | -8.8·10^-8 | -5.2·10^-8 |
| 1089.98 | 1.280·10^-5 | 1.648304 | 1.85852·10^-7 | 3.6113556667 | 3.6113556202 | 2.1·10^-8 | -2.3·10^-9 |

The three remainders fall like ε² log²(1/ε): the ratio of each to ε² log²(1/ε) stays nearly
constant (β grows like log(1/ε) on the zero-flux branch). Fits with relative weights on the
49 rings with ε < 0.05 and r_h ≤ 320, as A ε² log²(1/ε) + B ε² log(1/ε) + C ε²:
- W: A = -0.119, B = 0.076, C = -0.040; largest relative residual 1·10^-2; the ratio to ε² log²(1/ε) lies in [-0.111, -0.098]
- γ: A = 0.180, B = 0.043, C = 0.071; largest relative residual 5·10^-3; the ratio to ε² log²(1/ε) lies in [0.185, 0.203]
- ν: A = -0.116, B = 0.044, C = -0.066; largest relative residual 2·10^-3; the ratio to ε² log²(1/ε) lies in [-0.112, -0.108]

The differences fall to 10^-7 at R ≈ 320. Beyond R ≈ 550 those of W and γ scatter between 10^-9
and 5·10^-8, the accuracy of the computed We there; that of ν follows the law to 2·10^-9 at R = 1199.

**At the touching bubble** (`results/hires/asymptotics/asymptotics.md`): bubbles follow
We − We* ≈ 2.0364 z_pole; rings follow
We − We* ≈ 2.0364 r_h log(1/r_h). Fitted on the 35 rings with r_h < 10^-4 (one from the first computation) the ring coefficient
equals the bubble slope to 3·10^-7 (−1.1·10^-7 and +2.7·10^-7 with five and four terms); λ matches
to 10^-6 and the circulation to 3·10^-6.
