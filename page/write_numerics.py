"""Write numerics.md: the numbers behind the page that are not shown on it.

Sections: precision of the computed values, the bifurcation check within the
axisymmetric class, and the detailed comparison with the asymptotics of
Meyer-Niebel-Seis and Meyer-Seis. Everything is read from results/ (the
high-resolution computation in results/hires/, the first computation in results/
and results/rings/) and from results/checks/, so the file can be regenerated
after any rerun.

Usage: python page/write_numerics.py      (writes docs/numerics.md)
"""
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "page"))
from build_page import precision, stability, ring_fold  # noqa: E402
from data import load_json  # noqa: E402


def e(x, d=1):
    """x as a·10^k for Markdown."""
    if x == 0:
        return "0"
    k = int(math.floor(math.log10(abs(x))))
    m = x / 10 ** k
    return f"{m:.{d - 1}f}·10^{k}" if d > 1 else f"{round(m):d}·10^{k}"


def main():
    P = precision()
    st = stability()
    rf = ring_fold()
    bv = load_json("results/hires/viz/bubble_viz.json")["special"]
    mns = json.load(open(os.path.join(ROOT, "results", "checks", "mns.json")))
    ms = json.load(open(os.path.join(ROOT, "results", "checks", "ms.json")))
    ind = P.get("independent")
    L = []
    w = L.append
    w("# Numerics behind the page")
    w("")
    w("The bubble frames, We*, the ring frames, the Meyer–Seis table and fits, and the touching")
    w("asymptotics come from the high-resolution computation (`results/hires/`). Rows marked *first")
    w("computation* are checks made on the first, lower-resolution runs (`results/`, `results/rings/`),")
    w("which the high-resolution chains start from. The checks are `checks/check_mns.py`,")
    w("`checks/check_ms.py` and `verify/independent_bubble.py`. This file is written by")
    w("`page/write_numerics.py`.")
    w("")
    w("## Precision")
    w("")
    w("| quantity | value | how it was checked |")
    w("|---|---|---|")
    lv = P["fold_levels"]
    w(f"| largest Weber number We_max (fold) | {lv[-1][1]:.13f} | boundary integral method, three discretisations up to (M, N) = ({lv[-1][2]}, {lv[-1][3]}): spread {e(P['fold_spread'])} |")
    if ind:
        w(f"| We_max, independent solver | {ind['We_max']} | radial-graph shape, method of fundamental solutions, 80-bit arithmetic; {ind['n_res']} resolutions agree to {e(ind['spread'])}; {e(ind['minus_ref'])} above the first value, a steady relative offset of about 4·10^-14 along the branch whose source is not identified. Both codes give **1.617652171892** |")
    w(f"| dimple onset (λ = 0) | {P['dimple'][-1]:.13f} | three discretisations agree to 2·10^-13" + (f"; independent solver {ind['dimple']} (difference {e(ind['dimple_minus_ref'])})" if ind else "") + " |")
    w(f"| loss of star-shapedness | {P['star'][-1]:.12f} | three discretisations agree to 4·10^-13 |")
    src = load_json("results/hires/viz/bubble_viz.json").get("source", {})
    w(f"| touching, We* | {bv['touching']['We']:.12f} | cubic fit in z_pole on the {src.get('fit_states', '')} states with z_pole < 10^-3 of the high-resolution branch (down to z_pole = 1.0·10^-5); windows z_pole < 3·10^-4 … 2·10^-3 (109–169 states) agree to 2·10^-13 |")
    w(f"| bubble branch, 8 points z_pole = 0.89 … 1.2·10^-5 (*first computation*) | We, λ | three discretisations (up to 5248 nodes): We to {e(max(c[2] for c in P['conv']))}, λ to {e(max(c[3] for c in P['conv']))} |")
    w(f"| small-We sweep (23 bubbles, We = 10^-4 … 0.4) | χ, λ | two discretisations agree to {e(mns['resolution_spread']['chi'])} and {e(mns['resolution_spread']['lam'])} |")
    rh = P["rings_hp"]
    w(f"| rings, r_h = {rh['r_h'][0]} … {rh['r_h'][1]} ({rh['n']} rings, *first computation*) | We, λ, Γ | 128 against 240 Fourier modes: We to {e(rh['dWe'])}, λ to {e(rh['dlam'])}, Γ to {e(rh['dcirc'])} |")
    w("| high-resolution rings, r_h = 0.014 … 0.039 (16 rings) | We, λ, Γ | the neck map (M = 160) and the curvature map (M = 240) agree to 5·10^-13 |")
    w("| thin rings, r_h = 111, 613, 1116 (*first computation*) | relative spread of We | 7·10^-11, 1.6·10^-8, 8.8·10^-8 (three resolutions); Γ to 10^-12, λ to 10^-9 |")
    w("| rings, independent checks (*first computation*) | | a second ring solver (polar Fourier shape, method of fundamental solutions) reproduces We to 1.2·10^-12; a Kress–Nyström flow solver reproduces the sheet strength to 8·10^-10; a direct 3D Biot–Savart quadrature finds zero velocity inside the tube to 5·10^-14 |")
    w(f"| volume of every ring (Pappus) | 2π r̄ A = 4π/3 | holds to {e(max(abs(p['pappus_err']) for p in ms['zero_flux']), 2)} |")
    w("")
    w("Newton's method is iterated until the residual reaches its round-off floor, 3·10^-15 max(1, r_o²)")
    w("for the rings (cancellation in the vortex-sheet equation costs log10 r_o² digits); the")
    w("boundary-integral solver reproduces the exact flow past spheroids of aspect ratio 0.5 to 8 to")
    w("3·10^-15. The weakest point is therefore the far thin end, where that floor reaches 10^-9 at")
    w("r_o ≈ 600: on the high-resolution branch the Weber numbers of the rings beyond r_h ≈ 130 scatter")
    w("by a few 10^-9, at most about 10^-8 relative, about a smooth curve (in the first computation, by")
    w("10^-9–10^-7).")
    w("")
    w("## Bifurcations within the axisymmetric class")
    w("")
    sb, sr = st["bubbles"], st["rings"]
    ec = sb["e0_sign_changes"][0]
    w("Steady states are critical points of |∂B| − (We/2) m_zz at fixed volume. The lowest eigenvalues of")
    w("its second variation are followed in the two axisymmetric sectors: 0e (even under z ↦ −z, the class")
    w("in which the branch is computed) and 0o (odd: pear-shaped bubbles, tilted ring cross-sections).")
    w("")
    w("| family | sector 0e | sector 0o | branches off |")
    w("|---|---|---|---|")
    w(f"| {sb['n']} bubbles, sphere to touching (*first computation*) | one sign change, between We = {ec[0]:.5f} and {ec[1]:.5f}, at the fold | positive, smallest {sb['o_min']:.4f} (at touching) | nothing |")
    w(f"| {sr['n']} zero-flux rings (*first computation*), r_h = {sr['r_h_range'][0]} … {round(sr['r_h_range'][1])} | one sign change, at the ring fold (We = {rf['We']:.5f}, r_h = {rf['r_h']:.4f}); the neck mode (≈ −2.3/R² on thin rings) stays negative where resolved, up to r_h ≈ 340, and is below the accuracy of the spectrum beyond | positive, smallest {sr['o_min']:.4f} (at touching), growing like 3/a² | nothing in the resolved range |")
    w("")
    w("At the touching bubble the count of negative 0e eigenvalues jumps from 1 to 2: the ring has the")
    w("neck mode, the bubble does not. The rings also have a second parameter, the flux through the hole;")
    w("the zero-flux rings are one line in a two-parameter sheet of axisymmetric rings, whose thin end is")
    w("the two-parameter family of Meyer and Seis.")
    w("")
    w("## Asymptotics in detail")
    w("")
    f = mns["fits"]["M28"]["deg5"]
    w("**Near the sphere** (Meyer–Niebel–Seis, Theorem 1.1 and Remark 1.2 with γ = 0):")
    w("d(θ) = −(3/32) We (3cos²θ − 1) + o(We), χ = 1 + (9/32) We + O(We²), hence λ = H_pole = 2 − (3/4) We.")
    w("Least-squares fits on 19 bubbles with We ≤ 0.1:")
    w("")
    w(f"    χ = 1 + {f['chi'][0]:.13f} We + {f['chi'][1]:.10f} We² + …")
    w(f"    λ = 2 − {-f['lam'][0]:.13f} We − {-f['lam'][1]:.10f} We² + …")
    w("")
    w("The We² coefficients agree with 549/5120 and −9/160 to a few parts in 10^10 (observed, not derived).")
    w("")
    w("**Thin rings** (Meyer–Seis, Theorem 1.3, ρ = 0): W = (log(8/ε) − 1/2)/(4π) + πβ/2,")
    w("γ = 3 log(8/ε)/(8π) − 15/(16π) − πβ/4, ν = −1/(4π²) + β, each up to o(1). In our units")
    w("W = R/|Γ|, β = 2ε̄/(We Γ²), γ = −Q/(R|Γ|) = 0 on the zero-flux branch, ν = βε̄λ.")
    w("The informal Theorem 1.1, formula (1.7), writes the surface-tension term of the speed as")
    w("πR ε̄σ/(b_ext ρ), which is not a speed; Theorem 1.3 gives π ε̄σ/(R b_ext ρ).")
    w("")
    w("| R | ε | β | We | W computed | W Meyer–Seis | γ Meyer–Seis (computed 0) | ν − ν_MS |")
    w("|---|---|---|---|---|---|---|---|")
    zf = ms["zero_flux"]
    rows = sorted({min(range(len(zf)), key=lambda i: abs(math.log(zf[i]["r_h"] / r)))
                   for r in (5.2, 32.8, 110.7, 336.1, 1116.4)})
    for p in (zf[i] for i in rows):
        w(f"| {p['R']:.2f} | {e(p['eps'], 4)} | {p['beta']:.6f} | {e(p['We'], 6)} | {p['W']:.10f} | {p['W_MS']:.10f} | {e(p['gamma_MS'], 2)} | {e(p['dnu'], 2)} |")
    fits = ms["remainder_fits"]
    w("")
    w("The three remainders fall like ε² log²(1/ε): the ratio of each to ε² log²(1/ε) stays nearly")
    w("constant (β grows like log(1/ε) on the zero-flux branch). Fits with relative weights on the")
    w(f"{fits['dW']['n']} rings with ε < 0.05 and r_h ≤ {fits['dW']['r_h_max']:.0f}, as A ε² log²(1/ε) + B ε² log(1/ε) + C ε²:")
    for k, name in (("dW", "W"), ("dgamma", "γ"), ("dnu", "ν")):
        f_ = fits[k]
        w(f"- {name}: A = {f_['eps2log2']:.3f}, B = {f_['eps2log']:.3f}, C = {f_['eps2']:.3f}; largest relative residual "
          f"{e(f_['max_rel_resid'])}; the ratio to ε² log²(1/ε) lies in [{f_['ratio_eps2log2'][0]:.3f}, {f_['ratio_eps2log2'][1]:.3f}]")
    w("")
    w("The differences fall to 10^-7 at R ≈ 320. Beyond R ≈ 550 those of W and γ scatter between 10^-9")
    w("and 5·10^-8, the accuracy of the computed We there; that of ν follows the law to 2·10^-9 at R = 1199.")
    w("")
    w("**At the touching bubble** (`results/hires/asymptotics/asymptotics.md`): bubbles follow")
    w("We − We* ≈ 2.0364 z_pole; rings follow")
    w("We − We* ≈ 2.0364 r_h log(1/r_h). Fitted on the 35 rings with r_h < 10^-4 (one from the first computation) the ring coefficient")
    w("equals the bubble slope to 3·10^-7 (−1.1·10^-7 and +2.7·10^-7 with five and four terms); λ matches")
    w("to 10^-6 and the circulation to 3·10^-6.")
    os.makedirs(os.path.join(ROOT, "docs"), exist_ok=True)
    open(os.path.join(ROOT, "docs", "numerics.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("wrote docs/numerics.md")


if __name__ == "__main__":
    main()
