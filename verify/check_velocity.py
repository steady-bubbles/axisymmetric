"""Independent Biot-Savart flow check of computed zero-flux rings.

usage: python verify/check_velocity.py up:17 thin:7 ...   [--hi]  (--hi: tighter quadrature)
Writes results/verify/velocity_<tag>.json.
"""
import json
import os
import sys
import time
import warnings

import numpy as np
from numpy.polynomial.legendre import leggauss

warnings.filterwarnings('ignore')

if __name__ == '__main__' and {'-h', '--help'} & set(sys.argv[1:]):
    print(__doc__)
    sys.exit()
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

from bubbles.ring import RingMap, RingPanels, RingState  # noqa: E402  (only to rebuild st)
from biot_savart import ring_curve, velocity  # noqa: E402

FILES = dict(up='results/rings/rings_up.json', thin='results/rings/rings_thin.json',
             thin2='results/rings/rings_thin2.json', thin7='results/rings/rings_thin7.json',
             hp='results/rings/rings_up_hp.json')


def load(tag):
    f, i = tag.split(':')
    p = json.load(open(os.path.join(REPO, FILES[f])))[int(i)]
    a = np.array(p['a'])
    st = RingState(a, p['rho_h'], RingMap(np.array(p['map_b'])), RingPanels(np.array(p['breaks'])))
    return p, st


def inside(rpoly, zpoly, r0, z0):
    """even-odd rule against a closed polygon."""
    r1, z1 = rpoly, zpoly
    r2, z2 = np.roll(rpoly, -1), np.roll(zpoly, -1)
    cond = (z1 > z0) != (z2 > z0)
    with np.errstate(divide='ignore', invalid='ignore'):
        rx = r1 + (z0 - z1) * (r2 - r1) / (z2 - z1)
    return bool(np.sum(cond & (rx > r0)) % 2)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    hi = '--hi' in sys.argv
    qopt = dict(nq=24, nphi=24, fac=3.0, eps_fac=0.1) if hi else dict(nq=16, nphi=16, fac=1.5, eps_fac=0.2)
    allres = {}
    for tag in args:
        t0 = time.time()
        p, st = load(tag)
        g, P = st.g, st.P
        cv = ring_curve(st)
        gam = np.asarray(st.gamma)
        gmax = np.abs(gam).max()
        r_h, r_o = g.r_h, g.r_o
        a_t = (r_o - r_h) / 2           # tube half width
        zmax = g.z.max()
        res = dict(tag=tag, r_h=r_h, r_o=r_o, We=p['We'], lam=p['lam'], a_t=a_t, z_max=zmax,
                   gamma_max=gmax, gamma_min_abs=np.abs(gam).min(), circulation=st.circulation,
                   solver_residual=float(np.abs(st.F(p['We'], p['lam'])).max()), quad=qopt)
        print(f"=== {tag}: r_h={r_h:.5g} r_o={r_o:.5g} a_t={a_t:.5g} z_max={zmax:.5g} "
              f"We={p['We']:.5g} max|gamma|={gmax:.5g} circ={st.circulation:.8g}", flush=True)

        # closed polygon of the cross-section (for inside tests)
        rp = np.concatenate([cv.dense_r, cv.dense_r[::-1]])
        zp = np.concatenate([cv.dense_z, -cv.dense_z[::-1]])

        # inradius: maximise distance to the sheet over interior grid points
        rr = np.linspace(r_h, r_o, 41)[1:-1]
        zz = np.linspace(-zmax, zmax, 41)[1:-1]
        best = (0, None)
        for r0 in rr:
            for z0 in zz[::2]:
                if inside(rp, zp, r0, z0):
                    d = np.hypot(cv.dense_r - r0, np.abs(cv.dense_z) - abs(z0)).min()
                    d = min(d, np.hypot(cv.dense_r - r0, cv.dense_z + abs(z0)).min())
                    if d > best[0]:
                        best = (d, (r0, z0))
        # refine the centre by a local search on z = 0 (symmetric) and r
        c_r = best[1][0]
        for h in [a_t / 20, a_t / 100, a_t / 500]:
            cands = c_r + h * np.arange(-5, 6)
            ds = [cv.dist(c, 0.0) for c in cands]
            c_r = cands[int(np.argmax(ds))]
        a_in = cv.dist(c_r, 0.0)
        res['inradius'] = a_in
        res['centre_r'] = c_r
        print(f"  inradius {a_in:.6g} at r={c_r:.6g} (a_t={a_t:.6g})")

        # ---------------- (a) interior points ----------------
        tt = (np.arange(6) + 0.5) * np.pi / 6
        # surface points + normals by our own interpolation of the nodal data
        # (theta is interpolated from the solver's nodal tangent angles)
        idx = np.clip(np.searchsorted(P.b, tt), 0, P.np - 1)
        u_loc = (2 * tt - P.a[idx] - P.b[idx]) / P.h[idx]
        v = cv.interp(idx, u_loc)
        from biot_savart import lagrange_matrix
        Lm = lagrange_matrix(cv.xn, cv.bw, u_loc)
        th = np.einsum('nj,nj->n', Lm, g.theta.reshape(P.np, P.p)[idx])
        pts = [(c_r, 0.0)]
        for k in range(len(tt)):
            for sgn in (1, -1):
                xs = np.array([v[k, 0], sgn * v[k, 1]])
                nrm = np.array([-np.sin(th[k]), sgn * np.cos(th[k])])
                for f in (0.1, 0.5, 0.9):
                    pts.append(tuple(xs - f * a_in * nrm))
        rows = []
        for (r0, z0) in pts:
            ins = inside(rp, zp, r0, z0)
            u, d, ns, nph = velocity(cv, r0, z0, **qopt)
            rows.append(dict(r=r0, z=z0, inside=ins, d=d, d_over_ain=d / a_in, u=u.tolist(),
                             absu_rel=float(np.hypot(*u) / gmax), nsrc=ns, nphi=nph))
        A = np.array([q['absu_rel'] for q in rows])
        dd = np.array([q['d_over_ain'] for q in rows])
        res['a_interior'] = dict(n=len(rows), all_inside=all(q['inside'] for q in rows),
                                 d_over_ain_range=[float(dd.min()), float(dd.max())],
                                 max_absu_over_gmax=float(A.max()), median=float(np.median(A)),
                                 rows=rows)
        print(f"  (a) {len(rows)} interior pts, all inside={res['a_interior']['all_inside']}, "
              f"d/a_in in [{dd.min():.3f},{dd.max():.3f}]: max|u|/max|gamma| = {A.max():.2e} "
              f"(median {np.median(A):.1e})", flush=True)

        # ---------------- (b) near-surface points ----------------
        tsel = np.linspace(0, np.pi, 13)
        nodes = np.unique([int(np.argmin(np.abs(P.tf - t))) for t in tsel])
        dfr = np.array([1e-2, 5e-3, 2e-3, 1e-3])
        brows = []
        for i in nodes:
            xs = np.array([g.r[i], g.z[i]])
            T = np.array([np.cos(g.theta[i]), np.sin(g.theta[i])])
            nrm = np.array([-np.sin(g.theta[i]), np.cos(g.theta[i])])
            gi = gam[i]
            uo, ui = [], []
            for f in dfr:
                d = f * a_t
                u1, _, _, _ = velocity(cv, *(xs + d * nrm), d=d, **qopt)
                u2, _, _, _ = velocity(cv, *(xs - d * nrm), d=d, **qopt)
                uo.append(u1)
                ui.append(u2)
            uo, ui = np.array(uo), np.array(ui)
            # polynomial (cubic) extrapolation to d -> 0 of the outside velocity
            V = np.vander(dfr, 4)
            coef = np.linalg.solve(V, uo)
            u0 = coef[-1]
            coefi = np.linalg.solve(V, ui)
            brows.append(dict(t=float(P.tf[i]), r=float(xs[0]), z=float(xs[1]), gamma=float(gi),
                              un_out=(uo @ nrm / gmax).tolist(),
                              speed_mis=((np.hypot(uo[:, 0], uo[:, 1]) - abs(gi)) / gmax).tolist(),
                              vec_mis=(np.hypot(*(uo - gi * T).T) / gmax).tolist(),
                              u_in=(np.hypot(*ui.T) / gmax).tolist(),
                              u0=u0.tolist(), H_solver=float(g.H[i]),
                              extrap_vec_mis=float(np.hypot(*(u0 - gi * T)) / gmax),
                              extrap_un=float(u0 @ nrm / gmax),
                              extrap_in=float(np.hypot(*coefi[-1]) / gmax)))
        un = np.array([q['un_out'] for q in brows])
        sm = np.array([q['speed_mis'] for q in brows])
        vm = np.array([q['vec_mis'] for q in brows])
        ui = np.array([q['u_in'] for q in brows])
        res['b_near_surface'] = dict(d_over_at=dfr.tolist(), nodes=len(brows),
                                     max_abs_un_over_gmax=np.abs(un).max(axis=0).tolist(),
                                     max_abs_speed_mis=np.abs(sm).max(axis=0).tolist(),
                                     max_vec_mis=vm.max(axis=0).tolist(),
                                     max_u_inside=ui.max(axis=0).tolist(),
                                     extrap_vec_mis=max(q['extrap_vec_mis'] for q in brows),
                                     extrap_un=max(abs(q['extrap_un']) for q in brows),
                                     extrap_in=max(q['extrap_in'] for q in brows),
                                     rows=brows)
        print("  (b) d/a_t        : " + ' '.join(f"{x:9.1e}" for x in dfr))
        print("      max|u.n|/g   : " + ' '.join(f"{x:9.2e}" for x in np.abs(un).max(axis=0)))
        print("      max||u|-|g||/g: " + ' '.join(f"{x:9.2e}" for x in np.abs(sm).max(axis=0)))
        print("      max|u-gT|/g  : " + ' '.join(f"{x:9.2e}" for x in vm.max(axis=0)))
        print("      inside |u|/g : " + ' '.join(f"{x:9.2e}" for x in ui.max(axis=0)))
        print(f"      extrapolated to d=0: |u-gT|/g={res['b_near_surface']['extrap_vec_mis']:.2e} "
              f"|u.n|/g={res['b_near_surface']['extrap_un']:.2e} inside {res['b_near_surface']['extrap_in']:.2e}",
              flush=True)

        # ---------------- (e) extra: dynamic condition with our own curvature ----------------
        # own differentiation of the nodal (r, z) data (barycentric Lagrange on each panel)
        xn, bw = cv.xn, cv.bw
        Dm = (bw[None, :] / bw[:, None]) / (xn[:, None] - xn[None, :] + np.eye(len(xn)))
        np.fill_diagonal(Dm, 0.0)
        np.fill_diagonal(Dm, -Dm.sum(axis=1))
        sc = (2 / P.h)[:, None]
        R_ = g.r.reshape(P.np, P.p)
        Z_ = g.z.reshape(P.np, P.p)
        r1 = (R_ @ Dm.T) * sc
        z1 = (Z_ @ Dm.T) * sc
        r2 = (r1 @ Dm.T) * sc
        z2 = (z1 @ Dm.T) * sc
        sp = np.hypot(r1, z1)
        k1_own = (-(r1 * z2 - z1 * r2) / sp ** 3).ravel()
        k2_own = (-(z1 / sp) / R_).ravel()
        H_own = k1_own + k2_own
        # variant: differentiate the nodal tangent angle once
        th1 = ((g.theta.reshape(P.np, P.p) @ Dm.T) * sc).ravel()
        H_own_b = -th1 / sp.ravel() + k2_own
        We, lam = p['We'], p['lam']
        R_all = H_own - 0.5 * We * gam ** 2 - lam
        R_all_b = H_own_b - 0.5 * We * gam ** 2 - lam
        # at the (b) nodes with the Biot-Savart surface speed (extrapolated to d = 0)
        R_bs = np.array([H_own_b[i] - 0.5 * We * np.sum(np.array(q['u0']) ** 2) - lam
                         for q, i in zip(brows, nodes)])
        res['e_dynamic_bs_nodes_max_abs_residual'] = float(np.abs(R_bs).max())
        print(f"  (e) dynamic residual with Biot-Savart |u|^2 (extrapolated) and own H at {len(R_bs)} nodes: "
              f"max {np.abs(R_bs).max():.2e}")
        vol = -2 * np.pi * np.sum(P.wf * g.r ** 2 * z1.ravel())
        res['e_dynamic'] = dict(max_abs_H_own_minus_solver=float(np.abs(H_own - g.H).max()),
                                max_abs_H_own_b_minus_solver=float(np.abs(H_own_b - g.H).max()),
                                max_abs_residual_own_H=float(np.abs(R_all).max()),
                                max_abs_residual_own_H_b=float(np.abs(R_all_b).max()),
                                max_abs_residual_solver_nodal=float(np.abs(st.residual_nodal(We, lam)).max()),
                                scale_lam_plus_Wegam2=float(abs(lam) + 0.5 * We * gmax ** 2),
                                volume=float(vol), volume_err=float(vol / (4 * np.pi / 3) - 1),
                                ds_over_dt_vs_solver=float(np.abs(sp.ravel() - g.c).max() / g.c.max()))
        print(f"  (e) own curvature: |H_own-H_solver|max={res['e_dynamic']['max_abs_H_own_minus_solver']:.2e} "
              f"(via theta: {res['e_dynamic']['max_abs_H_own_b_minus_solver']:.2e}); max nodal |H-We/2 g^2-lam| = "
              f"{res['e_dynamic']['max_abs_residual_own_H']:.2e} / {res['e_dynamic']['max_abs_residual_own_H_b']:.2e} "
              f"(solver's own nodal {res['e_dynamic']['max_abs_residual_solver_nodal']:.2e}; scale {res['e_dynamic']['scale_lam_plus_Wegam2']:.3g}); volume/(4pi/3)-1 = {res['e_dynamic']['volume_err']:.2e}",
              flush=True)

        # ---------------- (c) flux through the hole ----------------
        # composite Gauss in r on [0, r_h], graded geometrically towards the
        # rim r = r_h (u_z varies on the tube scale a_t there)
        def flux_rule(nper):
            offs = [a_t * 2.0 ** k for k in range(-6, 60) if a_t * 2.0 ** k < 0.75 * r_h]
            br = np.unique(np.concatenate([[0.0], r_h - np.array(offs[::-1]), [r_h]]))
            x, w = leggauss(nper)
            a_, b_ = br[:-1], br[1:]
            rq = ((a_ + b_)[:, None] + (b_ - a_)[:, None] * x[None, :]) / 2
            wq = (b_ - a_)[:, None] / 2 * w[None, :]
            return rq.ravel(), wq.ravel()
        fl = {}
        for nper in (12, 20):
            rq, wq = flux_rule(nper)
            uz = np.array([velocity(cv, r0, 0.0, **qopt)[0][1] for r0 in rq])
            flux = 2 * np.pi * np.sum(wq * rq * uz)
            absflux = 2 * np.pi * np.sum(wq * rq * np.abs(uz))
            fl[nper] = (flux, absflux, uz, len(rq))
        flux, absflux, uz, nr = fl[20]
        res['c_flux'] = dict(flux_fine=fl[20][0], flux_coarse=fl[12][0], n_fine=fl[20][3], n_coarse=fl[12][3],
                             abs_flux=absflux,
                             rel_to_pi_rh2_gmax=flux / (np.pi * r_h ** 2 * gmax),
                             rel_to_pi_rh2_V=flux / (np.pi * r_h ** 2),
                             rel_to_absflux=flux / absflux,
                             uz_axis=float(velocity(cv, 0.0, 0.0, **qopt)[0][1]),
                             uz_axis_thin_ring_formula=float(1 + st.circulation / (2 * (r_h + r_o) / 2)),
                             uz_range=[float(uz.min()), float(uz.max())])
        print(f"  (c) flux(n={fl[20][3]})={fl[20][0]:+.3e} flux(n={fl[12][3]})={fl[12][0]:+.3e}  "
              f"int|u_z|dA={absflux:.4e}; flux/(pi r_h^2 max|g|)={res['c_flux']['rel_to_pi_rh2_gmax']:+.2e}, "
              f"flux/(pi r_h^2 V)={res['c_flux']['rel_to_pi_rh2_V']:+.2e}, flux/int|u_z|dA="
              f"{res['c_flux']['rel_to_absflux']:+.2e}; u_z in [{uz.min():.4f},{uz.max():.4f}] "
              f"u_z(axis)={res['c_flux']['uz_axis']:.6f} (1+Gamma/2R={res['c_flux']['uz_axis_thin_ring_formula']:.6f})",
              flush=True)

        # ---------------- (d) circulation ----------------
        loops = []
        for rad_fac in (1.5, 1.25):
            rad = rad_fac * a_t
            cr = (r_h + r_o) / 2
            if cr - rad <= 0.05 * cr:   # would hit the axis: ellipse squeezed on the inner side
                continue
            loops.append(('circle', rad_fac, cr, rad, rad))
        # an ellipse that always fits: inner crossing at r_h/2, outer at r_o + a_t/2
        ri, ro_ = r_h / 2, r_o + a_t / 2
        loops.append(('ellipse', None, (ri + ro_) / 2, (ro_ - ri) / 2, 1.5 * zmax + 0.25 * a_t))
        cres = []
        for kind, rf, cr, Ar, Az in loops:
            out = {}
            psi = 2 * np.pi * np.arange(64) / 64
            if any(inside(rp, zp, a, b) for a, b in zip(cr - Ar * np.cos(psi), Az * np.sin(psi))):
                print(f"  (d) {kind} loop intersects the tube -- discarded")
                continue
            for n in (128, 256):
                psi = 2 * np.pi * np.arange(n) / n
                # clockwise in (r, z): inner point -> top -> outer point (direction of T)
                xr = cr - Ar * np.cos(psi)
                xz = Az * np.sin(psi)
                dxr = Ar * np.sin(psi)
                dxz = Az * np.cos(psi)
                ok = all(not inside(rp, zp, a, b) for a, b in zip(xr[::4], xz[::4]))
                dmin = min(cv.dist(a, b) for a, b in zip(xr[::8], xz[::8]))
                U = np.array([velocity(cv, a, b, **qopt)[0] for a, b in zip(xr, xz)])
                circ = np.sum(U[:, 0] * dxr + U[:, 1] * dxz) * 2 * np.pi / n
                out[n] = circ
            if not ok:
                print(f"  (d) {kind} loop intersects the tube -- discarded")
                continue
            cres.append(dict(kind=kind, rad_fac=rf, centre=cr, Ar=Ar, Az=Az, outside=ok,
                             dmin_over_at=dmin / a_t, circ128=out[128], circ256=out[256],
                             rel_err=(out[256] - st.circulation) / abs(st.circulation)))
            print(f"  (d) {kind} rad_fac={rf} (Ar={Ar:.4g}, Az={Az:.4g}, outside={ok}, "
                  f"dmin/a_t={dmin / a_t:.2f}): circ256={out[256]:+.12f} circ128={out[128]:+.12f} "
                  f"st.circulation={st.circulation:+.12f} rel.err={cres[-1]['rel_err']:+.2e}", flush=True)
        res['d_circulation'] = cres
        res['time'] = time.time() - t0
        print(f"  time {res['time']:.0f}s", flush=True)
        allres[tag] = res
        os.makedirs(os.path.join(REPO, 'results', 'verify'), exist_ok=True)
        out = os.path.join(REPO, 'results', 'verify', f"velocity_{tag.replace(':', '')}{'_hi' if hi else ''}.json")
        json.dump(res, open(out, 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
