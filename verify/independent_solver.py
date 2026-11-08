"""Independent Newton solver for near-circular rings (polar Fourier cross-section + MFS flow).
Unknowns a0, c_1..c_N, lam, We; the area centroid R_c is prescribed (branch parameter).
rho(phi) = a0 (1 + sum c_n cos n phi) about (R_c, 0), phi from the outer rim.
Equations: cos-modes 0..N of H - (We/2)|u|^2 - lam; area = 2/(3 R_c) (volume 4pi/3 by Pappus);
centroid: int rho^3 cos phi = 0.   Usage: python verify/independent_solver.py tag N nsrc fsrc [warm]  (tags from geom.json)"""
import json, os, sys, warnings, time
import numpy as np
warnings.filterwarnings('ignore')
from mfs import solve, speed
REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')

tag, N, nsrc, fsrc = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])
CHORD = 1  # Jacobian (forward differences) computed once, reused
geo = {m['tag']: m for m in json.load(open(os.path.join(REPO, 'results', 'verify', 'geom.json')))}[tag]
Rc = geo['r_c']
J = max(4 * N, 128)
ph = np.pi * (np.arange(J) + 0.5) / J
nn = np.arange(N + 1)
COS = np.cos(np.outer(nn, ph))
phs = np.pi * (np.arange(nsrc) + 0.5) / nsrc
phc = np.pi * (np.arange(3 * J) + 0.5) / (3 * J)


def shape(a0, c, p):
    k = np.arange(1, N + 1)
    C = np.cos(np.outer(p, k)); S = np.sin(np.outer(p, k))
    rho = a0 * (1 + C @ c); d1 = -a0 * (S @ (k * c)); d2 = -a0 * (C @ (k ** 2 * c))
    return rho, d1, d2


def resid(Y, full=False):
    a0, c, lam, We = Y[0], Y[1:N + 1], Y[N + 1], Y[N + 2]
    rho, d1, d2 = shape(a0, c, ph)
    r, z = Rc + rho * np.cos(ph), rho * np.sin(ph)
    xp, zp = d1 * np.cos(ph) - rho * np.sin(ph), d1 * np.sin(ph) + rho * np.cos(ph)
    sp = np.hypot(xp, zp)
    nr, nz = zp / sp, -xp / sp
    k1 = (rho ** 2 + 2 * d1 ** 2 - rho * d2) / sp ** 3
    H = k1 + nr / r
    rs_, _, _ = shape(a0, c, phs)
    rs, zs = Rc + fsrc * rs_ * np.cos(phs), fsrc * rs_ * np.sin(phs)
    rcl, _, _ = shape(a0, c, phc)
    q = solve(Rc + rcl * np.cos(phc), rcl * np.sin(phc), rs, zs)
    u = speed(r, z, nr, nz, rs, zs, q)
    G = H - 0.5 * We * u ** 2 - lam
    Gn = (COS @ G) / J * np.where(nn == 0, 1, 2)
    area = np.pi * np.mean(rho ** 2)                       # int_0^pi rho^2 dphi = full area
    cen = np.pi * np.mean(rho ** 3 * np.cos(ph))
    F = np.concatenate([Gn, [area - 2 / (3 * Rc), cen]])
    if full:
        return F, dict(circ=2 * q.sum(), u=u, G=G)
    return F


a_eq = np.sqrt(2 / (3 * np.pi * Rc))
L = np.log(8 * Rc / a_eq)
We0 = 4 * a_eq * (L - 2) ** 2 / (3 * Rc ** 2 * (L - 2.5))
Y = np.concatenate([[a_eq], np.zeros(N), [1 / a_eq - 0.5 * We0 * (np.pi * Rc / (L - 2) / (2 * np.pi * a_eq)) ** 2, We0]])
if len(sys.argv) > 5 and sys.argv[5] == 'warm':   # start from the measured shape instead of the circle
    Y[0] = geo['a0']; Y[1:min(N, 8) + 1] = geo['c'][:min(N, 8)]; Y[N + 1] = geo['lam']; Y[N + 2] = geo['We']
t0 = time.time()
Jm = None
for it in range(40):
    F = resid(Y)
    if Jm is None or (CHORD == 0):
      Jm = np.zeros((N + 3, N + 3))
      for j in range(N + 3):
          h = 1e-7 * max(abs(Y[j]), 1e-3 * a_eq if j <= N else 1.0)
          Yp = Y.copy(); Yp[j] += h
          Jm[:, j] = (resid(Yp) - F) / h
    dY = np.linalg.solve(Jm, -F)
    Y = Y + dY
    print(f'  it {it}: |F|={np.abs(F).max():.2e} |dY|={np.abs(dY).max():.2e} We={Y[N+2]:.12g} lam={Y[N+1]:.12g}', flush=True)
    if np.abs(dY).max() < 1e-13 * max(1, abs(Y[N + 1])):
        break
F, info = resid(Y, full=True)
a0, c, lam, We = Y[0], Y[1:N + 1], Y[N + 1], Y[N + 2]
# aspect ratio from the polar shape
pd = np.linspace(0, np.pi, 20001)
rho, _, _ = shape(a0, c, pd)
r, z = Rc + rho * np.cos(pd), rho * np.sin(pd)
asp = (r.max() - r.min()) / (2 * z.max())
print(f'{tag} INDEPENDENT (N={N}, nsrc={nsrc}, f={fsrc}): |F|={np.abs(F).max():.1e} max|G| at J nodes*a={np.abs(info["G"]).max()*a0:.1e}  time {time.time()-t0:.1f}s')
print(f'   We   indep {We:.12g}   data {geo["We"]:.12g}   rel {We/geo["We"]-1:+.2e}')
print(f'   lam  indep {lam:.12g}   data {geo["lam"]:.12g}   rel {lam/geo["lam"]-1:+.2e}')
print(f'   circ indep {info["circ"]:.12g}   data {geo["circ"]:.12g}   rel {info["circ"]/geo["circ"]-1:+.2e}')
print(f'   r_h  indep {r.min():.12g}   data {geo["r_h"]:.12g};  r_o indep {r.max():.12g} data {geo["r_o"]:.12g}')
print(f'   a0   indep {a0:.12g}   data {geo["a0"]:.12g}')
for n in range(1, min(N, 8) + 1):
    print(f'   c_{n} indep {c[n-1]:+.10e}   data {geo["c"][n-1]:+.10e}   diff {c[n-1]-geo["c"][n-1]:+.1e}')
print(f'   aspect-1 indep {asp-1:.10e}  data {geo["aspect"]-1:.10e}')
print(f'   tail |c_n| n>N-3: {np.abs(c[-3:]).max():.1e}')
# store the comparison: results/verify/independent_solver.json, one entry per (tag, N, nsrc, fsrc)
out_fn = os.path.join(REPO, 'results', 'verify', 'independent_solver.json')
rows = json.load(open(out_fn)) if os.path.exists(out_fn) else []
rows = [o for o in rows if (o['tag'], o['N'], o['nsrc'], o['fsrc']) != (tag, N, nsrc, fsrc)]
rows.append(dict(tag=tag, r_h=geo['r_h'], N=N, nsrc=nsrc, fsrc=fsrc, F=float(np.abs(F).max()),
                 We=float(We), We_rel=float(We / geo['We'] - 1), lam=float(lam), lam_rel=float(lam / geo['lam'] - 1),
                 circ_rel=float(info['circ'] / geo['circ'] - 1),
                 c_diff=[float(c[n] - geo['c'][n]) for n in range(min(N, 8))],
                 c_diff_max=float(max(abs(c[n] - geo['c'][n]) for n in range(min(N, 8)))),
                 tail=float(np.abs(c[-3:]).max()), time_s=time.time() - t0))
json.dump(sorted(rows, key=lambda o: (o['r_h'], o['N'])), open(out_fn, 'w'), indent=1)
