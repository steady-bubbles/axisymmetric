"""Thin-ring prediction of c_2 vs measured c_2.
(A) asymptotic:  c2 = eps^2 * 3(12 mu^2 + 8 mu + 5) / (16 (9 mu - 1)),  mu = Lambda - 5/2,
    eps = a/R, Lambda = log(8R/a)   [leading order; -> eps^2 mu/4 for large Lambda]
(B) linearised about the EXACT circular torus (R = area centroid, a = area radius), flow by MFS
    (independent of the sheet solver), data We:
    c2 = -( [H_torus]_2 - (We/2) [|u|^2]_2 ) / (3/a - We <|u|^2>)
    also: We from the n = 1 (force) balance on the exact torus, and the torus circulation."""
import json, os, warnings
import numpy as np
warnings.filterwarnings('ignore')
from mfs import torus_speed_fourier
REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')

geo = json.load(open(os.path.join(REPO, 'results', 'verify', 'geom.json')))
rows = []
print(' tag     r_h     a/R      Lam   c2_meas     c2_asym(A)  ratio   c2_torus(B)  ratio   We_data    We_n1(torus)  rel   |Gam|data  |Gam|torus  rel')
for m in geo:
    if m['tag'] == 'thin:0' or m['r_h'] < 0.3:
        continue
    R, a = m['r_c'], m['a_eq']
    eps = a / R
    Lam = np.log(8 * R / a)
    mu = Lam - 2.5
    cA = eps ** 2 * 3 * (12 * mu ** 2 + 8 * mu + 5) / (16 * (9 * mu - 1))
    d = torus_speed_fourier(R, a, nsrc=64, fsrc=0.5, nq=512)
    ph = d['ph']
    Hc = np.cos(ph) / (R + a * np.cos(ph))
    H1, H2 = 2 * np.mean(Hc * np.cos(ph)), 2 * np.mean(Hc * np.cos(2 * ph))
    q0, q1, q2 = d['cu2'][0], d['cu2'][1], d['cu2'][2]
    We = m['We']
    g2 = H2 - 0.5 * We * q2
    cB = -g2 / (3 / a - We * q0)
    We_n1 = 2 * H1 / q1
    Gt = abs(d['G'])
    row = dict(tag=m['tag'], r_h=m['r_h'], eps=eps, Lam=Lam, c2=m['c'][1], c2_A=cA, c2_B=cB, We=We, We_n1=We_n1,
               Gam=abs(m['circ']), Gam_torus=Gt, aspect=m['aspect'], c3=m['c'][2], psi_err=d['psi_err'],
               cross=0.5 * We * (q2 - (H2 * 0)) , H2=H2, q0=q0, q1=q1, q2=q2)
    rows.append(row)
    print(f"{m['tag']:7s} {m['r_h']:7.4g} {eps:.3e} {Lam:6.3f} {m['c'][1]:.5e} {cA:.5e} {cA/m['c'][1]:.4f} "
          f"{cB:.5e} {cB/m['c'][1]:.5f}  {We:.5e} {We_n1:.5e} {We_n1/We-1:+.1e}  {abs(m['circ']):.6f} {Gt:.6f} {Gt/abs(m['circ'])-1:+.1e}",
          flush=True)
json.dump(rows, open(os.path.join(REPO, 'results', 'verify', 'predict.json'), 'w'), indent=1)
