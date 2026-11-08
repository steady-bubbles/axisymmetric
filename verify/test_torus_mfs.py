"""Validate the thin-torus expansion of the surface speed (zero flux, exact circular torus):
 |u| = (Gam/2 pi a)[1 + eps u1 cos phi + eps^2 (u20 + u2 cos 2phi) + O(eps^3)],
 u1 = 4 pi R/Gam - (Lambda - 1/2),  u2 = Lambda - pi R/Gam - 17/16  (= 15/16 on Gam = pi R/(Lambda-2))."""
import numpy as np, warnings
warnings.filterwarnings('ignore')
from mfs import torus_speed_fourier
R = 1.0
print(' a/R      Gam(MFS)   piR/(L-2)   circ_line   psi_err   U1/(G/2pia)/eps  u1_pred    U2/(G/2pia)/eps^2  u2_pred  [MFS nsrc 48 vs 64 diff in U2]')
for eps in [0.2, 0.1, 0.05, 0.02, 0.01, 0.005]:
    a = eps * R
    L = np.log(8 * R / a)
    res = {}
    for ns in (48, 64):
        res[ns] = torus_speed_fourier(R, a, nsrc=ns, fsrc=0.5)
    d = res[64]
    Gam = -d['G']
    u0 = Gam / (2 * np.pi * a)
    cu = d['cu'] * np.sign(d['cu'][0])
    U1 = cu[1] / u0 / eps
    U2 = cu[2] / u0 / eps ** 2
    U2b = res[48]['cu'][2] * np.sign(res[48]['cu'][0]) / (-res[48]['G'] / (2 * np.pi * a)) / eps ** 2
    u1p = 4 * np.pi * R / Gam - (L - 0.5)
    u2p = L - np.pi * R / Gam - 17 / 16
    print(f'{eps:6.3f}  {Gam:10.6f}  {np.pi*R/(L-2):10.6f}  {abs(d["circ_line"]):10.6f}  {d["psi_err"]:.1e}  '
          f'{U1:12.6f}  {u1p:10.6f}   {U2:12.6f}  {u2p:10.6f}  {abs(U2-U2b):.1e}')
