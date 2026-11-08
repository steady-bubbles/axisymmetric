"""Validation of biot_savart.py on the exact sphere solution (no solver code)."""
import numpy as np, time
from numpy.polynomial.legendre import leggauss
from biot_savart import Curve, velocity

npan, p = 24, 16
br = np.linspace(0, np.pi, npan + 1)
x, _ = leggauss(p)
al = (br[:-1, None] + br[1:, None]) / 2 + (br[1:] - br[:-1])[:, None] / 2 * x[None, :]
vals = dict(r=np.sin(al), z=-np.cos(al), c=np.ones_like(al), gamma=-1.5 * np.sin(al))
cv = Curve(br[:-1], br[1:], vals, mirror=False)

def exact(r, z):
    R2 = r * r + z * z
    R5 = R2 ** 2.5
    return np.array([0.5 * (-3 * z * r / R5), 1 + 0.5 * (1 / R2 ** 1.5 - 3 * z * z / R5)])

t0 = time.time()
print('interior points (u should be 0):')
for (r, z) in [(0.0, 0.0), (0.5, 0.3), (0.9, 0.0), (0.2, -0.95), (0.99, 0.05), (0.999 * np.sin(1.0), -0.999 * np.cos(1.0))]:
    u, d, ns, nph = velocity(cv, r, z)
    print(f'  ({r:.4f},{z:+.4f}) d={d:.1e} |u|={np.hypot(*u):.2e}  nsrc={ns} nphi={nph}')
print('exterior points vs exact potential flow:')
for (r, z) in [(1.5, 0.0), (0.0, 2.0), (1.001, 0.3), (1.0001 * np.sin(2.0), -1.0001 * np.cos(2.0)), (0.3, -1.2)]:
    u, d, ns, nph = velocity(cv, r, z)
    print(f'  ({r:.4f},{z:+.4f}) d={d:.1e} u={u} err={np.abs(u - exact(r, z)).max():.2e}')
print('time', time.time() - t0)
