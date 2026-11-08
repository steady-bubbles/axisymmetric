"""Near-core expansion of the unit vortex-ring stream function (R = 1), high precision.
Psi = (1/2pi)[log(8/rho) - 2] + (rho cos phi/4pi)[log(8/rho) - 1]
      + (1/2pi)[ q0(rho) + ((1/16) rho^2 log rho + D2 rho^2) cos 2phi ] + O(rho^3 log)
Extract D2 (and check the first two orders) by Fourier analysis in phi."""
import mpmath as mp
mp.mp.dps = 40

def psi(r, z, R=1):
    # Lamb: Psi = (1/2pi) sqrt(rR) [ (2/k - k) K(k) - (2/k) E(k) ],  k^2 = 4rR/((r+R)^2+z^2)
    m = 4 * r * R / ((r + R) ** 2 + z ** 2)
    k = mp.sqrt(m)
    return mp.sqrt(r * R) / (2 * mp.pi) * ((2 / k - k) * mp.ellipk(m) - 2 / k * mp.ellipe(m))

def fourier(rho, n, N=64):
    s = 0
    for j in range(N):
        ph = 2 * mp.pi * (j + mp.mpf(1) / 2) / N
        s += psi(1 + rho * mp.cos(ph), rho * mp.sin(ph)) * mp.cos(n * ph)
    return s * (2 if n else 1) / N

for rho in [mp.mpf('1e-2'), mp.mpf('1e-3'), mp.mpf('1e-4')]:
    f0 = fourier(rho, 0); f1 = fourier(rho, 1); f2 = fourier(rho, 2)
    c0 = f0 * 2 * mp.pi - (mp.log(8 / rho) - 2)
    c1 = f1 * 4 * mp.pi / rho - (mp.log(8 / rho) - 1)
    D2 = f2 * 2 * mp.pi / rho ** 2 - mp.log(rho) / 16
    print(f'rho={mp.nstr(rho,3)}: c0-resid/rho^2={mp.nstr(c0/rho**2,12)}  c1-resid/rho={mp.nstr(c1/rho,12)}  D2={mp.nstr(D2,15)}  d2=D2+log8/16={mp.nstr(D2+mp.log(8)/16,15)}')
