"""Newton solvers for the bubble equations."""
import numpy as np

from .bubble import BubbleState


def newton_fixed_we(a, lam, We, panels, pmap=None, tol=1e-13, maxit=25,
                    verbose=False, bie_opts=None):
    """Solve F(a, lam; We) = 0 by Newton's method with the analytic Jacobian."""
    a = np.array(a, float)
    M = len(a)
    hist = []
    for it in range(maxit):
        st = BubbleState(a, panels, pmap, bie_opts)
        F = st.residual(We, lam)
        J, _ = st.jacobian(We)
        dX = np.linalg.solve(J, -F)
        a += dX[:M]
        lam += dX[M]
        hist.append((np.abs(F).max(), np.abs(dX).max()))
        if verbose:
            print(f"  it {it}: |F|={hist[-1][0]:.3e} |dX|={hist[-1][1]:.3e}")
        if np.abs(dX).max() < tol:
            break
    st = BubbleState(a, panels, pmap, bie_opts)
    return a, lam, st, hist
