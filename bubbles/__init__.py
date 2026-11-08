"""Steady axisymmetric vacuum bubbles in an inviscid, irrotational fluid.

Numerical continuation of the branch of fore-aft symmetric bubbles solving

    H - (We/2) |grad phi|^2 = lambda   on S,   |B| = 4 pi / 3,

where phi is the exterior velocity potential (Neumann condition on S,
grad phi -> e_3 at infinity), H is twice the mean curvature (H = 2 on the
unit sphere) and We = rho V^2 R / sigma, following Meyer, Niebel & Seis,
"Steady bubbles and drops in inviscid fluids" (gamma = 0, hollow vortex case).
"""
