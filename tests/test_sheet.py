import json
import os
import warnings

import numpy as np

from bubbles.bubble import BubbleState
from bubbles.flowfield import StreamFunction
from bubbles.geometry import Panels, ParamMap
from bubbles.sheet import bubble_sheet_speed

warnings.filterwarnings('ignore')
ROOT = os.path.join(os.path.dirname(__file__), '..')


def test_stream_function_sphere():
    st = BubbleState(np.zeros(8), Panels.uniform(8))
    sf = StreamFunction(st)
    r = np.array([0.5, 1.2, 2.0, 0.1])
    z = np.array([1.3, 0.3, -1.0, 1.2])
    R = np.hypot(r, z)
    assert np.max(np.abs(sf(r, z) - 0.5 * r ** 2 * (1 - 1 / R ** 3))) < 1e-13
    assert np.max(np.abs(sf(np.array([0.3]), np.array([0.2])))) < 1e-13


def test_sheet_formulation_matches_potential_solver():
    """First-kind vortex-sheet solve reproduces the surface speed of the
    second-kind potential solver on the sphere and on a dimpled bubble."""
    st = BubbleState(np.zeros(8), Panels.uniform(8))
    gam, _ = bubble_sheet_speed(st)
    assert np.max(np.abs(gam - st.phi_s)) < 1e-10
    pts = json.load(open(os.path.join(ROOT, 'results', 'branch.json')))
    p = pts[80]
    st = BubbleState(np.array(p['a']), Panels(np.array(p['breaks_half'])), ParamMap(p['map_b']))
    gam, _ = bubble_sheet_speed(st)
    assert np.max(np.abs(gam - st.phi_s)) < 1e-9 * np.max(np.abs(st.phi_s))
