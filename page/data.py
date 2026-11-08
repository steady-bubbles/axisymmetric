"""Read the computed data from results/ of this repository.

The page shows the high-resolution computation: the bubble frames
results/hires/viz/bubble_viz.json, the ring frames results/hires/viz/rings_viz.json
and, for the thin-ring check, the merged ring branch results/hires/rings_merged.json.
The stability spectra, the branch points, the precision studies and the flux scans
come from the earlier runs in results/ and results/rings/.
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def path(rel):
    return os.path.join(ROOT, rel)


def load_json(rel):
    with open(path(rel), encoding="utf-8") as f:
        return json.load(f)
