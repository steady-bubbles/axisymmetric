"""Build the page 'Axisymmetric bubbles'.

Reads the computed data in results/ (page/data.py) and the checks in
results/checks/, fills page/page_template.html and writes one self-contained
HTML file: the figures' data are embedded, and MathJax is loaded from a CDN.
The published copy is https://steady-bubbles.github.io/axisymmetric/ (its
repository steady-bubbles/steady-bubbles.github.io holds only the built page).

Usage: python page/build_page.py [out.html]      (default: build/index.html)
"""
import json
import math
import os
import re
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "page"))
from data import load_json  # noqa: E402

CHECKS = os.path.join(ROOT, "results", "checks")


def rnd(x, n=10):
    return float(f"{x:.{n}g}")


def bubbles_frames():
    bv = load_json("results/hires/viz/bubble_viz.json")
    keep = ['We', 'lam', 'z_pole', 'r_eq', 'z_max', 'aspect_ext', 'kmax', 'u_max', 'r', 'z', 'lines']
    out = []
    for f in bv['frames']:
        d = {k: f[k] for k in keep}
        for k in ('We', 'lam', 'z_pole', 'r_eq', 'z_max', 'aspect_ext', 'kmax', 'u_max'):
            d[k] = rnd(d[k], 13)
        out.append(d)
    return out, bv['special'], bv['r_inf']


def ring_frames():
    rv = load_json("results/hires/viz/rings_viz.json")
    keep = ['r_h', 'We', 'lam', 'r_o', 'z_max', 'circulation', 'kmax', 'u_max', 'xy', 'r0', 'r',
            'z', 'lines', 'fwin', 'flines']
    out = []
    for f in rv['frames']:
        d = {k: f[k] for k in keep if k in f}
        for k in ('r_h', 'We', 'lam', 'r_o', 'z_max', 'circulation', 'kmax', 'u_max'):
            d[k] = rnd(d[k], 13)
        out.append(d)
    return out


def crossings():
    c = sorted(load_json("results/rings/crossing_rings.json"), key=lambda p: p['r_h'])
    return [dict(m=i + 1, r_h=rnd(p['r_h'], 6), We=rnd(p['We'], 6)) for i, p in enumerate(c)]


def ring_fold():
    rs = load_json("results/rings/ring_sectors_all.json")
    for c in rs['crossings']:
        if c['sector'] == '0eQ' and c.get('resolved'):
            return dict(r_h=rnd(c['r_h'], 6), We=rnd(c['We'], 8), lam=rnd(c['lam'], 6))


def stability():
    b = load_json("results/bubble_sectors_all.json")
    bub = dict(n=len(b),
               e0_sign_changes=[[rnd(b[i]['We'], 8), rnd(b[i + 1]['We'], 8)] for i in range(len(b) - 1)
                                if np.sign(b[i]['0e'][0]) != np.sign(b[i + 1]['0e'][0])],
               o_min=rnd(min(min(p['0o']) for p in b), 5),
               nneg0e=sorted({p['nneg']['0e'] for p in b}))
    rs = load_json("results/rings/ring_sectors_all.json")
    rows = sorted(rs['rows'], key=lambda x: x['r_h'])
    ch = {}
    for s in ('0eQ', '0eG', '0o'):
        nn = [x['nneg'][s] for x in rows]
        ch[s] = [[rnd(rows[i]['r_h'], 6), rnd(rows[i + 1]['r_h'], 6), nn[i], nn[i + 1]]
                 for i in range(len(nn) - 1) if nn[i] != nn[i + 1]]
    ring = dict(n=len(rows), changes=ch, o_min=rnd(min(min(x['mu']['0o']) for x in rows), 5),
                r_h_range=[rnd(rows[0]['r_h'], 4), rnd(rows[-1]['r_h'], 6)])
    return dict(bubbles=bub, rings=ring)


def touching(bub, rings, We_star):
    bt = [[rnd(f['z_pole'], 8), rnd(f['We'] - We_star, 8)] for f in bub if f['z_pole'] < 0.2]
    rt = [[rnd(f['r_h'] * math.log(1 / f['r_h']), 8), rnd(f['We'] - We_star, 8), rnd(f['r_h'], 8)]
          for f in rings if f['r_h'] < 0.05]
    return dict(bub=bt, ring=rt)


def mns_data():
    d = json.load(open(os.path.join(CHECKS, "mns.json")))
    run = d['runs']['M28']
    fit = d['fits']['M28']['deg5']
    return dict(We=[p['We'] for p in run],
                z_pole=[rnd(p['z_pole'], 13) for p in run],
                chi=[rnd(p['chi'], 16) for p in run],
                lam=[rnd(p['lam'], 16) for p in run],
                shape=[rnd(p['shape_err_over_We'], 6) for p in run],
                P2=[rnd(p['P2_coeff_over_We'], 12) for p in run],
                chi_coef=[rnd(c, 12) for c in fit['chi']],
                lam_coef=[rnd(c, 12) for c in fit['lam']],
                spread=d['resolution_spread'])


def ms_data():
    d = json.load(open(os.path.join(CHECKS, "ms.json")))
    zf = []
    seen = set()
    for p in d['zero_flux']:
        if p['r_h'] in seen:
            continue
        seen.add(p['r_h'])
        q = {k: rnd(p[k], 10) for k in ('r_h', 'eps', 'beta', 'W', 'W_MS', 'dW', 'gamma_MS',
                                         'nu', 'nu_MS', 'dnu', 'We', 'pappus_err', 'R', 'eps_bar')}
        # leading-order zero-flux Weber number of Meyer-Seis at this ring's R and eps_bar
        L = math.log(8 / p['eps'])
        q['We_MS'] = rnd(4 * p['eps_bar'] * (L - 2) ** 2 / (3 * p['R'] ** 2 * (L - 2.5)), 10) if L > 2.5 else None
        zf.append(q)
    sheet = {}
    for f, pts in d['sheet'].items():
        sheet[f] = [{k: rnd(p[k], 8) for k in ('Q', 'eps', 'beta', 'W', 'W_MS', 'gamma',
                                                'gamma_MS', 'nu', 'nu_MS')} for p in pts]
    return dict(zero_flux=zf, sheet=sheet, fits=d['remainder_fits'])


def ms_curve():
    """Zero-flux curve of the Meyer-Seis family at leading order (gamma_MS = 0):
    We = 4 a (L-2)^2 / (3 R^2 (L-5/2)), a = sqrt(2/(3 pi R)), L = log(8R/a)."""
    out = []
    for R in np.geomspace(1.2, 3000, 160):
        a = math.sqrt(2 / (3 * math.pi * R))
        L = math.log(8 * R / a)
        if L <= 2.6:
            continue
        We = 4 * a * (L - 2) ** 2 / (3 * R ** 2 * (L - 2.5))
        out.append([rnd(R - a, 6), rnd(We, 6), rnd(R + a, 6)])
    return out


def precision():
    sp = load_json("results/special_points.json")
    levels = list(sp)
    fold = [sp[k]['fold']['We'] for k in levels]
    conv = load_json("results/convergence.json")
    rp = load_json("results/rings/precision.json")
    out = dict(
        fold_levels=[[k, rnd(sp[k]['fold']['We'], 16), sp[k]['fold']['M'], sp[k]['fold']['N']] for k in levels],
        fold_spread=float(max(fold) - min(fold)),
        dimple=[rnd(sp[k]['dimple_onset']['We'], 16) for k in levels],
        star=[rnd(sp[k]['star_loss']['We'], 16) for k in levels],
        conv=[[rnd(c['z_pole'], 6), rnd(c['levels'][-1]['We'], 15), float(c['dWe']), float(c['dlam'])] for c in conv],
        rings_hp=dict(n=len(rp), dWe=float(max(abs(r['dWe']) for r in rp)),
                      dlam=float(max(abs(r['dlam']) for r in rp)),
                      dcirc=float(max(abs(r['dcirculation']) for r in rp)),
                      r_h=[rnd(rp[0]['r_h'], 4), rnd(rp[-1]['r_h'], 4)]))
    ind = os.path.join(CHECKS, "independent_bubble.json")
    if os.path.exists(ind):
        d = json.load(open(ind))
        from decimal import Decimal
        wd = [Decimal(r['fold']['We_max']) for r in d['fold_runs']]
        wm = [float(w) for w in wd]
        dim = [float(r['state']['We']) for r in d['dimple_runs']]
        ref = sp[levels[-1]]['fold']['We']
        out['independent'] = dict(
            We_max=d['fold_runs'][-1]['fold']['We_max'][:18],
            n_res=len(wm), spread=float(max(wd) - min(wd)),
            minus_ref=float(np.mean(wm) - ref), rel=float((np.mean(wm) - ref) / ref),
            dimple=f"{np.mean(dim):.15f}", dimple_minus_ref=float(np.mean(dim) - sp[levels[-1]]['dimple_onset']['We']),
            seconds=round(d['total_time_s']))
    return out


def prior_work():
    """The paragraph on related works (page/prior_work.html), or nothing."""
    f = os.path.join(ROOT, "page", "prior_work.html")
    return open(f, encoding="utf-8").read().strip() if os.path.exists(f) else ""


CITE = re.compile(r"\{\{cite:([^}]+)\}\}")


def load_references():
    return json.load(open(os.path.join(ROOT, "page", "references.json"), encoding="utf-8"))


def citation_order(tpl, refs):
    """The keys cited in tpl ({{cite:a,b}}), sorted alphabetically by the authors'
    surnames (then year); the page numbers the references in this order."""
    cited = {k.strip() for m in CITE.finditer(tpl) for k in m.group(1).split(",")}
    unknown = cited - set(refs)
    if unknown:
        raise KeyError(f"unknown reference keys {sorted(unknown)}")

    def surname(author):
        """'M. El Sawi' -> 'El Sawi', 'W. Thomson (Lord Kelvin)' -> 'Thomson'."""
        author = re.sub(r"\(.*?\)", "", author).strip()
        return " ".join(t for t in author.split() if not t.endswith("."))

    def sort_key(k):
        names = [surname(a) for a in refs[k]["authors"].split(",")]
        return ([n.casefold() for n in names], refs[k]["year"])

    return sorted(cited, key=sort_key)


def resolve_citations(tpl):
    """Replace {{cite:a,b}} by [n, m] and {{references}} by the list of exactly the
    cited works from data/references.json, numbered in citation_order."""
    import html as H
    refs = load_references()
    pat = CITE
    order = citation_order(tpl, refs)

    def cite(m):
        nums = sorted(order.index(k.strip()) + 1 for k in m.group(1).split(","))
        return '<span class="cite">[' + ", ".join(f'<a href="#ref-{n}">{n}</a>' for n in nums) + "]</span>"

    tpl = pat.sub(cite, tpl)
    items = []
    for n, k in enumerate(order, 1):
        r = refs[k]
        title = H.escape(r["title"])
        if r.get("url"):
            title = f'<a href="{H.escape(r["url"])}">{title}</a>'
        venue = H.escape(r["venue"])
        if r.get("arxiv") and r.get("doi"):
            venue += f', <a href="https://arxiv.org/abs/{r["arxiv"]}">arXiv:{r["arxiv"]}</a>'
        elif r.get("arxiv"):
            venue = f'{venue}, arXiv:{r["arxiv"]}' if venue != "preprint" else f'arXiv:{r["arxiv"]}'
        items.append(f'<li id="ref-{n}">{H.escape(r["authors"])}, {title}, {venue} ({H.escape(r["year"])}).</li>')
    return tpl.replace("{{references}}", '<ol class="refs">\n' + "\n".join(items) + "\n</ol>")


def main():
    bub, special, r_inf = bubbles_frames()
    rings = ring_frames()
    We_star = special['touching']['We']
    # lambda = 0 on the ring branch (linear interpolation in log r_h)
    lam0 = None
    for a, b in zip(rings, rings[1:]):
        if a['lam'] < 0 <= b['lam']:
            t = -a['lam'] / (b['lam'] - a['lam'])
            lam0 = dict(r_h=rnd(math.exp((1 - t) * math.log(a['r_h']) + t * math.log(b['r_h'])), 4),
                        We=rnd((1 - t) * a['We'] + t * b['We'], 6))
            break
    data = dict(
        generated=time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
        r_inf=r_inf,
        bubbles=bub, rings=rings,
        special=dict(fold=special['fold'], dimple=special['dimple_onset'], star=special['star_loss'],
                     touching=special['touching'], ring_fold=ring_fold(), ring_lam0=lam0,
                     touch_slope=2.03642109),
        crossings=crossings(),
        touch=touching(bub, rings, We_star),
        mns=mns_data(),
        ms=ms_data(),
        ms_curve=ms_curve(),
    )
    payload = json.dumps(data, separators=(',', ':'))
    tpl = open(os.path.join(ROOT, "page", "page_template.html"), encoding="utf-8").read()
    tpl = tpl.replace("{{prior_work}}", prior_work())
    tpl = resolve_citations(tpl)
    html = tpl.replace("/*__DATA__*/null", payload)
    # the template starts with <title> and <style> (the head), then the body from <div class="wrap">
    cut = html.index('<div class="wrap"')
    head, body = html[:cut].strip(), html[cut:].strip()
    page = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">\n'
            f'{head}\n</head>\n<body>\n{body}\n</body>\n</html>\n')
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "build", "index.html")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(page)
    print(f"wrote {os.path.relpath(out)} ({len(page) / 1e6:.2f} MB), {len(bub)} bubble and {len(rings)} ring frames")


if __name__ == "__main__":
    main()
