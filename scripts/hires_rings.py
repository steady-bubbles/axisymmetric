"""The zero-flux ring branch, densely sampled at high resolution, in chains.

Each chain runs scripts/run_rings.py segments one after the other; a segment
starts from the last ring of the one before. The chains are independent and run
concurrently, sharing BUBBLES_NPROC worker processes (default: all cores). Every
segment asks for a Fourier tail below 1e-13 (--tail=1e-13); M is the largest
number of Fourier modes it may use.

The rings of results/hires/rings_merged.json (scripts/merge_hires_rings.py) come
from these chains, each started from a ring of the first computation
(results/rings/):

  touch   r_h 9.4e-4 -> 1.0e-5  from rings_neck3.json:0   factor 1/1.07  neck map, M <= 400
  neck    r_h 5.6e-3 -> 9.4e-4  from rings_neck.json:0    factor 1/1.07  neck map, M <= 400
  downN2  r_h 0.0148 -> 5.6e-3  from rings_down.json:6    factor 1/1.07  neck map, M <= 400
  downN   r_h 0.039  -> 0.0149  from rings_up_hp.json:0   factor 1/1.07  neck map, M <= 400
  upA     r_h 0.039  -> 0.49    from rings_up_hp.json:0   factor 1.07    M = 240
  upB     r_h 0.49   -> 5.3     from rings_up_hp.json:18  factor 1.07    M = 240
  thin    r_h 5.2    -> 24      from rings_up_hp.json:35  factor 1.10    M = 240
  thinD   r_h 25     -> 120     from rings_thin.json:6    factor 1.10    M <= 240 (starts at 128)
  thinC   r_h 111    -> 1200    from rings_thin3.json:0   factor 1.10    M <= 240, thin-ring predictor

The thin chain was stopped by hand at r_h = 24 (thinD and thinC, which start from
rings with M = 128, cover the rest at a fraction of the cost); the touch chain
stopped at r_h = 1.0e-5, where its next step did not finish the mode refinement
(M 252 -> 316) in two hours. The down chain (r_h 0.039 -> 5.6e-3 with the
curvature map, M <= 240) is kept for comparison: below r_h ~ 0.009 it leaves
Fourier tails up to 8e-7, so the merge leaves those rings out, and where both are
resolved it agrees with downN to 5e-13. up covers the range of upA + upB in one
chain. The default --chains are the nine chains of the table.

With --hours=H the chains stop after H hours: the running ring is abandoned
(every finished ring is already saved), and a later run resumes from there.

Re-running continues: a segment that has not reached its end is resumed from its
last ring in a new part file (<chain><k>_p<j>.json). A part that adds no ring
stops its chain.

Usage: python scripts/hires_rings.py outdir [--chains=touch,neck,downN,...] [--weights=4,3,...] [--hours=H]
"""
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UP = 1.07
CHAINS = {
    'up': [dict(start='results/rings/rings_up_hp.json:0', factor=UP, stop=5.3, M=240, flags=['--tail=1e-13'])],
    # run to r_h = 24 (it was started towards 120 and stopped there by hand); thinD and thinC go on
    'thin': [dict(start='results/rings/rings_up_hp.json:35', factor=1.1, stop=24.0, M=240, flags=['--tail=1e-13'])],
    'down': [dict(start='results/rings/rings_up_hp.json:0', factor=1 / UP, stop=5.6e-3, M=240,
                  flags=['--tail=1e-13'])],
    'neck': [dict(start='results/rings/rings_neck.json:0', factor=1 / UP, stop=9.4e-4, M=400,
                  flags=['--neck', '--tail=1e-13'])],
    'touch': [dict(start='results/rings/rings_neck3.json:0', factor=1 / UP, stop=8.6e-6, M=400,
                   flags=['--neck', '--tail=1e-13'])],
    # the same branch cut into pieces at rings of the first computation, to run them in parallel
    'upA': [dict(start='results/rings/rings_up_hp.json:0', factor=UP, stop=0.486, M=240, flags=['--tail=1e-13'])],
    'upB': [dict(start='results/rings/rings_up_hp.json:18', factor=UP, stop=5.3, M=240, flags=['--tail=1e-13'])],
    'thinC': [dict(start='results/rings/rings_thin3.json:0', factor=1.1, stop=1120.0, M=240,
                   flags=['--thin', '--tail=1e-13'])],
    'thinD': [dict(start='results/rings/rings_thin.json:6', factor=1.1, stop=120.0, M=240, flags=['--tail=1e-13'])],
    # the down range again with the neck map: with the curvature map M = 240 leaves a Fourier
    # tail of 1e-12 at r_h = 0.026 rising to 8e-7 at 0.0055 (chain down)
    'downN': [dict(start='results/rings/rings_up_hp.json:0', factor=1 / UP, stop=0.0149, M=400,
                   flags=['--neck', '--tail=1e-13'])],
    'downN2': [dict(start='results/rings/rings_down.json:6', factor=1 / UP, stop=5.6e-3, M=400,
                    flags=['--neck', '--tail=1e-13'])],
}
WEIGHTS = {'up': 4, 'thin': 3, 'down': 2, 'neck': 3, 'touch': 4, 'upA': 4, 'upB': 4, 'thinC': 4, 'thinD': 4,
           'downN': 4, 'downN2': 4}
DEADLINE = [None]


def log(msg):
    print(time.strftime('%H:%M:%S'), msg, flush=True)


def reached(r_h, seg):
    return r_h >= seg['stop'] if seg['factor'] > 1 else r_h <= seg['stop']


def parts(outdir, name, k):
    """Existing part files of segment k, in order."""
    out, j = [], 0
    while os.path.exists(os.path.join(outdir, f'{name}{k}_p{j}.json')):
        out.append(os.path.join(outdir, f'{name}{k}_p{j}.json'))
        j += 1
    return out


def run_chain(name, outdir, nproc):
    prev_last = None                       # _last.npz of the previous segment's final part
    for k, seg in enumerate(CHAINS[name]):
        while True:
            if DEADLINE[0] and time.time() > DEADLINE[0]:
                log(f'{name}{k}: time is up')
                return name, None
            done = parts(outdir, name, k)
            if done:
                pts = json.load(open(done[-1]))
                if reached(pts[-1]['r_h'], seg):
                    prev_last = done[-1].replace('.json', '_last.npz')
                    break
                if len(pts) < 2 and len(done) > 1:
                    log(f'{name}{k}: part {len(done) - 1} added no ring; stopping the chain')
                    return name, False
                start = done[-1].replace('.json', '_last.npz')
            else:
                start = os.path.join(ROOT, seg['start']) if seg['start'] else prev_last
            out = os.path.join(outdir, f'{name}{k}_p{len(done)}.json')
            cmd = [sys.executable, '-u', os.path.join(ROOT, 'scripts', 'run_rings.py'), start, out,
                   repr(seg['factor']), repr(seg['stop']), str(seg['M'])] + seg['flags']
            env = dict(os.environ, BUBBLES_NPROC=str(nproc))
            log(f'{name}{k}: {" ".join(os.path.relpath(c, ROOT) if c.startswith(ROOT) else c for c in cmd[2:])}'
                f'  ({nproc} processes)')
            with open(out.replace('.json', '.log'), 'w') as fh:
                proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT)
                while proc.poll() is None:
                    if DEADLINE[0] and time.time() > DEADLINE[0]:
                        proc.terminate()
                        try:
                            proc.wait(60)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                        log(f'{name}{k}: time is up, part {len(done)} stopped')
                        return name, None
                    time.sleep(5)
                rc = proc.returncode
            n = len(json.load(open(out))) if os.path.exists(out) else 0
            log(f'{name}{k}: part {len(done)} exited with {rc}, {n} rings')
            if n < 2:
                log(f'{name}{k}: no progress; stopping the chain')
                return name, False
    log(f'{name}: complete')
    return name, True


def main():
    opts = dict(a[2:].split('=', 1) for a in sys.argv[1:] if a.startswith('--'))
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    outdir = os.path.abspath(args[0] if args else os.path.join(ROOT, 'results', 'hires', 'rings'))
    os.makedirs(outdir, exist_ok=True)
    names = opts.get('chains', 'touch,neck,downN2,downN,upA,upB,thin,thinD,thinC').split(',')
    if 'hours' in opts:
        DEADLINE[0] = time.time() + 3600 * float(opts['hours'])
    w = [float(x) for x in opts['weights'].split(',')] if 'weights' in opts else [WEIGHTS[n] for n in names]
    total = int(os.environ.get('BUBBLES_NPROC', os.cpu_count() or 1))
    nprocs = [max(1, round(total * x / sum(w))) for x in w]
    log(f'chains {names}, processes {nprocs} of {total}, output {os.path.relpath(outdir, ROOT)}')
    with ThreadPoolExecutor(len(names)) as ex:
        res = list(ex.map(lambda a: run_chain(*a), [(n, outdir, p) for n, p in zip(names, nprocs)]))
    log(f'finished: {res}')
    sys.exit(0 if all(ok is not False for _, ok in res) else 1)


if __name__ == '__main__':
    main()
