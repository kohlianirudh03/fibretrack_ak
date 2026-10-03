"""Link per-slice detections into fibre tracks within one scan.

Constant-velocity prediction (velocity from a linear fit to the last <=15 points),
greedy nearest assignment inside a 3 px gate (fibre spacing ~13-15 px, motion
<0.1 px/slice), gaps up to MAXGAP slices bridged. Then a second pass joins track
fragments whose extrapolated ends meet within 3 px across <= 40 slices.
"""
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from common import *
import numpy as np
from scipy.spatial import cKDTree

GATE, MAXGAP, MINLEN = 3.0, 10, 30

def link(z, y, x):
    order = np.argsort(z, kind="stable"); z, y, x = z[order], y[order], x[order]
    zs = np.unique(z); starts = np.searchsorted(z, zs); ends = np.r_[starts[1:], len(z)]
    tracks = []          # each: list of (z, y, x)
    active = []          # indices into tracks
    for zi, s, e in zip(zs, starts, ends):
        P = np.c_[y[s:e], x[s:e]]
        # drop stale
        active = [t for t in active if zi - tracks[t][-1][0] <= MAXGAP]
        used = np.zeros(len(P), bool)
        if active and len(P):
            pred = []
            for t in active:
                tr = np.asarray(tracks[t][-15:])
                if len(tr) >= 4:
                    A = np.c_[tr[:, 0] - tr[-1, 0], np.ones(len(tr))]
                    cy = np.linalg.lstsq(A, tr[:, 1], rcond=None)[0]; cx = np.linalg.lstsq(A, tr[:, 2], rcond=None)[0]
                    dz = zi - tr[-1, 0]; pred.append((cy[1] + cy[0] * dz, cx[1] + cx[0] * dz))
                else:
                    pred.append((tr[-1, 1], tr[-1, 2]))
            pred = np.array(pred)
            d, j = cKDTree(P).query(pred, distance_upper_bound=GATE)
            for k in np.argsort(d):
                if not np.isfinite(d[k]) or used[j[k]]: continue
                used[j[k]] = True; tracks[active[k]].append((zi, P[j[k], 0], P[j[k], 1]))
        for k in np.nonzero(~used)[0]:
            tracks.append([(zi, P[k, 0], P[k, 1])]); active.append(len(tracks) - 1)
    return [np.asarray(t, np.float32) for t in tracks]

def endfit(t, end, n=40):
    s = t[-n:] if end else t[:n]
    A = np.c_[s[:, 0], np.ones(len(s))]
    return np.linalg.lstsq(A, s[:, 1], rcond=None)[0], np.linalg.lstsq(A, s[:, 2], rcond=None)[0]

def join(tracks, maxdz=40, gate=3.0):
    tracks = [t for t in tracks if len(t) >= 8]
    changed = True
    while changed:
        changed = False
        ends = np.array([t[-1, 0] for t in tracks]); begs = np.array([t[0, 0] for t in tracks])
        fits_e = [endfit(t, True) for t in tracks]; fits_b = [endfit(t, False) for t in tracks]
        bpos = np.array([[t[0, 1], t[0, 2]] for t in tracks]); tree = cKDTree(bpos)
        cand = []
        for i, t in enumerate(tracks):
            (cy, cx) = fits_e[i]
            for j in tree.query_ball_point([t[-1, 1], t[-1, 2]], gate + 6):
                dz = begs[j] - ends[i]
                if j == i or dz <= 0 or dz > maxdz: continue
                py, px = cy[0] * begs[j] + cy[1], cx[0] * begs[j] + cx[1]
                d = np.hypot(py - bpos[j, 0], px - bpos[j, 1])
                if d < gate: cand.append((d + 0.02 * dz, i, j))
        cand.sort(); usedi, usedj = set(), set(); merge = {}
        for c, i, j in cand:
            if i in usedi or j in usedj: continue
            usedi.add(i); usedj.add(j); merge[i] = j
        if merge:
            changed = True
            heads = set(range(len(tracks))) - set(merge.values())
            new = []
            for h in heads:
                seq = [tracks[h]]; k = h
                while k in merge: k = merge[k]; seq.append(tracks[k])
                new.append(np.concatenate(seq))
            tracks = new
    return tracks

if __name__ == "__main__":
    import time
    name = sys.argv[1]
    d = np.load(OUT / f"data/det_{name}.npz")
    t0 = time.time()
    tr = link(d["z"].astype(np.int32), d["y"], d["x"])
    n0 = len(tr)
    tr = join(tr)
    tr = [t for t in tr if len(t) >= MINLEN]
    L = np.array([t[-1, 0] - t[0, 0] + 1 for t in tr])
    print(f"{name}: raw {n0} -> {len(tr)} tracks >= {MINLEN}; full-length (>=2150 slices) {np.sum(L >= 2150)}; median len {np.median(L):.0f}; {time.time()-t0:.0f}s")
    np.savez_compressed(OUT / f"data/tracks_{name}.npz", pts=np.concatenate(tr), off=np.cumsum([0] + [len(t) for t in tr]))
