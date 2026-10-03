"""Hybrid fibre tracker: optical-flow propagation + detection snapping.

Why: LoG detections are ~80% complete per slice, and as two touching fibres merge
the LoG peak drifts towards the merged blob before vanishing, so detection-only
linking fragments (929 of 1494 interior track ends had nothing within 4 px of the
extrapolated position). Dense Lucas-Kanade flow between slices STEP apart follows
the image texture regardless of the detector; snapping to a detection within SNAP
px removes flow drift whenever the detector is reliable.

Per scan: seeds = detections on long linked tracks at the seed slice; propagate
both ways; orphan linked tracks (not explained by any fibre) seed new fibres;
duplicates merged. Output out/data/hyb_<scan>.npz:
  Z (nz,), Y (nf,nz), X (nf,nz), S (nf,nz) bool snapped-to-detection
"""
import sys, time
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from common import *
import numpy as np, tifffile
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage.registration import optical_flow_ilk
from concurrent.futures import ProcessPoolExecutor

STEP, SNAP, FLOW_R, NSEED = 4, 6.0, 7, 5
FLOWDIR = OUT / "flow"


def _flow_job(args):
    name, z0, z1, box = args
    fl = files(name)

    def sl(z):
        return np.clip(np.asarray(tifffile.memmap(str(fl[z]), mode="r")[box[0]:box[1], box[2]:box[3]], np.float32), 8000, 65535)

    v, u = optical_flow_ilk(sl(z0), sl(z1), radius=FLOW_R, num_warp=3, gaussian=True, prefilter=False)
    return z0, ndi.gaussian_filter(v, 1.5).astype(np.float16), ndi.gaussian_filter(u, 1.5).astype(np.float16)


def flows(name, box, Z, workers=6):
    """forward flow Z[i] -> Z[i+1], memmapped on C:"""
    FLOWDIR.mkdir(exist_ok=True, parents=True)
    fn = FLOWDIR / f"flow_{name}.npy"
    H, W = box[1] - box[0], box[3] - box[2]
    if fn.exists():
        return np.load(fn, mmap_mode="r")
    tmp = FLOWDIR / f"flow_{name}.tmp.npy"
    F = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float16, shape=(len(Z) - 1, 2, H, W))
    jobs = [(name, int(Z[i]), int(Z[i + 1]), box) for i in range(len(Z) - 1)]
    idx = {int(z): i for i, z in enumerate(Z)}
    with ProcessPoolExecutor(workers) as ex:
        for k, (z0, v, u) in enumerate(ex.map(_flow_job, jobs, chunksize=4)):
            F[idx[z0], 0] = v
            F[idx[z0], 1] = u
            if k % 100 == 0:
                print(f"  flow {name} {k}/{len(jobs)}", flush=True)
    F.flush()
    del F
    tmp.rename(fn)
    return np.load(fn, mmap_mode="r")


def _snap(p, D, S_row, idx):
    if D is None or not len(D) or not len(idx):
        return p
    dd, k = cKDTree(D).query(p[idx], distance_upper_bound=SNAP)
    ok = np.isfinite(dd)
    taken = set()
    for m in np.argsort(dd):                      # one-to-one: closest claimant wins
        if not ok[m]:
            break
        if k[m] in taken:
            ok[m] = False
            continue
        taken.add(k[m])
    p[idx[ok]] = D[k[ok]]
    S_row[idx[ok]] = True
    return p


KP, KW, WMAX = 0.7, 0.03, 0.4


def _assoc(p, D, idx, gate):
    """one-to-one nearest detection within gate for fibres idx; returns (fibre idx, det rows)"""
    if D is None or not len(D) or not len(idx):
        return idx[:0], idx[:0]
    dd, k = cKDTree(D).query(p[idx], distance_upper_bound=gate)
    ok = np.isfinite(dd)
    taken = set()
    for m in np.argsort(dd):
        if not ok[m]:
            break
        if k[m] in taken:
            ok[m] = False
            continue
        taken.add(k[m])
    return idx[ok], k[ok]


def _flow_at(F, i, q, box):
    v = np.asarray(F[i, 0], np.float32); u = np.asarray(F[i, 1], np.float32)
    c = [q[:, 0] - box[0], q[:, 1] - box[2]]
    return np.c_[ndi.map_coordinates(v, c, order=1, mode="nearest"), ndi.map_coordinates(u, c, order=1, mode="nearest")]


def propagate(seeds, Z, F, det, box):
    """seeds (n, 3) = [start step index, y, x]. Alpha-beta filter per fibre:
    predict p' = p + flow(p) + w  (w = fibre's own velocity relative to the flow,
    which the locally-averaging flow cannot see); a detection within SNAP px of p'
    corrects p (gain KP) and w (gain KW). All fibres move together in one forward
    and one backward sweep so detection association is one-to-one."""
    seeds = np.asarray(seeds, float)
    i0 = seeds[:, 0].astype(int)
    n, nz = len(seeds), len(Z)
    Y = np.full((n, nz), np.nan, np.float32)
    X = Y.copy()
    S = np.zeros((nz, n), bool)
    Y[np.arange(n), i0] = seeds[:, 1]
    X[np.arange(n), i0] = seeds[:, 2]
    S[i0, np.arange(n)] = True
    lo = np.array([box[0] + 1, box[2] + 1]); hi = np.array([box[1] - 2, box[3] - 2])
    for direction in (+1, -1):
        p = np.full((n, 2), np.nan); w = np.zeros((n, 2))
        steps = range(0, nz - 1) if direction > 0 else range(nz - 1, 0, -1)
        for i in steps:
            new = np.nonzero(i0 == i)[0]
            p[new] = seeds[new, 1:]; w[new] = 0
            act = np.nonzero(i0 <= i)[0] if direction > 0 else np.nonzero(i0 >= i)[0]
            if not len(act):
                continue
            j = i + direction
            if direction > 0:
                pred = p[act] + _flow_at(F, i, p[act], box) + w[act]
            else:
                tgt = p[act]; q = tgt.copy()
                for _ in range(3):
                    q = tgt - _flow_at(F, j, q, box)
                pred = q + w[act]
            p[act] = pred
            fi, k = _assoc(p, det.get(int(Z[j])), act, SNAP)
            if len(fi):
                e = det[int(Z[j])][k] - p[fi]
                p[fi] += KP * e
                w[fi] = np.clip(w[fi] + KW * e, -WMAX, WMAX)
                S[j, fi] = True
            p[act] = np.clip(p[act], lo, hi)
            Y[act, j], X[act, j] = p[act, 0], p[act, 1]
    return Y, X, S.T.copy()


def dedup(Y, X, S, tol=4.0, frac=0.3):
    """drop fibres that sit within tol px of a better-supported fibre for > frac of the scan"""
    n = len(Y)
    keep = np.ones(n, bool)
    sup = S.mean(1)
    cols = range(0, Y.shape[1], 10)
    cnt = {}
    for j in cols:
        P = np.c_[Y[:, j], X[:, j]]
        for a, b in cKDTree(P).query_pairs(tol):
            cnt[(a, b)] = cnt.get((a, b), 0) + 1
    m = len(cols)
    for (a, b), c in sorted(cnt.items(), key=lambda t: -t[1]):
        if c > frac * m and keep[a] and keep[b]:
            keep[b if sup[a] >= sup[b] else a] = False
    return keep


def load_det(name):
    d = np.load(OUT / f"data/det_{name}.npz")
    box = tuple(int(b) for b in d["box"])
    zz = d["z"].astype(int)
    o = np.argsort(zz, kind="stable")
    zs, ys, xs = zz[o], d["y"][o], d["x"][o]
    u, st = np.unique(zs, return_index=True)
    en = np.r_[st[1:], len(zs)]
    det = {int(z_): np.c_[ys[a:b], xs[a:b]].astype(np.float64) for z_, a, b in zip(u, st, en)}
    return det, box


def _d2_job(args):
    import detect2 as D2
    name, z, box, P1 = args
    fl = files(name)
    a = np.mean([np.clip(np.asarray(tifffile.memmap(str(fl[k]), mode="r")[box[0]:box[1], box[2]:box[3]], np.float32), 8000, 65535)
                 for k in (z - 1, z, z + 1) if 0 <= k < len(fl)], 0)
    P2 = D2.second_pass(a, P1 - np.array([box[0], box[2]]))
    return z, P2 + np.array([box[0], box[2]])


def second_pass_dets(name, box, Z, det, workers=6):
    fn = OUT / f"data/det2_{name}.npz"
    if fn.exists():
        d = np.load(fn); return {int(z): d[str(int(z))] for z in Z if str(int(z)) in d.files}
    jobs = [(name, int(z), box, det.get(int(z), np.zeros((0, 2)))) for z in Z]
    out = {}
    with ProcessPoolExecutor(workers) as ex:
        for z, P2 in ex.map(_d2_job, jobs, chunksize=8):
            out[z] = P2
    np.savez_compressed(fn, **{str(k): v for k, v in out.items()})
    return out


def run(name, workers=6):
    t0 = time.time()
    det, box = load_det(name)
    Z = np.unique(np.r_[np.arange(0, 2200, STEP), 2199])
    F = flows(name, box, Z, workers)
    print(f"{name}: flows ready {time.time()-t0:.0f}s", flush=True)
    d2 = second_pass_dets(name, box, Z, det, workers)
    n2 = 0
    for z, P2 in d2.items():
        if len(P2):
            det[z] = np.r_[det.get(z, np.zeros((0, 2))), P2]; n2 += len(P2)
    print(f"{name}: second-pass detections added {n2} ({n2/len(Z):.0f} per step slice) {time.time()-t0:.0f}s", flush=True)
    t = np.load(OUT / f"data/tracks_{name}.npz")
    P, O = t["pts"], t["off"]
    tr = [P[O[i]:O[i + 1]] for i in range(len(O) - 1) if O[i + 1] - O[i] >= 100]
    # independent clean runs from several seed slices; union; physical pruning
    import prune as PR
    runs = []
    for i0 in np.linspace(0.08, 0.92, NSEED) * (len(Z) - 1):
        i0 = int(round(i0)); zseed = int(Z[i0])
        seeds = np.array([[i0, np.interp(zseed, q[:, 0], q[:, 1]), np.interp(zseed, q[:, 0], q[:, 2])]
                          for q in tr if q[0, 0] <= zseed <= q[-1, 0]])
        y_, x_, s_ = propagate(seeds, Z, F, det, box)
        runs.append((y_, x_, s_))
        print(f"  seed slice {zseed}: {len(seeds)} seeds, support median {np.median(s_.mean(1)):.2f}", flush=True)
    Y = np.concatenate([r[0] for r in runs]); X = np.concatenate([r[1] for r in runs]); S = np.concatenate([r[2] for r in runs])
    I = PR.path_intensity(name, Z, Y, X)
    keep = PR.prune(Y, X, S, I)
    Y, X, S = Y[keep], X[keep], S[keep]
    agree = np.zeros(len(Y), int)
    for (y_, x_, s_) in runs:
        cand = set()
        for j in np.linspace(0, len(Z) - 1, 7).astype(int):
            _, k = cKDTree(np.c_[y_[:, j], x_[:, j]]).query(np.c_[Y[:, j], X[:, j]])
            cand |= set(zip(range(len(Y)), k))
        ok = np.zeros(len(Y), bool)
        for f, k in cand:
            if not ok[f] and np.mean(np.hypot(y_[k] - Y[f], x_[k] - X[f]) < 1.5) >= 0.9:
                ok[f] = True
        agree += ok
    sup = S.mean(1)
    print(f"{name}: runs reproducing each fibre (of {NSEED}): " + str(np.bincount(agree, minlength=NSEED + 1).tolist()), flush=True)
    print(f"{name}: {len(Y)} fibres; support median {np.median(sup):.3f} p10 {np.percentile(sup, 10):.3f}; "
          f"support>0.5: {np.sum(sup > 0.5)}; {time.time()-t0:.0f}s", flush=True)
    np.savez_compressed(OUT / f"data/hyb_{name}.npz", Z=Z, Y=Y, X=X, S=S, box=np.array(box), agree=agree)


if __name__ == "__main__":
    for n in sys.argv[1:]:
        run(n)
