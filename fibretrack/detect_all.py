"""Detect fibre centres on every slice of one scan (local coordinates).

LoG sigma 4.0 px, min_distance 6 px (phantom-scored: recall 0.97-0.99, precision 1.00).
Sub-pixel by parabola on the LoG response. A 3-slice running mean along z is
used (fibres move < 0.1 px/slice so this costs no position accuracy).
Output: out/data/det_<scan>.npz  with z, y, x, resp (full-frame pixel coords).
"""
import sys, json, time; sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from common import *
import numpy as np, tifffile
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from concurrent.futures import ProcessPoolExecutor

SIG, MD, INT_THR = 4.0, 6, 40000

def sub(L, y, x):
    def p(a, b, c):
        d = a - 2 * b + c
        return 0.5 * (a - c) / d if d < 0 else 0.0
    return y + p(L[y - 1, x], L[y, x], L[y + 1, x]), x + p(L[y, x - 1], L[y, x], L[y, x + 1])

def work(args):
    name, z0, z1, box = args
    y0, y1, x0, x1 = box
    fl = files(name)
    def rd(z): return np.asarray(tifffile.memmap(str(fl[z]), mode="r")[y0:y1, x0:x1], np.float32)
    buf = {}
    out = []
    for z in range(z0, z1):
        for zz in (z - 1, z, z + 1):
            if 0 <= zz < len(fl) and zz not in buf: buf[zz] = rd(zz)
        for k in [k for k in buf if k < z - 1]: del buf[k]
        a = np.mean([buf[k] for k in (z - 1, z, z + 1) if k in buf], 0)
        L = -ndi.gaussian_laplace(a, SIG) * SIG * SIG
        pk = peak_local_max(L, min_distance=MD, threshold_abs=0, exclude_border=4)
        b = ndi.gaussian_filter(a, 2) > INT_THR
        pk = pk[b[pk[:, 0], pk[:, 1]]]
        for y, x in pk:
            yy, xx = sub(L, y, x)
            out.append((z, yy + y0, xx + x0, L[y, x]))
    return np.array(out, np.float32)

if __name__ == "__main__":
    name = sys.argv[1]
    bb = json.load(open(OUT / "data/bbox.json"))[name]
    pad = 60
    box = (max(0, bb["y0"] - pad), min(3660, bb["y1"] + pad), max(0, bb["x0"] - pad), min(3660, bb["x1"] + pad))
    t0 = time.time()
    chunks = [(name, z, min(2200, z + 100), box) for z in range(0, 2200, 100)]
    with ProcessPoolExecutor(int(sys.argv[2]) if len(sys.argv) > 2 else 6) as ex:
        res = list(ex.map(work, chunks))
    d = np.concatenate(res)
    np.savez_compressed(OUT / f"data/det_{name}.npz", z=d[:, 0].astype(np.int16), y=d[:, 1], x=d[:, 2], resp=d[:, 3], box=np.array(box))
    n = np.bincount(d[:, 0].astype(int))
    print(name, "box", box, "total", len(d), "per slice min/med/max", n.min(), int(np.median(n)), n.max(), f"{time.time()-t0:.0f}s")
