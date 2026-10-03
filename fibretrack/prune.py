"""Physical pruning of hybrid tracks: fibre centres cannot be closer than ~11 px
(13 px fibres), fibres are glass-bright, and a real fibre is confirmed by the
detector on a reasonable fraction of slices."""
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from common import *
import numpy as np, tifffile
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

def path_intensity(name, Z, Y, X, nsamp=10):
    I = []
    for j in np.linspace(5, len(Z) - 6, nsamp).astype(int):
        a = ndi.gaussian_filter(np.asarray(tifffile.memmap(str(files(name)[int(Z[j])]), mode="r"), np.float32), 1.5)
        I.append(ndi.map_coordinates(a, [Y[:, j], X[:, j]], order=1))
    return np.median(np.array(I), 0)

def prune(Y, X, S, I, tol=8.0, frac=0.10, min_sup=0.25, min_I=45000):
    sup = S.mean(1)
    keep = (sup >= min_sup) & (I >= min_I)
    cols = list(range(0, Y.shape[1], 5))
    cnt = {}
    idx = np.nonzero(keep)[0]
    for j in cols:
        P = np.c_[Y[idx, j], X[idx, j]]
        for a, b in cKDTree(P).query_pairs(tol):
            key = (idx[a], idx[b]); cnt[key] = cnt.get(key, 0) + 1
    for (a, b), c in sorted(cnt.items(), key=lambda t: -t[1]):
        if c > frac * len(cols) and keep[a] and keep[b]:
            keep[b if sup[a] >= sup[b] else a] = False
    return keep

if __name__ == "__main__":
    name = sys.argv[1]
    h = dict(np.load(OUT / f"data/hyb_{name}.npz"))
    I = path_intensity(name, h["Z"], h["Y"], h["X"])
    keep = prune(h["Y"], h["X"], h["S"], I)
    Y, X, S = h["Y"][keep], h["X"][keep], h["S"][keep]
    nn = []
    for j in range(0, len(h["Z"]), 25):
        P = np.c_[Y[:, j], X[:, j]]; nn.append(cKDTree(P).query(P, 2)[0][:, 1])
    nn = np.array(nn)
    print(f"{name}: {len(keep)} -> {keep.sum()} fibres; support med {np.median(S.mean(1)):.2f} p10 {np.percentile(S.mean(1),10):.2f}; "
          f"NN p1/p5/med {np.percentile(nn,1):.1f}/{np.percentile(nn,5):.1f}/{np.median(nn):.1f} px")
    np.savez_compressed(OUT / f"data/hybp_{name}.npz", Z=h["Z"], Y=Y, X=X, S=S, box=h["box"], I=I[keep])
