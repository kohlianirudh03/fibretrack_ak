"""Second-pass detection: finer LoG restricted to bright area not already covered
by a first-pass detection; a new peak must be glass-bright and >= MINSEP px from
every detection (fibres cannot overlap), so it cannot double-count."""
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage.feature import peak_local_max
SIG2, MINSEP, I2 = 3.2, 10.0, 48000
YY, XX = np.mgrid[-8:9, -8:9]; DISK = (YY ** 2 + XX ** 2) <= 6.5 ** 2

def second_pass(a, first_yx):
    g = ndi.gaussian_filter(a, 1.5)
    cov = np.zeros(a.shape, bool)
    if len(first_yx):
        iy = np.clip(np.rint(first_yx[:, 0]).astype(int), 0, a.shape[0] - 1); ix = np.clip(np.rint(first_yx[:, 1]).astype(int), 0, a.shape[1] - 1)
        cov[iy, ix] = True; cov = ndi.binary_dilation(cov, structure=DISK)
    L = -ndi.gaussian_laplace(a, SIG2) * SIG2 ** 2
    L[cov | (g < I2)] = 0
    pk = peak_local_max(L, min_distance=6, threshold_abs=0, exclude_border=4)
    if not len(pk): return np.zeros((0, 2))
    pk = pk[np.argsort(-L[pk[:, 0], pk[:, 1]])]
    keep = []
    allp = list(first_yx)
    tree = cKDTree(first_yx) if len(first_yx) else None
    for p in pk:
        if tree is not None and tree.query(p)[0] < MINSEP: continue
        if keep and np.min(np.hypot(*(np.array(keep) - p).T)) < MINSEP: continue
        keep.append(p.astype(float))
    return np.array(keep).reshape(-1, 2)
