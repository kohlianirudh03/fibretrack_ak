"""Per-slice fibre centre detection: LoG blob detector + sub-pixel centroid.

Glass fibres 12 um = 13 px at 0.9186 um. LoG sigma = r/sqrt(2) responds maximally
to a disc of radius r. Peaks must sit on fibre-bright material (multi-Otsu upper
class on the yarn), which rejects resin pockets and voids.
"""
import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max

R_PX = 6.5
SIG = R_PX / np.sqrt(2)

def detect(a, fib_thr, min_dist=8, resp_frac=0.25):
    a = a.astype(np.float32)
    L = -ndi.gaussian_laplace(a, SIG) * SIG**2
    bright = ndi.gaussian_filter(a, 1.5) > fib_thr
    thr = resp_frac * np.percentile(L[bright], 90) if bright.any() else np.inf
    pk = peak_local_max(L, min_distance=min_dist, threshold_abs=thr, exclude_border=4)
    if len(pk) == 0:
        return np.zeros((0, 3), np.float32)
    pk = pk[bright[pk[:, 0], pk[:, 1]]]
    # sub-pixel: intensity-weighted centroid in a 7x7 window of the LoG response
    out = []
    for y, x in pk:
        w = L[y - 3:y + 4, x - 3:x + 4]; w = np.clip(w - w.min(), 0, None)
        s = w.sum()
        if s <= 0: out.append((y, x, L[y, x])); continue
        yy, xx = np.mgrid[-3:4, -3:4]
        out.append((y + (w * yy).sum() / s, x + (w * xx).sum() / s, L[y, x]))
    return np.array(out, np.float32)
