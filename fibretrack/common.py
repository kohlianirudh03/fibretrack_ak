"""Project configuration for fibretrack scripts.

Reads fibretrack_config.json from the current directory (or $FIBRETRACK_CONFIG):
  {"voxel_um": 0.9186,
   "out": "C:/work/myyarn/out",
   "scans": {"top": "F:/scan_top", "mid": "F:/scan_mid"}}
"""
import json, os
from pathlib import Path
import numpy as np
import tifffile

_cfg_path = Path(os.environ.get("FIBRETRACK_CONFIG", "fibretrack_config.json"))
CFG = json.load(open(_cfg_path)) if _cfg_path.exists() else {"voxel_um": 1.0, "out": "out", "scans": {}}
VOX_UM = float(CFG["voxel_um"])
OUT = Path(CFG["out"])
SCANS = CFG["scans"]


def files(name):
    f = sorted(p for p in Path(SCANS[name]).iterdir() if p.suffix.lower() in (".tif", ".tiff"))
    assert f, name
    return f


def read(name, z, bin_=1, crop=None):
    a = tifffile.memmap(str(files(name)[z]), mode="r")
    if crop is not None:
        (y0, y1), (x0, x1) = crop
        a = a[y0:y1, x0:x1]
    a = np.asarray(a, np.float32)
    if bin_ > 1:
        h, w = (a.shape[0] // bin_) * bin_, (a.shape[1] // bin_) * bin_
        a = a[:h, :w].reshape(h // bin_, bin_, w // bin_, bin_).mean((1, 3))
    return a
