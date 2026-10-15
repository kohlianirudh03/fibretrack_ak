# fibretrack_ak

Fibre tracking, orientation fields and twisted-yarn theory tests for X-ray CT of yarns and
fibre composites (glass, carbon, hybrid). Also a Claude Code / claude.ai skill (`SKILL.md`).
Companion to [fibrestitcher_ak](https://github.com/kohlianirudh03/fibrestitcher_ak) (stitching).

| module | purpose |
|---|---|
| `detect.py`, `detect2.py` | LoG fibre-centre detection + second pass in uncovered fibre area |
| `link.py` | short slice-to-slice tracks (seeds) |
| `hybrid_track.py` | optical-flow + alpha-beta filter tracker with detection snapping, 5 seed runs, confidence |
| `prune.py` | physical pruning (non-overlap, fibre-bright path) |
| `structure_tensor.py` | 3-D structure tensor orientation (for fibres too fine to track, e.g. 7 um carbon) |
| `yarn_theory.py` | ideal helix, rigid-rotation twist, Treloar contraction, Hearle-Gupta-Merchant migration, Hamilton blend index |
| `common.py` | reads `fibretrack_config.json` (voxel size, scan folders, output dir) |

Worked examples: [gf40-twisted-glass-yarn](https://github.com/kohlianirudh03/gf40-twisted-glass-yarn) and
[hybrid100-carbon-glass-yarn](https://github.com/kohlianirudh03/hybrid100-carbon-glass-yarn).

```bash
pip install -r requirements.txt
python fibretrack/detect_all.py <scan>   # see the example repos for the full sequence
```

MIT licence.

## How to cite

If this package contributes to published work, please cite it. Machine-readable
metadata is in [`CITATION.cff`](CITATION.cff); GitHub renders a "Cite this
repository" button from it.

```
Kohli A. fibretrack_ak: fibre tracking, orientation and twisted-yarn theory tests for X-ray CT. Version 1.0.0, 2026. https://github.com/kohlianirudh03/fibretrack_ak
```

## Method references

The methods implemented here build on:

- Emerson MJ, Jespersen KM, Dahl AB, Conradsen K, Mikkelsen LP. Individual fibre segmentation from 3D X-ray computed tomography for characterising the fibre orientation in unidirectional composite materials. *Composites Part A* 2017;97:83-92. [doi:10.1016/j.compositesa.2016.12.028](https://doi.org/10.1016/j.compositesa.2016.12.028)
- Salling FB, Jeppesen N, Sonne MR, Hattel JH, Mikkelsen LP. Individual fibre inclination segmentation from X-ray computed tomography using principal component analysis. *Journal of Composite Materials* 2022;56:83-98. [doi:10.1177/00219983211052741](https://doi.org/10.1177/00219983211052741)
- Advani SG, Tucker III CL. The use of tensors to describe and predict fiber orientation in short fiber composites. *Journal of Rheology* 1987;31(8):751-784. [doi:10.1122/1.549945](https://doi.org/10.1122/1.549945)
- Hearle JWS, Grosberg P, Backer S. *Structural Mechanics of Fibers, Yarns, and Fabrics*. Wiley-Interscience, 1969.
- Lindeberg T. Feature detection with automatic scale selection. *International Journal of Computer Vision* 1998;30:79-116.
