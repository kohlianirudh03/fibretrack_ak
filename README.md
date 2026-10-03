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
