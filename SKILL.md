---
name: fibretrack_ak
description: Track individual fibres and measure fibre orientation, twist and migration in X-ray CT of yarns and fibre composites (glass, carbon, hybrid), then test the result against classical twisted-yarn theory (ideal helix, migration, blend segregation). Use when the user wants fibre trajectories, twist (turns per metre), helix angles, fibre migration, a digital twin or showcase media from CT slice stacks.
---

# fibretrack_ak

Built on two ESRF ID19 studies (Oct 2026, 0.9186 um voxels, Paganin delta/beta 1500):
a 40 tpm E-glass yarn (12 um fibres) and a 100 tpm carbon (7 um) / glass hybrid.
Stitch first with the `fibrestitcher_ak` skill.

## Choose the method from the image, not the voxel size

| fibre diameter / effective blur | method |
|---|---|
| fibres clearly separated discs (glass 12 um = 13 px) | per-fibre tracking (`hybrid_track.py`) |
| fibres marginal (carbon 7 um = 7.6 px at PSF sigma 2-3.7 px) | structure tensor (`structure_tensor.py`, sigma 2.5, rho 16, coarsen x4) |
| two phases (hybrid) | segment phases first (intensity + texture energy), then each phase with its own method |

Measure the PSF/contrast first; Paganin at delta/beta 1500 makes touching fibres differ by only ~5 % in grey level.

## Tracking recipe that worked (and why the obvious ones failed)

1. `detect.py` LoG sigma 4 px, min distance 6 px on a 3-slice mean; `detect2.py` second pass in uncovered glass.
2. `link.py` short tracks only as seeds. Detection-only linking fragments: when two fibres merge,
   the LoG peak drifts several px before vanishing, so extrapolation misses (929/1494 ends unmatched).
3. `hybrid_track.py`: Lucas-Kanade flow between slices 4 apart + per-fibre alpha-beta filter for the
   fibre's own motion (flow alone under-reads motion ~10 % because neighbours move differently),
   snap to a detection within 6 px, 5 independently seeded runs, `prune.py` (centres >= 8 px apart,
   glass-bright path). Confidence = runs reproducing the fibre.
4. Join scans by matching fibres inside each overlap with a rigid 2-D fit; the residual is the
   per-fibre stitch error (achieved 0.7-1.0 um median).

## Validation that must accompany any number

- Detector: score on a phantom whose image formation is calibrated to the real data, and *also*
  check coverage on real slices — a phantom with idealised touching-fibre contrast reported 0.99 recall
  while real coverage was far lower.
- Completeness: compare tracked fibres with glass area / single-fibre area; report the range.
- Twist: at least two methods sharing no processing (rigid rotation of tracked fibres; helix
  regression; structure tensor; rotation of an eccentric phase arrangement). Image-texture rotation
  over > 500 slices sits at its noise floor (NCC 0.12-0.2) — do not use it locally.
- Report twist along the length, not just one number: the 40 tpm yarn ran 49 -> 22 tpm top to bottom.

## Theory tests (`yarn_theory.py`)

- ideal helix tan(theta) = 2 pi r T: compare binned medians with the nominal line; report the scatter.
- Treloar contraction, Hearle-Gupta-Merchant migration (Ybar, D, I; complete migration D = 0.289).
- Hamilton-type radial distribution index for blends (+100 outside, 0 uniform, -100 centre) and
  single-fibre bending rigidity E d^4 to state what migration theory predicts.

## Outputs to produce

CSV trajectories (full + Excel-sized), per-fibre summary, GLB/STL/OBJ/PLY/Avizo models, an
offline HTML explorer (type a fibre number), a digital twin (centrelines, label volume, Abaqus B31
beam model, fidelity vs real CT), MP4 fly-through and rotating render, publication figures.

Set paths in `fibretrack_config.json` (see `fibretrack/common.py`).
