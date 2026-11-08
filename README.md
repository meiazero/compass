# compass

COMPASS intervals for an area metric on binary segmentation: the method library of
[garimpo-uncertainty](../README.md). Fork of [matthewyccheung/compass](https://github.com/matthewyccheung/compass)
(MIT, see `LICENSE`), adapted from the authors' medical code and paper.

![COMPASS overview](overview.png)

## Attribution

```bibtex
@article{cheung2025compass,
  title={COMPASS: Robust Feature Conformal Prediction for Medical Segmentation Metrics},
  author={Cheung, Matt Y and Veeraraghavan, Ashok and Balakrishnan, Guha},
  journal={arXiv preprint arXiv:2509.22240},
  year={2025}
}
```

## Contents

| Module | What |
|---|---|
| `compass2d/metrics.py` | Hard and differentiable area for a sigmoid output |
| `compass2d/directions.py` | Jacobian sum, direction `pca` (upstream's formula), direction `svm` (LinearSVC normal) |
| `compass2d/compass.py` | Endpoint interval, symmetric and asymmetric (Algorithm 6) binary search with a saturation flag, Algorithm 2 |
| `compass2d/calibration.py` | Group-weighted quantile (CDF pooling), repeated subsampling, leave-one-group-out, density ratio |

The library knows nothing about datasets, clusters or SLURM, and pins no torch build: the
application that installs it does. Install it as a uv path dependency. Check with
`uv run pytest -q && uvx ruff format --check . && uvx ruff check .`; the end-to-end test lives in
the parent package.
