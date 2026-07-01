# Visual Place Recognition — Adaptive Re-Ranking

This repository extends the base VPR project with an **adaptive re-ranking pipeline** that selectively applies image matching only to queries where the retrieval model is uncertain, improving R@1 while reducing unnecessary computation.

> Base repo: [FarInHeight/Visual-Place-Recognition-Project](https://github.com/FarInHeight/Visual-Place-Recognition-Project)

---

## Pipeline

![Adaptive Re-Ranking Pipeline](docs/pipeline.png)

The pipeline has two phases:

**Offline** — trained once on a labelled CSV:
- Logistic regressors (`hard`, `help`, `hurts`) fit on spatial uncertainty and/or inlier features
- Grid search over threshold τ and weight α → calibrated hyperparameters saved in `regressors/<model>_regressors.json`

**Online** — run per query set:
1. **Retrieval** — `main.py --save_for_uncertainty` produces `preds/` and `z_data.torch`
2. **Choice 1: feature set** — extract Spatial Uncertainty (RS, SD, SU) from `z_data.torch` and/or run IM top-1 to get inliers
3. **Choice 2: probability criterion** — apply logistic regressor with calibrated threshold to decide which queries to rerank
4. **Selective reranking** — run full IM top-20 only on selected queries
5. **R@1 evaluation** — combine reranked queries with skipped queries (using retrieval top-1)

### Results (sf_xs val, CosPlace ResNet18-512, superpoint-lg)

| | R@1 | R@5 | R@10 | R@20 |
|---|---|---|---|---|
| Retrieval baseline | 63.1 | — | — | — |
| Full reranking | 77.3 | — | — | — |
| **Adaptive (SU, P(help))** | **82.8** | **88.4** | **89.8** | **92.1** |

---

## Quickstart (Google Colab)

Open the notebook and run in order:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/roccnroll/Visual-Place-Recognition-Project/blob/main/adaptive_pipeline.ipynb)

1. Mount Drive + clone repo + install dependencies
2. *(Optional)* Download datasets
3. Run VPR retrieval
4. **Set config** — `RUN_DIR`, `FEATURE_SET`, `CRITERION`, `MATCHER`, `MODEL`
5. **Run Pipeline cell** — all steps run automatically based on your config

### Config parameters

| Parameter | Options | Description |
|---|---|---|
| `FEATURE_SET` | `inliers` \| `RS` \| `SD` \| `SU` \| `SU+inliers` | Which features to use for the decision |
| `CRITERION` | `P(hard)` \| `P(help)` \| `P(help)-aP(hurts)` \| `P(help)/P(hurts)>1` | Probability criterion |
| `MATCHER` | `superpoint-lg` \| `loftr` | Image matching model |
| `MODEL` | `cosplace` \| `megaloc` | VPR model (selects the regressors JSON) |
| `RUN_DIR` | path | Timestamp folder created by `main.py` in Drive/VPR/logs/ |

---

## Repository structure

```
├── adaptive_pipeline.ipynb          # Main Colab notebook
├── match_queries_preds.py           # IM top-20 reranking (from base repo)
├── reranking.py                     # Full reranking baseline (from base repo)
├── util.py                          # Geo utilities (from base repo)
├── regressors/
│   ├── cosplace_regressors.json     # Calibrated regressors for CosPlace
│   └── megaloc_regressors.json      # Calibrated regressors for MegaLoc
├── VPR-Adaptive-ReRanking/
│   ├── extract_su.py                # z_data.torch → su_scores.csv
│   ├── extract_inliers.py           # IM top-1 → inliers.csv
│   ├── select_queries.py            # Applies criterion → preds_filtered/
│   └── check_performance.py         # R@1 combining reranked + skipped
├── VPR-methods-evaluation/
│   └── main.py                      # VPR retrieval
└── image-matching-models/           # Submodule
```

### Regressor JSON format

```json
{
  "model": "CosPlace",
  "matchers": {
    "sp-lg": {
      "feature_sets": {
        "SU": {
          "regressors": {
            "hard": { "feat_cols": [...], "scaler_mean": [...], "scaler_scale": [...], "coef": [...], "intercept": ... },
            "help": { ... },
            "hurts": { ... }
          },
          "val_hparams": {
            "P(hard)":             { "tau": ..., "alpha": null },
            "P(help)":             { "tau": ..., "alpha": null },
            "P(help)-aP(hurts)":   { "tau": ..., "alpha": ... },
            "P(help)/P(hurts)>1":  { "tau": ..., "alpha": null }
          }
        }
      }
    }
  }
}
```

---

## Local setup

```sh
git clone --recursive https://github.com/roccnroll/Visual-Place-Recognition-Project.git
cd Visual-Place-Recognition-Project/image-matching-models
pip install -e .
pip install faiss-cpu
```

> [!NOTE]
> Dataset filename convention: `@UTM_easting@UTM_northing@zone_num@zone_letter@...`
> Only UTM coordinates are required; other fields can be empty.
