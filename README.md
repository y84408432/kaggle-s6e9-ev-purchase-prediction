# Kaggle S6E9 — EV Purchase Prediction

**Final Private AUC: 0.94544 · Rank: 350/3576 (Top 9.8%)**  
Competition: [Kaggle Playground Series S6E9 — Predicting Electric Vehicle Purchases](https://www.kaggle.com/competitions/playground-series-s6e9)

## 中文摘要

本项目基于 Kaggle Playground Series S6E9 电动汽车购买预测竞赛，采用 OOF（Out-of-Fold）验证框架，对 Logistic Regression、LightGBM、CatBoost、XGBoost 等模型进行比较，并研究收入特征精细化、交叉拟合目标统计、结构约束及模型预测融合等方法。

**最终成绩：Private AUC 0.94544，排名 350/3576（Top 9.8%）。**

注：最终竞赛提交使用了内部模型融合及外部公开预测结果，最终榜单成绩不等同于自主训练模型的 OOF 验证成绩。

完整研究过程、实验结果和复现说明见下方英文文档。

## Overview

An OOF-first tabular binary-classification study examining feature resolution, cross-fitted target statistics, structural constraints, prediction diversity, and rank ensembles. The target is `Will_Buy_EV`; evaluation is ROC-AUC.

### Research progression

| Stage | OOF AUC | Key result |
|---|---:|---|
| Logistic regression baseline | 0.938099 | Transparent reference |
| LightGBM exact view | ~0.946146 | High-resolution income features and fold-safe target statistics |
| CatBoost view | ~0.94612 | Categorical interactions and a distinct feature view |
| LightGBM coarse view | ~0.945831 | Coarser income / local target-rate representation |
| Additive XGBoost | 0.946215002 | Field-grouped interaction constraints, 10-fold OOF |
| Four-model internal ensemble | 0.946335057 | Rank blending with additive XGBoost at 40% |
| OOF-validated extended ensemble | 0.946413088 | Adds two *external/public OOF* views |

The final competition-stage submission combined the OOF-validated solution with additional **external public test predictions**, selected using leaderboard feedback. The final Private AUC (0.94544) is a competition result, **not** the unbiased CV score of the five self-trained models.

## Repository structure

| File | Purpose |
|---|---|
| [`01_logistic_baseline.ipynb`](01_logistic_baseline.ipynb) | Linear baseline and stratified CV |
| [`02_lgb_exact.ipynb`](02_lgb_exact.ipynb) | High-resolution LightGBM feature view |
| [`03_catboost.ipynb`](03_catboost.ipynb) | Categorical CatBoost feature view |
| [`04_lgb_coarse.ipynb`](04_lgb_coarse.ipynb) | Coarse LightGBM and local target windows |
| [`05_additive_xgb.ipynb`](05_additive_xgb.ipynb) | Structurally constrained XGBoost |
| [`06_final_research.ipynb`](06_final_research.ipynb) | OOF diagnostics, ablations and final ensembles |
| `assets/S6E9_research_assets_part*.npz` | Losslessly split OOF/test analysis arrays for notebook 06 |

## Quick start

The fastest entry point is **`06_final_research.ipynb`**: its required saved prediction arrays are in `assets/`, so it runs without re-training any model.

To run notebooks `01–05`, install Python dependencies (for example in a virtual environment):

```bash
python -m pip install -r requirements.txt
```

Download `train.csv`, `test.csv` and `sample_submission.csv` from the [official competition data page](https://www.kaggle.com/competitions/playground-series-s6e9/data), then place these three files **beside the notebooks**. Their default code setting is `DATA_DIR = Path(".")`.

Notebooks `02` and `04` additionally require `EV_Adoption_and_Range_Anxiety_Dataset.csv`, obtained from the [public EV Adoption Behavior and Range Anxiety dataset](https://www.kaggle.com/datasets/itzzomkar/ev-adoption-behavior-and-range-anxiety); place it beside the notebooks. These two pipelines apply exact feature-overlap removal before using original-data target statistics.

`03_catboost.ipynb` and `05_additive_xgb.ipynb` are configured for **NVIDIA CUDA**; use a Kaggle GPU runtime for the original training settings. `02` and `04` use CPU LightGBM. In Kaggle, update `DATA_DIR` and, if needed, `ORIGINAL_PATH` to point to the attached Input datasets.

The training notebooks save predictions under `outputs/`; these generated files are not committed to Git. The uploaded 01–05 notebooks have cleared execution histories to avoid distributing long training logs; their selected scores are recorded in their final Result sections. Notebook 06 includes displayed results and figures.

## Validation methodology

- Outer stratified OOF splits and inner cross-fitting for label-derived features.
- Fold-local preprocessing, encoding and feature statistics where required.
- Both standalone OOF scores and **incremental ensemble OOF** guide model decisions.
- A frozen late-stage promotion rule (`OOF gain >= 5e-5` and improvement in at least 4/5 reference folds) limits iterative rescue-tuning; it is an engineering threshold, not a significance test.
- External/public OOF assets and **test-only public leaderboard submissions** are explicitly distinguished from self-trained models.
- The exact-view train+test frequency encoding is transductive and label-free; it should not be copied unchanged into time-series deployment.

Reproduced scores of the cleaned training notebooks were consistent with the selected competition results; CatBoost differed slightly (approximately 4e-6 AUC). The saved analysis assets in notebook 06 represent the selected historical competition-stage predictions, so minor deviations from standalone retraining are expected.

## Data and attribution

Competition data and the original EV dataset are linked above rather than redistributed. Two extra OOF model views (Window XGB and Hybrid LightGBM) and subsequent test-only prediction blends come from publicly shared Kaggle prediction assets. They should not be interpreted as models trained within this repository. Some model recipes and engineered features were informed by public community notebooks/discussions; please refer to the Kaggle competition page for the public research context. The exact sources of public prediction files should be credited when known.

## Environment

Python, NumPy, pandas, SciPy, scikit-learn, LightGBM, CatBoost and XGBoost. Versions and CPU/GPU execution can influence model outputs slightly; the primary OOF results have been reproduced in the original competition environment.
