# MOBA Early-Game Diagnostics & Win Probability

**Author:** Cao Siyuan  
**Stack:** Python · Pandas · SciPy · Scikit-learn · Matplotlib · Seaborn

## Problem

How reliably can the first 10 minutes of a League of Legends match reveal its eventual outcome? This project analyzes **9,879 high-diamond ranked matches** to support early-game state diagnostics, competitive analysis, and responsible win-probability presentation.

The objective is not simply to maximize classification accuracy. It is to quantify how economic advantages convert into wins, assess additional information from map objectives and other game-state signals, and evaluate when predicted probabilities are reliable enough to be useful.

## Analysis workflow

1. **Data audit and feature engineering:** verify data quality and deterministic feature identities; construct 12 non-duplicated relative features across five game-state domains.
2. **Game-state diagnostics:** use vectorized Pandas transformations and grouped aggregation to compare economic lead conversion, comeback risk, and epic-objective context; report sample sizes and Wilson 95% confidence intervals.
3. **Incremental prediction value:** compare a prior baseline, gold-only and gold-plus-experience logistic regression, full logistic regression, and random forest on the same stratified holdout set. Hyperparameters are selected using 5-fold cross-validation and log loss on training data only.
4. **Reliability and interpretation:** evaluate probability calibration, paired-bootstrap performance differences, feature-family ablation, performance within economic strata, and the coverage–accuracy trade-off of high-confidence predictions.

## Key findings

For gold-leader diagnostics, the bins are (0, 500], (500, 1,500], (1,500, 3,000], and (3,000, ∞). The two exact economic ties are excluded from that leader-based analysis, but retained in predictive modeling.

| Question | Observed result |
| --- | --- |
| How stable is an economic lead? | The leading team's win rate rises from **52.6%** (0–500 gold) to **63.8%**, **78.8%**, and **90.8%** for progressively larger leads. Even a lead above 3,000 leaves a 9.2% comeback rate for the trailing team. |
| Do map objectives add context? | Within the 0–500 gold band, the gold leader's win rates are **43.9% / 53.6% / 60.1%** when its epic-objective status is trailing / even / leading (572 / 526 / 584 matches). These are conditional associations, not causal effects. |
| Does a richer model improve prediction? | Full logistic regression is selected by training CV and achieves **AUC 0.811, accuracy 72.2%, Brier 0.1775** on 2,470 holdout matches. Compared with gold-only LR, AUC improves by 0.0098 (paired-bootstrap 95% CI: 0.0049–0.0149), and Brier score decreases by 0.0043 (0.0020–0.0066). The small accuracy gain is uncertain. |
| Where is additional information useful? | In near-even holdout games (n = 403), AUC increases from 0.514 to 0.578 and Brier score decreases from 0.2503 to 0.2455. Feature-family ablation highlights economic/growth signals and map objectives as the largest conditional contributors to probability quality. |
| When should a prediction be displayed confidently? | At an illustrative confidence threshold of 0.80, the selected model covers **36.4%** of holdout games and achieves **89.4% accuracy** within that subset (803/898). The 95 high-confidence errors are separately exported for review. |

The results favor a transparent logistic model over greater model complexity for this dataset. They also show why accuracy alone is not enough when the intended output is a probability.

## Outputs and reproduction

- [Analysis notebook](MOBA_Early_Game_Diagnostics.ipynb): complete, executed workflow with figures, tables, and methodological explanations.
- [Key results](outputs/tables/key_results.csv): machine-generated headline metrics.
- [Diagnostic tables](outputs/tables/): data checks, economic and objective strata, model comparison, bootstrap intervals, ablation, prediction slices, and error cases.
- [Figures](outputs/figures/): six reproducible visualizations.

Install the dependencies, then open `MOBA_Early_Game_Diagnostics.ipynb` from the project root and **Run All**:

```bash
python -m pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace MOBA_Early_Game_Diagnostics.ipynb
```

The notebook regenerates all CSV tables and figures under `outputs/`. The original source CSV is retained; the previous exploratory analysis remains available in Git history.

## Limitations

The dataset records only a cumulative 10-minute snapshot from high-tier ranked matches. Event timing, pre-objective conditions, champion composition, player skill, patch version, and later-game actions are unavailable. Consequently, the project estimates **descriptive and predictive associations, not intervention effects**. Confidence intervals treat matches as independent observations because player/time clustering cannot be identified. Random holdout performance is an in-distribution evaluation; deployment would require external or time/patch-based validation.
