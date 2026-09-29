# Subscriber Lapse Prediction

Predicting which active users of a content platform will stop engaging, using only data available at the moment of prediction.
**[Live demo](https://subscriber-lapse-prediction.streamlit.app/)** - browse real held-out users and see the model's prediction against what actually happened.

## The problem

Content platforms don't have a "will churn" column. They have event logs. The labels have to be constructed, and that construction is most of the work.

This project uses MovieLens 25M as a stand-in for a streaming platform's engagement log: 25 million timestamped events from 162,000 users, 1995-2019.

## Approach

Pick an anchor date and treat it as the present:

| | Definition |
|---|---|
| **Anchor** | The moment of prediction |
| **Feature window** | 365 days before the anchor — the only data the model may use |
| **Eligible users** | Anyone active in the 90 days before the anchor |
| **Label** | 1 if the user has no activity in the 90 days after the anchor |

Nothing from after the anchor reaches the features.

## Why validation is time-based

The model trains on a snapshot from 2018-01-01 and is tested on a snapshot from 2019-01-01. A random split would let the model learn from the future to predict the past, which inflates results and cannot be reproduced in production.

| | Train | Test |
|---|---|---|
| Anchor | 2018-01-01 | 2019-01-01 |
| Eligible users | 5,045 | 5,160 |
| Lapse rate | 0.503 | 0.511 |

## Features

Thirteen behavioural features computed strictly before the anchor: recency, activity counts at four time scales, breadth (distinct movies, distinct active days), rating behaviour, a short-term trend, and lifetime facts (tenure, total ratings).

## Results

Random Forest, tuned by randomised search with 5-fold cross-validation.

| | Cross-validation (2018) | Test (2019) |
|---|---|---|
| ROC-AUC | 0.8925 | **0.8959** |
| Accuracy | | 0.816 |
| Recall (lapsed) | | 0.835 |
| Precision (lapsed) | | 0.811 |

Performance held a full year forward, which is the point of the backtest.

Random Forest beat logistic regression 0.8925 to 0.8802 — about two standard deviations of fold-to-fold noise, so a real difference rather than a coin flip.

## Findings

**A recency-only baseline reaches 0.757 ROC-AUC.** The full feature set reaches 0.896. Retention models are mostly recency models, and reporting that gap is the honest way to show the other twelve features earned their place.

**No single feature is load-bearing.** Permutation importance ranked `n_active_days` fifteen times above `recency_days`, but a drop-column test tells a different story:

| Feature removed | ROC-AUC | Change |
|---|---|---|
| none | 0.8959 | |
| `n_active_days` | 0.8921 | -0.0038 |
| `recency_days` | 0.8937 | -0.0022 |
| `per_active_day` | 0.8958 | -0.0001 |

Permutation importance was measuring redundancy. The features are correlated views of the same behaviour, so removing any one lets the rest reconstruct the signal. Drop-column is the more honest test when features overlap.

**Recent activity volume is a proxy for being new.** Lapse rate by 90-day activity is non-monotonic (0.50, 0.31, 0.49, 0.60, 0.61), which reads as "heavy users churn more." Median tenure explains it:

| 90-day ratings | Lapse rate | Median tenure (days) |
|---|---|---|
| 1-5 | 0.50 | 783 |
| 6-15 | 0.31 | 788 |
| 84+ | 0.61 | 62 |

Heavy recent activity means a new user working through a back catalogue. Tenure itself is strong and threshold-shaped: under 60 days, lapse rate is 0.78; past 590 days it settles near 0.30. Retention effort belongs in onboarding.

## What I'd do differently

- Test more than two anchors. Two snapshots show the model holds for one year, not that it holds generally.
- Try pruning the feature set. Removing any single feature barely moved performance, which suggests the thirteen are overlapping views of the same behaviour rather than independent signals. A smaller set would probably hold up and be easier to maintain.
- MovieLens ratings are a proxy for viewing. Real watch data would support session-level features this cannot.

## Repo layout

```
notebooks/
  01-label-construction.ipynb        windows, eligibility, labels
  02-feature-engineering.ipynb       behavioural features
  03-modeling-and-evaluation.ipynb   models, tuning, backtest
app.py                               Streamlit demo
demo/sample_users.csv                300 held-out users for the demo
src/download_data.py
reports/figures/                     saved plots
models/model.joblib                  trained model
data/                                not committed
```

## Setup

Reproduce the analysis:

```bash
pip install -r requirements-dev.txt
python src/download_data.py
jupyter notebook
```

Run the demo locally:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Data

MovieLens 25M, from GroupLens. Not committed — the licence prohibits redistribution.

Harper & Konstan (2015), *The MovieLens Datasets: History and Context*, ACM TiiS 5(4).