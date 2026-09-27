# Subscriber Lapse Prediction

Predicting which active users of a content platform will stop engaging, using only data available at the moment of prediction.

**Status:** in progress — labels and features complete, modelling next.

## The problem

Content platforms don't have a "will churn" column. They have event logs. The labels have to be constructed, and that construction is most of the work.

This project uses MovieLens 25M as a stand-in for a streaming platform's engagement log: 25 million timestamped events from 162,000 users, spanning 1995 to 2019.

## Approach

Pick an anchor date and treat it as the present:

| | Definition |
|---|---|
| **Anchor** | The moment of prediction |
| **Feature window** | 365 days before the anchor — the only data the model may use |
| **Eligible users** | Anyone active in the 90 days before the anchor |
| **Label** | 1 if the user has no activity in the 90 days after the anchor |

Nothing from after the anchor reaches the features.

### Snapshots

| | Train | Test |
|---|---|---|
| Anchor | 2018-01-01 | 2019-01-01 |
| Eligible users | 5,045 | 5,160 |
| Lapse rate | 0.503 | 0.511 |

The two lapse rates are close, so the snapshots are comparable. A drop from train to test will reflect the model rather than a shift in the underlying population.

## Why validation is time-based

The model is trained on a snapshot from one date and tested on a snapshot from a year later. A random split would let the model learn from the future to predict the past, which inflates results and cannot be reproduced in production. Retention models are backtested forward in time.

One practical constraint: the label window must fall inside the data. MovieLens 25M ends in November 2019, so an anchor later than roughly August 2019 would label every user as lapsed simply because the log runs out.

## Features

Thirteen behavioural features, all computed strictly before the anchor: recency, activity counts at four time scales (7/30/90/365 days), breadth (distinct movies, distinct active days), rating behaviour (mean, standard deviation), a short-term activity trend, and lifetime facts (tenure, total ratings).

### Findings so far

**Recency dominates.** Lapse rate runs from 0.19 for users last seen within 4 days to 0.79 for users last seen 61-89 days ago. Monotonic across every bucket.

**New users are the risk, with a threshold around six months.** Users with under 60 days of tenure lapse at 0.78; past roughly 590 days it settles near 0.30. The drop is concentrated in the first six months rather than spread evenly, which suggests retention effort belongs in onboarding rather than with long-tenured users.

**Recent activity volume is not what it appears.** Lapse rate by 90-day activity is non-monotonic (0.50, 0.31, 0.49, 0.60, 0.61), which reads as "heavy users churn more." Disaggregating by tenure explains it:

| 90-day ratings | Lapse rate | Median tenure (days) |
|---|---|---|
| 1-5 | 0.50 | 783 |
| 6-15 | 0.31 | 788 |
| 16-34 | 0.49 | 322 |
| 35-83 | 0.60 | 87 |
| 84+ | 0.61 | 62 |

High recent activity is a proxy for being new — users working through a back catalogue. The feature was measuring tenure, not engagement. This is a pooled statistic concealing a subgroup, and it implies the model needs to represent an activity-by-tenure interaction.

## Results

_To be filled in._

## Repo layout

```
notebooks/
  01-label-construction.ipynb        windows, eligibility, labels
  02-feature-engineering.ipynb       behavioural features from the history window
  03-modeling-and-evaluation.ipynb   baseline, model, threshold, results
src/
  download_data.py
data/      not committed
models/    not committed
```

## Setup

```bash
pip install -r requirements.txt
python src/download_data.py
jupyter notebook
```

## Data

MovieLens 25M, from GroupLens. Not committed — the licence prohibits redistribution. The download script fetches it.

Harper & Konstan (2015), *The MovieLens Datasets: History and Context*, ACM TiiS 5(4).