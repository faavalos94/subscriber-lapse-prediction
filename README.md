# Subscriber Lapse Prediction

Predicting which active users of a content platform will stop engaging, using only data available at the moment of prediction.

**Status:** in progress

## The problem

Content platforms don't have a "will churn" column. They have event logs. The labels have to be constructed, and that construction is most of the work.

This project uses MovieLens 25M as a stand-in for a streaming platform's engagement log: 25 million timestamped events from 162,000 users.

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

The model is trained on a snapshot from one date and tested on a snapshot from a year later. A random split would let the model learn from the future to predict the past, which inflates results and cannot be reproduced in production.

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