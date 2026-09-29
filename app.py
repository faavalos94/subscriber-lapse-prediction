"""Subscriber lapse prediction — interactive demo.

Browse real held-out users from the 2019 test snapshot, see the model's
risk score, and check it against what actually happened.
"""
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Subscriber Lapse Prediction",
                   page_icon="📉", layout="wide")

REPO = "https://github.com/faavalos94/subscriber-lapse-prediction"

# Plain-English labels for the model's feature names
LABELS = {
    "recency_days":     "Days since last activity",
    "tenure_days":      "Account age (days)",
    "n_90":             "Ratings in last 90 days",
    "n_30":             "Ratings in last 30 days",
    "n_active_days":    "Distinct active days",
    "per_active_day":   "Ratings per active day",
    "trend_30_90":      "Share of 90-day activity in last 30",
    "rating_mean":      "Average rating given",
    "lifetime_ratings": "Lifetime ratings",
}


@st.cache_resource
def load_model():
    bundle = joblib.load("models/model.joblib")
    return bundle["model"], bundle["features"]


@st.cache_data
def load_sample():
    return pd.read_csv("demo/sample_users.csv")


model, FEATURES = load_model()
users = load_sample()

# ---------------------------------------------------------------- sidebar
st.sidebar.header("Pick a user")
st.sidebar.caption("300 real users from the held-out 2019 snapshot.")

view = st.sidebar.radio("Show", ["All users", "Model got it right",
                                 "Model got it wrong"], label_visibility="collapsed")

pool = users.copy()
pool["predicted_lapse"] = (pool.predicted_risk >= 0.5).astype(int)
if view == "Model got it right":
    pool = pool[pool.predicted_lapse == pool.actual_lapsed]
elif view == "Model got it wrong":
    pool = pool[pool.predicted_lapse != pool.actual_lapsed]

if pool.empty:
    st.warning("No users match that filter.")
    st.stop()

choice = st.sidebar.selectbox("User", pool.user.tolist())
row = pool[pool.user == choice].iloc[0]

st.sidebar.divider()
st.sidebar.markdown("**What-if**")
st.sidebar.caption("Adjust this user's behaviour and watch the score move. "
                   "Other features are held constant.")
recency = st.sidebar.slider("Days since last activity", 0, 89,
                            int(row.recency_days))
tenure = st.sidebar.slider("Account age (days)", 0, 3000,
                           int(row.tenure_days))

# ------------------------------------------------------------------- main
st.title("Will this subscriber lapse?")
st.caption("Predicting 90-day disengagement from behaviour observed before "
           "the prediction date. Trained on a 2018 snapshot, tested on 2019.")

# Re-score with the slider values
modified = row[FEATURES].astype(float).copy()
modified["recency_days"] = recency
modified["tenure_days"] = tenure
risk = float(model.predict_proba(pd.DataFrame([modified])[FEATURES])[0, 1])

left, right = st.columns([1, 1.6], gap="large")

with left:
    st.metric("Predicted lapse risk", f"{risk:.0%}")
    st.progress(risk)

    if risk >= 0.70:
        st.error("**High risk.** Strong candidate for a retention offer.")
    elif risk >= 0.45:
        st.warning("**Uncertain.** Close to the decision boundary.")
    else:
        st.success("**Low risk.** Likely to stay engaged.")

    st.caption("Base rate across all users: 51%")

    unchanged = (recency == int(row.recency_days)
                 and tenure == int(row.tenure_days))
    if unchanged:
        actual = "lapsed" if row.actual_lapsed else "stayed active"
        correct = (risk >= 0.5) == bool(row.actual_lapsed)
        st.divider()
        st.markdown(f"**What actually happened:** this user {actual} "
                    f"in the 90 days after the prediction date.")
        st.markdown("✅ Model was right" if correct else "❌ Model was wrong")
    else:
        st.divider()
        st.info("Showing a modified user. Reset the sliders to see the "
                "real outcome.")

with right:
    st.subheader("This user's behaviour")
    shown = {LABELS[k]: modified[k] for k in LABELS if k in modified}
    st.dataframe(
        pd.DataFrame({"Value": shown}).rename_axis("Feature").round(2),
        use_container_width=True,
    )

st.divider()

# ------------------------------------------------------------ explanation
c1, c2, c3 = st.columns(3)
c1.metric("Test ROC-AUC", "0.896")
c2.metric("Recall (lapsed)", "0.835")
c3.metric("Accuracy", "0.816")

with st.expander("How this works"):
    st.markdown("""
The dataset has no "churn" column — only a log of timestamped events.
The labels are constructed by picking an **anchor date** and treating it
as the present:

| | Definition |
|---|---|
| **Feature window** | 365 days before the anchor — the only data the model sees |
| **Eligible users** | Anyone active in the 90 days before the anchor |
| **Label** | 1 if the user has no activity in the 90 days after the anchor |

The model trains on users as they looked on **2018-01-01** and is tested
on a different set of users a year later, on **2019-01-01**. A random
split would let the model learn from the future to predict the past.

Data: MovieLens 25M, used as a stand-in for a streaming engagement log.
""")
    st.markdown(f"[Full write-up and code on GitHub]({REPO})")
