"""Excursion risk at the decision epoch — baseline first, then the oracle.

THREE THINGS THIS MODULE INSISTS ON
-----------------------------------

1. THE BASELINE COMES FIRST, AND THE GAP GETS REPORTED. If gradient boosting
   beats logistic regression by two points, that is a finding worth stating, not
   an embarrassment to hide. Consultants respect the simple model that nearly
   works.

2. PROBABILITIES MUST BE CALIBRATED. This is not a refinement — it is load
   bearing, and its failure is silent. The optimizer computes
   `P(spoilage) x consignment value`, so a model that ranks perfectly but reports
   0.6 where the truth is 0.3 will double every expected-loss figure in the deck
   while showing a beautiful ROC curve. AUC cannot detect this. Brier and the
   reliability curve can.

3. SKILL IS SCORED AGAINST THE ORACLE, NOT AGAINST 1.0. The simulator contains
   genuinely unknowable components — the unobserved confounder, and downstream
   weather that has not happened yet. A model given both defines the CEILING.
   Reporting "AUC 0.83 against an achievable ceiling of 0.87" is a far more
   honest statement than "AUC 0.83", and it is the beginning of the answer to
   "your model just rediscovered your own simulator".
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

__all__ = [
    "NUMERIC_FEATURES",
    "CATEGORICAL_FEATURES",
    "DECISION_FEATURES",
    "ORACLE_EXTRA",
    "TARGET",
    "assert_no_leakage",
    "ModelReport",
    "evaluate",
    "reliability_curve",
]

NUMERIC_FEATURES = [
    "hub_budget_pct",
    "hub_mkt_c",
    "hub_peak_c",
    "hub_min_c",
    "hub_elapsed_h",
    "remaining_h",
    "running_var_min",
    "tau_h",
    "depart_month",
    "tarmac_hold",
    "consignment_value_usd",
]
CATEGORICAL_FEATURES = ["lane", "product", "packaging", "hub"]
DECISION_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

ORACLE_EXTRA = ["truth_confounder_u", "truth_post_amb_mean_c", "truth_post_amb_max_c"]
"""What the oracle is allowed to see and the real model is not: the unobserved
confounder and the weather that has not happened yet."""

TARGET = "post_hub_excursion"
"""Deliberately post-hub. Whether the shipment ALREADY breached before the hub is
observable at decision time, so predicting whole-journey excursion would be
scoring a model on something it can simply read off its own inputs."""


def assert_no_leakage(features: list[str]) -> None:
    """Fail loudly if an outcome-derived column reaches a feature list.

    Cheap, and it protects the one mistake that would invalidate every number
    downstream while leaving the metrics looking excellent.
    """
    banned_exact = {
        "budget_consumed_pct", "mkt_c", "peak_c", "min_c", "excursion",
        "post_hub_excursion", "destroyed", "minutes_above_spec", "freeze_degree_h",
    }
    bad = [f for f in features if f in banned_exact or f.startswith("truth_")]
    if bad:
        raise ValueError(
            f"outcome-derived columns in feature list: {bad}. These are measured at "
            "arrival; using them to predict arrival is leakage."
        )


def _pipeline(kind: str, features: list[str]) -> Pipeline:
    cats = [c for c in CATEGORICAL_FEATURES if c in features]
    nums = [c for c in features if c not in cats]
    if kind == "logistic":
        pre = ColumnTransformer(
            [("num", StandardScaler(), nums),
             ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=10), cats)]
        )
        return Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=2000, C=1.0))])

    pre = ColumnTransformer(
        [("num", "passthrough", nums),
         ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=10), cats)]
    )
    return Pipeline([
        ("pre", pre),
        ("clf", HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.06, max_leaf_nodes=15,
            min_samples_leaf=25, l2_regularization=1.0, random_state=0,
        )),
    ])


def _select_calibration(kind: str, features: list[str], tr: pd.DataFrame) -> str | bool:
    """Choose the calibration method by cross-validated Brier on TRAINING data.

    CALIBRATION IS NOT AUTOMATICALLY AN IMPROVEMENT, and assuming it is cost us a
    test failure worth keeping. Isotonic regression is non-parametric and
    flexible, which is what makes it able to fix badly-shaped probabilities — and
    also what makes it OVERFIT at modest sample sizes. HistGradientBoosting is
    already fairly well calibrated out of the box (unlike an SVM or naive Bayes),
    so there is little for isotonic to fix and plenty for it to break: at
    n=3,000 it made Brier *worse*.

    Platt scaling (sigmoid) has two parameters and cannot overfit the same way.
    Rather than pick one on faith, we measure — and report which won.
    """
    from sklearn.model_selection import cross_val_predict

    y = tr[TARGET].to_numpy()
    scores: dict[str | bool, float] = {}
    for method in ("isotonic", "sigmoid"):
        p = cross_val_predict(
            CalibratedClassifierCV(_pipeline(kind, features), method=method, cv=3),
            tr[features], y, cv=3, method="predict_proba",
        )[:, 1]
        scores[method] = brier_score_loss(y, p)
    p_raw = cross_val_predict(
        _pipeline(kind, features), tr[features], y, cv=3, method="predict_proba"
    )[:, 1]
    scores[False] = brier_score_loss(y, p_raw)
    return min(scores, key=lambda k: scores[k])


@dataclass
class ModelReport:
    name: str
    auc: float
    brier: float
    logloss: float
    n_train: int
    n_test: int
    calibrated: bool = False
    notes: str = ""
    probs: np.ndarray = field(default_factory=lambda: np.array([]))
    truth: np.ndarray = field(default_factory=lambda: np.array([]))

    def __str__(self) -> str:
        flag = "calibrated" if self.calibrated else "raw"
        return (f"{self.name:<34} AUC {self.auc:.4f}   Brier {self.brier:.4f}   "
                f"logloss {self.logloss:.4f}   [{flag}]")


def _fit_one(
    name: str, kind: str, features: list[str],
    tr: pd.DataFrame, te: pd.DataFrame, calibrate: bool | str,
) -> ModelReport:
    """`calibrate` is False, "isotonic", "sigmoid", or True to select by CV Brier."""
    pipe = _pipeline(kind, features)
    if calibrate is True:
        calibrate = _select_calibration(kind, features, tr)
    if calibrate:
        pipe = CalibratedClassifierCV(pipe, method=calibrate, cv=5)
    pipe.fit(tr[features], tr[TARGET])
    p = pipe.predict_proba(te[features])[:, 1]
    y = te[TARGET].to_numpy()
    return ModelReport(
        name=name,
        auc=float(roc_auc_score(y, p)),
        brier=float(brier_score_loss(y, p)),
        logloss=float(log_loss(y, p, labels=[0, 1])),
        n_train=len(tr), n_test=len(te), calibrated=calibrate,
        probs=p, truth=y,
    )


def evaluate(df: pd.DataFrame, seed: int = 0, lane_holdout: bool = True) -> list[ModelReport]:
    """Baseline, boosted, calibrated and oracle — on the same split."""
    assert_no_leakage(DECISION_FEATURES)
    d = df[df["product"] != "CAR-T dose"].dropna(subset=[TARGET]).copy()

    tr, te = train_test_split(d, test_size=0.3, random_state=seed, stratify=d[TARGET])

    picked = _select_calibration("boosted", DECISION_FEATURES, tr)
    label = {False: "uncalibrated wins", "isotonic": "isotonic", "sigmoid": "Platt"}[picked]

    out = [
        _fit_one("logistic regression (baseline)", "logistic", DECISION_FEATURES, tr, te, False),
        _fit_one("HistGradientBoosting", "boosted", DECISION_FEATURES, tr, te, False),
        _fit_one("HistGradientBoosting + isotonic", "boosted", DECISION_FEATURES, tr, te, "isotonic"),
        _fit_one("HistGradientBoosting + Platt", "boosted", DECISION_FEATURES, tr, te, "sigmoid"),
        _fit_one(f"HistGradientBoosting, SELECTED ({label})", "boosted",
                 DECISION_FEATURES, tr, te, picked),
        _fit_one("ORACLE (sees U + realised weather)", "boosted",
                 DECISION_FEATURES + ORACLE_EXTRA, tr, te, picked),
    ]
    out[-2].notes = "calibration method chosen by CV Brier, not assumed"
    out[-1].notes = "ceiling on achievable skill, not a deployable model"

    if lane_holdout:
        # Held-out LANES, not held-out rows. Random splits let a model memorise
        # lane-specific quirks and then flatter itself; this asks whether it
        # generalises to a corridor it has never seen — the question a client
        # actually cares about.
        gkf = GroupKFold(n_splits=4)
        idx_tr, idx_te = next(gkf.split(d, d[TARGET], groups=d["lane"]))
        r = _fit_one("HistGradientBoosting, UNSEEN LANES", "boosted",
                     DECISION_FEATURES, d.iloc[idx_tr], d.iloc[idx_te], picked)
        r.notes = "generalisation to corridors never trained on"
        out.append(r)

    return out


def reliability_curve(probs: np.ndarray, truth: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Predicted vs observed frequency, in equal-count bins.

    Equal-count rather than equal-width: with a skewed probability distribution,
    equal-width bins put almost every observation in the first bucket and the
    curve reports mostly noise.
    """
    q = np.quantile(probs, np.linspace(0, 1, bins + 1))
    q[0], q[-1] = -np.inf, np.inf
    idx = np.digitize(probs, q[1:-1])
    rows = []
    for b in range(bins):
        m = idx == b
        if m.sum() >= 5:
            rows.append({"bin": b, "n": int(m.sum()),
                         "predicted": float(probs[m].mean()),
                         "observed": float(truth[m].mean())})
    return pd.DataFrame(rows)
