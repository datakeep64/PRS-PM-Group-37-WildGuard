"""Runs the Part B model search and refit for notebook 10 (DESIGN_QUESTIONS.md B1-B5).

Takes a few hours, so it runs outside the notebook and saves its results in
DATA/processed/experiments/ as it goes: one file per candidate for the search and for the
2011 scores. Rerunning it resumes from what is saved. Notebook 10 then reads the results.

Run from SYSTEM CODE/notebooks:  python run_model_search.py
"""
import time
import pandas as pd
import eda_setup  # noqa: F401
import experiment_setup as X_
import model_selection as M
from wildguard import config
from wildguard.features import SHARED_FEATURES

OUT = config.PROCESSED_DIR / "experiments"
OUT.mkdir(exist_ok=True)
FEATS = SHARED_FEATURES + ["cons_sources"]
df = X_.load_lemis()
cons = df.groupby("shipment_id")["source"].nunique().rename("cons_sources")
df = df[df["animal"]].join(cons, on="shipment_id").reset_index(drop=True)
t0 = time.time()
log = lambda m: print(f"[{time.time() - t0:6.0f}s] {m}", flush=True)
search = M.search(df, FEATS, M.SIMPLICITY, log=log, cache_dir=OUT)
search.to_parquet(OUT / "model_search.parquet")
best = search.loc[search.groupby("candidate")["pr_auc"].idxmax()].set_index("candidate").loc[M.SIMPLICITY]
ev = df[df["year"] == 2011].reset_index(drop=True)
S = ev[["shipment_id", "label", "sp_listed", "purpose"]].copy()
for c in M.SIMPLICITY:
    part = OUT / f"score_2011_{c}.parquet"          # one file per candidate, so a restart resumes
    if not part.exists():
        row = best.loc[c]
        cfg = M.config_from_row({**row.to_dict(), "candidate": c})
        rounds = None if pd.isna(row["best_rounds"]) else int(row["best_rounds"])
        pd.DataFrame({c: M.refit_and_score(df, FEATS, c, cfg, rounds)}).to_parquet(part)
        log(f"refit {c}: done")
    S[c] = pd.read_parquet(part)[c].to_numpy()
S.to_parquet(OUT / "scores_2011.parquet")
log("all done")
