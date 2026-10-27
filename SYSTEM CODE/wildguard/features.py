"""Step 2 - risk-relevant feature engineering.

The core idea: a single field rarely signals risk, but a record that deviates from the norm
for *its own peer group* often does (e.g. 2 tortoises at $400/unit in a category where peers
ship 19-250 units at $6/unit). Features are therefore mostly *relative*:

  peer group = (genus, product group, unit group), backed off to (class, product, unit)
  route      = (exporter country, genus)

Relative features also transfer better across datasets than absolute ones: "10x the usual
quantity for this peer group" means the same thing in LEMIS and CITES even though the two
datasets have different absolute volume distributions.

FeatureBuilder.fit() learns the reference statistics from one population (LEMIS training
years, or the CITES scoring window); transform() applies them to any records.
"""
import numpy as np
import pandas as pd

MIN_PEER = 20      # back off to the class-level peer group below this many records
TOP_GENUS = 300
TOP_COUNTRY = 120

CAT_SHARED = ["class", "product_group", "unit_group", "purpose", "source", "appendix",
              "origin_region", "exporter_region", "genus_top", "exporter_top", "origin_top"]
NUM_SHARED = ["log_qty", "qty_z", "peer_n_log", "route_share", "route_n_log",
              "combo_rarity", "purpose_share_in_genus", "source_share_in_genus",
              "reexport_flag", "is_wild"]
CAT_LEMIS_ONLY = ["port"]
NUM_LEMIS_ONLY = ["month", "log_value", "log_unit_value", "unit_value_z", "value_missing"]

SHARED_FEATURES = CAT_SHARED + NUM_SHARED
FULL_FEATURES = SHARED_FEATURES + CAT_LEMIS_ONLY + NUM_LEMIS_ONLY

# Numeric deviation features used by the unsupervised detectors (no labels involved).
ANOMALY_FEATURES_SHARED = ["qty_z", "route_share", "combo_rarity",
                           "purpose_share_in_genus", "source_share_in_genus", "peer_n_log"]
ANOMALY_FEATURES_FULL = ANOMALY_FEATURES_SHARED + ["unit_value_z"]


def _robust_stats(df, keys, col, prefix):
    g = df.groupby(keys, observed=True)[col]
    med = g.median().rename(f"{prefix}_med")
    dev = (df[col] - g.transform("median")).abs()
    mad = dev.groupby([df[k] for k in keys], observed=True).median().rename(f"{prefix}_mad")
    n = g.size().rename(f"{prefix}_n")
    return pd.concat([med, mad, n], axis=1).reset_index()


class FeatureBuilder:
    def __init__(self, use_value=True):
        self.use_value = use_value

    # ------------------------------------------------------------------ fit
    def fit(self, ref: pd.DataFrame, vocab_from=None):
        """Learn reference norms from `ref`. `vocab_from` (another fitted FeatureBuilder) fixes
        the genus/country vocabulary to the training population's, so a category the model
        knows is never collapsed to OTHER just because it is rarer in the new population."""
        self.n_ref = len(ref)
        ref = ref.assign(_lq=np.log1p(ref["quantity_std"].clip(lower=0)))
        self.peer_genus = _robust_stats(ref, ["genus", "product_group", "unit_group"], "_lq", "q")
        self.peer_class = _robust_stats(ref, ["class", "product_group", "unit_group"], "_lq", "qc")

        if self.use_value:
            v = ref[(ref["value_usd"] > 0) & (ref["quantity_std"] > 0)]
            v = v.assign(_luv=np.log1p(v["value_usd"] / v["quantity_std"]))
            self.uv_genus = _robust_stats(v, ["genus", "product_group", "unit_group"], "_luv", "v")
            self.uv_class = _robust_stats(v, ["class", "product_group", "unit_group"], "_luv", "vc")

        genus_n = ref.groupby("genus", observed=True).size().rename("genus_n")
        route_n = ref.groupby(["exporter", "genus"], observed=True).size().rename("route_n")
        self.route = route_n.reset_index().merge(genus_n.reset_index(), on="genus")
        self.genus_n = genus_n.reset_index()

        combo = ref.groupby(["product_group", "purpose", "source"], observed=True).size()
        self.combo = (combo / len(ref)).rename("combo_share").reset_index()
        gp = ref.groupby(["genus", "purpose"], observed=True).size().rename("gp_n").reset_index()
        gs = ref.groupby(["genus", "source"], observed=True).size().rename("gs_n").reset_index()
        self.genus_purpose, self.genus_source = gp, gs

        if vocab_from is not None:
            self.top_genus = vocab_from.top_genus
            self.top_exporter = vocab_from.top_exporter
            self.top_origin = vocab_from.top_origin
        else:
            self.top_genus = set(ref["genus"].value_counts().index[:TOP_GENUS])
            self.top_exporter = set(ref["exporter"].value_counts().index[:TOP_COUNTRY])
            self.top_origin = set(ref["origin"].value_counts().index[:TOP_COUNTRY])
        return self

    # ------------------------------------------------------------ transform
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["log_qty"] = np.log1p(out["quantity_std"].clip(lower=0))

        out = out.merge(self.peer_genus, on=["genus", "product_group", "unit_group"], how="left")
        out = out.merge(self.peer_class, on=["class", "product_group", "unit_group"], how="left")
        use_g = out["q_n"].fillna(0) >= MIN_PEER
        med = np.where(use_g, out["q_med"], out["qc_med"])
        mad = np.where(use_g, out["q_mad"], out["qc_mad"])
        out["qty_z"] = (out["log_qty"] - med) / (1.4826 * np.asarray(mad, dtype=float) + 0.25)
        # counts are expressed per million reference records so that they mean the same thing
        # in populations of different size (LEMIS train 3.9M rows vs CITES 8.25M rows)
        per_m = 1e6 / self.n_ref
        out["peer_n_log"] = np.log1p(np.where(use_g, out["q_n"], out["qc_n"].fillna(0)) * per_m)
        out["peer_qty_median"] = np.expm1(np.asarray(med, dtype=float))   # for explanations only

        out = out.merge(self.route, on=["exporter", "genus"], how="left")
        out = out.merge(self.genus_n.rename(columns={"genus_n": "genus_n2"}), on="genus", how="left")
        out["route_n_log"] = np.log1p(out["route_n"].fillna(0) * per_m)
        out["route_share"] = (out["route_n"].fillna(0) / out["genus_n2"]).fillna(0)

        out = out.merge(self.combo, on=["product_group", "purpose", "source"], how="left")
        out["combo_rarity"] = -np.log10(out["combo_share"].fillna(1e-7))
        out = out.merge(self.genus_purpose, on=["genus", "purpose"], how="left")
        out = out.merge(self.genus_source, on=["genus", "source"], how="left")
        out["purpose_share_in_genus"] = (out["gp_n"].fillna(0) / out["genus_n2"]).fillna(0)
        out["source_share_in_genus"] = (out["gs_n"].fillna(0) / out["genus_n2"]).fillna(0)

        out["reexport_flag"] = (out["origin"] != out["exporter"]).astype(int)
        out["is_wild"] = (out["source"] == "W").astype(int)

        out["genus_top"] = np.where(out["genus"].isin(self.top_genus), out["genus"], "OTHER")
        out["exporter_top"] = np.where(out["exporter"].isin(self.top_exporter), out["exporter"], "OTHER")
        out["origin_top"] = np.where(out["origin"].isin(self.top_origin), out["origin"], "OTHER")

        if self.use_value:
            has_v = (out["value_usd"] > 0) & (out["quantity_std"] > 0)
            out["value_missing"] = (~has_v).astype(int)
            out["log_value"] = np.log1p(out["value_usd"].where(out["value_usd"] > 0))
            out["log_unit_value"] = np.log1p((out["value_usd"] / out["quantity_std"]).where(has_v))
            out = out.merge(self.uv_genus, on=["genus", "product_group", "unit_group"], how="left")
            out = out.merge(self.uv_class, on=["class", "product_group", "unit_group"], how="left")
            use_vg = out["v_n"].fillna(0) >= MIN_PEER
            vmed = np.where(use_vg, out["v_med"], out["vc_med"])
            vmad = np.where(use_vg, out["v_mad"], out["vc_mad"])
            out["unit_value_z"] = (out["log_unit_value"] - vmed) / (
                1.4826 * np.asarray(vmad, dtype=float) + 0.25)
            out["peer_unit_value_median"] = np.expm1(np.asarray(vmed, dtype=float))

        drop = [c for c in out.columns if c.startswith(("q_", "qc_", "v_", "vc_"))] + [
            "route_n", "genus_n", "genus_n2", "combo_share", "gp_n", "gs_n"]
        return out.drop(columns=[c for c in drop if c in out.columns])


def to_model_matrix(df: pd.DataFrame, features, categories=None):
    """Select model columns and cast categoricals to a fixed category set (learned on the
    training data) so XGBoost sees identical encodings at train and score time. Categories
    unseen in training become missing values."""
    X = df[features].copy()
    cats = [c for c in features if c in CAT_SHARED + CAT_LEMIS_ONLY]
    if categories is None:
        categories = {c: sorted(X[c].dropna().astype(str).unique()) for c in cats}
    for c in cats:
        X[c] = pd.Categorical(X[c].astype(str), categories=categories[c])
    for c in features:
        if c not in cats:
            X[c] = X[c].astype("float32")
    return X, categories
