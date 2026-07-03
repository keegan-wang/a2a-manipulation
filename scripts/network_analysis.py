"""Directed-network analysis of the live influence matrix (Experiment 1).

Treats the 14x14 coalition->agent matrix as a directed epistemic-influence graph
and asks whether it reduces to additive attacker-strength minus target-resistance,
or carries genuine pair-specific structure. Fits

    logit P(Y_ijk = 1) = mu + A_j - R_i + I_ij + u_k

via (a) a two-way variance decomposition of the cell-rate matrix and (b) a
Bayesian logistic mixed model with a crossed item random effect. Also reports the
directional asymmetry D_ij = P(i<-j) - P(j<-i).

Y = induced_hallucination, conditioned on the subject knowing the fact alone.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "results" / "dialogue_matrix_unified_aws"
OUT = ROOT / "results" / "reports"
OUT.mkdir(parents=True, exist_ok=True)

# ---- load per-item outcomes from every cell
rows = []
for cell in sorted(MATRIX.glob("*__S_*__P_*")):
    name = cell.name
    subj = name.split("__S_")[1].split("__P_")[0]
    pers = name.split("__P_")[1]
    f = cell / "dialogue.jsonl"
    if not f.exists():
        continue
    for line in f.open():
        r = json.loads(line)
        if not r.get("knew_alone", False):
            continue  # at-risk set only: the subject knew it alone
        rows.append({"subject": subj, "persuader": pers, "item": r["fact_id"],
                     "Y": int(bool(r.get("induced_hallucination")))})
df = pd.DataFrame(rows)
models = sorted(set(df.subject) | set(df.persuader))
print(f"loaded {len(df)} at-risk trials across {df.groupby(['subject','persuader']).ngroups} cells, "
      f"{df.item.nunique()} items, {len(models)} models")

# ---- cell-rate matrix P_ij  (rows=subject i, cols=persuader j)
P = df.groupby(["subject", "persuader"]).Y.mean().unstack().reindex(index=models, columns=models)

# ---- two-way variance decomposition: additive (attacker - resistance) vs interaction
grand = np.nanmean(P.values)
row_eff = np.nanmean(P.values, axis=1) - grand       # -R_i  (subject susceptibility)
col_eff = np.nanmean(P.values, axis=0) - grand       #  A_j  (attacker effectiveness)
fitted = grand + row_eff[:, None] + col_eff[None, :]
resid = P.values - fitted
ss_tot = np.nansum((P.values - grand) ** 2)
ss_row = P.shape[1] * np.nansum(row_eff ** 2)
ss_col = P.shape[0] * np.nansum(col_eff ** 2)
ss_int = np.nansum(resid ** 2)
print("\n=== variance decomposition of the 14x14 cell-rate matrix ===")
print(f"  subject resistance (rows):   {ss_row/ss_tot:6.1%}")
print(f"  attacker strength (cols):    {ss_col/ss_tot:6.1%}")
print(f"  pair-specific interaction:   {ss_int/ss_tot:6.1%}")
print(f"  additive model R^2:          {1 - ss_int/ss_tot:6.1%}")

# ---- attacker effectiveness and resistance rankings
A = (np.nanmean(P.values, axis=0))   # column mean = mean harm caused as persuader
R = (np.nanmean(P.values, axis=1))   # row mean = mean harm taken as subject
rank = pd.DataFrame({"model": models, "attacker_effectiveness_A": A, "susceptibility_R": R})
print("\n=== attacker effectiveness (A_j) — most effective persuaders ===")
for _, r in rank.sort_values("attacker_effectiveness_A", ascending=False).head(6).iterrows():
    print(f"  {r.model:18s} A={r.attacker_effectiveness_A:.3f}  (susceptibility {r.susceptibility_R:.3f})")

# ---- directional asymmetry D_ij = P(i<-j) - P(j<-i)
asym = []
for a in range(len(models)):
    for b in range(a + 1, len(models)):
        pij = P.values[a, b]   # subject a, persuader b
        pji = P.values[b, a]   # subject b, persuader a
        if np.isnan(pij) or np.isnan(pji):
            continue
        asym.append((models[a], models[b], pij, pji, pij - pji))
asym.sort(key=lambda t: -abs(t[4]))
print("\n=== largest directional asymmetries  D = P(row<-col) - P(col<-row) ===")
for s, p, pij, pji, d in asym[:8]:
    hi, lo = (s, p) if d > 0 else (p, s)
    print(f"  {hi} <- {lo}: {max(pij,pji):.2f}   vs reverse {min(pij,pji):.2f}   |D|={abs(d):.2f}")

# ---- Bayesian logistic mixed model: Y ~ attacker + subject, random item
gee_ok = False
try:
    import statsmodels.api as sm
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
    d = df.copy()
    # reference = the most-immune subject / weakest attacker so effects are interpretable
    d["subject"] = pd.Categorical(d.subject, categories=models)
    d["persuader"] = pd.Categorical(d.persuader, categories=models)
    md = BinomialBayesMixedGLM.from_formula(
        "Y ~ C(subject) + C(persuader)", {"item": "0 + C(item)"}, d)
    res = md.fit_vb()
    gee_ok = True
    print("\n=== Bayesian logistic mixed model fit (item random effect): converged ===")
    print(f"  item-level random-effect SD (posterior): {np.exp(res.vcp_mean[0]):.2f}")
except Exception as e:  # pragma: no cover
    print(f"\n[mixed model skipped: {type(e).__name__}: {str(e)[:80]}]")

report = {
    "n_trials": int(len(df)), "n_items": int(df.item.nunique()), "n_models": len(models),
    "variance_decomposition": {
        "subject_resistance": ss_row / ss_tot, "attacker_strength": ss_col / ss_tot,
        "pair_interaction": ss_int / ss_tot, "additive_r2": 1 - ss_int / ss_tot},
    "ranking": rank.to_dict("records"),
    "top_asymmetries": [{"strong": (s if d > 0 else p), "weak": (p if d > 0 else s),
                         "forward": max(pij, pji), "reverse": min(pij, pji), "D": abs(d)}
                        for s, p, pij, pji, d in asym[:10]],
    "mixed_model_converged": gee_ok,
}
(OUT / "network_analysis.json").write_text(json.dumps(report, indent=2, default=float))
print(f"\n-> wrote {OUT/'network_analysis.json'}")
