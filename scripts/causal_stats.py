"""Inferential statistics for the causal experiments (Study 2).

Items are fully crossed: the same 180 relational facts appear in every condition,
so every contrast is within-item paired. We therefore use:
  * exact McNemar tests on the discordant pairs for each pairwise contrast,
  * paired (within-item) risk differences with BCa-free percentile bootstrap CIs,
  * a GEE logistic regression (binomial/logit) clustered by item to test the
    framing x citation interaction under withheld source, for the susceptible
    models (avoids the perfect separation that source-visibility induces),
  * Holm-Bonferroni correction across the family of primary contrasts.

Outputs a human-readable report and a machine-readable JSON consumed by the paper.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from statsmodels.stats.contingency_tables import mcnemar
import statsmodels.api as sm
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = ROOT / "results" / "reports"
OUT.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(20260625)

POWERED = {
    "GPT-3.5-turbo": "causal_powered_openai_gpt35turbo",
    "gpt-4o-mini": "causal_powered_openai_gpt4omini",
    "GPT-5-nano": "causal_powered_openai_gpt5nano",
    "GPT-5.4-mini": "causal_powered_openai_gpt54mini",
    "GPT-5.5": "causal_powered_openai_gpt55",
    "Claude Haiku 4.5": "causal_powered_bedrock_haiku45",
    "Claude Sonnet 4.6": "causal_powered_bedrock_sonnet46",
    "Claude Opus 4.6": "causal_powered_bedrock_opus46",
    "Llama-3.1-8B": "causal_powered_bedrock_llama31_8b",
    "Llama-3.3-70B": "causal_powered_bedrock_llama33_70b",
    "DeepSeek-V3.2": "causal_powered_bedrock_deepseek",
    "Mistral-Large-3": "causal_powered_bedrock_mistral",
    "Gemini-2.5-Flash": "causal_powered_openrouter_gemini25flash",
    "Qwen3-235B": "causal_powered_openrouter_qwen3_235b",
}
ADAPT = {
    "gpt-4o-mini": "causal_adaptivity_openai_gpt4omini_subject_gpt55_persuader",
    "Claude Haiku 4.5": "causal_adaptivity_bedrock_haiku45_subject_gpt55_persuader",
    "GPT-5.5": "causal_adaptivity_openai_gpt55_subject_gpt55_persuader",
}
VERIFY = {
    "gpt-4o-mini": "causal_verification_openai_gpt4omini",
    "Claude Haiku 4.5": "causal_verification_bedrock_haiku45",
    "GPT-5.5": "causal_verification_openai_gpt55",
}


def load(run_dir: str) -> pd.DataFrame:
    rows = []
    with open(RES / run_dir / "causal.jsonl") as f:
        for line in f:
            r = json.loads(line)
            rows.append({
                "fact_id": r["fact_id"],
                "cell": r["condition"]["name"],
                "source": r["condition"]["source_availability"],
                "speaker": r["condition"]["speaker"],
                "citation": r["condition"]["citation"],
                "harmful": int(bool(r["harmful_revision"])),
                "final_parseable": int(bool(r.get("final_parseable", True))),
                "transition": r.get("transition"),
            })
    return pd.DataFrame(rows)


def paired(df: pd.DataFrame, cell_a: str, cell_b: str):
    """Within-item paired contrast B - A (harmful revision). Returns dict."""
    a = df[df.cell == cell_a].set_index("fact_id")["harmful"]
    b = df[df.cell == cell_b].set_index("fact_id")["harmful"]
    common = a.index.intersection(b.index)
    a, b = a.loc[common].to_numpy(), b.loc[common].to_numpy()
    n = len(common)
    # discordant pairs
    b01 = int(np.sum((a == 0) & (b == 1)))  # gained under B
    b10 = int(np.sum((a == 1) & (b == 0)))  # lost under B
    table = [[int(np.sum((a == 1) & (b == 1))), b10],
             [b01, int(np.sum((a == 0) & (b == 0)))]]
    res = mcnemar(table, exact=True)
    diff = float(b.mean() - a.mean())
    # percentile bootstrap CI on the paired difference
    idx = np.arange(n)
    boots = []
    for _ in range(5000):
        s = RNG.choice(idx, n, replace=True)
        boots.append(b[s].mean() - a[s].mean())
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {
        "a": cell_a, "b": cell_b, "n": n,
        "rate_a": float(a.mean()), "rate_b": float(b.mean()),
        "diff": diff, "diff_ci": [float(lo), float(hi)],
        "discordant_b_gains": b01, "discordant_b_loses": b10,
        "p": float(res.pvalue),
    }


def holm(pvals: dict[str, float]) -> dict[str, float]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adj, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        a = min(1.0, (m - i) * p)
        running = max(running, a)
        adj[k] = running
    return adj


def gee_interaction(df: pd.DataFrame):
    """GEE logistic clustered by item: harmful ~ speaker*citation on hidden cells."""
    d = df[df.source == "hidden"].copy()
    d = d[d.cell != "hidden_no_pressure"]
    d["expert"] = (d.speaker == "expert").astype(int)
    d["fab"] = (d.citation == "fabricated").astype(int)
    try:
        m = smf.gee("harmful ~ expert * fab", "fact_id", data=d,
                    family=sm.families.Binomial(),
                    cov_struct=sm.cov_struct.Exchangeable())
        r = m.fit(maxiter=200)
        out = {}
        for name in r.params.index:
            ci = r.conf_int().loc[name]
            out[name] = {
                "beta": float(r.params[name]),
                "or": float(np.exp(r.params[name])),
                "or_ci": [float(np.exp(ci[0])), float(np.exp(ci[1]))],
                "p": float(r.pvalues[name]),
            }
        return out
    except Exception as e:  # pragma: no cover
        return {"error": str(e)}


report = {"powered_contrasts": {}, "gee": {}, "adaptivity": {}, "verification": {},
          "baseline_corrected": {}, "gpt55_unparseable_bounds": {}}

pval_family = {}

print("=" * 78)
print("STUDY 2 — INFERENTIAL STATISTICS (within-item paired; same 180 items/cell)")
print("=" * 78)

for model, rd in POWERED.items():
    df = load(rd)
    contrasts = {
        "peer_vs_nopressure": ("hidden_no_pressure", "hidden_peer_none"),
        "expert_vs_nopressure": ("hidden_no_pressure", "hidden_expert_none"),
        "expert_vs_peer": ("hidden_peer_none", "hidden_expert_none"),
        "cite_under_expert": ("hidden_expert_none", "hidden_expert_fabricated"),
        "cite_under_peer": ("hidden_peer_none", "hidden_peer_fabricated"),
        "source_withholding": ("visible_expert_fabricated", "hidden_expert_fabricated"),
    }
    report["powered_contrasts"][model] = {}
    print(f"\n### {model}  ({rd})")
    base = df[df.cell == "hidden_no_pressure"]["harmful"].mean()
    print(f"   hidden no-pressure baseline: {base*180:.0f}/180 = {base:.3f}")
    report["baseline_corrected"][model] = {"baseline_rate": float(base)}
    for name, (ca, cb) in contrasts.items():
        r = paired(df, ca, cb)
        report["powered_contrasts"][model][name] = r
        pval_family[f"{model}:{name}"] = r["p"]
        print(f"   {name:22s} {r['rate_a']*180:5.0f}->{r['rate_b']*180:<5.0f} "
              f"Δ={r['diff']:+.3f} [{r['diff_ci'][0]:+.3f},{r['diff_ci'][1]:+.3f}] "
              f"disc(+{r['discordant_b_gains']}/-{r['discordant_b_loses']}) p={r['p']:.2e}")
    # GEE interaction for models with nonzero attack
    if df[df.source == "hidden"]["harmful"].sum() > 5:
        report["gee"][model] = gee_interaction(df)

# GPT-5.5 unparseable sensitivity bounds (hidden_expert_none)
df55 = load(POWERED["GPT-5.5"])
cell = df55[df55.cell == "hidden_expert_none"]
n = len(cell)
harmful = int(cell["harmful"].sum())
unparse = int((cell["transition"] == "correct_unparseable").sum())
report["gpt55_unparseable_bounds"]["hidden_expert_none"] = {
    "n": n, "harmful_strict": harmful, "unparseable": unparse,
    "rate_best_case": harmful / n,                       # unparseable = not harmful
    "rate_worst_case": (harmful + unparse) / n,          # unparseable = harmful
    "rate_among_parseable": harmful / (n - unparse),
}
print(f"\n### GPT-5.5 unparseable bounds (hidden_expert_none): "
      f"strict {harmful}/{n}={harmful/n:.3f}, "
      f"worst {(harmful+unparse)}/{n}={(harmful+unparse)/n:.3f}, "
      f"among-parseable {harmful}/{n-unparse}={harmful/(n-unparse):.3f}")

# Adaptivity: reframe as adaptive(live) vs static
print("\n" + "=" * 78)
print("ADAPTIVITY — adaptive(live) vs canned(static); replay is determinism check")
print("=" * 78)
for model, rd in ADAPT.items():
    df = load(rd)
    report["adaptivity"][model] = {}
    for name, (ca, cb) in {
        "static_vs_live": ("hidden_expert_none_live", "hidden_expert_none_static"),
        "replay_vs_live": ("hidden_expert_none_live", "hidden_expert_none_replay"),
    }.items():
        r = paired(df, ca, cb)
        report["adaptivity"][model][name] = r
        if name == "static_vs_live":
            pval_family[f"adapt:{model}:static_vs_live"] = r["p"]
        print(f"   {model:18s} {name:16s} {r['rate_a']*180:.0f}->{r['rate_b']*180:.0f} "
              f"Δ={r['diff']:+.3f} disc(+{r['discordant_b_gains']}/-{r['discordant_b_loses']}) p={r['p']:.2e}")

# Verification
print("\n" + "=" * 78)
print("VERIFICATION — forced source vs none (note: equals visible-source floor)")
print("=" * 78)
for model, rd in VERIFY.items():
    df = load(rd)
    r = paired(df, "hidden_expert_fabricated_no_verification",
               "hidden_expert_fabricated_forced_source")
    report["verification"][model] = r
    pval_family[f"verify:{model}"] = r["p"]
    print(f"   {model:18s} {r['rate_a']*180:.0f}->{r['rate_b']*180:.0f} "
          f"Δ={r['diff']:+.3f} p={r['p']:.2e}")

# Holm correction across the primary family
adj = holm(pval_family)
report["holm_adjusted_p"] = adj
print("\n" + "=" * 78)
print("HOLM-CORRECTED p-values (primary contrast family, m=%d)" % len(adj))
print("=" * 78)
for k in sorted(adj, key=lambda x: adj[x]):
    print(f"   {k:42s} raw={pval_family[k]:.2e}  holm={adj[k]:.2e}")

# GEE printout
print("\n" + "=" * 78)
print("GEE LOGISTIC (hidden cells, clustered by item): harmful ~ expert*fab")
print("=" * 78)
for model, g in report["gee"].items():
    print(f"\n### {model}")
    if "error" in g:
        print("   GEE failed:", g["error"]); continue
    for term, v in g.items():
        print(f"   {term:18s} OR={v['or']:8.2f} [{v['or_ci'][0]:.2f},{v['or_ci'][1]:.2f}] p={v['p']:.2e}")

(OUT / "causal_stats.json").write_text(json.dumps(report, indent=2))
print(f"\n-> wrote {OUT/'causal_stats.json'}")
