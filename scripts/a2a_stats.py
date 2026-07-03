"""Inferential stats for the A2A experiments (source x rationale, live A2A, verifier, cascade).

Within-item paired McNemar tests (same 180 items crossed across cells) with
bootstrap CIs, mirroring scripts/causal_stats.py.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy.stats import binomtest  # noqa: F401  (kept for parity)
from statsmodels.stats.contingency_tables import mcnemar

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
RNG = np.random.default_rng(20260626)


def load(run_dir):
    rows = {}
    for line in (RES / run_dir / "causal.jsonl").open():
        r = json.loads(line)
        rows.setdefault(r["condition"]["name"], {})[r["fact_id"]] = int(bool(r["harmful_revision"]))
    return rows


def paired(rows, a, b):
    items = sorted(set(rows[a]) & set(rows[b]))
    av = np.array([rows[a][i] for i in items]); bv = np.array([rows[b][i] for i in items])
    b01 = int(np.sum((av == 0) & (bv == 1))); b10 = int(np.sum((av == 1) & (bv == 0)))
    table = [[int(np.sum((av == 1) & (bv == 1))), b10], [b01, int(np.sum((av == 0) & (bv == 0)))]]
    p = float(mcnemar(table, exact=True).pvalue)
    boots = [bv[s].mean() - av[s].mean() for s in
             (RNG.choice(len(items), len(items), replace=True) for _ in range(3000))]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return dict(a_rate=float(av.mean()), b_rate=float(bv.mean()), diff=float(bv.mean() - av.mean()),
                ci=[float(lo), float(hi)], p=p, n=len(items))


SRT = {"gpt-4o-mini": "causal_srt_openai_gpt4omini", "Haiku 4.5": "causal_srt_bedrock_haiku45",
       "DeepSeek": "causal_srt_bedrock_deepseek", "Mistral-L3": "causal_srt_bedrock_mistral",
       "Llama-8B": "causal_srt_bedrock_llama31_8b"}
print("=" * 80)
print("SOURCE x RATIONALE — rationale effect (hidden: retained->removed) and source effect")
print("=" * 80)
for m, rd in SRT.items():
    r = load(rd)
    rat = paired(r, "hidden_retained_expert", "hidden_removed_expert")   # removing rationale
    src = paired(r, "visible_removed_expert", "hidden_removed_expert")   # removing source (no rationale)
    print(f"\n### {m}")
    print(f"   remove rationale (hidden): {rat['a_rate']*180:.0f}->{rat['b_rate']*180:.0f}  "
          f"Δ={rat['diff']:+.3f} [{rat['ci'][0]:+.2f},{rat['ci'][1]:+.2f}] p={rat['p']:.1e}")
    print(f"   remove source (no rat.):   {src['a_rate']*180:.0f}->{src['b_rate']*180:.0f}  "
          f"Δ={src['diff']:+.3f} [{src['ci'][0]:+.2f},{src['ci'][1]:+.2f}] p={src['p']:.1e}")

LIVE = {"gpt-4o-mini": "livea2a_gpt4omini_persGPT54mini", "Haiku 4.5": "livea2a_haiku45_persGPT54mini"}
print("\n" + "=" * 80)
print("LIVE A2A — live adaptive vs static, and neutral vs static (hidden+retained)")
print("=" * 80)
for m, rd in LIVE.items():
    r = load(rd)
    ls = paired(r, "static_expert", "live_expert")
    ns = paired(r, "static_expert", "neutral")
    print(f"\n### {m}")
    print(f"   live vs static:    {ls['a_rate']*180:.0f}->{ls['b_rate']*180:.0f}  "
          f"Δ={ls['diff']:+.3f} [{ls['ci'][0]:+.2f},{ls['ci'][1]:+.2f}] p={ls['p']:.1e}")
    print(f"   neutral vs static: {ns['a_rate']*180:.0f}->{ns['b_rate']*180:.0f}  "
          f"Δ={ns['diff']:+.3f} p={ns['p']:.1e}")

print("\n" + "=" * 80)
print("VERIFIER (smoke) & CASCADE (n=60) — descriptive")
print("=" * 80)
casc = json.loads((RES / "reports" / "cascade_main.json").read_text())["cells"]
for k, c in casc.items():
    print(f"   cascade {k:16s} worker_corrupt={c['worker_corruption_rate']:.2f} "
          f"system_error={c['system_error_rate']:.2f} P(prop|corrupt)={c['propagation_given_corrupted']:.2f}")
print("-> A2A stats computed")
