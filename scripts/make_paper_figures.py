"""Publication figures for the rewritten paper. Vector PDF, neutral styling;
all interpretation lives in the LaTeX captions, not the figure titles.

Figures:
  fig_influence_map.pdf      (Study 1) cross-lab directed influence matrix + susceptibility-vs-persuasiveness scatter
  fig_capability_channel.pdf (Study 1) persuader-capability gradient + peer-vs-authority dissociation
  fig_factorial.pdf          (Study 2) baseline + peer + expert by model with Wilson CIs; citation effect
  fig_adapt_verify.pdf       (Study 2) static>=live adaptivity + forced-source verification
  fig_boundary.pdf           (scoping) induced revision by belief type
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 200,
})
BLUE, RED, GREEN, PURPLE, ORANGE, GRAY = (
    "#2166AC", "#B2182B", "#1A9850", "#762A83", "#E08214", "#5A5A5A")


def wilson(x, n, z=1.96):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = x / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return p, max(0, c - h), min(1, c + h)


def load_json(*cands):
    for c in cands:
        p = RES / c if not c.startswith("/") else Path(c)
        if (RES / c).exists():
            return json.loads((RES / c).read_text())
    return None


def summary_cells(run_dir):
    s = json.loads((RES / run_dir / "summary.json").read_text())
    return {k: (v["x"], v["n"]) for k, v in s["headline"]["cells"].items()}


SHORT = {"gpt-3.5-turbo": "GPT-3.5", "gpt-4o-mini": "GPT-4o-mini",
         "gpt-5-nano": "GPT-5-nano", "gpt-5.4-mini": "GPT-5.4-mini", "gpt-5.5": "GPT-5.5",
         "claude-haiku-4.5": "Haiku-4.5", "claude-sonnet-4.6": "Sonnet-4.6",
         "claude-opus-4.6": "Opus-4.6", "llama-3.1-8b": "Llama-8B",
         "llama-3.3-70b": "Llama-70B", "deepseek-v3.2": "DeepSeek", "mistral-large-3": "Mistral-L3",
         "qwen3-235b": "Qwen3-235B",
         # legacy labels (old cross-lab appendix matrix)
         "mistral-small": "Mistral-Sm", "qwen-2.5-72b": "Qwen-72B",
         "deepseek-chat": "DeepSeek", "gemini-2.5-flash": "Gemini-Fl", "claude-sonnet-4": "Sonnet-4"}


# ----------------------------------------------------------------------- Fig 1
def fig_influence_map():
    cl = load_json("dialogue_matrix_unified_aws/matrix.json")
    models = cl["models"]
    grid = cl["grid"]
    M = np.array([[grid.get(s, {}).get(p, np.nan) for p in models] for s in models])
    suscept = np.nanmean(M, axis=1)          # row mean = how movable as subject
    persuade = np.nanmean(M, axis=0)         # col mean = how persuasive
    order_r = np.argsort(-suscept)
    order_c = np.argsort(-persuade)
    Mo = M[np.ix_(order_r, order_c)]
    rlab = [SHORT[models[i]] for i in order_r]
    clab = [SHORT[models[i]] for i in order_c]

    fig = plt.figure(figsize=(7.4, 3.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0], wspace=0.42)
    ax = fig.add_subplot(gs[0])
    im = ax.imshow(Mo, cmap="Reds", vmin=0, vmax=0.6, aspect="auto")
    ax.set_xticks(range(len(models)), clab, rotation=45, ha="right")
    ax.set_yticks(range(len(models)), rlab)
    for i in range(len(models)):
        for j in range(len(models)):
            v = Mo[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.0,
                        color="white" if v > 0.36 else "#333333")
    ax.set_xlabel("persuader  (sorted by persuasiveness →)")
    ax.set_ylabel("subject  (↑ sorted by susceptibility)")
    ax.set_title("(a) directed influence map (12 models, 5 labs)")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("harmful-revision rate", fontsize=7.5)
    cb.ax.tick_params(labelsize=7)

    ax2 = fig.add_subplot(gs[1])
    ax2.scatter(persuade, suscept, s=34, color=BLUE, zorder=3)
    # hand-tuned label offsets to de-crowd the near-immune cluster (bottom strip)
    off = {"GPT-3.5": (0.006, -0.004, "left"), "Llama-8B": (0.006, 0.004, "left"),
           "GPT-5-nano": (0.006, 0.006, "left"), "Mistral-L3": (0.004, 0.028, "left"),
           "GPT-4o-mini": (0.004, 0.022, "left"), "Llama-70B": (-0.004, 0.020, "right"),
           "Sonnet-4.6": (0.004, 0.020, "left"), "DeepSeek": (0.004, -0.028, "left"),
           "GPT-5.5": (0.004, 0.006, "left"), "GPT-5.4-mini": (0.002, -0.030, "center"),
           "Opus-4.6": (-0.004, 0.022, "right"), "Haiku-4.5": (-0.004, -0.024, "right")}
    for i, m in enumerate(models):
        dx, dy, ha = off.get(SHORT[m], (0.006, 0.010, "left"))
        ax2.annotate(SHORT[m], (persuade[i], suscept[i]), (persuade[i] + dx, suscept[i] + dy),
                     fontsize=6.4, ha=ha,
                     arrowprops=dict(arrowstyle="-", lw=0.3, color="#999")
                     if abs(dy) > 0.018 else None)
    ax2.set_xlabel("persuasiveness (mean as persuader)")
    ax2.set_ylabel("susceptibility (mean as subject)")
    ax2.set_title("(b) the two axes are orthogonal")
    ax2.set_xlim(0.04, 0.24)
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, lw=0.4, alpha=0.4)
    fig.savefig(OUT / "fig_influence_map.pdf", bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------- Fig 2
def fig_capability_channel():
    m = load_json("dialogue_matrix_unified_aws/matrix.json")["grid"]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)

    def gradient(axx, subj, ladder, color, title):
        rates = [m[subj][p] for p in ladder]
        axx.plot(range(len(ladder)), rates, "o-", color=color, lw=1.8, ms=6)
        for i, v in enumerate(rates):
            axx.annotate(f"{v:.2f}", (i, v), (i, v + 0.04), ha="center", fontsize=6.5)
        axx.set_xticks(range(len(ladder)), [SHORT[p] for p in ladder], rotation=35, ha="right")
        axx.set_xlabel("persuader  (weaker → stronger)")
        axx.set_title(title)
        axx.grid(True, axis="y", lw=0.4, alpha=0.4)

    # two graded subjects; persuaders ordered by capability (left→right)
    ladder_a = ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-5-nano", "gpt-5.4-mini", "gpt-5.5", "deepseek-v3.2"]
    ladder_b = ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-5-nano", "gpt-5.4-mini", "deepseek-v3.2",
                "claude-sonnet-4.6", "gpt-5.5"]
    gradient(ax, "gpt-5-nano", ladder_a, PURPLE, "(a) subject: GPT-5-nano")
    gradient(ax2, "llama-3.1-8b", ladder_b, ORANGE, "(b) subject: Llama-3.1-8B")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("harmful-revision rate")
    fig.tight_layout()
    fig.savefig(OUT / "fig_capability_channel.pdf", bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------- Fig 3
MODELS12 = [
    ("GPT-3.5", "causal_powered_openai_gpt35turbo"),
    ("GPT-4o-mini", "causal_powered_openai_gpt4omini"),
    ("GPT-5-nano", "causal_powered_openai_gpt5nano"),
    ("GPT-5.4-mini", "causal_powered_openai_gpt54mini"),
    ("GPT-5.5", "causal_powered_openai_gpt55"),
    ("Haiku 4.5", "causal_powered_bedrock_haiku45"),
    ("Sonnet 4.6", "causal_powered_bedrock_sonnet46"),
    ("Opus 4.6", "causal_powered_bedrock_opus46"),
    ("Llama-8B", "causal_powered_bedrock_llama31_8b"),
    ("Llama-70B", "causal_powered_bedrock_llama33_70b"),
    ("DeepSeek", "causal_powered_bedrock_deepseek"),
    ("Mistral-L3", "causal_powered_bedrock_mistral"),
    ("Gemini-Fl", "causal_powered_openrouter_gemini25flash"),
    ("Qwen3-235B", "causal_powered_openrouter_qwen3_235b"),
]


def fig_factorial():
    cells = {lab: summary_cells(run) for lab, run in MODELS12}
    labs = [lab for lab, _ in MODELS12]
    # sort by expert rate (descending) so the susceptibility gradient reads top-down
    labs.sort(key=lambda L: cells[L]["hidden_expert_none"][0], reverse=True)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.6, 4.4),
                                  gridspec_kw={"width_ratios": [1.25, 1.0]})

    # (a) horizontal grouped bars: baseline / peer / expert (withheld, no citation)
    series = [("expert", "hidden_expert_none", RED),
              ("peer", "hidden_peer_none", BLUE),
              ("no pressure", "hidden_no_pressure", GRAY)]
    y = np.arange(len(labs)); h = 0.26
    for k, (lab, cell, col) in enumerate(series):
        ps, xerr = [], [[], []]
        for L in labs:
            xx, n = cells[L][cell]
            p, lo, hi = wilson(xx, n)
            ps.append(p); xerr[0].append(p - lo); xerr[1].append(hi - p)
        ax.barh(y + (1 - k) * h, ps, h, label=lab, color=col,
                xerr=xerr, capsize=1.5, error_kw={"elinewidth": 0.7, "ecolor": "#444"})
    ax.set_yticks(y, labs)
    ax.invert_yaxis()
    ax.set_xlabel("harmful-revision rate (source withheld)")
    ax.set_xlim(0, 1.02)
    ax.set_title("(a) authority vs. peer vs. baseline")
    ax.legend(frameon=False, loc="lower right", handlelength=1.2)
    ax.grid(True, axis="x", lw=0.4, alpha=0.4)

    # (b) citation effect: Δ = (expert+cite) − expert, per model (never amplifies)
    deltas = []
    for L in labs:
        e = cells[L]["hidden_expert_none"][0] / 180
        ec = cells[L]["hidden_expert_fabricated"][0] / 180
        deltas.append(ec - e)
    colors = [RED if d > 0.02 else (GREEN if d < -0.02 else GRAY) for d in deltas]
    ax2.barh(y, deltas, 0.6, color=colors)
    ax2.axvline(0, color="#222", lw=0.8)
    ax2.set_yticks(y, labs)
    ax2.invert_yaxis()
    ax2.set_xlabel(r"$\Delta$ from adding a fabricated citation")
    ax2.set_xlim(-0.5, 0.12)
    ax2.set_title("(b) citation never amplifies")
    ax2.grid(True, axis="x", lw=0.4, alpha=0.4)
    fig.tight_layout()
    fig.savefig(OUT / "fig_factorial.pdf", bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------- Fig 4
def fig_adapt_verify():
    adapt = {"gpt-4o-mini": "causal_adaptivity_openai_gpt4omini_subject_gpt55_persuader",
             "Claude Haiku 4.5": "causal_adaptivity_bedrock_haiku45_subject_gpt55_persuader",
             "GPT-5.5": "causal_adaptivity_openai_gpt55_subject_gpt55_persuader"}
    verify = {"gpt-4o-mini": "causal_verification_openai_gpt4omini",
              "Claude Haiku 4.5": "causal_verification_bedrock_haiku45",
              "GPT-5.5": "causal_verification_openai_gpt55"}
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.0))

    ms = list(adapt)
    conds = [("live adaptive", "hidden_expert_none_live", BLUE),
             ("replayed", "hidden_expert_none_replay", GRAY),
             ("static (canned)", "hidden_expert_none_static", RED)]
    x = np.arange(len(ms)); w = 0.26
    for k, (lab, cell, col) in enumerate(conds):
        vals = []
        for m in ms:
            c = summary_cells(adapt[m]); vals.append(c[cell][0] / c[cell][1])
        ax.bar(x + (k - 1) * w, vals, w, label=lab, color=col)
    ax.set_xticks(x, [m.replace("Claude ", "") for m in ms], rotation=18, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("harmful-revision rate")
    ax.set_title("(a) a canned line $\\geq$ live adaptation")
    ax.legend(frameon=False, loc="upper right")
    ax.grid(True, axis="y", lw=0.4, alpha=0.4)

    x2 = np.arange(len(ms)); w2 = 0.36
    no_v, fs = [], []
    for m in ms:
        c = summary_cells(verify[m])
        no_v.append(c["hidden_expert_fabricated_no_verification"][0] / 180)
        fs.append(c["hidden_expert_fabricated_forced_source"][0] / 180)
    ax2.bar(x2 - w2 / 2, no_v, w2, label="no verification", color=RED)
    ax2.bar(x2 + w2 / 2, fs, w2, label="forced source", color=GREEN)
    ax2.set_xticks(x2, [m.replace("Claude ", "") for m in ms], rotation=18, ha="right")
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("harmful-revision rate")
    ax2.set_title("(b) re-grounding returns to the visible floor")
    ax2.legend(frameon=False, loc="upper right")
    ax2.grid(True, axis="y", lw=0.4, alpha=0.4)
    fig.tight_layout()
    fig.savefig(OUT / "fig_adapt_verify.pdf", bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------- Fig 5
def fig_boundary():
    conds = ["line length\n(perceptual)", "verifiable fact\n(strong prior)",
             "fictional fact,\nsource visible", "fictional fact,\nsource withheld"]
    gpt35 = [0.00, 0.00, 0.00, 0.82]
    gpt4o = [0.00, 0.00, 0.00, 0.00]
    x = np.arange(len(conds)); w = 0.38
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    ax.bar(x - w / 2, gpt35, w, label="GPT-3.5", color=RED)
    ax.bar(x + w / 2, gpt4o, w, label="GPT-4o-mini", color=BLUE)
    ax.set_xticks(x, conds)
    ax.set_ylabel("harmful-revision rate")
    ax.set_ylim(0, 1.0)
    ax.set_title("an arbitrary AND non-re-checkable belief is required")
    ax.legend(frameon=False)
    ax.grid(True, axis="y", lw=0.4, alpha=0.4)
    fig.tight_layout()
    fig.savefig(OUT / "fig_boundary.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_influence_map()
    fig_capability_channel()
    fig_factorial()
    fig_adapt_verify()
    fig_boundary()
    print("figures ->", OUT)
    for f in sorted(OUT.glob("fig_*.pdf")):
        print("  ", f.name, f.stat().st_size, "bytes")
