"""Figures for the reframed A2A paper. Vector PDF, neutral styling."""
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
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.titlesize": 9,
                     "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
                     "legend.fontsize": 7.5, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 200})
BLUE, RED, GREEN, PURPLE, ORANGE, GRAY = (
    "#2166AC", "#B2182B", "#1A9850", "#762A83", "#E08214", "#5A5A5A")


def cells(run):
    s = json.loads((RES / run / "summary.json").read_text())
    return {k: v["x"] / v["n"] for k, v in s["headline"]["cells"].items()}


# ---- Fig: source x rationale (5 subjects, 4 states)
def fig_sourcerationale():
    subs = [("GPT-4o-mini", "causal_srt_openai_gpt4omini"), ("Haiku 4.5", "causal_srt_bedrock_haiku45"),
            ("DeepSeek", "causal_srt_bedrock_deepseek"), ("Mistral-L3", "causal_srt_bedrock_mistral"),
            ("Llama-8B", "causal_srt_bedrock_llama31_8b")]
    states = [("visible+rationale", "visible_retained_expert", BLUE),
              ("visible, no rat.", "visible_removed_expert", "#7FB3D5"),
              ("hidden+rationale", "hidden_retained_expert", GREEN),
              ("hidden, no rat.", "hidden_removed_expert", RED)]
    data = {lab: cells(run) for lab, run in subs}
    x = np.arange(len(subs)); w = 0.2
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    for k, (lab, cell, col) in enumerate(states):
        ax.bar(x + (k - 1.5) * w, [data[s][cell] for s, _ in subs], w, label=lab, color=col)
    ax.set_xticks(x, [s for s, _ in subs], rotation=12)
    ax.set_ylabel("harmful-revision rate (expert)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Source vs. rationale: keeping either defends — for capable models")
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.0), columnspacing=1.0)
    ax.grid(True, axis="y", lw=0.4, alpha=0.4)
    fig.tight_layout(); fig.savefig(OUT / "fig_a2a_sourcerationale.pdf", bbox_inches="tight"); plt.close(fig)


# ---- Fig: live adaptive vs static vs neutral (2 subjects)
def fig_live():
    subs = [("GPT-4o-mini", "livea2a_gpt4omini_persGPT54mini"), ("Haiku 4.5", "livea2a_haiku45_persGPT54mini")]
    conds = [("no pressure", "no_pressure", GRAY), ("neutral agent", "neutral", "#9B59B6"),
             ("static message", "static_expert", BLUE), ("live adaptive", "live_expert", RED)]
    data = {lab: cells(run) for lab, run in subs}
    x = np.arange(len(subs)); w = 0.2
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    for k, (lab, cell, col) in enumerate(conds):
        ax.bar(x + (k - 1.5) * w, [data[s][cell] for s, _ in subs], w, label=lab, color=col)
    ax.set_xticks(x, [s for s, _ in subs])
    ax.set_ylabel("harmful-revision rate")
    ax.set_ylim(0, 1.1)
    ax.set_title("Live adaptive persuasion $\\gg$ static (hidden+rationale)")
    ax.legend(frameon=False, ncol=2, loc="upper center")
    ax.grid(True, axis="y", lw=0.4, alpha=0.4)
    fig.tight_layout(); fig.savefig(OUT / "fig_a2a_live.pdf", bbox_inches="tight"); plt.close(fig)


# ---- Fig: verifier + cascade (two panels)
def fig_verifier_cascade():
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.0))
    # verifier (powered, 3 subjects)
    vsubs = [("GPT-4o-mini", "causal_verifier_openai_gpt4omini"),
             ("Haiku 4.5", "causal_verifier_bedrock_haiku45"),
             ("Mistral-L3", "causal_verifier_bedrock_mistral")]
    vdata = {lab: cells(run) for lab, run in vsubs}
    series = [("attacker only", "attacker_only", RED),
              ("ungrounded verifier", "verifier_ungrounded", ORANGE),
              ("grounded verifier", "verifier_grounded", GREEN)]
    x = np.arange(len(vsubs)); w = 0.26
    for k, (lab, cell, col) in enumerate(series):
        ax.bar(x + (k - 1) * w, [vdata[s][cell] for s, _ in vsubs], w, label=lab, color=col)
    ax.set_xticks(x, [s for s, _ in vsubs], rotation=10)
    ax.set_ylim(0, 1.15); ax.set_ylabel("harmful-revision rate")
    ax.set_title("(a) Verifier: grounding, not authority")
    ax.legend(frameon=False, ncol=1, loc="upper right", fontsize=6.5)
    ax.grid(True, axis="y", lw=0.4, alpha=0.4)
    # cascade
    casc = json.loads((RES / "reports" / "cascade_main.json").read_text())["cells"]
    order = [("no\nattack", "no_attack"), ("attack", "attack"), ("attack +\nverifier", "attack_verifier")]
    wc = [casc[k]["worker_corruption_rate"] for _, k in order]
    se = [casc[k]["system_error_rate"] for _, k in order]
    x = np.arange(3); w = 0.36
    ax2.bar(x - w/2, wc, w, label="worker corrupted", color=ORANGE)
    ax2.bar(x + w/2, se, w, label="aggregator wrong\n(GPT-5.5, immune)", color=RED)
    ax2.set_xticks(x, [o[0] for o in order])
    ax2.set_ylim(0, 1.1); ax2.set_ylabel("rate")
    ax2.set_title("(b) Cascade: propagation to an immune agent")
    ax2.legend(frameon=False, loc="upper right")
    ax2.grid(True, axis="y", lw=0.4, alpha=0.4)
    fig.tight_layout(); fig.savefig(OUT / "fig_a2a_verifier_cascade.pdf", bbox_inches="tight"); plt.close(fig)


# ---- Fig: network variance decomposition (small, for Exp 1)
def fig_network():
    net = json.loads((RES / "reports" / "network_analysis.json").read_text())
    vd = net["variance_decomposition"]
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    parts = [("target\nresistance $R_i$", vd["subject_resistance"], BLUE),
             ("pair-specific\n$I_{ij}$", vd["pair_interaction"], PURPLE),
             ("attacker\nstrength $A_j$", vd["attacker_strength"], GRAY)]
    ax.bar(range(3), [p[1] for p in parts], 0.6, color=[p[2] for p in parts])
    for i, p in enumerate(parts):
        ax.text(i, p[1] + 0.01, f"{p[1]*100:.0f}%", ha="center", fontsize=8)
    ax.set_xticks(range(3), [p[0] for p in parts])
    ax.set_ylim(0, 1.0); ax.set_ylabel("share of matrix variance")
    ax.set_title("Influence is target-dominated")
    ax.grid(True, axis="y", lw=0.4, alpha=0.4)
    fig.tight_layout(); fig.savefig(OUT / "fig_network_decomp.pdf", bbox_inches="tight"); plt.close(fig)


def fig_channel():
    """Subagent-channel asymmetry: same fabrication, peer vs subagent tool-result."""
    s = json.loads((RES / "subagent_channel_powered" / "summary.json").read_text())
    subs = ["Opus-4.6", "Sonnet-4.6", "GPT-5.4"]
    chans = [("honest subagent", "honest_tool", GRAY),
             ("fabrication as peer", "peer_msg", BLUE),
             ("fabrication via subagent tool-call", "subagent_tool", RED)]
    x = np.arange(len(subs)); w = 0.26
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    for k, (lab, key, col) in enumerate(chans):
        vals = [s[m][key]["x"] / s[m][key]["n"] for m in subs]
        ax.bar(x + (k - 1) * w, vals, w, label=lab, color=col)
    ax.set_xticks(x, subs)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("harmful-revision rate")
    ax.set_title("Same attacker agent, same fabrication: trusted as a subagent, not as a peer")
    ax.legend(frameon=False, ncol=1, loc="upper right", fontsize=7)
    ax.grid(True, axis="y", lw=0.4, alpha=0.4)
    fig.tight_layout(); fig.savefig(OUT / "fig_a2a_channel.pdf", bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    fig_sourcerationale(); fig_live(); fig_verifier_cascade(); fig_network(); fig_channel()
    print("A2A figures ->", OUT)
    for f in ("fig_a2a_sourcerationale", "fig_a2a_live", "fig_a2a_verifier_cascade", "fig_network_decomp"):
        p = OUT / f"{f}.pdf"; print("  ", f, p.stat().st_size if p.exists() else "MISSING")
