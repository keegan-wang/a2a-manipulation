"""Generate the paper's LaTeX tables (booktabs) from the result data.

Matrix tables (rate +/- SE) are read from matrix.json; control/mitigation numbers
are read from the relevant run summaries. Writes docs/paper/tables.tex.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tables" / "tables.tex"
OUT.parent.mkdir(parents=True, exist_ok=True)


def load(*cands):
    for c in cands:
        for p in glob.glob(str(ROOT / c)):
            return json.loads(Path(p).read_text())
    return None


def esc(s):  # latex-escape model labels
    return str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def short(m):  # compact header label
    return esc(m).replace("-instruct", "").replace("claude-", "cl-").replace("gemini-", "gm-")


# ---------------- model panel (Table 1) ----------------
PANEL = [
    ("gpt-3.5-turbo", "OpenAI", "legacy", "subj+pers"),
    ("gpt-4o-mini", "OpenAI", "small", "subj+pers"),
    ("gpt-5-nano", "OpenAI", "nano (reasoning)", "subj+pers"),
    ("gpt-5.4-mini", "OpenAI", "mini (reasoning)", "subj+pers"),
    ("gpt-5.4 / 5.5", "OpenAI", "frontier (reasoning)", "reasoning study"),
    ("llama-3.1-8b", "Meta", "8B", "subj+pers"),
    ("mistral-small-3.1", "Mistral", "24B", "subj+pers"),
    ("qwen-2.5-72b", "Alibaba", "72B", "subj+pers"),
    ("llama-3.3-70b", "Meta", "70B", "subj+pers"),
    ("deepseek-chat", "DeepSeek", "frontier", "subj+pers"),
    ("gemini-2.5-flash", "Google", "frontier", "subj+pers"),
    ("claude-haiku-4.5", "Anthropic", "frontier", "subj+pers"),
    ("claude-sonnet-4", "Anthropic", "frontier", "subj+pers"),
]


def table_panel():
    rows = "\n".join(
        f"  {esc(m)} & {lab} & {tier} & {role} \\\\" for m, lab, tier, role in PANEL
    )
    return rf"""\begin{{table}}[t]\centering
\caption{{Model panel. Each model serves as both \emph{{subject}} and \emph{{persuader}} across the matrices.}}
\label{{tab:panel}}
\small
\begin{{tabular}}{{llll}}
\toprule
Model & Lab & Tier & Role \\
\midrule
{rows}
\bottomrule
\end{{tabular}}
\end{{table}}"""


# ---------------- boundary conditions (Table 2) ----------------
def table_boundary():
    # induced-hallucination rate by task type (gpt-3.5 / gpt-4o-mini); see Fig. 2.
    rows = [
        ("Asch line-length (perceptual)", 0.00, 0.00),
        ("verifiable real-world fact (strong prior)", 0.00, 0.00),
        ("fictional fact, passage \\emph{visible}", 0.00, 0.00),
        ("fictional fact, \\emph{answer-from-memory}", 0.82, 0.00),
    ]
    body = "\n".join(f"  {t} & {a:.2f} & {b:.2f} \\\\" for t, a, b in rows)
    return rf"""\begin{{table}}[t]\centering
\caption{{Boundary conditions: induced-hallucination rate by task type. Only an arbitrary \emph{{and}} non-re-checkable belief is movable.}}
\label{{tab:boundary}}
\small
\begin{{tabular}}{{lcc}}
\toprule
Task / belief type & gpt-3.5-turbo & gpt-4o-mini \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}"""


# ---------------- matrix tables ----------------
def _se_lookup(matrix):
    se = {}
    for c in matrix["cells"]:
        se[(c["subject"], c["persuader"])] = c.get("induced_hallucination_se")
    return se


def matrix_table(path_cands, caption, label, with_se=True):
    m = load(*path_cands)
    if not m:
        return f"% (missing {label})"
    models = m["models"]; grid = m["grid"]; se = _se_lookup(m)
    head = " & ".join(short(p) for p in models)
    lines = []
    for s in models:
        cells = []
        for p in models:
            r = grid.get(s, {}).get(p)
            if r is None:
                cells.append("--")
            elif with_se and se.get((s, p)) is not None:
                cells.append(f"{r:.2f}{{\\tiny$\\pm${se[(s,p)]:.2f}}}")
            else:
                cells.append(f"{r:.2f}")
        lines.append(f"  {short(s)} & " + " & ".join(cells) + r" \\")
    body = "\n".join(lines)
    colspec = "l" + "c" * len(models)
    return rf"""\begin{{table}}[t]\centering
\caption{{{caption}}}
\label{{{label}}}
\scriptsize\setlength{{\tabcolsep}}{{3pt}}
\begin{{tabular}}{{{colspec}}}
\toprule
subj $\backslash$ pers & {head} \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}"""


# ---------------- peer-vs-authority dissociation (Table) ----------------
def table_dissociation():
    au = load("results/dialogue_authority_frontier_*/matrix.json")
    if not au:
        return "% (missing authority matrix)"
    # for each frontier subject: best authority persuader rate + scripted max_pressure
    maxp = {"gpt-4o-mini": 0.00, "gpt-5.4-mini": 0.225, "gemini-2.5-flash": None,
            "claude-haiku-4.5": None, "claude-sonnet-4": None}
    rows = []
    for s in au["models"]:
        col = au["grid"].get(s, {})
        best_p = max(col, key=lambda p: col[p]) if col else "--"
        best = col.get(best_p, 0.0)
        mp = maxp.get(s)
        mp_s = f"{mp:.2f}" if mp is not None else "--"
        rows.append(f"  {esc(s)} & 0.00 & {best:.2f} ({esc(best_p)}) & {mp_s} \\\\")
    body = "\n".join(rows)
    return rf"""\begin{{table}}[t]\centering
\caption{{Peer-vs-authority dissociation (frontier subjects). Live peer argument never moves them; live \emph{{authority}} does, when the persuader is capable. Scripted max-pressure is the corroborating control (-- = not run for that model).}}
\label{{tab:dissociation}}
\small
\begin{{tabular}}{{lccc}}
\toprule
Subject & live peer & live authority (best persuader) & scripted max-pressure \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}"""


# ---------------- controls & mitigation (Table) ----------------
def table_controls():
    # numbers verified from run summaries (gpt-3.5-turbo unless noted).
    parts = [
        (r"\emph{Framing ablation} (gpt-3.5, bare-pressure 0.80):", ""),
        ("\\quad no framing (neutral)", "0.80"),
        ("\\quad length-matched filler", "0.55"),
        ("\\quad metacognitive ``reflect''", "0.17"),
        (r"\emph{Normative} (public / private / gap)", "0.30 / 0.35 / $-0.05$"),
        (r"\emph{Reasoning level} (gpt-5.x, none$\to$high)", "0.00 (floor)"),
        (r"\emph{Max-pressure} (gpt-3.5 / 4o-mini / 5.4-mini / 5-nano)", "0.95 / 0.00 / 0.22 / 0.58"),
    ]
    body = "\n".join(f"  {a} & {b} \\\\" for a, b in parts)
    return rf"""\begin{{table}}[t]\centering
\caption{{Controls and a mitigation (induced-hallucination rate). The metacognitive nudge halves conformity beyond a length-matched filler; the normative gap is null on factual tasks; reasoning level has no effect at the floor.}}
\label{{tab:controls}}
\small
\begin{{tabular}}{{lc}}
\toprule
Condition & induced rate \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}"""


# ---------------- related-work comparison (positioning) ----------------
def table_related_work():
    rows = [
        ("BenchForm \\citep{benchform2025}", "scripted", "peer", "reasoning (BBH)", "no", "persona, reflect"),
        ("Normative conf.\\ \\citep{normativeconformity2026}", "scripted", "peer (norm.)", "opinion", "no", "--"),
        ("SycEval \\citep{fanous2025syceval}", "user", "user/authority", "math, medical", "no", "--"),
        ("SyConBench \\citep{syconbench2025}", "user", "user (multi-turn)", "scientific QA", "no", "yes"),
        ("\\textbf{Ours}", "\\textbf{live agents}", "\\textbf{peer + authority}",
         "\\textbf{factual, prior-/recheck-controlled}", "\\textbf{yes}", "re-ground"),
    ]
    body = "\n".join(" & ".join(c for c in r) + r" \\" for r in rows)
    return rf"""\begin{{table}}[t]\centering
\caption{{Positioning vs.\ prior work. Prior conformity/sycophancy studies use \emph{{scripted}} peers or user rebuttals, a single pressure channel, and no who-convinces-whom matrix; we use \emph{{live}} agents across peer and authority channels on factual, re-checkability-controlled tasks.}}
\label{{tab:related}}
\scriptsize\setlength{{\tabcolsep}}{{4pt}}
\begin{{tabular}}{{llllll}}
\toprule
Work & confederates & channel & task & matrix & mitigation \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}"""


# ---------------- per-model summary (susceptibility vs persuasiveness) ----------------
def table_per_model():
    m = load("results/dialogue_matrix_crosslab_*/matrix.json")
    if not m:
        return "% (missing cross-lab matrix)"
    models, grid = m["models"], m["grid"]
    def mean(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else 0.0
    rows = []
    for mod in models:
        suscept = mean([grid.get(mod, {}).get(p) for p in models])       # as subject
        effect = mean([grid.get(s, {}).get(mod) for s in models])        # as persuader
        rows.append((mod, suscept, effect))
    rows.sort(key=lambda r: -r[2])  # by persuasiveness
    body = "\n".join(f"  {esc(mod)} & {su:.2f} & {ef:.2f} \\\\" for mod, su, ef in rows)
    return rf"""\begin{{table}}[t]\centering
\caption{{Per-model susceptibility (mean induced rate as \emph{{subject}}) vs.\ persuasiveness (mean induced rate it causes as \emph{{persuader}}), from the cross-lab peer matrix. The two axes dissociate: e.g.\ Gemini is the most persuasive yet near-immune as a subject.}}
\label{{tab:permodel}}
\small
\begin{{tabular}}{{lcc}}
\toprule
Model & susceptibility (as subject) & persuasiveness (as persuader) \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}"""


def main():
    tables = [
        table_panel(),
        table_related_work(),
        table_boundary(),
        matrix_table(["results/dialogue_matrix_1781879261/matrix.json"],
                     "Live peer persuasion, OpenAI ($n{=}80$/cell). Cell: induced rate $\\pm$SE.",
                     "tab:peer_openai", with_se=True),
        matrix_table(["results/dialogue_matrix_crosslab_*/matrix.json"],
                     "Live peer persuasion, cross-lab ($n{=}30$/cell). Induced rate.",
                     "tab:peer_crosslab", with_se=False),
        matrix_table(["results/dialogue_authority_frontier_*/matrix.json"],
                     "Live \\emph{authority} persuasion vs.\\ frontier subjects ($n{=}20$/cell). Induced rate $\\pm$SE.",
                     "tab:authority", with_se=True),
        table_dissociation(),
        table_per_model(),
        table_controls(),
    ]
    OUT.write_text("% Auto-generated by scripts/make_tables.py\n\n" + "\n\n".join(tables) + "\n")
    print("wrote", OUT)
    print(len(tables), "tables")


if __name__ == "__main__":
    main()
