"""Step 3: figures (static PNG/SVG under results/figures/<lang>/, English and German).

Encoding, consistent across all figures:
  * color = model (BF16 blue, Q8_0 orange, Q4_0 aqua — the first three slots of a validated
    categorical palette, colorblind-safe pairwise)
  * line style / hatch = lens condition (solid = fixed BF16 lens, dashed = refitted, dotted = logit
    lens, dash-dot = fixed lens + BF16 unembedding)
  * noise floor (BF16 lens B) in gray
Every figure has a legend; the underlying numbers are in results/tables/*.csv.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from jspace_quant import analysis, config  # noqa: E402

LANGS = ("en", "de")

MODEL_COLOR = {"bf16": "#2a78d6", "q8_0": "#eb6834", "q4_0": "#1baf7a"}
NOISE_COLOR = "#8a8a85"
TEXT = "#2b2b29"
MUTED = "#6b6b66"
GRID = "#e4e3de"
ROLE_STYLE = {"reference": "-", "fixed": "-", "refit": "--", "identity": ":", "fixed_ref_unembed": "-.",
              "noise": "-"}
ROLE_HATCH = {"reference": "", "fixed": "", "refit": "///", "identity": "...", "noise": "",
              "fixed_ref_unembed": "xx"}

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "font.size": 10, "axes.titlesize": 11,
    "axes.edgecolor": MUTED, "axes.labelcolor": TEXT, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": TEXT, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "lines.linewidth": 2.0, "figure.facecolor": "white", "axes.facecolor": "white",
})

# --- labels ----------------------------------------------------------------------------------

STRINGS = {
    "layer": ("Layer", "Schicht"),
    "noise": ("Noise floor (BF16 lens B)", "Rauschgrenze (BF16-Lens B)"),
    "reference": ("Reference (BF16, BF16 lens A)", "Referenz (BF16, BF16-Lens A)"),
    "fixed": ("{m}, fixed BF16 lens", "{m}, feste BF16-Lens"),
    "fixed_ref_unembed": ("{m}, fixed BF16 lens, BF16 unembedding", "{m}, feste BF16-Lens, BF16-Unembedding"),
    "refit": ("{m}, refitted lens", "{m}, neu gefittete Lens"),
    "identity": ("Logit lens {m}", "Logit Lens {m}"),
    "pass_title": ("Intermediate concepts in the lens readouts — pass@{k}, {scope}",
                   "Zwischenkonzepte in den Lens-Readouts — pass@{k}, {scope}"),
    "scope_all": ("all layers", "alle Schichten"),
    "scope_band": ("workspace band", "Workspace-Band"),
    "set_all": ("all", "alle"),
    "set_core": ("core", "Kern"),
    "hit_y": ("Share of concepts with rank ≤ 10", "Anteil Konzepte mit Rang ≤ 10"),
    "hit_title": ("Where do the intermediate concepts become readable? (shaded = workspace band)",
                  "Wo werden die Zwischenkonzepte lesbar? (graues Feld = Workspace-Band)"),
    "top1_ref": ("Top-1 = reference", "Top-1 = Referenz"),
    "overlap_ref": ("Top-10 overlap with reference", "Top-10-Overlap mit Referenz"),
    "kl_ref": ("KL(reference ‖ arm) [nats]", "KL(Referenz ‖ Arm) [nats]"),
    "agreement_title": ("Drift of the lens readouts relative to the reference ({split})",
                        "Drift der Lens-Readouts gegenüber der Referenz ({split})"),
    "heldout": ("held-out text", "Held-out-Text"),
    "evals": ("eval positions", "Eval-Positionen"),
    "fid_y": ("Top-1 = own model output", "Top-1 = eigener Modell-Output"),
    "fid_title": ("Readout fidelity to the model's own output ({split})",
                  "Readout-Fidelity zum eigenen Output ({split})"),
    "rank_log": ("Rank (log)", "Rang (log)"),
    "acc_y": ("Answer correct (greedy)", "Antwort korrekt (greedy)"),
    "ppl": ("Perplexity (held-out)", "Perplexity (held-out)"),
    "ppl_rel": ("Perplexity relative to BF16", "Perplexity relativ zu BF16"),
    "behavior_title": ("Behavior check: how much does quantization change the model itself?",
                       "Verhaltenskontrolle: Wie stark ändert Quantisierung das Modell selbst?"),
    "legacy": ("Legacy family (Q8_0, Q5_0, Q4_0)", "Legacy-Familie (Q8_0, Q5_0, Q4_0)"),
    "kquant": ("K family (Q6_K … Q2_K)", "K-Familie (Q6_K … Q2_K)"),
    "dr_top1": ("Top-1 = reference (workspace band)", "Top-1 = Referenz (Workspace-Band)"),
    "dr_kl": ("KL(reference ‖ quant), band [nats]", "KL(Referenz ‖ Quant), Band [nats]"),
    "dr_pass": ("pass@10 core, difference to reference", "pass@10 Kern, Differenz zur Referenz"),
    "bpw": ("Bits per weight", "Bits pro Gewicht"),
    "dr_title": ("Dose-response: how do the readouts change with bit width?",
                 "Dosis-Wirkung: Wie verändern sich die Readouts mit der Bitbreite?"),
    "rel_err": ("Residual stream: ‖h_quant − h_BF16‖ / ‖h_BF16‖ (no lens)",
                "Residual Stream: ‖h_quant − h_BF16‖ / ‖h_BF16‖ (ohne Lens)"),
    "fixed_top1": ("Fixed BF16 lens: top-1 = reference", "Feste BF16-Lens: Top-1 = Referenz"),
    "ladder_title": ("Bit-width ladder per layer (held-out; dashed = legacy, solid = K family)",
                     "Bitbreiten-Leiter pro Schicht (Held-out; gestrichelt = Legacy-, durchgezogen = K-Familie)"),
    "abl_title": ("Component ablation: where does the Q4_0 drift come from?",
                  "Komponenten-Ablation: Woher kommt die Q4_0-Drift?"),
    "loss_title": ("Which concepts suffer from quantization? (core sets, fixed BF16 lens)",
                   "Welche Konzepte leiden unter Quantisierung? (Kern-Sets, feste BF16-Lens)"),
    "loss_strength_y": ("Δ log10(rank) in band (> 0 = worse)", "Δ log10(Rang) im Band (> 0 = schlechter)"),
    "loss_strength_t": ("Rank change by strength in the reference", "Rang-Änderung nach Stärke in der Referenz"),
    "loss_freq_y": ("Share of concepts lost", "Anteil verlorener Konzepte"),
    "loss_freq_t": ("Loss (rank ≤ 10 → > 10) by token frequency in corpus A",
                    "Verlust (Rang ≤ 10 → > 10) nach Token-Häufigkeit in Korpus A"),
}
MODEL_LABEL_DE = {"q4_0-attn": "nur Attention Q4_0", "q4_0-ffn": "nur MLP Q4_0",
                  "q4_0-embd": "nur Embedding Q4_0"}
BIN_LABEL_DE = dict(zip(analysis.STRENGTH_BINS, ["Rang 1", "Rang 2–10", "Rang 11–100", "Rang > 100"]))
BIN_LABEL_DE |= dict(zip(analysis.FREQUENCY_BINS, ["selten", "eher selten", "eher häufig", "häufig"]))


def t(key: str, lang: str, **kw) -> str:
    return STRINGS[key][LANGS.index(lang)].format(**kw)


def model_label(m: str, lang: str) -> str:
    if lang == "de" and m in MODEL_LABEL_DE:
        return MODEL_LABEL_DE[m]
    return config.MODEL_INFO[m].label


def arm_label(name: str, lang: str) -> str:
    arm = config.ARMS_BY_NAME[name]
    if arm.role in ("reference", "noise"):
        return t(arm.role, lang)
    return t(arm.role, lang, m=model_label(arm.model, lang))


def bin_label(b: str, lang: str) -> str:
    return BIN_LABEL_DE[b] if lang == "de" else b


def set_label(s: str, lang: str) -> str:
    return {config.AGG_ALL: t("set_all", lang), config.AGG_CORE: t("set_core", lang)}.get(s, s)


# --- helpers ---------------------------------------------------------------------------------

def _color(arm_name: str) -> str:
    arm = config.ARMS_BY_NAME[arm_name]
    return NOISE_COLOR if arm.role == "noise" else MODEL_COLOR[arm.model]


def _save(fig, name: str, lang: str) -> None:
    out = config.FIGURES_DIR / lang
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(out / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)


def _band(ax, band) -> None:
    ax.axvspan(band[0] - 0.5, band[1] + 0.5, color="#f1f0eb", zorder=0, lw=0)


MAIN_ARMS = ["bf16|bf16-A", "bf16|bf16-B", "q8_0|bf16-A", "q8_0|q8_0-A", "q4_0|bf16-A", "q4_0|q4_0-A",
             "bf16|identity"]


# --- main study ------------------------------------------------------------------------------

def pass_at_k(pak: pd.DataFrame, k: int, lang: str) -> None:
    """Grouped bars: pass@k per eval set, one bar per arm, 95 % bootstrap CI."""
    for scope, key, suffix in ((config.SCOPE_ALL, "scope_all", "all"), (config.SCOPE_BAND, "scope_band", "band")):
        sub = pak[(pak.k == k) & (pak.scope == scope)]
        sets = [s for s in config.EVAL_SETS + [config.AGG_ALL, config.AGG_CORE] if s in set(sub["set"])]
        fig, ax = plt.subplots(figsize=(11, 4.2))
        width = 0.8 / len(MAIN_ARMS)
        for i, name in enumerate(MAIN_ARMS):
            arm = config.ARMS_BY_NAME[name]
            d = sub[sub.arm == name].set_index("set").loc[sets]
            x = np.arange(len(sets)) + (i - (len(MAIN_ARMS) - 1) / 2) * width
            ax.bar(x, d["pass_at_k"], width * 0.92, color=_color(name), hatch=ROLE_HATCH[arm.role],
                   edgecolor="white", linewidth=0, label=arm_label(name, lang))
            ax.errorbar(x, d["pass_at_k"], yerr=[d["pass_at_k"] - d["ci_lo"], d["ci_hi"] - d["pass_at_k"]],
                        fmt="none", ecolor=TEXT, elinewidth=0.8, capsize=0)
        ax.set_xticks(np.arange(len(sets)), [set_label(s, lang) for s in sets])
        ax.set_ylim(0, 1)
        ax.set_ylabel(f"pass@{k}")
        ax.set_title(t("pass_title", lang, k=k, scope=t(key, lang)))
        ax.legend(ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1))
        _save(fig, f"pass_at_{k}_{suffix}", lang)


def hit_rate_layers(hit: pd.DataFrame, band, lang: str) -> None:
    """Hit rate (concept rank ≤ 10) per layer, pooled over all eval concepts."""
    sub = hit[hit.set == config.AGG_ALL]
    fig, ax = plt.subplots(figsize=(9, 4))
    _band(ax, band)
    for name in MAIN_ARMS + ["q4_0|identity"]:
        arm = config.ARMS_BY_NAME[name]
        d = sub[sub.arm == name].sort_values("layer")
        ax.plot(d["layer"], d["hit"], ROLE_STYLE[arm.role], color=_color(name), label=arm_label(name, lang))
    ax.set_xlabel(t("layer", lang))
    ax.set_ylabel(t("hit_y", lang))
    ax.set_title(t("hit_title", lang))
    ax.legend(fontsize=8, ncol=2)
    _save(fig, "hit_rate_per_layer", lang)


def agreement_layers(curves: pd.DataFrame, band, split: str, lang: str) -> None:
    """Agreement with the reference per layer, with 95 % cluster-bootstrap band."""
    arms = ["bf16|bf16-B", "q8_0|bf16-A", "q8_0|bf16-A|U=bf16", "q8_0|q8_0-A",
            "q4_0|bf16-A", "q4_0|bf16-A|U=bf16", "q4_0|q4_0-A"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, metric in zip(axes, ("top1_ref", "overlap_ref", "kl_ref")):
        _band(ax, band)
        for name in arms:
            arm = config.ARMS_BY_NAME[name]
            d = curves[(curves.split == split) & (curves.arm == name) & (curves.metric == metric)].sort_values("layer")
            if metric == "kl_ref":
                # last block: all arms of one model read identically (KL = 0) -> not log-plottable
                d = d[d["mean"] > 0]
            ax.plot(d["layer"], d["mean"], ROLE_STYLE[arm.role], color=_color(name), label=arm_label(name, lang))
            ax.fill_between(d["layer"], d["ci_lo"].clip(lower=1e-6) if metric == "kl_ref" else d["ci_lo"],
                            d["ci_hi"], color=_color(name), alpha=0.12, lw=0)
        ax.set_title(t(metric, lang))
        ax.set_xlabel(t("layer", lang))
        if metric == "kl_ref":
            ax.set_yscale("log")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=4, fontsize=8)
    fig.suptitle(t("agreement_title", lang, split=t(split, lang)), y=1.02)
    _save(fig, f"agreement_per_layer_{split}", lang)


def fidelity_layers(curves: pd.DataFrame, band, split: str, lang: str) -> None:
    """Does the readout at layer l already predict the model's own output?"""
    arms = ["bf16|bf16-A", "q8_0|bf16-A", "q8_0|q8_0-A", "q4_0|bf16-A", "q4_0|q4_0-A",
            "bf16|identity", "q4_0|identity"]
    fig, ax = plt.subplots(figsize=(9, 4))
    _band(ax, band)
    for name in arms:
        arm = config.ARMS_BY_NAME[name]
        d = curves[(curves.split == split) & (curves.arm == name) & (curves.metric == "top1_own")].sort_values("layer")
        ax.plot(d["layer"], d["mean"], ROLE_STYLE[arm.role], color=_color(name), label=arm_label(name, lang))
    ax.set_xlabel(t("layer", lang))
    ax.set_ylabel(t("fid_y", lang))
    ax.set_title(t("fid_title", lang, split=t(split, lang)))
    ax.legend(fontsize=8, ncol=2)
    _save(fig, f"fidelity_per_layer_{split}", lang)


EXAMPLES = [("multilingual", "fr-time-after-night", "night"), ("multihop", "tajmahal-continent", "India"),
            ("multilingual", "frn-month-after-jun", "June"), ("typo", "typo-local", "local")]


def concept_trajectories(cr: pd.DataFrame, band, lang: str) -> None:
    """Rank of single intermediate concepts across layers (log scale, smaller = better)."""
    arms = ["bf16|bf16-A", "bf16|identity", "q8_0|bf16-A", "q4_0|bf16-A", "q4_0|q4_0-A"]
    if "q2_k|bf16-A" in set(cr.arm):
        arms.append("q2_k|bf16-A")
    fig, axes = plt.subplots(1, len(EXAMPLES), figsize=(4 * len(EXAMPLES), 3.8), sharey=True)
    for ax, (eval_set, item, word) in zip(np.atleast_1d(axes), EXAMPLES):
        d0 = cr[(cr.set == eval_set) & (cr.name == item)]
        _band(ax, band)
        for name in arms:
            arm = config.ARMS_BY_NAME[name]
            d = d0[(d0.arm == name) & (d0.word == word)].sort_values("layer")
            color = "#0d366b" if arm.model == "q2_k" else _color(name)
            # reference drawn wide underneath so it stays visible where curves coincide (Q8_0)
            lw = 4.0 if arm.role == "reference" else 2.0
            label = arm_label(name, lang) if arm.model in MODEL_COLOR else model_label(arm.model, lang)
            ax.plot(d["layer"], d["rank"], ROLE_STYLE[arm.role], color=color, label=label, lw=lw)
        ax.set_yscale("log")
        ax.invert_yaxis()
        ax.set_title(f"{eval_set}: “{word}”", fontsize=10)
        ax.set_xlabel(t("layer", lang))
    np.atleast_1d(axes)[0].set_ylabel(t("rank_log", lang))
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3, fontsize=8)
    _save(fig, "concept_trajectories", lang)


def behavior(beh: pd.DataFrame, lang: str) -> None:
    """Behavior check: answer accuracy per set and held-out perplexity per model."""
    acc = beh[beh.metric.str.startswith("accuracy_")]
    sets = sorted(acc.metric.str.removeprefix("accuracy_").unique())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [3, 1.2]})
    width = 0.8 / len(config.MAIN_MODELS)
    for i, m in enumerate(config.MAIN_MODELS):
        d = acc[acc.model == m].set_index("metric").loc[[f"accuracy_{s}" for s in sets]]
        x = np.arange(len(sets)) + (i - (len(config.MAIN_MODELS) - 1) / 2) * width
        ax1.bar(x, d["value"], width * 0.92, color=MODEL_COLOR[m], label=m.upper())
        ax1.errorbar(x, d["value"], yerr=[d["value"] - d["ci_lo"], d["ci_hi"] - d["value"]],
                     fmt="none", ecolor=TEXT, elinewidth=0.8)
    ax1.set_xticks(np.arange(len(sets)), sets)
    ax1.set_ylim(0, 1)
    ax1.set_ylabel(t("acc_y", lang))
    ax1.legend()
    ppl = beh[beh.metric == "heldout_ppl"].set_index("model").loc[config.MAIN_MODELS]
    ax2.bar(range(len(ppl)), ppl["value"], color=[MODEL_COLOR[m] for m in ppl.index])
    ax2.errorbar(range(len(ppl)), ppl["value"], yerr=[ppl["value"] - ppl["ci_lo"], ppl["ci_hi"] - ppl["value"]],
                 fmt="none", ecolor=TEXT, elinewidth=0.8)
    ax2.set_xticks(range(len(ppl)), [m.upper() for m in ppl.index])
    ax2.set_ylabel(t("ppl", lang))
    fig.suptitle(t("behavior_title", lang), y=1.02)
    _save(fig, "behavior", lang)


# --- extension: bit-width ladder, ablation, representations, concept analysis ---------------

FAMILY_COLOR = {"legacy": "#eb6834", "kquant": "#6250d6"}  # orange / violet, validated
# ordinal blue ramp: light = many bits, dark = few; same bit width = same step
LADDER_COLOR = {"q8_0": "#86b6ef", "q6_k": "#5598e7", "q5_0": "#2a78d6", "q5_k": "#2a78d6",
                "q4_0": "#1c5cab", "q4_k": "#1c5cab", "q3_k": "#104281", "q2_k": "#0d366b"}
LADDER = ["q8_0", "q6_k", "q5_0", "q5_k", "q4_0", "q4_k", "q3_k", "q2_k"]
ABLATION = ["q4_0-embd", "q4_0-attn", "q4_0-ffn", "q4_0"]
ABLATION_STYLE = {  # color + line style + marker (color alone is not enough per palette validation)
    "q4_0-embd": ("#eda100", ":", "s"), "q4_0-attn": ("#6250d6", "-.", "^"),
    "q4_0-ffn": ("#e87ba4", "--", "D"), "q4_0": ("#1baf7a", "-", "o"),
}


def _ladder_style(m: str, lang: str) -> dict:
    fam = config.MODEL_INFO[m].family
    return {"color": LADDER_COLOR[m], "ls": "--" if fam == "legacy" else "-", "label": model_label(m, lang)}


def dose_response(models: pd.DataFrame, band_agr: pd.DataFrame, pak: pd.DataFrame, beh: pd.DataFrame,
                  lang: str) -> None:
    """Four measures against bits per weight, one line per quant family; noise floor in gray."""
    bpw = models.set_index("model")["bits_per_weight"]
    agr = band_agr[band_agr.split == "heldout"].set_index("arm")
    p10 = pak[(pak.k == 10) & (pak.set == config.AGG_CORE) & (pak.scope == config.SCOPE_ALL)].set_index("arm")
    ppl = beh[beh.metric == "heldout_ppl"].set_index("model")
    panels = [
        ("dr_top1", lambda m: agr.loc[f"{m}|bf16-A", "top1_ref"], agr.loc[config.NOISE_ARM, "top1_ref"], None),
        ("dr_kl", lambda m: agr.loc[f"{m}|bf16-A", "kl_ref"], agr.loc[config.NOISE_ARM, "kl_ref"], "log"),
        ("dr_pass", lambda m: p10.loc[f"{m}|bf16-A", "delta_ref"], p10.loc[config.NOISE_ARM, "delta_ref"], None),
        ("ppl_rel", lambda m: ppl.loc[m, "vs_ref"], None, "log"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(17, 3.9))
    for ax, (key, value, noise, scale) in zip(axes, panels):
        for fam in ("legacy", "kquant"):
            ms = sorted([m for m in LADDER if config.MODEL_INFO[m].family == fam and m in bpw.index],
                        key=lambda m: bpw[m])
            if ms:
                ax.plot([bpw[m] for m in ms], [value(m) for m in ms], "-o", color=FAMILY_COLOR[fam],
                        label=t(fam, lang), ms=7)
                for m in ms:
                    # legacy labels below, K labels above: Q5_0/Q5_K and Q4_0/Q4_K share an x value
                    off = (0, -13) if fam == "legacy" else (0, 7)
                    ax.annotate(model_label(m, lang), (bpw[m], value(m)), textcoords="offset points",
                                xytext=off, fontsize=7, color=MUTED, ha="center")
        if noise is not None:
            ax.axhline(noise, color=NOISE_COLOR, lw=1.5, label=t("noise", lang))
        if scale:
            ax.set_yscale(scale)
        ax.set_title(t(key, lang), fontsize=10)
        ax.set_xlabel(t("bpw", lang))
        ax.invert_xaxis()  # left many bits, right strong quantization
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3, fontsize=8)
    fig.suptitle(t("dr_title", lang), y=1.03)
    _save(fig, "dose_response", lang)


def ladder_layers(curves: pd.DataFrame, repsim: pd.DataFrame, band, lang: str) -> None:
    """Per layer: relative error of the residual stream (no lens) and top-1 agreement of the fixed
    lens with the reference, for every step of the ladder."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.2))
    rs = repsim[repsim.split == "heldout"]
    cv = curves[(curves.split == "heldout") & (curves.metric == "top1_ref")]
    for ax in (ax1, ax2):
        _band(ax, band)
    for m in [m for m in LADDER if m in set(rs.model)]:
        st = _ladder_style(m, lang)
        d = rs[rs.model == m].sort_values("layer")
        ax1.plot(d["layer"], d["rel_error"], **st)
        d = cv[cv.arm == f"{m}|bf16-A"].sort_values("layer")
        ax2.plot(d["layer"], d["mean"], **st)
    noise = cv[cv.arm == config.NOISE_ARM].sort_values("layer")
    ax2.plot(noise["layer"], noise["mean"], color=NOISE_COLOR, lw=1.5, label=t("noise", lang))
    ax1.set_yscale("log")
    ax1.set_title(t("rel_err", lang), fontsize=10)
    ax2.set_title(t("fixed_top1", lang), fontsize=10)
    for ax in (ax1, ax2):
        ax.set_xlabel(t("layer", lang))
    handles, labels = ax2.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=5, fontsize=8)
    fig.suptitle(t("ladder_title", lang), y=1.02)
    _save(fig, "ladder_per_layer", lang)


def ablation(curves: pd.DataFrame, repsim: pd.DataFrame, beh: pd.DataFrame, band, lang: str) -> None:
    """Which component causes the Q4_0 drift? Embedding / attention / MLP only in Q4_0, vs. full Q4_0."""
    present = [m for m in ABLATION if m in set(repsim.model)]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2), gridspec_kw={"width_ratios": [1.2, 1.2, 1]})
    rs = repsim[repsim.split == "heldout"]
    cv = curves[(curves.split == "heldout") & (curves.metric == "top1_ref")]
    for ax in axes[:2]:
        _band(ax, band)
    for m in present:
        c, ls, mk = ABLATION_STYLE[m]
        d = rs[rs.model == m].sort_values("layer")
        axes[0].plot(d["layer"], d["rel_error"], ls, color=c, marker=mk, markevery=5, ms=6, label=model_label(m, lang))
        d = cv[cv.arm == f"{m}|bf16-A"].sort_values("layer")
        axes[1].plot(d["layer"], d["mean"], ls, color=c, marker=mk, markevery=5, ms=6, label=model_label(m, lang))
    noise = cv[cv.arm == config.NOISE_ARM].sort_values("layer")
    axes[1].plot(noise["layer"], noise["mean"], color=NOISE_COLOR, lw=1.5, label=t("noise", lang))
    axes[0].set_yscale("log")
    axes[0].set_title(t("rel_err", lang), fontsize=10)
    axes[1].set_title(t("fixed_top1", lang), fontsize=10)
    for ax in axes[:2]:
        ax.set_xlabel(t("layer", lang))
    axes[1].legend(fontsize=8)
    ppl = beh[beh.metric == "heldout_ppl"].set_index("model")
    x = np.arange(len(present))
    vals = [ppl.loc[m, "vs_ref"] for m in present]
    axes[2].bar(x, vals, color=[ABLATION_STYLE[m][0] for m in present])
    axes[2].errorbar(x, vals, yerr=[[ppl.loc[m, "vs_ref"] - ppl.loc[m, "vs_ref_ci_lo"] for m in present],
                                    [ppl.loc[m, "vs_ref_ci_hi"] - ppl.loc[m, "vs_ref"] for m in present]],
                     fmt="none", ecolor=TEXT, elinewidth=0.8)
    axes[2].axhline(1, color=MUTED, lw=1)
    axes[2].set_xticks(x, [model_label(m, lang).replace(" Q4_0", "") for m in present], rotation=15, fontsize=8)
    axes[2].set_title(t("ppl_rel", lang), fontsize=10)
    fig.suptitle(t("abl_title", lang), y=1.02)
    _save(fig, "ablation", lang)


def concept_loss(effects: pd.DataFrame, core: list[str], lang: str) -> None:
    """Concepts the reference reads in the band (rank ≤ 10) and the fixed lens loses on the quant
    (rank > 10) — broken down by strength in the reference and by token frequency."""
    eff = effects[effects.set.isin(core) & (effects.role.isin(["fixed", "noise"]))]
    models = [m for m in LADDER if m in set(eff.model)]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.2))
    width = 0.8 / (len(models) + 1)
    series = [(m, eff[(eff.model == m) & (eff.role == "fixed")]) for m in models]
    series.append(("noise", eff[eff.role == "noise"]))

    def style(m: str) -> dict:
        legacy = m != "noise" and config.MODEL_INFO[m].family == "legacy"
        return {"color": NOISE_COLOR if m == "noise" else LADDER_COLOR[m], "hatch": "//" if legacy else "",
                "edgecolor": "white", "linewidth": 0,
                "label": t("noise", lang) if m == "noise" else model_label(m, lang)}

    for i, (m, d) in enumerate(series):
        v = d.groupby("ref_strength", observed=False)["delta_log10_rank"].mean().reindex(analysis.STRENGTH_BINS)
        ax1.bar(np.arange(4) + (i - len(series) / 2 + 0.5) * width, v, width * 0.92, **style(m))
    ax1.axhline(0, color=MUTED, lw=1)
    ax1.set_xticks(np.arange(4), [bin_label(b, lang) for b in analysis.STRENGTH_BINS])
    ax1.set_ylabel(t("loss_strength_y", lang))
    ax1.set_title(t("loss_strength_t", lang), fontsize=10)
    ax1.legend(fontsize=7, ncol=3)
    if "frequency_q" in eff:
        read = eff[eff.ref_rank <= 10]
        for i, (m, _) in enumerate(series):
            d = read[read.role == "noise"] if m == "noise" else read[(read.model == m) & (read.role == "fixed")]
            v = d.groupby("frequency_q", observed=False)["lost"].mean().reindex(analysis.FREQUENCY_BINS)
            ax2.bar(np.arange(4) + (i - len(series) / 2 + 0.5) * width, v, width * 0.92, **style(m))
        ax2.set_xticks(np.arange(4), [bin_label(b, lang) for b in analysis.FREQUENCY_BINS])
        ax2.set_ylabel(t("loss_freq_y", lang))
        ax2.set_title(t("loss_freq_t", lang), fontsize=10)
    fig.suptitle(t("loss_title", lang), y=1.02)
    _save(fig, "concept_loss", lang)


def all_figures(langs: tuple[str, ...] = LANGS) -> None:
    T = config.TABLES_DIR
    summary = json.loads((T / "summary.json").read_text())
    band = tuple(summary["workspace_band"])
    pak = pd.read_csv(T / "pass_at_k.csv")
    hit = pd.read_csv(T / "hit_rate_per_layer.csv")
    curves = pd.read_csv(T / "agreement_per_layer.csv")
    cr = pd.read_csv(T / "concept_ranks.csv.gz")
    beh = pd.read_csv(T / "behavior.csv")
    models = pd.read_csv(T / "models.csv")
    repsim = pd.read_csv(T / "representation_similarity.csv")
    band_agr = pd.read_csv(T / "agreement_band.csv")
    effects = pd.read_csv(T / "concept_effects.csv")
    for lang in langs:
        for k in config.PASS_AT_K:
            pass_at_k(pak, k, lang)
        hit_rate_layers(hit, band, lang)
        for split in analysis.SPLITS:
            agreement_layers(curves, band, split, lang)
            fidelity_layers(curves, band, split, lang)
        concept_trajectories(cr, band, lang)
        behavior(beh[beh.model.isin(config.MAIN_MODELS)], lang)
        # extension (only if those models were captured)
        if len(set(models.model) & set(LADDER)) > 3:
            dose_response(models, band_agr, pak, beh, lang)
            ladder_layers(curves, repsim, band, lang)
        if set(ABLATION) - {"q4_0"} & set(models.model):
            ablation(curves, repsim, beh, band, lang)
        concept_loss(effects, summary["core_sets"], lang)
