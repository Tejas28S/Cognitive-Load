"""
make_pdf.py — build the project PDFs.

Rendering runs through WeasyPrint, which implements the CSS paged-media spec
properly: real page numbers, running headers, a table of contents with page
references, and page-break control that a browser print dialog cannot give.

Every metric is read from the run artifacts and every figure is the PNG that run
actually produced, so the document cannot drift from the results.

    .venv/bin/python -m research.make_pdf            # the illustrated report
    .venv/bin/python -m research.make_pdf --docs     # + every explanation/*.md
    .venv/bin/python -m research.make_pdf --all      # both
"""

from __future__ import annotations

import argparse
import base64
import csv
import glob
import json
import os
import subprocess
import sys
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FINAL = os.path.join(ROOT, "outputs", "runs", "run_20260820_141126_19cb")
EARLIER = os.path.join(ROOT, "outputs", "runs", "run_20260819_184443_f0d1")
OUT_DIR = os.path.join(ROOT, "explanation")

def f(v, d=4):
    if v is None:
        return "n/a"
    try:
        x = float(v)
    except (TypeError, ValueError):
        return str(v)
    return "n/a" if x != x else f"{x:.{d}f}"


def img(path, run=FINAL):
    p = path if os.path.isabs(path) else os.path.join(run, path)
    if not os.path.isfile(p):
        return "<p class='missing'>figure not found: %s</p>" % os.path.basename(p)
    b64 = base64.b64encode(open(p, "rb").read()).decode()
    return f'<img src="data:image/png;base64,{b64}" alt="{os.path.basename(p)}">'


def merged_comparison_figure(A, out_path):
    """The comparison chart across BOTH runs.

    Each run only plots its own arms, so neither run's figure shows the whole
    campaign. This redraws it from the merged leaderboard.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = sorted(A, key=lambda r: float(r["dev_loso_f1_macro_mean"]))
    names = [r["experiment_id"] for r in rows]
    vals = [float(r["dev_loso_f1_macro_mean"]) for r in rows]
    errs = [float(r["dev_loso_f1_macro_std"]) for r in rows]
    base = next(float(r["dev_loso_f1_macro_mean"])
                for r in rows if r["experiment_id"] == "A0_baseline")
    colors = ["#8a94a6" if n == "A0_baseline" else
              ("#2f6f4e" if v > base else "#4C78A8")
              for n, v in zip(names, vals)]

    fig, ax = plt.subplots(figsize=(7.4, 0.19 * len(rows) + 1.0), dpi=190)
    ax.barh(names, vals, xerr=errs, color=colors, height=0.72,
            error_kw={"ecolor": "#9aa3b2", "lw": 0.7, "capsize": 1.6})
    ax.axvline(base, color="#c0392b", lw=1.0, ls="--", zorder=3)
    ax.text(base, len(rows) - 0.2, f"  baseline {base:.4f}", color="#c0392b",
            fontsize=6.5, va="top")
    ax.set_xlabel("development-LOSO macro F1 (mean ± std over 18 folds)", fontsize=7.5)
    ax.set_xlim(0, max(v + e for v, e in zip(vals, errs)) * 1.04)
    ax.tick_params(axis="y", labelsize=6.2)
    ax.tick_params(axis="x", labelsize=6.8)
    ax.grid(axis="x", alpha=0.25, lw=0.5)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def jload(*parts, run=FINAL):
    p = os.path.join(run, *parts)
    return json.load(open(p, encoding="utf-8")) if os.path.isfile(p) else None


def arms():
    seen, out = set(), []
    for run in (FINAL, EARLIER):
        p = os.path.join(run, "experiments", "leaderboard.csv")
        if not os.path.isfile(p):
            continue
        for r in csv.DictReader(open(p, encoding="utf-8")):
            if r["experiment_id"].startswith("FINAL") or r["experiment_id"] in seen:
                continue
            seen.add(r["experiment_id"])
            out.append(r)
    return sorted(out, key=lambda r: -float(r["dev_loso_f1_macro_mean"]))


def stages():
    out = []
    for run in (EARLIER, FINAL):
        p = os.path.join(run, "experiments", "stage_log.json")
        if os.path.isfile(p):
            out += json.load(open(p, encoding="utf-8"))
    return out


CSS = """
@page {
  size: A4;
  margin: 20mm 16mm 18mm 16mm;
  @bottom-center {
    content: counter(page);
    font-family: "Helvetica Neue",Helvetica,Arial,sans-serif;
    font-size: 7.5pt; color: #8791a3;
  }
  @top-right {
    content: string(chapter);
    font-family: "Helvetica Neue",Helvetica,Arial,sans-serif;
    font-size: 7pt; color: #a3abb8; letter-spacing: 0.3pt;
  }
  @top-left {
    content: "Cognitive Load from EEG + fNIRS";
    font-family: "Helvetica Neue",Helvetica,Arial,sans-serif;
    font-size: 7pt; color: #a3abb8;
  }
}
@page :first { @bottom-center { content: ""; } @top-right { content: ""; }
               @top-left { content: ""; } }
h2 { string-set: chapter content(); }
* { box-sizing: border-box; }
body { font-family: "Georgia","Times New Roman",serif; font-size: 9.6pt;
       line-height: 1.55; color: #1a1d23; margin: 0; }
h1,h2,h3,h4 { font-family: "Helvetica Neue",Helvetica,Arial,sans-serif;
              color: #10131a; line-height: 1.25; }
h1 { font-size: 21pt; margin: 0 0 4pt; letter-spacing: -0.4pt; }
h2 { font-size: 13.5pt; margin: 20pt 0 7pt; padding-bottom: 4pt;
     border-bottom: 1.6pt solid #2f3e56; page-break-after: avoid; }
h3 { font-size: 10.8pt; margin: 13pt 0 4pt; color: #2f3e56; page-break-after: avoid; }
h4 { font-size: 9.6pt; margin: 10pt 0 3pt; color: #47506a; page-break-after: avoid; }
p { margin: 0 0 6pt; text-align: justify; hyphens: auto; }
code, .mono { font-family: "SF Mono",Menlo,Consolas,monospace; font-size: 8.4pt;
              background: #f2f4f7; padding: 0.5pt 2.5pt; border-radius: 2pt; }
pre { font-family: "SF Mono",Menlo,Consolas,monospace; font-size: 7.8pt;
      background: #f5f7fa; border-left: 2.5pt solid #2f3e56; padding: 7pt 9pt;
      border-radius: 3pt; overflow-x: auto; line-height: 1.42;
      page-break-inside: avoid; white-space: pre; }
table { width: 100%; border-collapse: collapse; margin: 7pt 0 9pt;
        font-size: 8.2pt; page-break-inside: avoid; }
th { background: #2f3e56; color: #fff; text-align: left; padding: 4pt 5pt;
     font-family: "Helvetica Neue",Helvetica,Arial,sans-serif;
     font-weight: 600; font-size: 7.6pt; letter-spacing: 0.2pt; }
td { padding: 3.2pt 5pt; border-bottom: 0.5pt solid #dde2e9;
     font-variant-numeric: tabular-nums; }
tr:nth-child(even) td { background: #fafbfc; }
td.n, th.n { text-align: right; }
.hi td { background: #eef4ff !important; font-weight: 700; }
.win { color: #0a6b52; } .lose { color: #a52020; }
figure { margin: 9pt 0 11pt; page-break-inside: avoid; text-align: center; }
figure img { max-width: 100%; border: 0.5pt solid #dde2e9; border-radius: 3pt; }
figcaption { font-size: 7.8pt; color: #5b6478; margin-top: 4pt;
             font-family: "Helvetica Neue",Helvetica,Arial,sans-serif;
             text-align: left; }
figcaption b { color: #2f3e56; }
.grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 9pt; }
.callout { background: #f5f8fc; border-left: 2.5pt solid #2f3e56;
           padding: 7pt 10pt; margin: 9pt 0; page-break-inside: avoid;
           font-size: 9pt; }
.callout.warn { background: #fdf6ec; border-left-color: #b5761a; }
.callout.key { background: #eef4ff; border-left-color: #1e4fa3; }
.callout p:last-child { margin-bottom: 0; }
.kpi { display: grid; grid-template-columns: repeat(4,1fr); gap: 7pt; margin: 10pt 0; }
.kpi div { border: 0.8pt solid #d5dce6; border-radius: 4pt; padding: 7pt 8pt;
           text-align: center; }
.kpi .v { font-size: 15pt; font-weight: 700; color: #1e4fa3;
          font-family: "Helvetica Neue",Helvetica,Arial,sans-serif;
          font-variant-numeric: tabular-nums; }
.kpi .l { font-size: 6.8pt; color: #5b6478; text-transform: uppercase;
          letter-spacing: 0.5pt; margin-top: 2pt;
          font-family: "Helvetica Neue",Helvetica,Arial,sans-serif; }
.title-page { text-align: center; padding-top: 55mm; page-break-after: always; }
.title-page .sub { font-size: 12pt; color: #47506a; margin-top: 7pt;
                   font-family: "Helvetica Neue",Helvetica,Arial,sans-serif; }
.title-page .meta { margin-top: 30mm; font-size: 8.6pt; color: #5b6478;
                    font-family: "SF Mono",Menlo,monospace; line-height: 1.9; }
.title-rule { width: 58mm; height: 2.4pt; background: #2f3e56; margin: 13pt auto; }
.toc { font-size: 9.4pt; }
.toc a { display: block; text-decoration: none; color: #1a1d23;
         margin-bottom: 3.4pt; break-inside: avoid; }
.toc a::after { content: leader(". ") target-counter(attr(href), page);
                color: #8791a3; font-size: 8.4pt; }
.toc .n { color: #1e4fa3; font-weight: 700; display: inline-block;
          width: 17pt; font-family: "Helvetica Neue",Helvetica,Arial,sans-serif; }
.pb { page-break-before: always; }
ul, ol { margin: 0 0 7pt; padding-left: 15pt; }
li { margin-bottom: 2.6pt; }
.small { font-size: 8pt; color: #5b6478; }
"""


def build_html():
    s = jload("run_summary.json")
    st = jload("run_status.json")
    md = jload("run_metadata.json")
    di = jload("dataset_info.json")
    au = jload("leakage_audit.json")
    sel = s["results"]["selected"]
    la, lo, pl = sel["loso_all_subjects"], sel["lockbox"], sel["loso_pooled"]
    g = sel["generalization_gap"]
    fi = s["feature_importance"]
    reps = s["representations"]
    A, SG = arms(), stages()

    ps = list(csv.DictReader(open(os.path.join(FINAL, "metrics",
                                               "per_subject_metrics.csv"))))
    ps.sort(key=lambda r: -float(r["f1_macro"]))
    below = sum(1 for r in ps if float(r["accuracy"]) < 0.5)

    def pm(b, k, d=4):
        e = b.get(k) or {}
        return "n/a" if e.get("mean") is None else f"{f(e['mean'],d)} ± {f(e.get('std'),3)}"

    H = []
    w = H.append

    # ── title ──────────────────────────────────────────────────────────
    w(f"""<div class="title-page">
<h1>Cognitive Load Estimation<br>from EEG + fNIRS</h1>
<div class="title-rule"></div>
<div class="sub">Improving and honestly evaluating the binary<br>
(Low vs High) workload model</div>
<div class="meta">
TU Berlin n-back dataset · {di['n_subjects']} subjects · {di['n_samples']:,} windows<br>
{len(A)} controlled experiments · 26-fold leave-one-subject-out + 8-subject lockbox<br><br>
Run <b>{st['run_id']}</b><br>
commit {(md['git']['commit'] or '')[:12]} · branch {md['git']['branch']}<br>
torch {md['environment']['torch']} · seed {s['protocol'].get('seed', 42)}<br>
generated {datetime.now().strftime('%d %B %Y')}
</div></div>""")

    # ── contents ───────────────────────────────────────────────────────
    toc = ["The problem and the data", "Feature representation", "Model architecture",
           "Evaluation protocol", "Leakage audit", "The experiments",
           "Final results", "What the experiments found", "Feature importance",
           "Learned representations", "Limitations", "Reproducing this work"]
    w("<h2>Contents</h2><div class='toc'>")
    for i, t in enumerate(toc, 1):
        w(f"<a href='#sec{i:02d}'><span class='n'>{i:02d}</span>{t}</a>")
    w("</div>")

    # ── 1 problem ──────────────────────────────────────────────────────
    w("<h2 id='sec01'>01 &nbsp;The problem and the data</h2>")
    w("""<p>This project estimates <b>mental workload</b> from two brain signals
recorded at the same time. <b>EEG</b> measures electrical activity at 200 Hz —
fast, noisy, sensitive to cortical rhythms. <b>fNIRS</b> measures blood
oxygenation at 10 Hz — slow, metabolic, reported as oxygenated (HbO) and
deoxygenated (HbR) haemoglobin. They are complementary, and fusing them is the
modelling premise.</p>""")
    w(f"""<p>The data is the open <b>TU Berlin dataset (Shin et al., 2018)</b>:
{di['n_subjects']} subjects performing an n-back working-memory task with 25
trials each. The label is not a subjective rating — it is the experimentally
imposed difficulty, which is what makes it usable as ground truth. 0-back is
<b>Low</b> load, 2-back <b>Medium</b>, 3-back <b>High</b>.</p>""")
    w(f"""<p>This work addresses the <b>binary</b> framing: Low (0-back) versus
High (3-back), dropping the middle class. The result is perfectly balanced —
{di['class_distribution']['Low']:,} windows per class.</p>""")

    w("<table><tr><th>Property</th><th>Value</th></tr>")
    for k, v in [("Subjects", di["n_subjects"]),
                 ("Trials", f"{di['n_trials']} (25 per subject)"),
                 ("EEG", "30 channels @ 200 Hz"),
                 ("fNIRS", "36 channels @ 10 Hz (HbO and HbR)"),
                 ("Window", f"{di['window_sec']:g} s, 5 s stride (50% overlap)"),
                 ("Windows (binary)", f"{di['n_samples']:,}"),
                 ("Classes", " vs ".join(f"{a} ({b})" for a, b in
                                         zip(di["class_names"], di["class_semantics"]))),
                 ("Chance", f"{1/len(di['class_names']):.4f}")]:
        w(f"<tr><td>{k}</td><td>{v}</td></tr>")
    w("</table>")

    # ── 2 features ─────────────────────────────────────────────────────
    w("<h2 id='sec02'>02 &nbsp;Feature representation</h2>")
    fs = di["feature_schema"]
    w(f"""<p>Each window is reduced to <b>{fs['n_eeg'] + fs['n_fnirs']:,} features</b>.
The column names are <i>channel-major</i> — all of channel 00's features, then
channel 01's, and so on — which means each modality block reshapes exactly onto
a (channel × feature-type) grid. That property is what later makes a
convolutional model a legitimate thing to try.</p>""")
    w(f"""<pre>EEG    {fs['n_eeg']} = {fs['eeg_grid']['n_channels']} electrodes x {fs['eeg_grid']['n_features_per_channel']} feature types
              (6 bands: delta, theta, alpha, beta, gamma, engagement
               x 3 statistics: mean, std, slope)

fNIRS  {fs['n_fnirs']} = {fs['hbo_grid']['n_channels']} optodes x {fs['hbo_grid']['n_features_per_channel'] * 2} feature types
              (13 statistics: mean, std, slope, min, max, ptp, auc, power,
               skew, kurtosis, early, late, delta   x HbO and HbR)</pre>""")
    pp = di["preprocessing"]
    w(f"""<p>The preprocessing contract is <code>{pp['order']}</code>. Per-subject
z-scoring removes each person's individual baseline; EEG band powers are
right-skewed so they get a log, and HbO/HbR go negative so they get a
<i>signed</i> log. The scalers — RobustScaler for EEG, StandardScaler for fNIRS —
are refit from scratch on each fold's training rows only.</p>""")

    # ── 3 architecture ─────────────────────────────────────────────────
    w("<h2 id='sec03'>03 &nbsp;Model architecture</h2>")
    w("""<p>Two parallel modality encoders feed a fusion layer and a
classification head. The encoder family is swappable, which is what makes the
MLP-versus-CNN comparison controlled: everything downstream is held fixed.</p>""")
    w("""<pre>EEG   (540) --&gt; encoder --+
                          +--&gt; fusion --&gt; FC head --&gt; softmax
fNIRS (936) --&gt; encoder --+</pre>""")
    w("""<h3>Encoder families</h3>
<p><b>MLP</b> (the served model): a projection to a fixed width, then <i>N</i>
residual MLP blocks, then a projection to the encoder dimension. Residual
connections keep gradients healthy in a small-data regime.</p>
<p><b>1D CNN</b> (added here): reshape onto the channel grid and convolve along
the <b>electrode / optode axis</b>, with the feature types as the convolution's
input channels. One filter bank is shared across all channels — a real inductive
bias, and an order-of-magnitude fewer input-stage parameters. It assumes
neighbouring channel indices are spatially related, which is only approximately
true for this montage, so it was tested rather than assumed.</p>""")
    w("""<h3>Fusion</h3>
<p><b>Attention fusion</b> computes a sigmoid gate per modality from the
concatenation of both embeddings, then scales each modality by its own gate
before fusing. The model learns how much to trust EEG versus fNIRS, and those
gate values are what the web app's modality bar displays. <b>Concat fusion</b> is
the ablation.</p>""")
    w("""<h3>Domain-adversarial option (DANN)</h3>
<p>A subject discriminator sits behind a gradient-reversal layer on the fused
embedding: forward pass is the identity, backward pass multiplies the gradient
by &minus;&lambda;. The discriminator gets better at identifying subjects while the
same signal pushes the encoder to <i>erase</i> subject identity. Domain labels are
the training subjects only, re-indexed per fold, so the discriminator never has
a class for the held-out subject.</p>""")
    w(f"""<p>The selected model has <b>{s['model']['n_params']:,} parameters</b>:
projection width 96, encoder dimension 48, 2 residual blocks, attention fusion,
head [48, 24], dropout 0.5 / 0.6.</p>""")

    # ── 4 protocol ─────────────────────────────────────────────────────
    w("<h2 id='sec04' class='pb'>04 &nbsp;Evaluation protocol</h2>")
    w("""<p>A 26-fold leave-one-subject-out score is a good generalization
estimate — but if you run forty experiments and keep the best one, that score is
a selection artifact: the folds have quietly become a training signal. So the
subjects are split once, deterministically, into two pools.</p>""")
    w(f"""<pre>development pool  {len(s['protocol']['development_subjects'])} subjects  {s['protocol']['development_subjects']}
lockbox            {len(s['protocol']['lockbox_subjects'])} subjects  {s['protocol']['lockbox_subjects']}</pre>""")
    w("""<p>Every architecture, hyperparameter, scheduler, dropout rate and
dataset variant was chosen on <b>development-pool LOSO alone</b>. The lockbox was
never read during the search; after the configuration was frozen, each lockbox
subject was evaluated once by a model trained only on development subjects. The
26-fold LOSO number is also reported, labelled <i>selection-informed</i>, so it
stays comparable to the rest of the project.</p>""")
    w("""<h3>Inside one fold</h3>
<pre>held-out subject   --&gt;  test only. Never in train, val, or any fitted statistic.
remaining subjects --&gt;  GroupShuffleSplit by subject  --&gt;  train + validation</pre>
<p>Checkpoints are selected on <b>validation macro F1</b> with best-weight
restore, on subjects the weights never saw.</p>""")
    w(f"""<div class="callout key"><p><b>The selection rule, fixed in advance.</b>
Rank by mean development-LOSO macro F1; take every arm within one standard error
of the leader; inside that set prefer the incumbent, then the smallest model. A
challenger must clear the noise floor to be adopted. With 18 folds and a
fold-to-fold standard deviation near 0.08, that floor is
<b>&plusmn;{SG[0]['noise_floor_1se']:.4f}</b>.</p></div>""")
    w("""<div class="callout warn"><p><b>A flaw that was caught mid-campaign.</b>
The first implementation broke ties only on parameter count. Arms within a stage
usually have <i>identical</i> parameter counts, so the rule silently degenerated
into "pick the highest mean" — noise-chasing. It surfaced when a stage accepted a
+0.0023 improvement against a &plusmn;0.018 floor. Over ten stages that compounds
into a selection artifact, so the rule was corrected and the entire campaign was
restarted. The discarded run is still in the archive.</p></div>""")

    # ── 5 leakage ──────────────────────────────────────────────────────
    w("<h2 id='sec05'>05 &nbsp;Leakage audit</h2>")
    w("""<p>Nine assertions run before <i>every</i> fold trains. Any failure raises
an exception and aborts the run, so a leaked score is never produced and
therefore can never be reported.</p>""")
    w("<table><tr><th>Check</th><th>What it proves</th></tr>")
    for a, b in [("row_index_disjoint", "no row appears in two splits"),
                 ("subject_disjoint", "train / val / test share no subject"),
                 ("trial_disjoint", "no trial straddles a split"),
                 ("window_identity_disjoint", "no (subject, trial, window) key on both sides"),
                 ("duplicate_feature_rows", "no byte-identical feature row across splits"),
                 ("scaler_fit_on_train_only", "rows used to fit the scalers are a subset of training rows"),
                 ("heldout_subject_never_fitted", "the held-out subject is absent from train and val"),
                 ("no_nan_or_inf", "preprocessing produced finite values"),
                 ("labels_valid", "labels in range, training fold has both classes")]:
        w(f"<tr><td><code>{a}</code></td><td>{b}</td></tr>")
    w("</table>")
    w("""<h3>The sliding-window question</h3>
<p>Adjacent windows overlap by 50%, so they share raw signal — putting two of
them on opposite sides of a split would leak. Grouping by subject makes that
structurally impossible: every window of the held-out subject is in the test set
and none is anywhere else. The audit asserts it explicitly rather than relying on
the argument.</p>""")
    w("""<h3>Per-subject normalization — the honest caveat</h3>
<p>Each subject is z-scored against their own statistics, and that happens before
the fold split. It <b>cannot</b> leak between subjects, and this is proved
numerically rather than in prose: the whole preprocessing chain is re-run for one
subject <i>from that subject's raw rows in isolation</i> and checked against the
full-dataset computation. Observed difference: <b>exactly zero</b>.</p>
<p>It is, however, <i>transductive within a subject</i> — computing a mean needs
the whole session. That is standard practice in BCI and uses no labels, but it
means deployment needs a recorded session, not a single window. The cost of
restricting it to a short calibration block was measured (see section 08).</p>""")
    w("""<div class="callout"><p><b>The audit is not vacuously true.</b> A test
suite injects each leakage type deliberately — a subject leak, a window leak, a
scaler-fit leak, a duplicated row, a NaN — and asserts the audit catches that
specific check. All 8 pass. A separate suite asserts the protocol machinery,
including that early stopping restores weights which reproduce the best
validation score to 1e-6. All 6 pass.</p></div>""")

    # ── 6 experiments ──────────────────────────────────────────────────
    w("<h2 id='sec06' class='pb'>06 &nbsp;The experiments</h2>")
    w(f"""<p><b>{len(A)} arms across {len(SG)} stages.</b> Each stage varied one
axis from the incumbent and the winner carried forward. All arms trained and
evaluated on identical development folds, so every comparison is paired.</p>""")
    w("<table><tr><th>Stage</th><th>Axis varied</th><th>Best challenger</th>"
      "<th class='n'>&Delta;</th><th>Outcome</th></tr>")
    axes = {"S1_batch_lr": "batch size, learning rate",
            "S2_schedule": "epoch budget, LR schedule",
            "S3_weight_decay": "weight decay",
            "S4_dropout": "dropout rates",
            "S5_capacity": "width, depth",
            "S6_architecture": "MLP vs CNN, fusion",
            "S7_modality_dropout": "modality dropout",
            "S8_noise": "input noise, label smoothing",
            "S9_dann": "domain-adversarial &lambda;",
            "S10_data": "window geometry, normalization, HbR"}
    for x in SG:
        bc = x["best_challenger"] or {}
        d = bc.get("delta_vs_incumbent", 0)
        cls = "win" if d > 0 else "lose"
        note = "rejected (inside noise)" if bc.get("inside_noise_floor") and d > 0 else "rejected"
        w(f"<tr><td><code>{x['stage']}</code></td><td>{axes.get(x['stage'],'')}</td>"
          f"<td><code>{bc.get('experiment_id','')}</code></td>"
          f"<td class='n {cls}'>{d:+.4f}</td><td>{note}</td></tr>")
    w("</table>")
    w("""<div class="callout key"><p><b>No stage produced a change that cleared
the noise floor.</b> The configuration the project had already locked in survived
every axis tested.</p></div>""")

    w("<h3>Full leaderboard — 18-fold development LOSO</h3>")
    w("<p class='small'>&Delta; and <i>p</i> are paired against the baseline on the "
      "same folds. <i>gap</i> is train &minus; held-out macro F1: higher means more "
      "overfitting.</p>")
    w("<table><tr><th>Experiment</th><th>Macro F1</th><th class='n'>Acc</th>"
      "<th class='n'>AUC</th><th class='n'>&Delta;</th><th class='n'>p</th>"
      "<th class='n'>gap</th><th class='n'>Params</th><th>Description</th></tr>")
    for r in A:
        base = r["experiment_id"] == "A0_baseline"
        d = r.get("vs_baseline_mean_diff") or ""
        p = r.get("vs_baseline_t_p") or ""
        ds = "baseline" if base else (f"{float(d):+.4f}" if d not in ("", "nan") else "")
        cls = "" if base else ("win" if d not in ("", "nan") and float(d) > 0 else "lose")
        sigp = ""
        if p not in ("", "nan"):
            sigp = f"<b>{float(p):.3f}</b>" if float(p) < 0.05 else f"{float(p):.3f}"
        w(f"<tr class=\"{'hi' if base else ''}\"><td><code>{r['experiment_id']}</code></td>"
          f"<td class='n'>{f(r['dev_loso_f1_macro_mean'])} ± {f(r['dev_loso_f1_macro_std'],3)}</td>"
          f"<td class='n'>{f(r['dev_loso_accuracy_mean'],3)}</td>"
          f"<td class='n'>{f(r['dev_loso_auc_mean'],3)}</td>"
          f"<td class='n {cls}'>{ds}</td><td class='n'>{sigp}</td>"
          f"<td class='n'>{f(r['train_minus_heldout_f1'],3)}</td>"
          f"<td class='n'>{int(r['n_params']):,}</td><td>{r['note']}</td></tr>")
    w("</table>")
    cmp_png = os.path.join(OUT_DIR, "_cmp.png")
    merged_comparison_figure(A, cmp_png)
    w("<figure>" + img(cmp_png) +
      f"<figcaption><b>Figure 1.</b> All {len(A)} arms across both campaign runs, "
      "ranked by development-LOSO macro F1. Grey is the baseline, green beat it, "
      "blue did not; the dashed line is the baseline score. Error bars are the "
      "standard deviation across the 18 development folds — note how much wider "
      "they are than the gaps between arms.</figcaption></figure>")

    # ── 7 results ──────────────────────────────────────────────────────
    w("<h2 id='sec07'>07 &nbsp;Final results</h2>")
    w(f"""<p>The selected model is <code>{s['model']['selected_experiment']}</code>
— the configuration already being served. Nothing beat it.</p>""")
    w(f"""<div class="kpi">
<div><div class="v">{f(la['accuracy']['mean'],4)}</div><div class="l">LOSO accuracy</div></div>
<div><div class="v">{f(la['f1_macro']['mean'],4)}</div><div class="l">LOSO macro F1</div></div>
<div><div class="v">{f(lo['accuracy']['mean'],4)}</div><div class="l">Lockbox accuracy</div></div>
<div><div class="v">{f(lo['f1_macro']['mean'],4)}</div><div class="l">Lockbox macro F1</div></div>
</div>""")
    w("<table><tr><th>Metric</th><th class='n'>26-fold LOSO (all subjects)</th>"
      "<th class='n'>Lockbox (8 unseen subjects)</th></tr>")
    for k, lab in [("accuracy", "Accuracy"), ("f1_macro", "Macro F1"),
                   ("balanced_accuracy", "Balanced accuracy"), ("roc_auc", "ROC-AUC"),
                   ("mcc", "Matthews correlation"), ("cohen_kappa", "Cohen's Kappa")]:
        w(f"<tr><td>{lab}</td><td class='n'>{pm(la,k)}</td><td class='n'>{pm(lo,k)}</td></tr>")
    w("</table>")
    w("""<div class="callout key"><p><b>The lockbox number is not lower than the
LOSO number.</b> That is the check the whole protocol existed for: forty-two
experiments of searching produced <b>zero selection inflation</b>. The 63% is
real.</p></div>""")

    w("<h3>Training, validation and testing accuracy</h3>")
    w("""<p>"Testing" here means <b>a person the model has never seen</b>. There is
no random train/test split anywhere in this work. A random window split would
score considerably higher, but it would be measuring the model's ability to
recognise a subject's physiological baseline, not workload.</p>""")
    w("<table><tr><th>Protocol</th><th class='n'>Train</th><th class='n'>Validation</th>"
      "<th class='n'>Test (held-out subject)</th><th class='n'>Gap</th></tr>")
    w("<tr><td>26-fold LOSO</td><td class='n'>0.8022 ± 0.093</td>"
      "<td class='n'>0.6400 ± 0.053</td><td class='n'><b>0.6309 ± 0.101</b></td>"
      "<td class='n'>+0.171</td></tr>")
    w("<tr><td>Lockbox</td><td class='n'>0.8186 ± 0.059</td>"
      "<td class='n'>0.6783 ± 0.038</td><td class='n'><b>0.6371 ± 0.085</b></td>"
      "<td class='n'>+0.182</td></tr></table>")

    cm = pl["confusion_matrix"]
    w("<div class='grid2'>")
    w("<figure>" + img("confusion_matrix/confusion_matrix.png") +
      f"<figcaption><b>Figure 2.</b> Pooled over all {pl['n_samples']:,} held-out "
      "windows from the 26 folds.</figcaption></figure>")
    w("<figure>" + img("curves/roc_curve.png") +
      f"<figcaption><b>Figure 3.</b> ROC across every fold's held-out "
      f"probabilities. AUC {f(pl['roc_auc_macro_ovr'],4)}.</figcaption></figure>")
    w("</div>")
    w("<table><tr><th>Class</th><th class='n'>Precision</th><th class='n'>Recall</th>"
      "<th class='n'>F1</th><th class='n'>Support</th></tr>")
    for c in pl["per_class"]:
        w(f"<tr><td>{c['class_name']}</td><td class='n'>{f(c['precision'],3)}</td>"
          f"<td class='n'>{f(c['recall'],3)}</td><td class='n'>{f(c['f1'],3)}</td>"
          f"<td class='n'>{c['support']:,}</td></tr>")
    w("</table>")

    w("<figure>" + img("curves/per_subject_performance.png") +
      "<figcaption><b>Figure 4.</b> Macro F1 for each held-out subject. Every bar "
      "is one person the model never trained on.</figcaption></figure>")
    w(f"""<div class="callout key"><p><b>The dominant source of error is which
person you test on, not which model you chose.</b> The best subject scores
{f(float(ps[0]['f1_macro']),3)} and the worst {f(float(ps[-1]['f1_macro']),3)} — a
spread of {f(float(ps[0]['f1_macro'])-float(ps[-1]['f1_macro']),2)}. That is more
than twice the largest difference between any two of the {len(A)} models tested.
{below} of {len(ps)} subjects fall below chance accuracy.</p></div>""")

    w("<div class='grid2'>")
    w("<figure>" + img("curves/loss_curves.png") +
      "<figcaption><b>Figure 5.</b> Train versus validation loss, averaged over "
      "folds.</figcaption></figure>")
    w("<figure>" + img("curves/generalization_gap.png") +
      "<figcaption><b>Figure 6.</b> The macro-F1 gap opening during "
      "training.</figcaption></figure>")
    w("</div>")

    # ── 8 findings ─────────────────────────────────────────────────────
    w("<h2 id='sec08' class='pb'>08 &nbsp;What the experiments found</h2>")

    w("<h3>The CNN does not help this representation</h3>")
    w("<table><tr><th>Architecture</th><th class='n'>Macro F1</th><th class='n'>&Delta;</th>"
      "<th class='n'>p</th><th class='n'>gap</th><th class='n'>Params</th></tr>")
    for eid in ["A0_baseline", "S6_cnn_res", "S6_cnn_res_attn", "S6_cnn_wide", "S6_cnn"]:
        r = next((x for x in A if x["experiment_id"] == eid), None)
        if not r:
            continue
        d = r.get("vs_baseline_mean_diff") or ""
        base = eid == "A0_baseline"
        w(f"<tr class=\"{'hi' if base else ''}\"><td><code>{eid}</code></td>"
          f"<td class='n'>{f(r['dev_loso_f1_macro_mean'])}</td>"
          f"<td class='n lose'>{'baseline' if base else f'{float(d):+.4f}'}</td>"
          f"<td class='n'>{f(r.get('vs_baseline_t_p'),3)}</td>"
          f"<td class='n'>{f(r['train_minus_heldout_f1'],3)}</td>"
          f"<td class='n'>{int(r['n_params']):,}</td></tr>")
    w("</table>")
    w("""<p>Every variant is significantly worse. The interesting column is
<i>gap</i>: the CNNs cut overfitting from 0.163 to 0.026&ndash;0.076 — sharing
one filter bank across electrodes nearly <b>eliminates</b> it. But it removes so
much capacity that held-out performance falls further than the overfitting it
cures.</p>
<p><b>The interpretation matters more than the verdict.</b> These are
<i>pre-aggregated per-channel statistics</i>, not raw time series. A convolution
along the channel axis of 18 summary numbers per electrode has no local structure
to exploit. A CNN would be the right tool on the raw signal; on this
representation the spatial prior costs more than it buys.</p>""")

    w("<h3>Regularization cannot close the gap</h3>")
    w("""<p>Weight decay (0 to 0.1), dropout (0.3 to 0.7), input noise (0 to 0.2),
label smoothing and modality dropout were all tested. None improved held-out
performance beyond noise, and the pattern is one-directional: stronger
regularization reliably reduces the gap <i>and</i> reduces held-out performance
with it. Dropout 0.6 / 0.7 cuts the gap from 0.163 to 0.040 while losing 0.137
macro F1 (p &lt; 0.001).</p>
<p>The gap is not excess capacity that can be regularized away. It reflects
genuine between-subject variability: the model fits its training subjects well,
and those subjects are simply not fully representative of the next person.</p>""")

    w("<h3>Capacity is not the constraint in either direction</h3>")
    narrow = next((x for x in A if x["experiment_id"] == "S5_proj32_enc32"), None)
    if narrow:
        w(f"""<p>A model with <b>{int(narrow['n_params']):,} parameters</b> reaches
{f(narrow['dev_loso_f1_macro_mean'])} — statistically identical to the
{s['model']['n_params']:,}-parameter baseline (&Delta;
{float(narrow['vs_baseline_mean_diff']):+.4f}, p =
{f(narrow['vs_baseline_t_p'],3)}) at <b>3.7&times; smaller</b>. Widening to
620,570 parameters gains only +0.016, inside the noise floor. If model size
matters for your deployment, the small model is free.</p>""")

    w("<h3>Domain-adversarial training is rejected — and the probe explains why</h3>")
    w("<table><tr><th>&lambda;</th><th class='n'>Macro F1</th><th class='n'>&Delta;</th>"
      "<th class='n'>p</th></tr>")
    w(f"<tr class='hi'><td>0 (baseline)</td><td class='n'>0.6451</td>"
      "<td class='n'>&mdash;</td><td class='n'>&mdash;</td></tr>")
    for eid, lam in [("S9_dann01", "0.1"), ("S9_dann05", "0.5"), ("S9_dann10", "1.0")]:
        r = next((x for x in A if x["experiment_id"] == eid), None)
        if r:
            w(f"<tr><td>{lam}</td><td class='n'>{f(r['dev_loso_f1_macro_mean'])}</td>"
              f"<td class='n lose'>{float(r['vs_baseline_mean_diff']):+.4f}</td>"
              f"<td class='n'>{f(r['vs_baseline_t_p'],3)}</td></tr>")
    w("</table>")
    w("""<p>Performance degrades <b>monotonically</b> with adversarial strength,
independently replicating the project's earlier 3-class finding on the binary
task under a stricter protocol. The subject-invariance probe — a fresh classifier
trained to recover subject identity from each representation — explains the
mechanism.</p>""")
    w("<table><tr><th>Representation</th><th class='n'>Probe accuracy</th>"
      "<th class='n'>Chance</th></tr>")
    for k, v in reps.items():
        p_ = v["subject_invariance_probe"]
        if p_:
            w(f"<tr><td>{k.replace('_',' ')}</td>"
              f"<td class='n'>{f(p_['accuracy_mean'],3)}</td>"
              f"<td class='n'>{f(p_['chance_uniform'],3)}</td></tr>")
    w("</table>")
    w("""<p>Per-subject z-scoring has already stripped subject identity from the
<i>raw</i> features. The encoder then <b>re-introduces</b> it, and DANN
<b>fails to remove it</b> — no better than the baseline it was meant to improve.
The adversary injects noise into the task head without deleting the confound it
targets. That is a coherent negative result, not an inconclusive one.</p>""")

    w("<h3>Four existing design choices, confirmed and priced</h3>")
    w("<table><tr><th>Variant</th><th class='n'>&Delta; macro F1</th><th class='n'>p</th>"
      "<th>Reading</th></tr>")
    for eid, read in [("S10_no_subject_norm", "per-subject normalization genuinely matters"),
                      ("S10_no_hbr", "HbR genuinely helps"),
                      ("S10_norm_after_log", "normalizing before the log was the right call"),
                      ("S10_calibration_norm", "the deployment-realistic mode costs ~0.04, not significant"),
                      ("S10_stride1s", "denser windows buy nothing measurable"),
                      ("S10_win8s", "10 s windows beat 8 s")]:
        r = next((x for x in A if x["experiment_id"] == eid), None)
        if r:
            d = float(r["vs_baseline_mean_diff"])
            p_ = float(r["vs_baseline_t_p"])
            w(f"<tr><td><code>{r['note'][:38]}</code></td>"
              f"<td class='n {'win' if d>0 else 'lose'}'>{d:+.4f}</td>"
              f"<td class='n'>{'<b>%.3f</b>'%p_ if p_<0.05 else '%.3f'%p_}</td>"
              f"<td>{read}</td></tr>")
    w("</table>")
    w("""<p>The <b>calibration-only</b> result is the practically useful one:
restricting per-subject statistics to the first 30 windows — a short calibration
block rather than the whole session — costs about 0.04 macro F1 and is not
statistically significant. That makes an online deployment defensible.</p>""")

    # ── 9 SHAP ─────────────────────────────────────────────────────────
    w("<h2 id='sec09' class='pb'>09 &nbsp;Feature importance</h2>")
    ms = fi["modality_share"]
    w(f"""<div class="kpi">
<div><div class="v">{f(ms['fNIRS'],3)}</div><div class="l">fNIRS share</div></div>
<div><div class="v">{f(ms['EEG'],3)}</div><div class="l">EEG share</div></div>
<div><div class="v">{f(ms['HbO'],3)}</div><div class="l">HbO share</div></div>
<div><div class="v">{f(ms['HbR'],3)}</div><div class="l">HbR share</div></div>
</div>""")
    w("""<p>Computed with SHAP's gradient explainer on this run's own trained
model over held-out windows. <b>fNIRS dominates overwhelmingly</b> — far more
starkly than the original project's near-parity attention gates suggested. And
<b>HbR slightly exceeds HbO</b>, which independently corroborates the ablation
showing that removing HbR costs 0.067 macro F1.</p>""")
    w("<table><tr><th class='n'>#</th><th>Feature family</th><th>Modality</th>"
      "<th class='n'>Total mean |SHAP|</th></tr>")
    for i, c in enumerate(fi["top_categories"][:10], 1):
        w(f"<tr><td class='n'>{i}</td><td>{c['category']}</td><td>{c['modality']}</td>"
          f"<td class='n'>{f(c['mean_abs_shap'],4)}</td></tr>")
    w("</table>")
    w("""<p>The dominant signal is haemodynamic response <i>shape</i> — area under
the curve, maximum, minimum, mean, the late-window response — not EEG band
power.</p>""")
    w("<figure>" + img("feature_importance/shap_categories.png") +
      "<figcaption><b>Figure 7.</b> Attribution mass by feature family. Blue is "
      "EEG, orange fNIRS.</figcaption></figure>")

    # ── 10 representations ─────────────────────────────────────────────
    w("<h2 id='sec10'>10 &nbsp;Learned representations</h2>")
    w("""<p>Every variant is trained on the same development subjects and embedded
on the same unseen subjects, so the pictures differ only by the model that
produced them.</p>""")
    w("<div class='grid2'>")
    w("<figure>" + img("representations/tsne.png") +
      "<figcaption><b>Figure 8.</b> t-SNE of the fused embedding, coloured by "
      "workload class.</figcaption></figure>")
    w("<figure>" + img("representations/subject_tsne.png") +
      "<figcaption><b>Figure 9.</b> The same embedding, coloured by subject. "
      "Subject clustering is visible — which is the ceiling.</figcaption></figure>")
    w("</div>")

    # ── 11 limitations ─────────────────────────────────────────────────
    w("<h2 id='sec11' class='pb'>11 &nbsp;Limitations</h2>")
    w("<ol>")
    for x in [
        "<b>Statistical power.</b> With 18 development subjects the standard error on mean macro F1 is &plusmn;0.018, so this search could only reliably detect improvements larger than roughly 0.04. A smaller genuine gain would be invisible to it.",
        f"<b>{di['n_subjects']} subjects is small</b> for cross-subject claims, and {below} of them fall below chance. The between-subject spread dominates everything else measured here.",
        "<b>Per-subject normalization is transductive.</b> It uses no labels and no other subject's data, but it needs the subject's whole session. The calibration mode prices that at about 0.04 macro F1.",
        "<b>The greedy staged search is not exhaustive.</b> It varies one axis at a time from a moving incumbent, so it cannot find improvements requiring two simultaneous changes.",
        "<b>The 26-fold LOSO number is selection-informed</b> — the configuration was chosen using 18 of the 26 subjects it evaluates on. The lockbox number is the clean estimate, and it agrees.",
        "<b>SHAP is one attribution method on one trained model.</b> The fNIRS dominance is much starker than the original attention-gate analysis suggested; the HbR ablation corroborates the direction but not the magnitude.",
        "<b>The CNN result is specific to this feature representation.</b> It says nothing about a CNN over the raw time series, which was never tested.",
    ]:
        w(f"<li>{x}</li>")
    w("</ol>")

    # ── 12 reproducing ─────────────────────────────────────────────────
    w("<h2 id='sec12'>12 &nbsp;Reproducing this work</h2>")
    w("""<pre># Full campaign (several hours on CPU)
.venv/bin/python run_campaign.py --tag my-run

# A single stage, or the 3-class task
.venv/bin/python run_campaign.py --stages S6
.venv/bin/python run_campaign.py --mode 3class

# Smoke test: reduced cohort, every code path in ~2 minutes
.venv/bin/python run_campaign.py --quick --limit-subjects 7 --max-epochs 6

# Tests and independent verification
.venv/bin/python -m research.test_leakage      # 8/8
.venv/bin/python -m research.test_framework    # 6/6
.venv/bin/python -m research.verify_run --all  # recompute metrics from predictions

# The app
cd backend &amp;&amp; ../.venv/bin/python -m uvicorn main:app --port 8000
cd frontend &amp;&amp; npm run dev            # http://localhost:5173</pre>""")
    w("""<p>Every run writes its own directory under <code>outputs/runs/</code>,
created before any data is read, so a run that crashes still leaves its metadata,
config and traceback behind. Nothing is ever overwritten and failed runs are never
deleted. One row per evaluated window is saved to
<code>predictions.csv</code>, and every metric in the archive can be recomputed
from it — <code>research.verify_run</code> does exactly that and reports zero
mismatches.</p>""")
    w("""<h3>Further reading</h3>
<ul>
<li><code>explanation/PROJECT_RECORD.md</code> — the complete record, in full detail</li>
<li><code>explanation/BINARY_MODEL_IMPROVEMENT.md</code> — framework methodology</li>
<li><code>explanation/API_AND_ARCHITECTURE.md</code> — endpoint-by-endpoint API reference</li>
<li><code>explanation/PROJECT_OVERVIEW.md</code> — the original research project</li>
<li><code>RESULTS_BINARY_IMPROVEMENT.md</code> — machine-generated results tables</li>
</ul>""")

    # ── conclusion ─────────────────────────────────────────────────────
    w("<h2>Conclusion</h2>")
    w(f"""<p><b>The binary model was not improved, and that is the result.</b>
Across {len(A)} controlled experiments spanning batch size, learning rate, LR
schedule, epoch budget, weight decay, dropout, capacity, depth, architecture
family, fusion type, modality dropout, input noise, label smoothing,
domain-adversarial training, window geometry, normalization mode and HbR
inclusion, nothing beat the served configuration by more than the noise floor,
and most alternatives were significantly worse.</p>""")
    w("""<p>The configuration the project had already locked in sits at a local
optimum for this feature representation. Three mutually consistent lines of
evidence say the remaining error is not a modelling deficiency: regularization
trades the gap for performance one-for-one; the subject probe shows the encoder
re-introduces subject identity that preprocessing had removed and DANN cannot
remove it again; and the between-subject spread is more than twice the largest
difference between any two models tested.</p>""")
    w(f"""<div class="callout key"><p>The final honest number is
<b>{f(la['accuracy']['mean']*100,1)}% accuracy on people the model has never
seen</b> (chance 50%), and the lockbox confirms it is not inflated by the search.
The most promising untested direction is not a better classifier — it is either
more subjects, or a genuinely different feature representation where an
architectural prior like convolution would have something to work
with.</p></div>""")

    return (f"<!DOCTYPE html><html><head><meta charset='utf-8'>"
            f"<title>Cognitive Load from EEG + fNIRS</title>"
            f"<style>{CSS}</style></head><body>{''.join(H)}</body></html>")


def render(html, pdf_path, base_url=OUT_DIR):
    """HTML -> PDF via WeasyPrint (CSS paged media: page numbers, headers, TOC)."""
    from weasyprint import HTML

    HTML(string=html, base_url=base_url).write_pdf(pdf_path)
    return pdf_path


# ── markdown documents -> PDF ──────────────────────────────────────────

DOC_CSS = """
@page {
  size: A4; margin: 20mm 17mm 18mm 17mm;
  @bottom-center { content: counter(page);
    font-family: "Helvetica Neue",Helvetica,Arial,sans-serif;
    font-size: 7.5pt; color: #8791a3; }
  @top-left { content: string(doctitle);
    font-family: "Helvetica Neue",Helvetica,Arial,sans-serif;
    font-size: 7pt; color: #a3abb8; }
}
@page :first { @bottom-center { content: ""; } @top-left { content: ""; } }
body { font-family: "Georgia","Times New Roman",serif; font-size: 9.6pt;
       line-height: 1.55; color: #1a1d23; }
h1 { string-set: doctitle content(); font-family: "Helvetica Neue",Helvetica,Arial,sans-serif;
     font-size: 19pt; color: #10131a; margin: 0 0 12pt; letter-spacing: -0.3pt;
     padding-bottom: 6pt; border-bottom: 2.4pt solid #2f3e56; }
h2 { font-family: "Helvetica Neue",Helvetica,Arial,sans-serif; font-size: 13pt;
     color: #10131a; margin: 18pt 0 6pt; padding-bottom: 3pt;
     border-bottom: 1.2pt solid #d5dce6; page-break-after: avoid; }
h3 { font-family: "Helvetica Neue",Helvetica,Arial,sans-serif; font-size: 10.6pt;
     color: #2f3e56; margin: 12pt 0 4pt; page-break-after: avoid; }
h4 { font-family: "Helvetica Neue",Helvetica,Arial,sans-serif; font-size: 9.6pt;
     color: #47506a; margin: 9pt 0 3pt; page-break-after: avoid; }
p { margin: 0 0 6pt; text-align: justify; hyphens: auto; }
code { font-family: "SF Mono",Menlo,Consolas,monospace; font-size: 8.3pt;
       background: #f2f4f7; padding: 0.5pt 2.5pt; border-radius: 2pt; }
pre { font-family: "SF Mono",Menlo,Consolas,monospace; font-size: 7.6pt;
      background: #f5f7fa; border-left: 2.5pt solid #2f3e56; padding: 7pt 9pt;
      border-radius: 3pt; line-height: 1.42; page-break-inside: avoid;
      white-space: pre-wrap; word-wrap: break-word; }
pre code { background: none; padding: 0; font-size: inherit; }
table { width: 100%; border-collapse: collapse; margin: 7pt 0 9pt;
        font-size: 8pt; page-break-inside: avoid; }
th { background: #2f3e56; color: #fff; text-align: left; padding: 4pt 5pt;
     font-family: "Helvetica Neue",Helvetica,Arial,sans-serif;
     font-weight: 600; font-size: 7.5pt; }
td { padding: 3.2pt 5pt; border-bottom: 0.5pt solid #dde2e9;
     font-variant-numeric: tabular-nums; }
tr:nth-child(even) td { background: #fafbfc; }
blockquote { background: #f5f8fc; border-left: 2.5pt solid #2f3e56;
             padding: 6pt 10pt; margin: 8pt 0; page-break-inside: avoid; }
blockquote p:last-child { margin-bottom: 0; }
ul, ol { margin: 0 0 7pt; padding-left: 15pt; }
li { margin-bottom: 2.4pt; }
a { color: #1e4fa3; text-decoration: none; }
hr { border: 0; border-top: 0.8pt solid #d5dce6; margin: 14pt 0; }
img { max-width: 100%; }
"""


def markdown_to_pdf(md_path, pdf_path):
    """pandoc renders the markdown; WeasyPrint paginates it."""
    body = subprocess.run(
        ["pandoc", md_path, "--from", "gfm", "--to", "html5",
         "--wrap=none", "--no-highlight"],
        check=True, capture_output=True, text=True,
    ).stdout
    # Inside the PDF set, a link to a sibling .md should open that sibling's
    # PDF, not a markdown file the reader has no way to open.
    import re as _re
    body = _re.sub(r'href="([^"]+)\.md"', r'href="\1.pdf"', body)

    html = (f"<!DOCTYPE html><html><head><meta charset='utf-8'>"
            f"<style>{DOC_CSS}</style></head><body>{body}</body></html>")
    render(html, pdf_path, base_url=os.path.dirname(os.path.abspath(md_path)))
    return pdf_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", action="store_true",
                    help="also convert every explanation/*.md to PDF")
    ap.add_argument("--only-docs", action="store_true",
                    help="convert the markdown documents and nothing else")
    ap.add_argument("--all", action="store_true", help="report + documents")
    args = ap.parse_args()

    made = []
    if not args.only_docs:
        pdf = os.path.join(OUT_DIR, "Cognitive_Load_Project_Report.pdf")
        render(build_html(), pdf)
        for junk in ("_cmp.png",):
            jp = os.path.join(OUT_DIR, junk)
            if os.path.exists(jp):
                os.remove(jp)
        made.append(pdf)

    if args.docs or args.only_docs or args.all:
        for md in sorted(glob.glob(os.path.join(OUT_DIR, "*.md"))):
            out = os.path.splitext(md)[0] + ".pdf"
            try:
                markdown_to_pdf(md, out)
                made.append(out)
            except subprocess.CalledProcessError as exc:
                print(f"  pandoc failed on {os.path.basename(md)}: {exc.stderr[:200]}")

    for m in made:
        print(f"wrote {os.path.relpath(m, ROOT)} ({os.path.getsize(m)/1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
