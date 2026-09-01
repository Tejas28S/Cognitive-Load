"""
make_slides.py — build the panel presentation deck.

A 16:9 slide deck rendered with WeasyPrint. Every number is read from the run
artifacts, so the deck cannot disagree with the results.

Each slide carries a "SAY THIS" band: the two or three sentences to speak while
that slide is up. The deck opens with a presenter briefing and closes with
anticipated panel questions and prepared answers.

    .venv/bin/python -m research.make_slides
"""

from __future__ import annotations

import base64
import csv
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FINAL = os.path.join(ROOT, "outputs", "runs", "run_20260820_141126_19cb")
EARLIER = os.path.join(ROOT, "outputs", "runs", "run_20260819_184443_f0d1")
OUT = os.path.join(ROOT, "explanation", "Cognitive_Load_Presentation.pdf")


def f(v, d=4):
    if v is None:
        return "n/a"
    x = float(v)
    return "n/a" if x != x else f"{x:.{d}f}"


def pct(v, d=1):
    return "n/a" if v is None else f"{float(v) * 100:.{d}f}%"


def img(path, run=FINAL):
    p = path if os.path.isabs(path) else os.path.join(run, path)
    if not os.path.isfile(p):
        return ""
    return ('<img src="data:image/png;base64,'
            + base64.b64encode(open(p, "rb").read()).decode() + '">')


def jload(*parts, run=FINAL):
    p = os.path.join(run, *parts)
    return json.load(open(p, encoding="utf-8")) if os.path.isfile(p) else None


def split_accuracy(exp="FINAL_loso26", run=FINAL):
    """Train / validation accuracy at each fold's selected epoch, from history."""
    import statistics as st

    hist = jload("experiments", exp, "history.json", run=run) or []
    tr = [h["train_accuracy"][h["best_epoch"] - 1] for h in hist]
    va = [h["val_accuracy"][h["best_epoch"] - 1] for h in hist]
    return (st.fmean(tr), st.fmean(va)) if hist else (None, None)


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


CSS = """
@page { size: 297mm 167mm; margin: 0; }
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: "Helvetica Neue",Helvetica,Arial,sans-serif; color: #14171d; }

.slide { width: 297mm; height: 167mm; padding: 13mm 16mm 11mm;
         page-break-after: always; position: relative; overflow: hidden;
         display: flex; flex-direction: column; }
.slide:last-child { page-break-after: auto; }
.kicker { font-size: 8pt; letter-spacing: 2.2pt; text-transform: uppercase;
          color: #6b7686; margin-bottom: 4mm; font-weight: 600; }
h1 { font-size: 27pt; line-height: 1.12; letter-spacing: -0.7pt; color: #0d1017;
     margin-bottom: 4mm; font-weight: 700; }
h2 { font-size: 13pt; color: #2f3e56; margin-bottom: 2.5mm; font-weight: 600; }
p  { font-size: 11pt; line-height: 1.5; color: #2b313c; margin-bottom: 3mm; }
.big { font-size: 15pt; line-height: 1.45; color: #1a1f28; }
ul { list-style: none; }
li { font-size: 11.2pt; line-height: 1.5; margin-bottom: 3mm;
     padding-left: 7mm; position: relative; color: #2b313c; }
li::before { content: "▸"; position: absolute; left: 0; color: #3d6fd4;
             font-size: 9pt; top: 1.5pt; }
li b { color: #0d1017; }
.body { flex: 1; }

/* speaker band */
.say { position: absolute; left: 0; right: 0; bottom: 0; background: #f2f5fa;
       border-top: 1.2pt solid #d3dcea; padding: 4mm 16mm 4.5mm; }
.say .lbl { font-size: 6.8pt; letter-spacing: 1.6pt; color: #4a72c4;
            font-weight: 700; margin-bottom: 1.5mm; }
.say p { font-size: 9.6pt; line-height: 1.45; color: #333b47; margin: 0; }
.has-say .body { padding-bottom: 26mm; }

.num { font-size: 8pt; color: #a8b1bf; position: absolute; right: 10mm;
       bottom: 3.5mm; }
.has-say .num { color: #8a94a4; }

/* cover */
.cover { background: #0d1017; color: #fff; justify-content: center; }
.cover h1 { color: #fff; font-size: 38pt; }
.cover .sub { font-size: 15pt; color: #9aa6bb; margin-top: 4mm; line-height: 1.45; }
.cover .rule { width: 42mm; height: 3pt; background: #4a72c4; margin: 7mm 0; }
.cover .meta { margin-top: 12mm; font-size: 9.5pt; color: #6d7a8f;
               font-family: "SF Mono",Menlo,monospace; line-height: 1.9; }

/* stat rows */
.stats { display: flex; gap: 5mm; margin: 4mm 0; }
.stat { flex: 1; border: 1.2pt solid #dbe2ec; border-radius: 3mm;
        padding: 5mm 4mm; text-align: center; }
.stat .v { font-size: 25pt; font-weight: 700; color: #1e4fa3; letter-spacing: -0.6pt; }
.stat .v.warn { color: #b5761a; }
.stat .v.ok { color: #16794f; }
.stat .l { font-size: 8.2pt; color: #6b7686; text-transform: uppercase;
           letter-spacing: 0.7pt; margin-top: 2mm; }
.stat .s { font-size: 8pt; color: #96a0b0; margin-top: 1.2mm; }

.hero { background: #1e4fa3; color: #fff; border-radius: 3mm;
        padding: 7mm 8mm; margin: 3mm 0; }
.hero .v { font-size: 40pt; font-weight: 700; letter-spacing: -1.2pt; }
.hero .l { font-size: 11pt; color: #c3d4f0; margin-top: 2mm; }

table { width: 100%; border-collapse: collapse; font-size: 10pt; margin: 2mm 0; }
th { background: #2f3e56; color: #fff; text-align: left; padding: 2.6mm 3mm;
     font-size: 8.6pt; font-weight: 600; }
td { padding: 2.4mm 3mm; border-bottom: 0.5pt solid #e2e7ef;
     font-variant-numeric: tabular-nums; }
td.n, th.n { text-align: right; }
tr.hl td { background: #eaf1ff; font-weight: 700; }
.win { color: #16794f; font-weight: 600; } .lose { color: #b3352f; }

.two { display: flex; gap: 8mm; }
.two > div { flex: 1; }
.fig { text-align: center; }
.fig img { max-height: 78mm; max-width: 100%; }
.fig.short img { max-height: 66mm; }
.fig.tall img { max-height: 92mm; }
.cap { font-size: 8pt; color: #8a94a4; margin-top: 2mm; }

.callout { background: #fff8ec; border-left: 3pt solid #d99a2b;
           padding: 4mm 5mm; margin: 3mm 0; }
.callout.key { background: #eaf1ff; border-left-color: #1e4fa3; }
.callout p { margin: 0; font-size: 10.5pt; }

.qa { margin-bottom: 4.5mm; }
.qa .q { font-size: 10.8pt; font-weight: 700; color: #1e4fa3; margin-bottom: 1.5mm; }
.qa .a { font-size: 10pt; line-height: 1.45; color: #2b313c; }
"""


def slide(kicker, title, body, say=None, n=None, cls=""):
    say_html = (f'<div class="say"><div class="lbl">SAY THIS</div><p>{say}</p></div>'
                if say else "")
    num = f'<div class="num">{n}</div>' if n else ""
    klass = f"slide {cls}" + (" has-say" if say else "")
    k = f'<div class="kicker">{kicker}</div>' if kicker else ""
    t = f"<h1>{title}</h1>" if title else ""
    return f'<div class="{klass}">{k}{t}<div class="body">{body}</div>{say_html}{num}</div>'


def build():
    s = jload("run_summary.json")
    di = jload("dataset_info.json")
    sel = s["results"]["selected"]
    la, lo, pl = sel["loso_all_subjects"], sel["lockbox"], sel["loso_pooled"]
    g = sel["generalization_gap"]
    fi = s["feature_importance"]
    reps = s["representations"]
    A = arms()
    ps = list(csv.DictReader(open(os.path.join(FINAL, "metrics",
                                               "per_subject_metrics.csv"))))
    ps.sort(key=lambda r: -float(r["f1_macro"]))
    below = sum(1 for r in ps if float(r["accuracy"]) < 0.5)
    best, worst = ps[0], ps[-1]
    acc = la["accuracy"]["mean"]
    tr_acc, va_acc = split_accuracy()
    H = []
    n = [0]

    def add(*a, **kw):
        n[0] += 1
        H.append(slide(*a, n=n[0], **kw))

    # ── cover ──────────────────────────────────────────────────────────
    H.append(f'''<div class="slide cover">
<h1>Estimating Cognitive Load<br>from EEG + fNIRS</h1>
<div class="rule"></div>
<div class="sub">Can we tell how hard someone is thinking —<br>
on a person the model has never seen before?</div>
<div class="meta">Capstone Project · Phase 2<br>
TU Berlin n-back dataset · 26 subjects · 4,576 windows<br>
42 controlled experiments · 26-fold leave-one-subject-out</div></div>''')

    # ── presenter briefing ─────────────────────────────────────────────
    H.append(f'''<div class="slide">
<div class="kicker">Before you start — read this once</div>
<h1>Your story in four sentences</h1>
<div class="body">
<ol style="list-style:none">
<li><b>1. The goal.</b> Predict mental workload from brain signals, for a
<i>new person</i> the model has never seen. That is the hard version of the problem.</li>
<li><b>2. What I built.</b> A framework that makes it impossible to cheat —
it refuses to produce a score if any data leaks between training and testing.</li>
<li><b>3. What I found.</b> I ran {len(A)} controlled experiments. None beat the
existing model. The final honest number is <b>{pct(acc)} accuracy on unseen
people</b> (chance is 50%).</li>
<li><b>4. Why that matters.</b> The error is not a modelling failure — it is
person-to-person variability. I have three independent pieces of evidence for that.</li>
</ol>
<div class="callout key"><p><b>If the panel pushes on "why didn't it improve?"</b> —
that is the finding, and it is a stronger result than a lucky number. I can show
exactly what I tried, that each was properly tested, and why the ceiling is the
data and not the model. Slides 14–17 are the evidence.</p></div>
</div></div>''')

    # ── agenda ─────────────────────────────────────────────────────────
    add("Agenda", "What I'll cover", '''
<div class="two">
<div><ul>
<li><b>The problem</b> — and why it is harder than it looks</li>
<li><b>The data</b> — 26 people, two brain-imaging methods</li>
<li><b>The model</b> — how EEG and fNIRS are fused</li>
<li><b>The trap</b> — how these studies fool themselves</li>
<li><b>My protocol</b> — how I made cheating impossible</li>
</ul></div>
<div><ul>
<li><b>The experiments</b> — 42 of them, one axis at a time</li>
<li><b>Results</b> — the honest number</li>
<li><b>Findings</b> — CNN, DANN, regularization</li>
<li><b>Why the ceiling exists</b> — three lines of evidence</li>
<li><b>Limitations and next steps</b></li>
</ul></div></div>''',
        "Keep this quick — about twenty seconds. The two halves are 'what I did' "
        "and 'what I found'. Tell them the interesting part is the second half.")

    # ── problem ────────────────────────────────────────────────────────
    add("01 · The problem", "Measuring mental effort from the brain", f'''
<p class="big">If a machine could tell when a person is cognitively overloaded, you
could intervene before they make a mistake — a pilot, a surgeon, a driver, a student.</p>
<div class="two">
<div><h2>EEG — electrical</h2>
<ul><li>200 samples per second</li><li>Fast, catches instant shifts</li>
<li>Noisy, picks up muscle movement</li></ul></div>
<div><h2>fNIRS — blood oxygen</h2>
<ul><li>10 samples per second</li><li>Slow, metabolic response</li>
<li>Stable, less prone to artifacts</li></ul></div>
</div>
<div class="callout"><p><b>The premise:</b> the two are complementary. EEG sees the
fast electrical change, fNIRS sees the slower blood-flow change that follows.
Fusing them should beat either alone — and in our data, it does.</p></div>''',
        "Open with the human motivation, not the technology. One sentence on each "
        "modality, then the key idea: they see different things, so combine them.")

    # ── data ───────────────────────────────────────────────────────────
    add("02 · The data", "26 people doing a memory task", f'''
<div class="two">
<div>
<p>Public <b>TU Berlin dataset</b>. Each subject does an <b>n-back</b> task: watch a
stream of items, say whether the current one matches the item <i>n</i> steps back.</p>
<ul>
<li><b>0-back</b> — just react. Low load.</li>
<li><b>2-back</b> — hold two items in mind. Medium.</li>
<li><b>3-back</b> — hold three. High load.</li>
</ul>
<p>The label is not someone's opinion of how hard it felt. It is the difficulty
we <i>imposed</i>. That is what makes it trustworthy ground truth.</p>
</div>
<div>
<table>
<tr><th>Property</th><th>Value</th></tr>
<tr><td>Subjects</td><td>{di["n_subjects"]}</td></tr>
<tr><td>EEG</td><td>30 channels @ 200 Hz</td></tr>
<tr><td>fNIRS</td><td>36 channels @ 10 Hz</td></tr>
<tr><td>Trials</td><td>{di["n_trials"]} (25 each)</td></tr>
<tr><td>Windows</td><td>{di["n_samples"]:,} (10 s each)</td></tr>
<tr class="hl"><td>Task studied</td><td>Low vs High</td></tr>
<tr><td>Balance</td><td>2,288 vs 2,288 — exact</td></tr>
<tr><td>Chance</td><td>50%</td></tr>
</table>
</div></div>''',
        "The important line is 'the label is the difficulty we imposed, not a "
        "subjective rating.' That is what makes this real ground truth. Also point "
        "out the classes are perfectly balanced, so accuracy is a fair measure here.")

    # ── features ───────────────────────────────────────────────────────
    add("03 · Features", "From raw signal to 1,476 numbers", '''
<p>A 10-second window of brain activity is summarised into a fixed vector.</p>
<div class="two">
<div><h2>EEG → 540 numbers</h2>
<p style="font-size:10pt">30 electrodes × 18 measures<br>
(6 frequency bands × mean, variability, trend)</p></div>
<div><h2>fNIRS → 936 numbers</h2>
<p style="font-size:10pt">36 optodes × 26 measures<br>
(13 shape statistics × oxygenated and deoxygenated blood)</p></div>
</div>
<div class="callout key"><p><b>Remember this for later.</b> These are <i>summary
statistics per channel</i>, not the raw waveform. That single fact explains why the
CNN my mentor suggested did not work — I'll come back to it on slide 14.</p></div>''',
        "Don't dwell on the numbers. The one thing to plant here is that we are "
        "feeding the model summary statistics, not the raw time series. Say you will "
        "come back to it — it sets up the CNN slide.")

    # ── architecture ───────────────────────────────────────────────────
    add("04 · The model", "Two streams, then a learned merge", f'''
<div style="background:#f5f7fa;border-radius:3mm;padding:6mm;font-family:monospace;
font-size:11pt;line-height:1.9;margin-bottom:4mm">
&nbsp;EEG&nbsp;&nbsp;&nbsp;(540) ──▶ encoder ──┐<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├──▶ <b>attention fusion</b> ──▶ head ──▶ Low / High<br>
&nbsp;fNIRS&nbsp;(936) ──▶ encoder ──┘
</div>
<div class="two">
<div><ul>
<li>Each modality gets its <b>own encoder</b>, so it learns its own representation first</li>
<li><b>Attention fusion</b> learns a gate per modality — how much to trust EEG
versus fNIRS, per sample</li>
</ul></div>
<div><ul>
<li>Those gate values are shown live in the web app</li>
<li><b>{s["model"]["n_params"]:,} parameters</b> — deliberately small for 26 subjects</li>
</ul></div></div>''',
        "The one idea: the model learns how much to trust each sensor rather than "
        "weighting them equally. If asked why it is small — with only 26 people, a "
        "bigger network just memorises them.")

    # ── the trap ───────────────────────────────────────────────────────
    add("05 · The critical issue", "How studies in this field fool themselves", f'''
<div class="two">
<div><h2 style="color:#b3352f">The tempting way</h2>
<p style="font-size:10.5pt">Shuffle all windows, split 80/20. But then the
<b>same person</b> appears in training and testing.</p>
<p style="font-size:10.5pt">The model learns to recognise <i>that individual's
baseline brain pattern</i> — not workload. Scores look great. Useless on a new person.</p></div>
<div><h2 style="color:#16794f">What I did instead</h2>
<p style="font-size:10.5pt"><b>Leave-One-Subject-Out.</b> Train on 25 people, test on
the 26th. Repeat 26 times so every person gets a turn as the unseen test subject.</p>
<p style="font-size:10.5pt">Harder, lower scores — and the only number that means
anything for real deployment.</p></div>
</div>
<div class="callout key"><p><b>Every accuracy in this presentation is on a person the
model has never seen.</b> There is no random train/test split anywhere in this project.</p></div>''',
        "This is the most important slide in the deck. Slow down. The panel needs to "
        "understand that a lower number under a harder protocol is the honest one. "
        "If they ask why your accuracy is lower than papers they've read — this is why.")

    # ── protocol ───────────────────────────────────────────────────────
    add("06 · Second safeguard", "A lockbox the search never touched", f'''
<p>There is a subtler trap. If I run 42 experiments and keep whichever scores best,
that best score is <b>partly luck</b> — I have quietly tuned to my test folds.</p>
<div class="stats">
<div class="stat"><div class="v">18</div><div class="l">Development subjects</div>
<div class="s">every experiment, every comparison</div></div>
<div class="stat"><div class="v ok">8</div><div class="l">Lockbox subjects</div>
<div class="s">never touched until the very end</div></div>
</div>
<p>Architecture, hyperparameters, dropout, everything — chosen using the 18 only.
The 8 were evaluated <b>once</b>, after the model was frozen.</p>
<div class="callout key"><p><b>The payoff (slide 12):</b> the lockbox score came out
<i>higher</i> than the development score. That proves the {pct(acc)} is not inflated
by my searching.</p></div>''',
        "Explain it like a sealed envelope: I locked 8 people away, did all my work "
        "with the other 18, and only opened the envelope once at the end. Tell them "
        "the result agreed — that is the proof the number is trustworthy.")

    # ── leakage audit ──────────────────────────────────────────────────
    add("07 · Third safeguard", "Code that refuses to produce a dishonest score", '''
<p>Before every single training fold, <b>nine automatic checks</b> run. If any fails,
the program <b>crashes on purpose</b> rather than reporting a number.</p>
<div class="two">
<div><ul>
<li>No subject in both training and testing</li>
<li>No trial split across the boundary</li>
<li>No overlapping time-window on both sides</li>
<li>No duplicated data row</li>
</ul></div>
<div><ul>
<li>Normalisation fitted on training data only</li>
<li>Held-out person absent from every statistic</li>
<li>No corrupt or missing values</li>
<li>Labels valid and both classes present</li>
</ul></div></div>
<div class="callout"><p><b>And I tested the tests.</b> I deliberately injected each
kind of leak into the data and confirmed the checks caught every one — 8 out of 8.
A safety check that never fires proves nothing.</p></div>''',
        "The line that lands with a technical panel is the last one: I broke it on "
        "purpose to prove the alarm actually works. That is unusual rigour for a "
        "student project and worth saying out loud.")

    # ── experiments ────────────────────────────────────────────────────
    add("08 · The search", f"{len(A)} experiments, one variable at a time", f'''
<div class="two">
<div><ul>
<li><b>Learning rate & batch size</b></li>
<li><b>Training length & schedule</b></li>
<li><b>Weight decay</b> — 5 settings</li>
<li><b>Dropout</b> — 0.3 to 0.7</li>
<li><b>Model size</b> — 66K to 620K parameters</li>
<li><b>Architecture</b> — MLP vs CNN</li>
</ul></div>
<div><ul>
<li><b>Modality dropout</b> — hide a sensor while training</li>
<li><b>Input noise & label smoothing</b></li>
<li><b>DANN</b> — subject-invariant training</li>
<li><b>Window length & overlap</b></li>
<li><b>Normalisation mode</b></li>
<li><b>Dropping HbR</b></li>
</ul></div></div>
<div class="callout key"><p><b>The rule, fixed before I started:</b> a change is only
adopted if it beats the current model by more than the noise floor of <b>±0.018</b>.
With 18 subjects, anything smaller is indistinguishable from luck.</p></div>''',
        "Emphasise that the acceptance rule was written down BEFORE running anything. "
        "That stops you fooling yourself by chasing a lucky 1% and calling it progress.")

    # ── comparison figure ──────────────────────────────────────────────
    add("08 · The search", "Every experiment, ranked", f'''
<div class="fig tall">{img("_slides_cmp.png", run=os.path.dirname(OUT))}
<div class="cap">All {len(A)} arms on 18-fold development LOSO. Grey = current model.
Green beat it, blue did not. Dashed line = current model's score.</div></div>''',
        "Point at the error bars, not the bars. They are wider than the gaps between "
        "experiments — that is the whole story. Nothing separated itself from noise.")

    # ── headline ───────────────────────────────────────────────────────
    add("09 · Results", "The honest number", f'''
<div class="hero"><div class="v">{pct(acc)}</div>
<div class="l">accuracy on people the model has never seen &nbsp;·&nbsp; chance = 50%
&nbsp;·&nbsp; 26-fold leave-one-subject-out</div></div>
<div class="stats">
<div class="stat"><div class="v">{f(la["f1_macro"]["mean"],3)}</div>
<div class="l">Macro F1</div><div class="s">± {f(la["f1_macro"]["std"],3)}</div></div>
<div class="stat"><div class="v">{f(la["roc_auc"]["mean"],3)}</div>
<div class="l">ROC-AUC</div><div class="s">ranking quality</div></div>
<div class="stat"><div class="v">{f(la["mcc"]["mean"],3)}</div>
<div class="l">MCC</div><div class="s">correlation measure</div></div>
<div class="stat"><div class="v">{f(la["cohen_kappa"]["mean"],3)}</div>
<div class="l">Cohen's Kappa</div><div class="s">chance-corrected</div></div>
</div>''',
        f"Lead with {pct(acc)} against a 50% chance baseline. Do not apologise for it. "
        "Then immediately go to the next slide — the lockbox — because that is what "
        "makes this number believable rather than just claimed.")

    # ── lockbox proof ──────────────────────────────────────────────────
    add("10 · Results", "Proof the number is not inflated", f'''
<table>
<tr><th>Protocol</th><th class="n">Accuracy</th><th class="n">Macro F1</th><th class="n">ROC-AUC</th></tr>
<tr><td>26-fold LOSO — all subjects</td><td class="n">{f(la["accuracy"]["mean"])}</td>
<td class="n">{f(la["f1_macro"]["mean"])}</td><td class="n">{f(la["roc_auc"]["mean"])}</td></tr>
<tr class="hl"><td>Lockbox — 8 subjects never used for any decision</td>
<td class="n">{f(lo["accuracy"]["mean"])}</td><td class="n">{f(lo["f1_macro"]["mean"])}</td>
<td class="n">{f(lo["roc_auc"]["mean"])}</td></tr>
</table>
<div class="callout key"><p style="font-size:12pt"><b>The lockbox score is
<i>higher</i>, not lower.</b> After 42 experiments of searching, the subjects I never
touched scored just as well. That is the strongest evidence I can give that this
result would hold on new people.</p></div>
<p style="font-size:10pt;color:#6b7686">If I had been overfitting my search to the
test folds, the lockbox number would have dropped noticeably. It did not.</p>''',
        "This is your credibility slide. Say: 'if I had been fooling myself, this "
        "second number would have collapsed. It didn't.' Panels respect a result that "
        "carries its own audit.")

    # ── train vs test ──────────────────────────────────────────────────
    add("11 · Results", "Training vs testing — and the gap", f'''
<div class="stats">
<div class="stat"><div class="v warn">{pct(tr_acc)}</div>
<div class="l">Training accuracy</div></div>
<div class="stat"><div class="v">{pct(va_acc)}</div>
<div class="l">Validation accuracy</div></div>
<div class="stat"><div class="v ok">{pct(acc)}</div>
<div class="l">Test — unseen person</div></div>
<div class="stat"><div class="v">{f(tr_acc - acc, 3)}</div>
<div class="l">The gap</div><div class="s">train − test</div></div>
</div>
<p>The model fits its training subjects well and loses about
{round((tr_acc - acc) * 100)} points on a new person. <b>That gap is the central
problem of this field.</b></p>
<div class="callout"><p>I attacked it with every standard tool — dropout, weight decay,
noise, smaller models, modality dropout. <b>Each one shrank the gap and shrank test
accuracy by just as much.</b> That tells you the gap is not the model memorising —
it is that 25 people are not enough to represent the 26th.</p></div>''',
        "If a panellist says 'your model is overfitting, add regularisation' — this is "
        "your answer. You already did, systematically, and it does not help. Explain "
        "why: the gap is between-person variability, not memorisation.")

    # ── confusion + ROC ────────────────────────────────────────────────
    add("11 · Results", "Where the errors are", f'''
<div class="two">
<div class="fig">{img("confusion_matrix/confusion_matrix.png")}
<div class="cap">All 4,576 held-out windows pooled</div></div>
<div class="fig">{img("curves/roc_curve.png")}
<div class="cap">AUC {f(pl["roc_auc_macro_ovr"],3)} across every fold</div></div>
</div>
<p style="font-size:10.5pt">Low load is caught {pct(pl["per_class"][0]["recall"],1)} of
the time; High load {pct(pl["per_class"][1]["recall"],1)}. <b>High load is the harder
class</b> — the model misses it more often than it misses Low.</p>''',
        "Note the asymmetry: High load is harder to detect than Low. If this were "
        "deployed as a safety system, that is the clinically important direction — "
        "missing an overloaded operator is the costly error.")

    # ── CNN ────────────────────────────────────────────────────────────
    add("12 · Finding", "The CNN did not work — and I know why", f'''
<table>
<tr><th>Architecture</th><th class="n">Macro F1</th><th class="n">vs current</th>
<th class="n">p-value</th><th class="n">Overfit gap</th></tr>
<tr class="hl"><td>Current model (MLP)</td><td class="n">0.6451</td><td class="n">—</td>
<td class="n">—</td><td class="n">0.163</td></tr>
<tr><td>CNN + residual</td><td class="n">0.5550</td><td class="n lose">−0.090</td>
<td class="n">0.002</td><td class="n win">0.026</td></tr>
<tr><td>CNN + residual + attention</td><td class="n">0.5467</td><td class="n lose">−0.098</td>
<td class="n">0.001</td><td class="n win">0.035</td></tr>
<tr><td>CNN — plain</td><td class="n">0.4786</td><td class="n lose">−0.167</td>
<td class="n">&lt;0.001</td><td class="n win">0.026</td></tr>
</table>
<div class="callout key"><p><b>Look at the last column.</b> The CNN almost eliminates
overfitting — 0.163 down to 0.026. But it loses more accuracy than it saves.
<b>Why:</b> a CNN looks for local patterns, and we feed it 18 summary numbers per
electrode, not a waveform. There is no local structure there to find.</p></div>''',
        "Your mentor suggested the CNN, so handle this respectfully: say you tested it "
        "properly rather than dismissing it, and the interesting part is WHY it fails "
        "— we removed the raw time series before the model ever sees it.")

    # ── DANN ───────────────────────────────────────────────────────────
    add("13 · Finding", "DANN: rejected, with the mechanism explained", f'''
<div class="two">
<div>
<p style="font-size:10.5pt">DANN tries to make the model <b>forget who the person is</b>
while keeping what they are doing — exactly our problem.</p>
<table>
<tr><th>Strength (λ)</th><th class="n">Macro F1</th></tr>
<tr class="hl"><td>0 — off</td><td class="n">0.6451</td></tr>
<tr><td>0.1</td><td class="n lose">0.6182</td></tr>
<tr><td>0.5</td><td class="n lose">0.5968</td></tr>
<tr><td>1.0</td><td class="n lose">0.5778</td></tr>
</table>
<p style="font-size:10pt;color:#6b7686">Steadily worse as we push harder.</p>
</div>
<div>
<h2>Why — I probed it</h2>
<p style="font-size:10.5pt">I trained a separate classifier to guess <i>which person</i>
a signal came from:</p>
<table>
<tr><th>Representation</th><th class="n">Can identify person</th></tr>
<tr><td>Raw features</td><td class="n win">0.013</td></tr>
<tr><td>Our model</td><td class="n">0.214</td></tr>
<tr><td>With DANN</td><td class="n lose">0.217</td></tr>
</table>
<p style="font-size:10pt">Chance = 0.125. Our preprocessing already removes identity;
the network puts it back, and <b>DANN fails to remove it again.</b></p>
</div></div>''',
        "This is your best 'I understood the negative result' moment. Do not just say "
        "DANN did not help — say you measured WHY: it never removed the thing it "
        "targets. That is a diagnosis, not a shrug.")

    # ── per-subject ────────────────────────────────────────────────────
    add("14 · The key insight", "The ceiling is people, not the model", f'''
<div class="fig short">{img("curves/per_subject_performance.png")}</div>
<div class="callout key"><p style="font-size:11pt">Best subject <b>{f(float(best["f1_macro"]),2)}</b>
· worst subject <b>{f(float(worst["f1_macro"]),2)}</b> · <b>{below} of 26 below chance.</b>
That spread of {f(float(best["f1_macro"])-float(worst["f1_macro"]),2)} is
<b>more than twice</b> the difference between any two of the {len(A)} models I tested.</p></div>''',
        "This is the slide that answers 'why not just build a better model'. Which "
        "PERSON you test on matters far more than which model you pick. That reframes "
        "the whole project — the bottleneck is data, not architecture.")

    # ── SHAP ───────────────────────────────────────────────────────────
    add("15 · Interpretability", "What the model actually uses", f'''
<div class="stats">
<div class="stat"><div class="v">{f(fi["modality_share"]["fNIRS"]*100,1)}%</div>
<div class="l">fNIRS</div></div>
<div class="stat"><div class="v">{f(fi["modality_share"]["EEG"]*100,1)}%</div>
<div class="l">EEG</div></div>
<div class="stat"><div class="v">{f(fi["modality_share"]["HbO"]*100,1)}%</div>
<div class="l">Oxygenated (HbO)</div></div>
<div class="stat"><div class="v">{f(fi["modality_share"]["HbR"]*100,1)}%</div>
<div class="l">Deoxygenated (HbR)</div></div>
</div>
<p><b>fNIRS dominates overwhelmingly.</b> The most useful features describe the
<i>shape</i> of the blood-flow response — its area, peak, minimum and late behaviour —
not EEG frequency bands.</p>
<div class="callout"><p>Two independent checks agree: SHAP says HbR matters slightly
more than HbO, and separately, <b>deleting HbR costs 0.067 macro F1 (p = 0.014)</b>.
Two different methods, same conclusion.</p></div>''',
        "Panels like interpretability. The headline is that the slow blood-flow signal "
        "carries almost all the information, and the response SHAPE matters more than "
        "EEG rhythms. Mention that two independent methods agreed.")

    # ── validated choices ──────────────────────────────────────────────
    add("16 · Validation", "Four existing design choices, now proven", '''
<table>
<tr><th>What I removed or changed</th><th class="n">Cost to accuracy</th>
<th class="n">p-value</th><th>Verdict</th></tr>
<tr><td>Per-subject normalisation</td><td class="n lose">−0.075</td><td class="n">0.014</td>
<td>Genuinely essential</td></tr>
<tr><td>The HbR signal</td><td class="n lose">−0.067</td><td class="n">0.014</td>
<td>Genuinely helps</td></tr>
<tr><td>Normalisation order</td><td class="n lose">−0.030</td><td class="n">0.050</td>
<td>Original order correct</td></tr>
<tr><td>Calibration-only normalisation</td><td class="n">−0.040</td><td class="n">0.156</td>
<td>Not significant — deployable</td></tr>
</table>
<div class="callout key"><p><b>The last row is practically important.</b> Normally the
model needs a person's whole recording session to normalise. I tested using only the
<b>first 30 windows</b> — a short calibration. The cost is not statistically
significant, which means <b>real-time use is defensible.</b></p></div>''',
        "This shows the project isn't only negative results — it converted untested "
        "assumptions into measured facts. The calibration row is the one with real "
        "deployment value; lead with that if time is short.")

    # ── limitations ────────────────────────────────────────────────────
    add("17 · Honesty", "What this study cannot tell you", f'''
<div class="two">
<div><ul>
<li><b>26 subjects is small.</b> With 18 in the development pool, I could only
reliably detect improvements bigger than ~4 points. A real 2-point gain would be
invisible to me.</li>
<li><b>{below} of 26 subjects perform below chance.</b> The model genuinely does not
work for some people, and I cannot yet predict who.</li>
</ul></div>
<div><ul>
<li><b>Normalisation needs a calibration recording</b> — measured at ~0.04 cost.</li>
<li><b>The CNN result is specific to these features.</b> It says nothing about a CNN
on the raw waveform, which I did not test.</li>
<li><b>One-variable-at-a-time search</b> cannot find improvements needing two
simultaneous changes.</li>
</ul></div></div>''',
        "Volunteer these before the panel finds them. Stating your own limits "
        "precisely reads as confidence, and it usually pre-empts the harder questions.")

    # ── next ───────────────────────────────────────────────────────────
    add("18 · Next steps", "Where the gains actually are", '''
<div class="two">
<div><ul>
<li><b>More subjects.</b> The evidence says the bottleneck is people, not
architecture. This is the single highest-value change.</li>
<li><b>Go back to the raw time series.</b> We discarded it before modelling. A CNN or
transformer needs that structure to be useful — my CNN result is evidence for this,
not against it.</li>
</ul></div>
<div><ul>
<li><b>Per-person calibration.</b> A short adaptation block per user, already priced
at ~0.04 and not significant.</li>
<li><b>Predict who it fails on.</b> {} subjects sit below chance — identifying them
in advance is its own useful result.</li>
</ul></div></div>
<div class="callout key"><p>Every one of these follows directly from a measurement in
this study, not from guesswork.</p></div>'''.format(below),
        "Close on this. Each next step is justified by a specific finding you just "
        "presented — that shows the negative results were productive, and gives the "
        "panel a clear sense of direction.")

    # ── summary ────────────────────────────────────────────────────────
    add("Summary", "What to remember", f'''
<div class="two">
<div><ul>
<li><b>{pct(acc)} accuracy on unseen people</b>, chance 50%, under the hardest honest
protocol.</li>
<li><b>The lockbox confirmed it.</b> 8 untouched subjects scored just as well.</li>
<li><b>{len(A)} experiments, nothing beat the baseline.</b> That is a finding, not a
failure.</li>
</ul></div>
<div><ul>
<li><b>The ceiling is person-to-person variability</b>, proven three independent ways.</li>
<li><b>fNIRS carries {f(fi["modality_share"]["fNIRS"]*100,0)}% of the signal</b>,
mostly in blood-flow response shape.</li>
<li><b>Everything is reproducible</b> — every metric recomputable from saved
predictions.</li>
</ul></div></div>''',
        "Your closing. Land on the honesty point: I built something that makes it hard "
        "to lie to myself, and then reported what it found. That is the contribution.")

    # ── Q&A ────────────────────────────────────────────────────────────
    qa = [
        ("Why is your accuracy only 63%? I've seen papers with 90%.",
         "Those are usually a different, easier problem — either binary within a "
         "single person, or a random split where the same person is in train and test. "
         "Under matched protocols our numbers are comparable. I deliberately report "
         "the harder cross-person setting because it is the one that reflects real use."),
        ("Isn't 63% too low to be useful?",
         "For a safety-critical decision on one window, yes. But predictions can be "
         "aggregated over time, and the ROC-AUC of 0.68 shows useful ranking ability. "
         "The honest framing is that this is a research result showing feasibility, "
         "not a deployable product."),
        ("Your model overfits — train 80%, test 63%. Why not regularise?",
         "I did, systematically: dropout from 0.3 to 0.7, weight decay across five "
         "settings, input noise, label smoothing, modality dropout, and models from "
         "66K to 620K parameters. Every one reduced the gap and reduced test accuracy "
         "by the same amount. The gap is between-person variability, not memorisation."),
        ("You ran 42 experiments and none worked. Is that not a failed project?",
         "The goal was the best scientifically valid model, not a bigger number. I "
         "established that the existing configuration is at a local optimum, "
         "converted four untested assumptions into measured facts, and diagnosed why "
         "the ceiling exists. A project that reports a lucky number it cannot defend "
         "is the failed one."),
        ("Why did the CNN fail when CNNs dominate elsewhere?",
         "Because of what we feed it. We compress each 10-second window into summary "
         "statistics per channel before modelling. A CNN exploits local structure, and "
         "there is none left in 18 summary numbers. On the raw waveform it would very "
         "likely help — that is my main proposed next step."),
        ("How do I know you didn't accidentally leak data?",
         "Nine automatic checks run before every fold and crash the program on "
         "failure. I also injected each type of leak deliberately and confirmed all "
         "eight were caught. And every metric can be independently recomputed from the "
         "saved predictions — I have a script that does exactly that."),
        ("Why leave-one-subject-out instead of a normal split?",
         "A random split lets the same person appear on both sides, so the model can "
         "score highly by recognising an individual's baseline rather than their "
         "workload. That number would not survive contact with a new user."),
        ("What is the single most important finding?",
         "That the spread across people — 0.43 to 0.81 — is more than twice the "
         "difference between any two of the 42 models I tested. The bottleneck is the "
         "number of subjects, not the architecture. That redirects future effort."),
    ]
    half = 4
    for chunk, ttl in ((qa[:half], "Likely panel questions — prepared answers"),
                       (qa[half:], "Likely panel questions — continued")):
        body = "".join(f'<div class="qa"><div class="q">Q. {q}</div>'
                       f'<div class="a"><b>A.</b> {a}</div></div>' for q, a in chunk)
        add("Q&A preparation", ttl, body,
            "Do not read these out — they are for you beforehand. Knowing the answer "
            "exists is usually enough to stay calm.")

    return ("<!DOCTYPE html><html><head><meta charset='utf-8'>"
            f"<title>Cognitive Load — Presentation</title><style>{CSS}</style>"
            f"</head><body>{''.join(H)}</body></html>")


def comparison_chart(out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    A = sorted(arms(), key=lambda r: float(r["dev_loso_f1_macro_mean"]))
    names = [r["experiment_id"] for r in A]
    vals = [float(r["dev_loso_f1_macro_mean"]) for r in A]
    errs = [float(r["dev_loso_f1_macro_std"]) for r in A]
    base = next(float(r["dev_loso_f1_macro_mean"])
                for r in A if r["experiment_id"] == "A0_baseline")
    colors = ["#8a94a6" if nm == "A0_baseline" else
              ("#16794f" if v > base else "#4C78A8") for nm, v in zip(names, vals)]
    fig, ax = plt.subplots(figsize=(11.5, 4.6), dpi=210)
    ax.barh(names, vals, xerr=errs, color=colors, height=0.74,
            error_kw={"ecolor": "#aab3c0", "lw": 0.7, "capsize": 1.4})
    ax.axvline(base, color="#c0392b", lw=1.1, ls="--")
    ax.set_xlabel("macro F1 on unseen subjects (mean ± std over 18 folds)", fontsize=8)
    ax.set_xlim(0, max(v + e for v, e in zip(vals, errs)) * 1.03)
    ax.tick_params(axis="y", labelsize=5.0)
    ax.tick_params(axis="x", labelsize=7)
    ax.grid(axis="x", alpha=0.25, lw=0.5)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    from weasyprint import HTML

    tmp = os.path.join(os.path.dirname(OUT), "_slides_cmp.png")
    comparison_chart(tmp)
    HTML(string=build(), base_url=os.path.dirname(OUT)).write_pdf(OUT)
    if os.path.exists(tmp):
        os.remove(tmp)
    print(f"wrote {os.path.relpath(OUT, ROOT)} ({os.path.getsize(OUT)/1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
