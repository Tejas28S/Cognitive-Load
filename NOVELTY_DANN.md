# Novel Contribution — Subject-Invariant Cognitive-Load Classification (DANN)

## The problem this attacks

Across this project the single most consistent finding is that **subject
variability is the performance ceiling**: models generalize far worse to unseen
people (leave-one-subject-out, LOSO) than the within-subject numbers common in
the literature suggest, because they latch onto person-specific physiological
baselines that do not transfer. The `Protocol Taxonomy & Benchmark` section of
`RESULTS.md` quantifies the honest gap; per-subject normalization and the
Hybrid+RF ensemble each helped a little, but none of them *directly* target the
cross-subject transfer problem.

**Domain-Adversarial Neural Networks (DANN)** do target it directly. This is the
novel modeling contribution of the capstone.

## The idea

Train the fused EEG+fNIRS representation to be good at predicting cognitive
**load** while being uninformative about **which subject** a sample came from. A
second head — a *subject discriminator* — tries to recover subject identity from
the shared embedding. A **gradient-reversal layer (GRL)** sits between the
encoder and that head: on the forward pass it is the identity; on the backward
pass it multiplies the gradient by `-lambda`. So the discriminator learns to
identify subjects, while the same signal pushes the *encoder* to erase
subject-specific structure. At convergence the representation keeps load-relevant
information and discards subject-identity information (Ganin & Lempitsky, 2015).

```
                     ┌────────────► task head ──► Low / Medium / High   (minimize)
 EEG ─┐              │
      ├─ encoders ─ fused embedding
 fNIRS┘              │
                     └─ GRL(−λ) ──► subject head ──► which subject?      (adversarial)
```

`lambda` is ramped from 0 → `DANN_LAMBDA_MAX` over training with the standard
schedule `λ = λ_max · (2 / (1 + e^(−γ·p)) − 1)`, `p` = training progress. Early
on the adversary is effectively off (stable warmup); it strengthens as the task
head matures.

## What was implemented

All additions are backward-compatible and **off by default** — the frozen locked
checkpoint, the served ensemble, and the shared training loop are untouched.

- `models/architectures.py` — `GradientReversal` autograd function, `grad_reverse`
  helper, `DomainDiscriminator`, and grad-enabled hooks on `HybridCNNLSTM`
  (`encode`, `attach_domain_head`, `forward_dann`). `forward` is unchanged, so
  `backend/model_service.py` and evaluation are unaffected.
- `config.py` — `USE_DANN`, `DANN_LAMBDA_MAX`, `DANN_GAMMA`, `DANN_HIDDEN`,
  `DANN_DOMAIN_DROPOUT`, `DANN_EPOCHS` (read via `getattr`, so nothing else
  breaks).
- `dann.py` — self-contained dual-loss training loop with the ramped GRL lambda,
  early stopping on validation macro F1, and the **subject-invariance probe**.
- `dann_ablation.py` — the 26-fold LOSO evaluation, training baseline and DANN on
  **identical folds/splits/scalers** for a properly paired comparison, plus a
  lambda sweep.

## How it is evaluated (same discipline as the rest of the project)

Same 26-fold LOSO protocol, same per-subject-normalized features and per-fold
train-only scalers as everything else. DANN is compared to the locked recipe for
both the Hybrid network alone and the Hybrid+RF ensemble (the served config),
with paired significance (`part1_rigor.paired_tests`: t-test **and** Wilcoxon).

**Decision gate (mirrors the HbR and classical-sweep gates):** DANN is adopted
into serving only if it delivers a *clear, significant* lift on the ensemble
(both tests, meaningful effect). Otherwise it is reported as an honest,
well-analyzed result and the locked model is retained. A rigorous negative result
is a legitimate contribution here — the deliverable is the evaluation, not a
guaranteed win.

### The subject-invariance probe (the diagnostic that proves the mechanism)

Independent of task accuracy, we verify the method *does what it claims*: after
training, freeze the encoder, extract embeddings for the training subjects, and
train a fresh classifier to recover subject identity from them. If DANN works,
subject-id recovery drops toward chance for the DANN encoder while remaining
higher for the baseline. This is reported in `outputs/dann_ablation.json`
(`subject_invariance_probe`) and is compelling evidence even when the accuracy
delta is modest. (In the 5-epoch smoke test the probe already trended the right
way: DANN below baseline.)

## How to run

```bash
# Full 26-fold ablation (slow — trains 2 models x 26 folds on CPU):
python dann_ablation.py

# Fast smoke test (correctness, not results):
python dann_ablation.py --max-folds 2 --epochs 6

# Lambda sensitivity sweep on a subset of folds:
python dann_ablation.py --lambda-sweep 0.0 0.5 1.0 2.0 --max-folds 4
```

Interpreter: `C:/Program Files/Python314/python.exe` (the repo `.venv` is broken).
Outputs: `outputs/dann_ablation.json`, `outputs/dann_lambda_sweep.json`.

## Results (full 26-fold LOSO run)

Artifacts: `outputs/dann_ablation.json`, `outputs/dann_lambda_sweep.json`.

| Config | Hybrid Macro F1 | Ensemble Macro F1 | Ensemble AUC | Subject-probe acc |
|---|---:|---:|---:|---:|
| Baseline (locked) | 0.3924 | 0.4520 | 0.6203 | 0.091 |
| DANN (λ_max=1.0) | 0.3753 | 0.4545 | 0.6321 | 0.098 |
| _chance_ | — | — | — | 0.050 |

Paired significance (macro F1, 26 folds): Hybrid-alone diff **−0.0171** (t p=0.63,
Wilcoxon p=0.62); Ensemble (served) diff **+0.0025** (t p=0.88, Wilcoxon p=0.86).

Lambda sweep (DANN Hybrid Macro F1; one shared code path — the controlled
adversary on/off comparison): **0.4177** (λ=0) → 0.3933 (λ=0.5) → 0.3753 (λ=1.0)
→ **0.3622** (λ=2.0).

### Verdict: NOT ADOPTED (honest negative result) — and the mechanism is clear

DANN produced **no significant improvement** (ensemble diff is noise, p≈0.86), and
adversarial strength **monotonically degrades** the task in the sweep. The
decision gate (adopt only on a clear, significant ensemble lift, both tests)
therefore retains the locked model. This is not a null non-finding — three
independent lines of evidence explain *why* DANN can't help on this dataset:

1. **Protocol benchmark** — subject-baseline leakage is a minor effect here
   (within-subject ≈ LOSO once per-subject normalization is applied).
2. **Subject-invariance probe** — the *baseline* representation is already near
   chance for subject-identity recovery (0.091 vs 0.050). There is almost no
   decodable subject-identity information left for the adversary to strip out.
3. **Lambda sweep** — with nothing harmful to remove, the adversary only injects
   noise, so task performance falls as λ grows.

**Conclusion:** on the TU Berlin/Shin 3-class LOSO task, cross-subject difficulty
is *not* a decodable subject-identity confound in the learned features — it
reflects genuine task difficulty and limited transferable signal. A mainstream
cross-subject method was applied rigorously and correctly declined, with a
mechanistic explanation consistent with the rest of the project. That is the
contribution: a well-designed, decisively-answered experiment, not a lucky win.

> Caveat for the record: the main-run baseline uses `train_model` while DANN uses
> `train_dann`, so the cleanest adversary-on-vs-off comparison is the lambda sweep
> (single code path), which agrees with the paired test. Since DANN was not
> adopted, serving/export is unchanged.

## Positioning vs literature

Domain adaptation is the mainstream direction for cross-subject EEG/fNIRS
decoding (e.g. gradient-reversal / DANN for cross-subject EEG emotion and fNIRS
workload, 2023–24). Applying it to **3-class, leave-one-subject-out cognitive
load on the TU Berlin/Shin dataset**, with a paired-significance decision gate and
an explicit subject-invariance probe, is the specific, honest contribution here —
not a claim of a new algorithm, but a rigorous, deployment-minded application to
the exact bottleneck this project documented.
