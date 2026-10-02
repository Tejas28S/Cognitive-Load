# Adding a new dataset

How to make the pipeline read a dataset it has never seen. The design goal:
**a dataset whose naming is a variation on what the alias registry already
covers needs no code change** — only config edits under `schema/`.

```
raw file  ->  DatasetParser  ->  SchemaNormalizer  ->  ChannelProjector  ->  [existing feature extraction]  ->  [existing frozen model]
              (reads it)        (names -> canonical)   (channels -> bundle)      UNCHANGED                       UNCHANGED
```

---

## Quick start

```bash
# What the ingestion layer already knows about
python predict.py --list-datasets

# Predict from any supported file (repeat --input for multi-modality recordings)
python predict.py --input path/to/data.csv
python predict.py --input eeg.mat --input nirs.mat --max-windows 50
python predict.py --input data.csv --version 3class --json result.json
```

Exit codes: `0` prediction produced, `2` refused (input could not be mapped
confidently), `1` error.

Via the API (additive — the existing endpoints are unchanged):

```bash
curl -X POST "http://localhost:8000/api/predict/universal" \
     -F "files=@eeg.mat" -F "files=@nirs.mat"
curl "http://localhost:8000/api/predict/universal/datasets"
```

---

## Step 1 — Inspect the raw columns

```python
from ingestion import DatasetParser
r = DatasetParser().parse("path/to/your_file.csv")   # or .mat / .edf / .npz

print(r.kind)               # "features" (extracted table) or "signals" (raw)
print(r.detected_dataset)   # matched profile, or None
print(r.layout)             # wide / long
if r.kind == "features":
    print(list(r.dataframe.columns)[:20])
else:
    print(r.recording.summary())
```

Then see how much of it the registry already understands:

```python
from ingestion import AliasIndex
idx = AliasIndex()
for c in list(r.dataframe.columns)[:40]:
    m = idx.resolve(c)
    print(f"{c:30s} -> {m.canonical or 'UNMATCHED':15s} ({m.how})")
```

## Step 2 — Add aliases

Edit [`schema/channel_aliases.yaml`](../schema/channel_aliases.yaml). You do
**not** need entries for pure case/separator/unit variation — `EEG_Fp1`,
`eeg.fp1`, `FP1` and `Fp1_uV` all already collapse to `Fp1`. Add an entry only
for a genuinely different spelling:

```yaml
eeg_electrodes:
  T7:
    aliases: [T3, EEG T7-Ref]        # add your dataset's spelling here
    spatial_neighbours: [TP7, FT7, C5]
```

Also map the dataset's **label values**, not just its column names:

```yaml
label_values:
  my_dataset:
    description: what the conditions mean, and any assumption being made
    map:
      "easy": 0
      "hard": 2
```

## Step 3 — Add a dataset profile (raw recordings only)

Extracted feature tables usually need nothing more than aliases. Raw recordings
need a profile in [`schema/datasets/`](../schema/datasets/) describing the file
layout — copy [`stroop.yaml`](../schema/datasets/stroop.yaml) as a template. The
keys that matter:

| Key | Purpose |
|---|---|
| `detection.mat_struct_keys` / `mat_field_signature` | how the parser recognizes the file without being told |
| `eeg.fields` | where `data`, `sample_rate`, `channel_names` live inside the struct |
| `eeg.data_orientation` | `samples_by_channels` or `channels_by_samples` |
| `fnirs.needs_beer_lambert` | true if the file ships raw optical intensity rather than HbO/HbR |
| `labels.source` | `marker_file`, `eeg_events`, or a label column |
| `labels.value_map` | which `label_values` block in the alias registry to use |
| `*.resample_to_hz` | the bundle's rate, if the dataset differs |

## Step 4 — Test it

```bash
.venv/Scripts/python.exe -m pytest tests/ -v
python predict.py --input your_file.mat
```

Add a case to `tests/test_ingestion.py` guarded by a `skipif` on the data being
present, following `test_parse_stroop_eeg_autodetects`.

---

## Understanding the outcome

Every prediction carries a `schema_mapping_report` and `projection_reports`.
Read them — they say exactly how the input was interpreted.

### Coverage and verdicts

The projector reports, per modality, how much of the model's expected channel
layout your dataset actually fills. Thresholds live in
[`schema/canonical_schema.yaml`](../schema/canonical_schema.yaml) under
`projection.thresholds`:

| Verdict | Coverage | Behaviour |
|---|---|---|
| `confident` | >= 0.90 | prediction emitted normally |
| `degraded` | 0.70 - 0.90 | emitted, flagged, coverage numbers attached |
| `refuse` | < 0.70 | **no prediction** — a structured error names what is missing |

Coverage counts an exact electrode match as 1.0 and a declared spatial-neighbour
substitution as 0.5, because a substitute is genuinely less informative than the
electrode the model was trained on.

### Why refusal is a feature

The alternative to refusing is filling the missing channels with zeros and
returning a number anyway. After the per-subject z-score, 0.0 is a perfectly
legitimate value — so a zero-filled channel is **indistinguishable from real
data** to both the model and the reader. The result looks like a prediction,
carries a confidence score, and means nothing.

That is why unfillable channels are held as NaN all the way through, and why the
pipeline stops rather than laundering them into the model.

### The current honest limit: fNIRS across datasets

EEG channels can be mapped across datasets because electrode names are a shared
standard, and the model's positional `ch00..ch29` were recovered back to real
electrode names (see [`tu_berlin_assumptions.md`](tu_berlin_assumptions.md) §0).

fNIRS cannot. The serving bundle stores optode-pair indices with **no montage
metadata**, so there is no defensible way to say which of your dataset's optodes
corresponds to the model's `hbo_ch07`. The projector therefore refuses unless a
profile declares an explicit `fnirs.montage_map`, and it will not treat a
matching channel *count* as evidence of matching channel *identity*.

**Worked example — the Stroop dataset:**

| | Model expects | Stroop supplies | Result |
|---|---|---|---|
| EEG | 30 channels (TU Berlin montage) | 34 channels | 16 exact + 14 neighbour = **0.767, degraded** |
| fNIRS | 36 optode positions | 20 frontal channels, raw wavelengths | **0.0, refuse** |

So Stroop parses, normalizes and projects its EEG with no manual renaming at
all — and then correctly refuses at the fNIRS stage.

### What to do when a dataset is refused

Refusal on channel coverage means the dataset is not compatible with a model
trained on a different sensor montage. That is a real scientific limit, not a
bug in the ingestion layer. The valid route is a **serving bundle trained on
that dataset's own feature space**:

1. Extract features for the new dataset with `ingestion.features.extract_feature_frame`
   (it is channel-count agnostic — 34 EEG / 20 fNIRS channels is fine).
2. Train on that feature space using the existing training pipeline
   (`config.py` + `data_loader.py` detect column counts from the CSV).
3. Export with `export_for_serving.py` into a new bundle directory.
4. Register it in `_BUNDLES` in `backend/main.py` and in
   `feature_level.bundle_registry` in `canonical_schema.yaml`.

The serving layer is already bundle-driven (`UploadService` reads its column
list and scalers from `feature_schema.json` at runtime), so a new bundle needs
no serving-code change.

---

## Files in this layer

| File | Role |
|---|---|
| `schema/canonical_schema.yaml` | the target schema; channel order; projection thresholds |
| `schema/channel_aliases.yaml` | alias registry + label value maps (**edit this first**) |
| `schema/datasets/*.yaml` | per-dataset file-layout profiles |
| `ingestion/canonical.py` | schema loading, name matching, `CanonicalRecording` |
| `ingestion/parser.py` | format readers, dataset detection, Beer-Lambert |
| `ingestion/normalizer.py` | alias matching -> canonical names, mapping report |
| `ingestion/projection.py` | channel projection onto a bundle, coverage, verdicts |
| `ingestion/features.py` | windowing bridge (reuses `backend/mat_extractor.py` maths) |
| `ingestion/pipeline.py` | end-to-end orchestration |
| `predict.py` | CLI |
| `tests/test_ingestion.py` | the suite guarding all of the above |
