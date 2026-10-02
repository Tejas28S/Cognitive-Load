"""
Tests for the universal dataset ingestion layer.

The contract being protected, in order of importance:

  1. The existing TU Berlin flow produces IDENTICAL predictions to before.
     This is the regression that matters most — the ingestion layer is only
     worth having if it changes nothing for the dataset already in production.
  2. A new dataset (Stroop) is read and normalized without manual renaming.
  3. Renamed / reordered columns are recovered through the alias registry.
  4. A missing required field produces a clear, actionable error — not a
     stack trace, and never a silently-wrong prediction.
  5. Unrecognized columns are ignored but always reported.

Run:  .venv/Scripts/python.exe -m pytest tests/ -v
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "backend"))

from ingestion.canonical import (          # noqa: E402
    AliasIndex,
    IngestionError,
    load_canonical_schema,
    load_dataset_profiles,
    normalize_token,
    validate_against_bundle,
)
from ingestion.normalizer import SchemaNormalizer      # noqa: E402
from ingestion.parser import DatasetParser             # noqa: E402
from ingestion.projection import ChannelProjector      # noqa: E402

DEMO_CSV = os.path.join(ROOT, "demo_data", "upload_sample_2class.csv")
STROOP_EEG = os.path.join(ROOT, "data", "stroop_raw", "eeg_mat",
                          "Raw EEG Data In mat", "S1.mat")
STROOP_FNIRS = os.path.join(ROOT, "data", "stroop_raw", "fnirs_mat",
                            "Raw fNIRS Data In mat", "S1.mat")
TUB_EEG = os.path.join(ROOT, "data", "EEG_01-26_MATLAB", "VP001-EEG", "cnt_nback.mat")
TUB_NIRS = os.path.join(ROOT, "data", "NIRS_01-26_MATLAB", "VP001-NIRS", "cnt_nback.mat")

needs_demo = pytest.mark.skipif(not os.path.exists(DEMO_CSV),
                                reason="demo_data/ sample not present")
needs_stroop = pytest.mark.skipif(
    not (os.path.exists(STROOP_EEG) and os.path.exists(STROOP_FNIRS)),
    reason="Stroop raw data not downloaded (data/stroop_raw/)")
needs_tub_raw = pytest.mark.skipif(
    not (os.path.exists(TUB_EEG) and os.path.exists(TUB_NIRS)),
    reason="TU Berlin raw .mat data not present (data/)")


@pytest.fixture(scope="module")
def target_columns():
    with open(os.path.join(ROOT, "serving", "feature_schema.json"),
              encoding="utf-8") as f:
        s = json.load(f)
    return s["eeg_columns"] + s["fnirs_columns"]


@pytest.fixture(scope="module")
def parser():
    return DatasetParser()


@pytest.fixture(scope="module")
def normalizer():
    return SchemaNormalizer()


@pytest.fixture(scope="module")
def predictor():
    from ingestion.pipeline import UniversalPredictor
    return UniversalPredictor(version="binary")


# ── Schema integrity ────────────────────────────────────────────────────────

def test_canonical_schema_matches_serving_bundle():
    """The written schema must still describe the live bundle."""
    ok, problems = validate_against_bundle()
    assert ok, f"canonical_schema.yaml has drifted from serving/: {problems}"


def test_channel_order_covers_every_model_position():
    schema = load_canonical_schema()
    order = schema["feature_level"]["eeg"]["channel_order"]
    assert len(order) == schema["feature_level"]["eeg"]["n_channels"] == 30
    assert order["ch00"] == "Fp1"      # recovered from TU Berlin `clab`
    assert order["ch29"] == "VEOG"


def test_dataset_profiles_load():
    profiles = load_dataset_profiles()
    assert {"tu_berlin", "stroop"} <= set(profiles)
    assert profiles["tu_berlin"]["is_reference_dataset"] is True


# ── Alias matching ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("spelling", ["Fp1", "FP1", "fp1", "EEG_Fp1", "eeg.fp1",
                                      "Fp1_uV", "  FP1  ", "EEG-FP1"])
def test_electrode_aliases_all_resolve_to_one_canonical(spelling):
    idx = AliasIndex()
    m = idx.resolve(spelling)
    assert m.matched and m.canonical == "Fp1", f"{spelling!r} -> {m.canonical!r}"


def test_older_1020_names_resolve():
    idx = AliasIndex()
    assert idx.resolve("T3").canonical == "T7"      # pre-1991 naming
    assert idx.resolve("T5").canonical == "P7"
    assert idx.resolve("HEO").canonical == "HEOG"


def test_unknown_channel_is_not_guessed():
    """No edit-distance fallback: a near-miss must NOT resolve."""
    idx = AliasIndex()
    for junk in ["Fp99", "blorp", "weird_sensor_42", "Xz"]:
        assert not idx.resolve(junk).matched, f"{junk!r} was wrongly matched"


def test_metadata_aliases():
    idx = AliasIndex()
    for spelling in ["subject", "participant_id", "subj", "SubjectID"]:
        assert idx.resolve(spelling).canonical == "subject_id"
    for spelling in ["label", "condition", "workload", "target"]:
        assert idx.resolve(spelling).canonical == "Cognitive_Load"


def test_normalize_token_strips_units_and_separators():
    assert normalize_token("Fp1_uV", ["uv"]) == "fp1"
    assert normalize_token("EEG.Fp1", []) == "eegfp1"
    assert normalize_token("EEG.Fp1", [], ["eeg"], strip_tags=True) == "fp1"


# ── Feature-column parsing ──────────────────────────────────────────────────

@pytest.mark.parametrize("raw,expected", [
    ("eeg_ch00_alpha_mean", "eeg_ch00_alpha_mean"),
    ("EEG_CH0_Alpha_Mean", "eeg_ch00_alpha_mean"),
    ("eeg.ch00.alpha.mean", "eeg_ch00_alpha_mean"),
    ("eeg.ch29.engagement.slope", "eeg_ch29_engage_slope"),
    ("hbo_ch00_delta", "hbo_ch00_delta"),          # `delta` as an fNIRS STAT
    ("HBO.CH0.DELTA", "hbo_ch00_delta"),
    ("eeg_ch00_delta_mean", "eeg_ch00_delta_mean"),  # `delta` as an EEG BAND
    ("hbr_channel_5_kurtosis", "hbr_ch05_kurt"),
])
def test_feature_column_parsing(normalizer, raw, expected):
    assert normalizer.parse_feature_column(raw) == expected


def test_non_feature_columns_return_none(normalizer):
    for junk in ["subject_id", "weird_sensor_42", "blorp", "window_id"]:
        assert normalizer.parse_feature_column(junk) is None


# ── Parser ──────────────────────────────────────────────────────────────────

@needs_demo
def test_parse_wide_csv(parser):
    r = parser.parse(DEMO_CSV)
    assert r.kind == "features" and r.layout == "wide"
    assert r.detected_dataset == "tu_berlin"
    assert len(r.dataframe) > 0


def test_parse_long_csv_preserves_every_timepoint(parser):
    """Regression: the long->wide pivot must not collapse timepoints."""
    long_df = pd.DataFrame({
        "time": np.repeat([0, 1, 2], 3),
        "subject": "S1",
        "channel": ["Fp1", "Cz", "O1"] * 3,
        "value": np.arange(9.0),
    })
    r = parser.parse(content=long_df.to_csv(index=False).encode(),
                     filename="long.csv")
    assert r.layout == "long"
    assert len(r.dataframe) == 3, "pivot dropped timepoints"
    assert {"Fp1", "Cz", "O1"} <= set(r.dataframe.columns)
    assert r.dataframe.loc[r.dataframe.time == 1, "Fp1"].iloc[0] == 3.0


def test_unsupported_extension_is_actionable(parser):
    with pytest.raises(IngestionError) as exc:
        parser.parse(content=b"x", filename="scan.dcm")
    assert "supported" in exc.value.detail


def test_npz_roundtrip(parser, tmp_path):
    path = tmp_path / "f.npz"
    np.savez(path, data=np.arange(6.0).reshape(2, 3),
             columns=np.array(["Fp1", "Cz", "O1"]))
    r = parser.parse(str(path))
    assert list(r.dataframe.columns) == ["Fp1", "Cz", "O1"]


@needs_tub_raw
def test_parse_tu_berlin_raw_mat(parser):
    r = parser.parse(TUB_EEG)
    assert r.detected_dataset == "tu_berlin" and r.kind == "signals"
    rec = r.recording
    assert rec.eeg.shape[1] == 30 and rec.eeg_fs == 200
    # Electrode identity must survive — it is what makes projection possible.
    assert rec.eeg_channels[0] == "Fp1"
    assert rec.eeg_channels[29] == "VEOG"


@needs_stroop
def test_parse_stroop_eeg_autodetects(parser):
    r = parser.parse(STROOP_EEG)
    assert r.detected_dataset == "stroop" and r.kind == "signals"
    rec = r.recording
    assert rec.eeg.shape[1] == 34 and rec.eeg_fs == 1000
    assert rec.subject_id == "S1"
    # Stroop labels live in the EEG event codes.
    assert len(rec.events) == 64
    assert {e["type"] for e in rec.events} == {"1", "2"}


@needs_stroop
def test_parse_stroop_fnirs_applies_beer_lambert(parser):
    r = parser.parse(STROOP_FNIRS)
    rec = r.recording
    assert rec.hbo.shape[1] == 20 and rec.fnirs_fs == 100
    assert rec.provenance["beer_lambert"]["applied"] is True
    # Converted concentrations should be physiologically plausible.
    assert abs(float(np.percentile(rec.hbo, 50))) < 5.0
    # The assumed separation must be surfaced, not buried.
    assert any("ASSUMED" in w for w in rec.warnings)


# ── Normalizer ──────────────────────────────────────────────────────────────

@needs_demo
def test_tu_berlin_csv_normalizes_completely(parser, normalizer, target_columns):
    df = parser.parse(DEMO_CSV).dataframe
    out, report = normalizer.normalize_features(df, target_columns)
    assert report.ok
    assert report.coverage["features"] == 1.0
    assert report.missing_required == []


@needs_demo
def test_renamed_and_shuffled_columns_still_normalize(parser, normalizer,
                                                     target_columns):
    """Step 7 case 3: underscores->dots, uppercase, unpadded indices, shuffled."""
    df = parser.parse(DEMO_CSV).dataframe.copy()
    renames = {}
    for c in df.columns:
        if c.startswith(("eeg_", "hbo_", "hbr_")):
            parts = c.split("_")
            parts[1] = "CH" + str(int(parts[1][2:]))   # ch00 -> CH0
            renames[c] = ".".join(parts).upper()
    df = df.rename(columns=renames)
    df = df[list(np.random.RandomState(0).permutation(df.columns))]

    out, report = normalizer.normalize_features(df, target_columns)
    assert report.ok, f"missing after rename: {report.missing_required[:5]}"
    assert report.coverage["features"] == 1.0
    # Canonical order must be restored regardless of input order.
    assert list(out.columns[:len(target_columns)]) == target_columns


@needs_demo
def test_missing_required_field_names_it_clearly(parser, normalizer,
                                                 target_columns):
    """Step 7 case 4: a clear error naming the field, never a stack trace."""
    df = parser.parse(DEMO_CSV).dataframe.drop(
        columns=["eeg_ch00_alpha_mean", "hbo_ch03_slope"])
    out, report = normalizer.normalize_features(df, target_columns)
    assert not report.ok
    assert "eeg_ch00_alpha_mean" in report.missing_required
    assert "hbo_ch03_slope" in report.missing_required
    assert len(report.missing_required) == 2


@needs_demo
def test_extra_unrecognized_columns_are_reported_not_fatal(parser, normalizer,
                                                           target_columns):
    """Step 7 case 5: still parses, and the ignored columns leave a trace."""
    df = parser.parse(DEMO_CSV).dataframe.copy()
    df["weird_sensor_42"] = 1.0
    df["blorp"] = 2.0
    out, report = normalizer.normalize_features(df, target_columns)
    assert report.ok
    assert "weird_sensor_42" in report.unmatched_input
    assert "blorp" in report.unmatched_input
    assert report.mapping["blorp"] == "UNMATCHED"
    assert any("not recognized" in w for w in report.warnings)


@needs_stroop
def test_stroop_channels_normalize_without_manual_renaming(parser, normalizer):
    rec, report = normalizer.normalize_recording(parser.parse(STROOP_EEG).recording)
    assert report.mapping["FP1"] == "Fp1"
    assert report.mapping["FZ"] == "Fz"
    assert report.mapping["HEO"] == "HEOG"
    assert report.unmatched_input == [], "every Stroop electrode should be known"


def test_label_value_mapping(normalizer):
    mapped, unmapped = normalizer.map_label_values(["1", "2", "incongruent"],
                                                   "stroop")
    assert list(mapped) == [0, 2, 2]
    assert unmapped == []
    # Unmappable values become -1, never a guess.
    mapped, unmapped = normalizer.map_label_values(["banana"], "stroop")
    assert list(mapped) == [-1] and unmapped == ["banana"]


def test_unknown_label_map_is_actionable(normalizer):
    with pytest.raises(IngestionError):
        normalizer.map_label_values(["1"], "no_such_dataset")


# ── Projection ──────────────────────────────────────────────────────────────

@needs_tub_raw
def test_reference_dataset_projects_at_full_coverage(parser, normalizer):
    proj = ChannelProjector()
    rec, _ = normalizer.normalize_recording(parser.parse(TUB_EEG).recording)
    arr, report = proj.project_eeg(rec)
    assert report.coverage == 1.0
    assert report.verdict == "confident"
    assert report.n_exact == 30 and report.n_missing == 0
    assert not np.isnan(arr).any()


@needs_stroop
def test_stroop_eeg_projects_as_degraded(parser, normalizer):
    """16 exact + 14 neighbour substitutions -> degraded, not silently fine."""
    proj = ChannelProjector()
    rec, _ = normalizer.normalize_recording(parser.parse(STROOP_EEG).recording)
    arr, report = proj.project_eeg(rec)
    assert report.n_exact == 16
    assert report.n_neighbour == 14
    assert report.verdict == "degraded"
    assert any("APPROXIMATE" in w for w in report.warnings)


@needs_stroop
def test_stroop_fnirs_projection_refuses(parser):
    """The honest outcome: 20 frontal channels cannot become 36 unknown ones."""
    proj = ChannelProjector()
    profiles = load_dataset_profiles()
    rec = parser.parse(STROOP_FNIRS).recording
    hbo, hbr, report = proj.project_fnirs(rec, profiles["stroop"])
    assert hbo is None and report.coverage == 0.0
    assert report.verdict == "refuse"
    assert report.notes["reason"] == "no_montage_map"


def test_matching_channel_count_is_not_treated_as_identity():
    """Channel COUNT is not channel IDENTITY — the key safety property."""
    from ingestion.canonical import CanonicalRecording
    proj = ChannelProjector()
    rec = CanonicalRecording(hbo=np.zeros((100, 36), dtype=np.float32),
                             hbr=np.zeros((100, 36), dtype=np.float32),
                             fnirs_fs=10.0)
    # A non-reference dataset with a coincidentally matching count.
    hbo, hbr, report = proj.project_fnirs(rec, {"name": "someone_else"})
    assert report.verdict == "refuse"
    assert any("not channel identity" in w for w in report.warnings)


def test_resample_preserves_missing_channel_markers():
    proj = ChannelProjector()
    data = np.random.RandomState(0).randn(1000, 3).astype(np.float32)
    data[:, 1] = np.nan                      # an unfillable position
    out, info = proj.resample(data, 1000, 200)
    assert info["resampled"] and out.shape[0] == 200
    assert np.all(np.isnan(out[:, 1])), "missing marker lost in resampling"
    assert not np.isnan(out[:, 0]).any()


# ── End-to-end ──────────────────────────────────────────────────────────────

@needs_demo
def test_regression_identical_predictions_to_existing_path(predictor):
    """THE regression test: the new path must not move a single prediction."""
    from model_service import ModelService
    from upload_service import UploadService

    df = pd.read_csv(DEMO_CSV)
    feats, _, _, _ = UploadService().preprocess(df)
    old = ModelService().predict_batch(feats.tolist())

    new = predictor.predict([{"path": DEMO_CSV}])

    assert new["ok"] and new["verdict"] == "confident"
    assert len(new["predictions"]) == len(old)
    for o, n in zip(old, new["predictions"]):
        assert o["predicted_index"] == n["predicted_index"]
        for cls, p in o["probabilities"].items():
            assert n["probabilities"][cls] == pytest.approx(p, abs=1e-12)


@needs_tub_raw
def test_raw_mat_pair_predicts_end_to_end(predictor):
    res = predictor.predict([{"path": TUB_EEG}, {"path": TUB_NIRS}],
                            max_windows=10)
    assert res["ok"] and res["verdict"] == "confident"
    assert res["n_rows"] == 10
    assert all(p["modality"] in {"eeg", "fnirs"} for p in res["projection_reports"])
    assert all(p["coverage"] == 1.0 for p in res["projection_reports"])


@needs_stroop
def test_stroop_end_to_end_refuses_with_full_report(predictor):
    """Step 7 case 2, answered honestly.

    Stroop parses, normalizes and projects its EEG without any manual column
    renaming — but its 20 frontal fNIRS channels cannot be mapped onto the
    model's 36 unknown optode positions. The pipeline must therefore REFUSE
    with a structured report, rather than zero-fill 936 features and return a
    confident-looking number.
    """
    res = predictor.predict([{"path": STROOP_EEG}, {"path": STROOP_FNIRS}])

    assert res["refused"] is True
    assert res["error_code"] == "insufficient_channel_coverage"
    assert res["detail"]["blame_modality"] == "fnirs"
    # EEG got as far as it could, and says so with numbers.
    assert res["detail"]["eeg"]["coverage"] == pytest.approx(0.7667, abs=1e-3)
    assert res["detail"]["eeg"]["n_missing"] == 0
    assert res["detail"]["fnirs"]["coverage"] == 0.0
    # The refusal must tell the user what to do about it.
    assert "retrain" in res["detail"]["remedy"].lower() or \
           "own feature space" in res["detail"]["remedy"]


@needs_demo
def test_unknown_dataset_missing_fields_refuses_cleanly(predictor, tmp_path):
    """A totally unknown table must fail clearly, not crash."""
    df = pd.DataFrame({"alpha": [1.0, 2.0], "beta": [3.0, 4.0],
                       "who_knows": ["a", "b"]})
    path = tmp_path / "mystery.csv"
    df.to_csv(path, index=False)

    res = predictor.predict([{"path": str(path)}])
    assert res["refused"] is True
    assert res["error_code"] == "missing_required_features"
    assert res["detail"]["n_missing"] == 1476
    assert isinstance(res["schema_mapping_report"]["mapping"], dict)
