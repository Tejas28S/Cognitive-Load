"""
parser.py: DatasetParser — read ANY supported file into one intermediate form.

Responsibility boundary: this module only READS. It does not rename columns, map
labels, or decide what a channel means — that is the SchemaNormalizer's job.
What it guarantees is that downstream code never has to know about file formats.

Supported inputs
    .csv/.tsv/.txt  tabular (delimiter sniffed), wide or long layout
    .npz/.npy       arrays, with channel names taken from a companion key
    .mat            MATLAB — BBCI `cnt_*` structs, EEGLAB `EEG` structs, and
                    raw cell arrays, dispatched by dataset profile signature
    .edf/.bdf       via mne, if installed (a clear error if not)

Outputs one of:
    ParseResult.dataframe  — a raw DataFrame with ORIGINAL column names, for
                             already-extracted feature tables
    ParseResult.recording  — a CanonicalRecording, for raw continuous signals

Raw-signal reads do apply the physical conversions a dataset needs to reach
canonical form (notably Beer-Lambert for datasets shipping optical intensity
rather than haemoglobin), because those are properties of the FILE, not of the
schema. Every such conversion is recorded in `provenance`.
"""

from __future__ import annotations

import io
import os
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import scipy.io as sio

from .canonical import (
    AliasIndex,
    CanonicalRecording,
    IngestionError,
    load_dataset_profiles,
    normalize_token,
)

TABULAR_EXT = {".csv", ".tsv", ".txt"}
ARRAY_EXT = {".npz", ".npy"}
MAT_EXT = {".mat"}
EDF_EXT = {".edf", ".bdf"}
SUPPORTED_EXT = TABULAR_EXT | ARRAY_EXT | MAT_EXT | EDF_EXT


@dataclass
class ParseResult:
    """What the parser produces. Exactly one of dataframe/recording is set."""
    kind: str                                   # "features" | "signals"
    dataframe: pd.DataFrame | None = None
    recording: CanonicalRecording | None = None
    source_format: str = ""
    detected_dataset: str | None = None
    layout: str = ""                            # wide | long | n/a
    warnings: list[str] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)


# ── Modified Beer-Lambert ───────────────────────────────────────────────────

def beer_lambert(intensity_wl1, intensity_wl2, cfg):
    """Convert dual-wavelength optical intensity to HbO/HbR concentrations.

    ΔOD(λ) = (ε_HbO(λ)·ΔHbO + ε_HbR(λ)·ΔHbR) · d · DPF(λ)

    Solved per channel by inverting the 2x2 extinction matrix. Returns
    micromolar changes relative to each channel's own baseline (its mean), which
    is what the canonical schema specifies and what the downstream fNIRS feature
    stats assume.

    `cfg` is the dataset profile's fnirs.beer_lambert block.
    """
    wl = cfg["wavelengths_nm"]
    dpf = cfg["dpf"]
    d = float(cfg["source_detector_separation_cm"])
    ext = cfg["extinction_coefficients"]

    e1 = ext[str(wl[0])]     # [eps_HbO, eps_HbR] at wavelength 1
    e2 = ext[str(wl[1])]

    # Optical density change, referenced to each channel's mean intensity. A
    # non-positive intensity is unphysical; clip rather than emit NaN/-inf.
    def _od(x):
        x = np.asarray(x, dtype=np.float64)
        base = np.nanmean(x, axis=0, keepdims=True)
        base = np.where(np.abs(base) < 1e-12, 1e-12, base)
        ratio = np.clip(x / base, 1e-12, None)
        return -np.log10(ratio)

    od1, od2 = _od(intensity_wl1), _od(intensity_wl2)

    a = np.array([
        [e1[0] * d * dpf[0], e1[1] * d * dpf[0]],
        [e2[0] * d * dpf[1], e2[1] * d * dpf[1]],
    ], dtype=np.float64)
    inv = np.linalg.inv(a)

    stacked = np.stack([od1, od2], axis=-1)          # (samples, channels, 2)
    conc = stacked @ inv.T                            # mol/L
    hbo = conc[..., 0] * 1e6                          # -> micromolar
    hbr = conc[..., 1] * 1e6
    return hbo.astype(np.float32), hbr.astype(np.float32)


# ── Parser ──────────────────────────────────────────────────────────────────

class DatasetParser:

    def __init__(self, profiles=None, alias_index=None):
        self.profiles = profiles or load_dataset_profiles()
        self.aliases = alias_index or AliasIndex()

    # -- entry point ---------------------------------------------------------

    def parse(self, path=None, content=None, filename=None, dataset=None):
        """Read a file (from disk, or from bytes + filename) into a ParseResult.

        `dataset` forces a dataset profile; omit it to auto-detect.
        """
        name = filename or (os.path.basename(path) if path else None)
        if not name:
            raise IngestionError("Cannot parse without a path or a filename.")
        ext = os.path.splitext(name)[1].lower()

        if ext not in SUPPORTED_EXT:
            raise IngestionError(
                f"Unsupported file type '{ext}'.",
                {"supported": sorted(SUPPORTED_EXT), "filename": name},
            )

        raw = content if content is not None else open(path, "rb").read()

        if ext in TABULAR_EXT:
            return self._parse_tabular(raw, name, ext)
        if ext in ARRAY_EXT:
            return self._parse_array(raw, name, ext)
        if ext in MAT_EXT:
            return self._parse_mat(raw, name, dataset)
        return self._parse_edf(raw, name, path)

    # -- tabular -------------------------------------------------------------

    def _parse_tabular(self, raw, name, ext):
        text = raw.decode("utf-8-sig", errors="replace")
        sep = "\t" if ext == ".tsv" else self._sniff_delimiter(text)
        df = pd.read_csv(io.StringIO(text), sep=sep, engine="python")
        if df.empty:
            raise IngestionError(f"'{name}' parsed to zero rows.", {"filename": name})

        layout, df, notes = self._detect_layout(df)
        return ParseResult(
            kind="features",
            dataframe=df,
            source_format=f"tabular{ext}",
            detected_dataset=self._detect_dataset_from_columns(df.columns),
            layout=layout,
            warnings=notes,
            provenance={"delimiter": sep, "n_rows": int(len(df)),
                        "n_columns": int(df.shape[1])},
        )

    @staticmethod
    def _sniff_delimiter(text):
        head = "\n".join(text.splitlines()[:5])
        counts = {d: head.count(d) for d in [",", "\t", ";", "|"]}
        best = max(counts, key=counts.get)
        return best if counts[best] > 0 else ","

    def _detect_layout(self, df):
        """Wide (channels as columns) vs long (channels as values); pivot if long.

        Long is recognized by a column that NAMES channels whose values mostly
        resolve to known channels — not merely by a column called "channel",
        since that could hold anything.
        """
        notes = []
        candidates = [
            c for c in df.columns
            if normalize_token(c, self.aliases.unit_suffixes) in
            {"channel", "chan", "ch", "electrode", "elec", "sensor"}
        ]
        for col in candidates:
            vals = df[col].astype(str).unique()
            if len(vals) < 2:
                continue
            resolved = sum(1 for v in vals if self.aliases.resolve(v).matched)
            if resolved / len(vals) < 0.5:
                continue

            value_cols = [
                c for c in df.columns
                if c != col and pd.api.types.is_numeric_dtype(df[c])
            ]
            if not value_cols:
                continue
            # Prefer an explicitly value-named column; otherwise the last numeric
            # one. Picking wrong here silently destroys data, so the preferred
            # names are checked first rather than relying on column position.
            preferred = {"value", "amplitude", "signal", "data", "reading", "y", "v"}
            value_col = next(
                (c for c in value_cols
                 if normalize_token(c, self.aliases.unit_suffixes) in preferred),
                value_cols[-1],
            )
            # EVERY remaining column is an index column, numeric ones included —
            # a numeric `time`/`sample` column is what distinguishes one row from
            # the next, and dropping it would collapse the whole recording.
            index_cols = [c for c in df.columns if c not in (col, value_col)]

            if not index_cols:
                df = df.assign(_row=np.arange(len(df)))
                index_cols = ["_row"]

            wide = df.pivot_table(
                index=index_cols, columns=col, values=value_col, aggfunc="first"
            ).reset_index()
            wide.columns.name = None
            notes.append(
                f"Detected LONG layout (channels in column '{col}', values in "
                f"'{value_col}'); pivoted to wide on {index_cols}."
            )
            return "long", wide, notes

        return "wide", df, notes

    def _detect_dataset_from_columns(self, columns):
        colset = {str(c) for c in columns}
        for pname, prof in self.profiles.items():
            sig = ((prof.get("detection") or {}).get("csv_column_signature")) or []
            if sig and all(s in colset for s in sig):
                return pname
        return None

    # -- npz / npy -----------------------------------------------------------

    def _parse_array(self, raw, name, ext):
        buf = io.BytesIO(raw)
        if ext == ".npy":
            arr = np.load(buf, allow_pickle=False)
            df = pd.DataFrame(arr, columns=[f"col{i:03d}" for i in range(arr.shape[1])])
            return ParseResult(
                kind="features", dataframe=df, source_format="npy", layout="wide",
                warnings=["`.npy` carries no column names; columns were named "
                          "positionally and will almost certainly not match the "
                          "alias registry. Prefer .npz with a names array, or a CSV."],
                provenance={"shape": list(arr.shape)},
            )

        z = np.load(buf, allow_pickle=False)
        keys = list(z.keys())
        # Convention: a data array plus an optional names array.
        data_key = next((k for k in ("data", "features", "X", "arr_0") if k in keys), keys[0])
        names_key = next(
            (k for k in ("columns", "names", "channels", "clab", "labels") if k in keys),
            None,
        )
        arr = np.asarray(z[data_key])
        if arr.ndim != 2:
            raise IngestionError(
                f"'{name}': expected a 2-D array in '{data_key}', got shape {arr.shape}.",
                {"filename": name, "key": data_key, "shape": list(arr.shape)},
            )
        if names_key is not None:
            cols = [str(c) for c in np.asarray(z[names_key]).ravel()]
            if len(cols) != arr.shape[1]:
                raise IngestionError(
                    f"'{name}': '{names_key}' has {len(cols)} names for "
                    f"{arr.shape[1]} columns.",
                    {"filename": name},
                )
        else:
            cols = [f"col{i:03d}" for i in range(arr.shape[1])]

        df = pd.DataFrame(arr, columns=cols)
        return ParseResult(
            kind="features", dataframe=df, source_format="npz", layout="wide",
            detected_dataset=self._detect_dataset_from_columns(df.columns),
            provenance={"keys": keys, "data_key": data_key, "names_key": names_key},
        )

    # -- MATLAB --------------------------------------------------------------

    def _parse_mat(self, raw, name, dataset=None):
        try:
            mat = sio.loadmat(io.BytesIO(raw), struct_as_record=False, squeeze_me=True)
        except NotImplementedError as exc:
            # v7.3 files are HDF5; scipy cannot read them.
            raise IngestionError(
                f"'{name}' looks like a MATLAB v7.3 (HDF5) file, which scipy "
                f"cannot read. Re-save it as v7 in MATLAB "
                f"(`save('file.mat','var','-v7')`), or install h5py.",
                {"filename": name, "error": str(exc)},
            ) from exc

        keys = [k for k in mat if not k.startswith("__")]
        if not keys:
            raise IngestionError(f"'{name}' contains no MATLAB variables.",
                                 {"filename": name})

        profile_name = dataset or self._detect_dataset_from_mat(mat, keys)
        if profile_name is None:
            raise IngestionError(
                f"Could not identify the layout of '{name}'. Its top-level "
                f"variables are {keys}, which match no dataset profile in "
                f"schema/datasets/.",
                {"filename": name, "mat_keys": keys,
                 "known_profiles": sorted(self.profiles)},
            )

        prof = self.profiles[profile_name]
        eeg_fmt = (prof.get("eeg") or {}).get("format")
        fnirs_fmt = (prof.get("fnirs") or {}).get("format")

        # A .mat holds ONE modality; which one is decided by what is inside it.
        if eeg_fmt == "mat_eeglab" and prof["eeg"]["struct_key"] in keys:
            rec = self._read_eeglab(mat, prof, name)
        elif fnirs_fmt == "mat_cell_array" and prof["fnirs"]["struct_key"] in keys:
            rec = self._read_cell_fnirs(mat, prof, name)
        elif eeg_fmt == "mat_bbci" or fnirs_fmt == "mat_bbci":
            rec = self._read_bbci(mat, prof, keys, name)
        else:
            raise IngestionError(
                f"Profile '{profile_name}' does not describe how to read "
                f"'{name}' (top-level variables {keys}).",
                {"filename": name, "profile": profile_name, "mat_keys": keys},
            )

        rec.dataset = profile_name
        rec.subject_id = rec.subject_id or self._subject_from_filename(name, prof)
        return ParseResult(
            kind="signals", recording=rec, source_format="mat",
            detected_dataset=profile_name, layout="n/a",
            warnings=rec.warnings, provenance=rec.provenance,
        )

    def _detect_dataset_from_mat(self, mat, keys):
        best, best_score = None, 0
        for pname, prof in self.profiles.items():
            det = prof.get("detection") or {}
            want_keys = det.get("mat_struct_keys") or []
            score = 0
            for wk in want_keys:
                if wk in keys:
                    score += 2
                elif wk.endswith("*") and any(k.startswith(wk[:-1]) for k in keys):
                    score += 2
                elif "*" in wk:
                    pat = re.compile("^" + wk.replace("*", ".*") + "$")
                    if any(pat.match(k) for k in keys):
                        score += 2
            # Field signature is the tie-breaker and the guard against two
            # datasets that happen to share a variable name.
            sig = det.get("mat_field_signature") or []
            if sig:
                for k in keys:
                    v = mat[k]
                    fields = getattr(v, "_fieldnames", None)
                    if fields is None and isinstance(v, np.ndarray) and v.dtype == object:
                        first = v.flat[0] if v.size else None
                        fields = getattr(first, "_fieldnames", None)
                    if fields and sum(1 for s in sig if s in fields) >= max(2, len(sig) // 2):
                        score += 3
                        break
            if score > best_score:
                best, best_score = pname, score
        return best if best_score >= 2 else None

    @staticmethod
    def _subject_from_filename(name, prof):
        pat = ((prof.get("subjects") or {}).get("id_pattern"))
        if not pat:
            return None
        m = re.search(pat, os.path.splitext(os.path.basename(name))[0])
        return m.group(0) if m else None

    def _read_eeglab(self, mat, prof, name):
        """EEGLAB `EEG` struct — data, srate, chanlocs.labels, event."""
        spec = prof["eeg"]
        eeg = mat[spec["struct_key"]]
        data = np.asarray(eeg.data, dtype=np.float32)

        # EEGLAB stores channels x samples; canonical is samples x channels.
        if spec.get("data_orientation") == "channels_by_samples":
            data = data.T
        elif data.shape[0] < data.shape[1]:
            data = data.T

        chanlocs = eeg.chanlocs
        channels = [str(getattr(c, "labels", f"ch{i:02d}"))
                    for i, c in enumerate(np.atleast_1d(chanlocs))]
        fs = float(eeg.srate)

        events = []
        raw_events = getattr(eeg, "event", None)
        if raw_events is not None:
            for e in np.atleast_1d(raw_events):
                etype = getattr(e, "type", None)
                lat = getattr(e, "latency", None)
                if etype is None or lat is None:
                    continue
                events.append({
                    "type": str(etype).strip(),
                    "latency_sec": float(lat) / fs,
                    "latency_samples": float(lat),
                })

        rec = CanonicalRecording(
            eeg=data, eeg_channels=channels, eeg_fs=fs, events=events,
            provenance={
                "reader": "mat_eeglab", "source_file": name,
                "n_channels_in_file": len(channels), "sample_rate_hz": fs,
                "event_latency_units": "samples @ EEG rate",
            },
        )
        if data.shape[1] != len(channels):
            rec.warnings.append(
                f"Channel count mismatch: data has {data.shape[1]} columns but "
                f"chanlocs lists {len(channels)} labels."
            )
        return rec

    def _read_cell_fnirs(self, mat, prof, name):
        """Raw fNIRS cell array with interleaved wavelength columns."""
        spec = prof["fnirs"]
        layout = spec["layout"]
        cells = np.atleast_1d(mat[spec["struct_key"]])

        lo, hi = layout["data_columns"]
        cols = []
        for i in range(lo, hi + 1):
            arr = np.asarray(cells[i], dtype=np.float64).ravel()
            cols.append(arr)
        lengths = {len(c) for c in cols}
        if len(lengths) != 1:
            raise IngestionError(
                f"'{name}': fNIRS data columns have inconsistent lengths {sorted(lengths)}.",
                {"filename": name},
            )
        block = np.stack(cols, axis=1)      # (samples, 40)

        # Odd/even interleave, expressed relative to the data block's own start.
        interleave = layout.get("wavelength_interleave") or {}
        first_is_odd = str(interleave.get("wl_785", "odd")).lower() == "odd"
        idx_a = np.arange(0, block.shape[1], 2)
        idx_b = np.arange(1, block.shape[1], 2)
        wl1, wl2 = (block[:, idx_a], block[:, idx_b]) if first_is_odd else \
                   (block[:, idx_b], block[:, idx_a])

        prov = {
            "reader": "mat_cell_array", "source_file": name,
            "n_optode_channels": int(wl1.shape[1]),
            "sample_rate_hz": float(spec["sample_rate_hz"]),
        }
        warnings = []

        if spec.get("needs_beer_lambert"):
            bl = spec["beer_lambert"]
            hbo, hbr = beer_lambert(wl1, wl2, bl)
            prov["beer_lambert"] = {
                "applied": True,
                "wavelengths_nm": bl["wavelengths_nm"],
                "dpf": bl["dpf"],
                "source_detector_separation_cm": bl["source_detector_separation_cm"],
            }
            warnings.append(
                "fNIRS was converted from optical intensity to HbO/HbR via the "
                "modified Beer-Lambert law using an ASSUMED source-detector "
                f"separation of {bl['source_detector_separation_cm']} cm — the "
                "dataset does not document per-channel separations."
            )
        else:
            hbo, hbr = wl1.astype(np.float32), wl2.astype(np.float32)
            prov["beer_lambert"] = {"applied": False}

        sync = None
        if layout.get("sync_column") is not None:
            sync = np.asarray(cells[layout["sync_column"]], dtype=np.float64).ravel()
            prov["sync_trace_samples"] = int(sync.size)

        rec = CanonicalRecording(
            hbo=hbo, hbr=hbr, fnirs_fs=float(spec["sample_rate_hz"]),
            provenance=prov, warnings=warnings,
        )
        if sync is not None:
            rec.provenance["sync_available"] = True
        sync_spec = prof.get("synchronization") or {}
        if sync_spec.get("status") == "unvalidated":
            rec.warnings.append(
                "Cross-modal alignment for this dataset is UNVALIDATED "
                f"({sync_spec.get('method')}). {sync_spec.get('note', '').strip()}"
            )
        return rec

    def _read_bbci(self, mat, prof, keys, name):
        """BBCI `cnt_*` struct — the TU Berlin layout (EEG or NIRS)."""
        cnt_key = next((k for k in keys if k.startswith("cnt_")), None)
        if cnt_key is None:
            raise IngestionError(
                f"'{name}': no `cnt_*` struct found (variables: {keys}).",
                {"filename": name, "mat_keys": keys},
            )
        cnt = mat[cnt_key]
        if isinstance(cnt, np.ndarray) and cnt.dtype == object and cnt.size:
            cnt = cnt.flat[0]
        fields = set(getattr(cnt, "_fieldnames", []) or [])

        # NIRS files carry oxy/deoxy sub-structs; EEG files carry x/clab.
        if {"oxy", "deoxy"} & fields:
            oxy, deoxy = cnt.oxy, cnt.deoxy
            hbo = np.asarray(oxy.x, dtype=np.float32)
            hbr = np.asarray(deoxy.x, dtype=np.float32)
            fs = float(np.atleast_1d(oxy.fs).flat[0])
            return CanonicalRecording(
                hbo=hbo, hbr=hbr, fnirs_fs=fs,
                provenance={"reader": "mat_bbci_nirs", "source_file": name,
                            "sample_rate_hz": fs,
                            "n_channels_in_file": int(hbo.shape[1]),
                            "beer_lambert": {"applied": False,
                                             "reason": "already in HbO/HbR space"}},
            )

        data = np.asarray(cnt.x, dtype=np.float32)
        if data.ndim == 1:
            data = data[:, None]
        fs = float(np.atleast_1d(cnt.fs).flat[0])
        channels = [str(c) for c in np.atleast_1d(cnt.clab)] if "clab" in fields \
            else [f"ch{i:02d}" for i in range(data.shape[1])]
        if data.shape[0] < data.shape[1]:
            data = data.T
        return CanonicalRecording(
            eeg=data, eeg_channels=channels, eeg_fs=fs,
            provenance={"reader": "mat_bbci_eeg", "source_file": name,
                        "sample_rate_hz": fs,
                        "n_channels_in_file": len(channels)},
        )

    # -- EDF -----------------------------------------------------------------

    def _parse_edf(self, raw, name, path):
        try:
            import mne
        except ImportError as exc:
            raise IngestionError(
                "Reading .edf/.bdf requires the `mne` package, which is not "
                "installed. Install it with `pip install mne`, or convert the "
                "recording to .mat/.csv first.",
                {"filename": name, "missing_package": "mne"},
            ) from exc

        # mne needs a real path; materialize uploaded bytes to a temp file.
        tmp = None
        if path is None:
            import tempfile
            fd, tmp = tempfile.mkstemp(suffix=os.path.splitext(name)[1])
            with os.fdopen(fd, "wb") as f:
                f.write(raw)
            path = tmp
        try:
            reader = mne.io.read_raw_bdf if name.lower().endswith(".bdf") \
                else mne.io.read_raw_edf
            rawobj = reader(path, preload=True, verbose="ERROR")
            data = rawobj.get_data().T.astype(np.float32)     # -> samples x channels
            # mne returns volts for EEG; canonical is microvolts.
            data = data * 1e6
            channels = list(rawobj.ch_names)
            fs = float(rawobj.info["sfreq"])

            events = []
            try:
                ann = rawobj.annotations
                for onset, desc in zip(ann.onset, ann.description):
                    events.append({"type": str(desc), "latency_sec": float(onset),
                                   "latency_samples": float(onset) * fs})
            except Exception:
                pass

            rec = CanonicalRecording(
                eeg=data, eeg_channels=channels, eeg_fs=fs, events=events,
                provenance={"reader": "edf_mne", "source_file": name,
                            "sample_rate_hz": fs,
                            "n_channels_in_file": len(channels),
                            "unit_conversion": "volts -> microvolts (x1e6)"},
            )
            return ParseResult(
                kind="signals", recording=rec, source_format="edf",
                detected_dataset=None, layout="n/a",
                warnings=rec.warnings, provenance=rec.provenance,
            )
        finally:
            if tmp and os.path.exists(tmp):
                os.unlink(tmp)
