# ============================================================
#  data_loader.py  –  Load, split, and scale EEG+fNIRS data
#  Auto-detects EEG/fNIRS columns by prefix when not explicit
# ============================================================

import numpy as np
import pandas as pd
import re
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    StratifiedGroupKFold,
    GroupShuffleSplit,
)
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import torch
from torch.utils.data import Dataset, DataLoader
from config import Config


def configured_class_weights(labels, cfg: Config = None):
    """Build class weights according to config, using train labels only."""
    cfg = cfg or Config()
    if not cfg.USE_CLASS_WEIGHTS:
        return None

    mode = getattr(cfg, "CLASS_WEIGHT_MODE", "balanced").lower()
    if mode == "manual":
        weights = np.asarray(
            getattr(cfg, "CLASS_WEIGHTS", [1.0] * cfg.NUM_CLASSES),
            dtype=np.float32,
        )
        if len(weights) != cfg.NUM_CLASSES:
            raise ValueError("Config.CLASS_WEIGHTS must match Config.NUM_CLASSES")
    else:
        weights = np.ones(cfg.NUM_CLASSES, dtype=np.float32)
        present_classes = np.unique(labels)
        balanced = compute_class_weight("balanced", classes=present_classes, y=labels)
        weights[present_classes] = balanced.astype(np.float32)
        if mode == "balanced_medium_boost" and cfg.NUM_CLASSES > 1:
            weights[1] *= float(getattr(cfg, "MEDIUM_CLASS_WEIGHT_BOOST", 1.0))
        elif mode != "balanced":
            raise ValueError(
                "Config.CLASS_WEIGHT_MODE must be 'manual', 'balanced', "
                "or 'balanced_medium_boost'"
            )

    return torch.tensor(weights, dtype=torch.float32)


# ── Dataset class ─────────────────────────────────────────────

class CognitiveLoadDataset(Dataset):
    def __init__(self, eeg_data, fnirs_data, labels, domains=None):
        self.eeg    = torch.tensor(eeg_data,   dtype=torch.float32)
        self.fnirs  = torch.tensor(fnirs_data,  dtype=torch.float32)
        self.labels = torch.tensor(labels,      dtype=torch.long)
        self.domains = torch.tensor(domains, dtype=torch.long) if domains is not None else None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if self.domains is not None:
            return self.eeg[idx], self.fnirs[idx], self.labels[idx], self.domains[idx]
        return self.eeg[idx], self.fnirs[idx], self.labels[idx]


# ── Core loader ───────────────────────────────────────────────

class CognitiveLoadDataLoader:

    def __init__(self, cfg: Config = None):
        self.cfg          = cfg or Config()
        self.eeg_scaler   = RobustScaler()
        self.fnirs_scaler = StandardScaler()
        self.eeg_cols     = None
        self.fnirs_cols   = None
        self.group_cols    = None
        self.eeg_sequence_shape = None
        self.subjects = None
        self.domains = None
        self.n_domains = 0

    # ── Auto-detect columns from CSV header ───────────────────

    def _resolve_columns(self, all_cols):
        """Use explicit lists if set in config, else detect by prefix."""
        cfg = self.cfg

        eeg_cols = getattr(cfg, 'EEG_COLUMNS', None)
        fn_cols  = getattr(cfg, 'MODALITY2_COLUMNS', None)

        if not eeg_cols:
            eeg_cols = sorted(
                [c for c in all_cols if c.startswith(cfg.EEG_FEATURE_PREFIX)],
                key=lambda x: int(x.split('_')[-1]) if x.split('_')[-1].isdigit() else 0
            )
        if not fn_cols:
            fn_cols = sorted(
                [c for c in all_cols if c.startswith(cfg.FNIRS_FEATURE_PREFIX)],
                key=lambda x: int(x.split('_')[-1]) if x.split('_')[-1].isdigit() else 0
            )

        if not eeg_cols:
            raise ValueError(
                f"No EEG columns found with prefix '{cfg.EEG_FEATURE_PREFIX}'.\n"
                f"Set Config.EEG_COLUMNS explicitly or fix Config.EEG_FEATURE_PREFIX."
            )
        if not fn_cols:
            raise ValueError(
                f"No fNIRS columns found with prefix '{cfg.FNIRS_FEATURE_PREFIX}'.\n"
                f"Set Config.MODALITY2_COLUMNS or fix Config.FNIRS_FEATURE_PREFIX."
            )
        return eeg_cols, fn_cols

    def _infer_eeg_sequence_shape(self, eeg_cols):
        """Detect v2 EEG columns and return (timesteps, features_per_timestep)."""
        parsed = []
        for col in eeg_cols:
            match = re.match(r"^eeg_t(\d+)_(.+)$", col)
            if not match:
                self.eeg_sequence_shape = None
                return None
            parsed.append((int(match.group(1)), match.group(2)))

        timesteps = sorted({t for t, _ in parsed})
        base_features = sorted({name for _, name in parsed})
        expected = {(t, name) for t in timesteps for name in base_features}
        observed = set(parsed)
        if observed != expected:
            raise ValueError("EEG sequence columns are incomplete or inconsistent")

        self.eeg_sequence_shape = (len(timesteps), len(base_features))
        return self.eeg_sequence_shape

    # ── Load raw data ─────────────────────────────────────────

    def _resolve_group_columns(self, df):
        """Return metadata columns that should stay together across splits."""
        cfg = self.cfg
        configured = getattr(cfg, "SPLIT_GROUP_COLUMNS", None)
        if configured:
            group_cols = [c for c in configured if c in df.columns]
        else:
            candidates = [getattr(cfg, "SUBJECT_COLUMN", None), "trial_id"]
            group_cols = [c for c in candidates if c and c in df.columns]

        self.group_cols = group_cols
        return group_cols

    def _make_groups(self, df):
        group_cols = self._resolve_group_columns(df)
        if not group_cols:
            return None

        groups = df[group_cols].astype(str).agg("|".join, axis=1).to_numpy()
        print(f"   Split groups : {', '.join(group_cols)} ({len(np.unique(groups)):,} groups)")
        return groups

    def _apply_feature_transforms(self, eeg_raw, fnirs_raw):
        cfg = self.cfg

        if getattr(cfg, "EEG_LOG_TRANSFORM", False):
            eeg_mask = np.array(
                [c.lower().startswith("eeg_") for c in self.eeg_cols],
                dtype=bool,
            )
            if eeg_mask.any():
                eeg_raw = eeg_raw.copy()
                values = eeg_raw[:, eeg_mask]
                log_mode = getattr(cfg, "EEG_LOG_MODE", "auto").lower()
                use_signed = log_mode == "signed" or (
                    log_mode == "auto" and np.nanmin(values) < 0
                )
                if use_signed:
                    eeg_raw[:, eeg_mask] = np.sign(values) * np.log1p(np.abs(values))
                    print(f"   EEG transform : signed log1p on {int(eeg_mask.sum())} columns")
                else:
                    eeg_raw[:, eeg_mask] = np.log1p(
                        np.clip(values, a_min=0.0, a_max=None)
                    )
                    print(f"   EEG transform : log1p on {int(eeg_mask.sum())} columns")

        if getattr(cfg, "FNIRS_SIGNED_LOG_TRANSFORM", False):
            fnirs_mask = np.array(
                [c.lower().startswith(("hbo_", "hbr_")) for c in self.fnirs_cols],
                dtype=bool,
            )
            if fnirs_mask.any():
                fnirs_raw = fnirs_raw.copy()
                values = fnirs_raw[:, fnirs_mask]
                fnirs_raw[:, fnirs_mask] = np.sign(values) * np.log1p(np.abs(values))
                print(f"   fNIRS transform: signed log1p on {int(fnirs_mask.sum())} columns")

        return eeg_raw, fnirs_raw

    def _per_subject_normalize(self, eeg_raw, fnirs_raw, subjects, stage):
        if subjects is None:
            raise ValueError(
                "PER_SUBJECT_NORMALIZE=True requires a subject column in the dataset"
            )

        eeg_norm = eeg_raw.copy()
        fnirs_norm = fnirs_raw.copy()
        eps = float(getattr(self.cfg, "PER_SUBJECT_NORMALIZE_EPS", 1e-8))
        unique_subjects = pd.unique(subjects)

        def normalize_block(block):
            mean = np.nanmean(block, axis=0)
            std = np.nanstd(block, axis=0, ddof=1)
            std = np.nan_to_num(std, nan=eps, posinf=eps, neginf=eps)
            std[std < eps] = eps
            return (block - mean) / std

        for subject in unique_subjects:
            mask = subjects == subject
            eeg_norm[mask] = normalize_block(eeg_norm[mask])
            fnirs_norm[mask] = normalize_block(fnirs_norm[mask])

        print(
            "   Per-subject normalize: "
            f"z-score EEG + fNIRS for {len(unique_subjects)} subjects ({stage})"
        )
        return eeg_norm, fnirs_norm

    def split_indices(self, labels, groups=None):
        """Create train/val/test indices, using grouped splits when metadata exists."""
        cfg = self.cfg
        idx = np.arange(len(labels))

        if groups is None:
            train_val_idx, test_idx = train_test_split(
                idx,
                test_size    = cfg.TEST_SIZE,
                stratify     = labels,
                random_state = cfg.RANDOM_SEED,
            )
            train_idx, val_idx = train_test_split(
                train_val_idx,
                test_size    = cfg.VAL_SIZE,
                stratify     = labels[train_val_idx],
                random_state = cfg.RANDOM_SEED,
            )
            return train_idx, val_idx, test_idx

        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=cfg.TEST_SIZE,
            random_state=cfg.RANDOM_SEED,
        )
        train_val_rel, test_rel = next(splitter.split(idx, labels, groups))
        train_val_idx = idx[train_val_rel]
        test_idx = idx[test_rel]

        val_splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=cfg.VAL_SIZE,
            random_state=cfg.RANDOM_SEED,
        )
        train_rel, val_rel = next(
            val_splitter.split(
                train_val_idx,
                labels[train_val_idx],
                groups[train_val_idx],
            )
        )
        train_idx = train_val_idx[train_rel]
        val_idx = train_val_idx[val_rel]

        return train_idx, val_idx, test_idx

    def _print_split_distribution(self, name, y):
        bits = []
        for cls, class_name in enumerate(self.cfg.CLASS_NAMES):
            n = int((y == cls).sum())
            bits.append(f"{class_name}={n}")
        print(f"   {name:<5}: " + "  ".join(bits))

    def load_raw(self, return_groups=False):
        cfg = self.cfg
        print(f"📂 Loading dataset: {cfg.MERGED_CSV}")

        if cfg.CHUNK_LOAD:
            print(f"   Mode: CHUNKED (chunk_size={cfg.CHUNK_SIZE:,})")
            chunks = []
            for i, chunk in enumerate(pd.read_csv(cfg.MERGED_CSV,
                                                   chunksize=cfg.CHUNK_SIZE)):
                chunks.append(chunk)
                print(f"   Chunk {i+1} loaded ({len(chunk):,} rows)", end="\r")
            df = pd.concat(chunks, ignore_index=True)
            print()
        else:
            print(f"   Mode: STANDARD (full read)")
            df = pd.read_csv(cfg.MERGED_CSV)

        # Resolve column lists
        eeg_cols, fn_cols = self._resolve_columns(df.columns)
        self.eeg_cols   = eeg_cols
        self.fnirs_cols = fn_cols

        if cfg.LABEL_COLUMN not in df.columns:
            raise ValueError(
                f"Label column '{cfg.LABEL_COLUMN}' not found.\n"
                f"Available: {list(df.columns[:10])} ..."
            )

        subject_col = getattr(cfg, "SUBJECT_COLUMN", "subject_id")
        subjects = None
        domains = None
        if subject_col in df.columns:
            subjects = df[subject_col].astype(str).to_numpy()
            unique_subjects = np.unique(subjects)
            subj_to_id = {s: i for i, s in enumerate(unique_subjects)}
            domains = np.array([subj_to_id[s] for s in subjects], dtype=np.int64)
            self.subjects = subjects
            self.domains = domains
            self.n_domains = len(unique_subjects)

        groups = self._make_groups(df)

        print(f"   EEG features  : {len(eeg_cols)}")
        print(f"   fNIRS features: {len(fn_cols)}")
        print(f"   Total samples : {len(df):,}")
        eeg_sequence_shape = self._infer_eeg_sequence_shape(eeg_cols)
        if eeg_sequence_shape:
            steps, features = eeg_sequence_shape
            print(f"   EEG sequence : {steps} timesteps x {features} features")

        eeg_raw   = df[eeg_cols].values.astype(np.float32)
        fnirs_raw = df[fn_cols].values.astype(np.float32)
        labels    = df[cfg.LABEL_COLUMN].values.astype(np.int64)

        if getattr(cfg, "BINARY_CLASSIFICATION", False):
            keep_idx = labels != 1
            eeg_raw = eeg_raw[keep_idx]
            fnirs_raw = fnirs_raw[keep_idx]
            labels = labels[keep_idx]
            if subjects is not None:
                subjects = subjects[keep_idx]
                self.subjects = subjects
            if domains is not None:
                domains = domains[keep_idx]
                self.domains = domains
            if groups is not None:
                groups = groups[keep_idx]
            
            # Remap High class from 2 to 1
            labels[labels == 2] = 1
            print("   [Binary] Filtered 'Medium' (1) and remapped 'High' (2 -> 1).")

        # Clean NaNs / Infs
        eeg_raw   = np.nan_to_num(eeg_raw,   nan=0.0, posinf=0.0, neginf=0.0)
        fnirs_raw = np.nan_to_num(fnirs_raw, nan=0.0, posinf=0.0, neginf=0.0)

        subject_norm_stage = getattr(
            cfg,
            "PER_SUBJECT_NORMALIZE_STAGE",
            "after_log",
        ).lower()
        if getattr(cfg, "PER_SUBJECT_NORMALIZE", False):
            if subject_norm_stage not in {"before_log", "after_log"}:
                raise ValueError(
                    "Config.PER_SUBJECT_NORMALIZE_STAGE must be 'before_log' "
                    "or 'after_log'"
                )
            if subject_norm_stage == "before_log":
                eeg_raw, fnirs_raw = self._per_subject_normalize(
                    eeg_raw,
                    fnirs_raw,
                    subjects,
                    subject_norm_stage,
                )

        eeg_raw, fnirs_raw = self._apply_feature_transforms(eeg_raw, fnirs_raw)

        if (
            getattr(cfg, "PER_SUBJECT_NORMALIZE", False)
            and subject_norm_stage == "after_log"
        ):
            eeg_raw, fnirs_raw = self._per_subject_normalize(
                eeg_raw,
                fnirs_raw,
                subjects,
                subject_norm_stage,
            )

        print(f"\n✅ Loaded successfully")
        print(f"   Labels: {np.unique(labels)}  distribution:", end="")
        for cls, name in enumerate(cfg.CLASS_NAMES):
            n = (labels == cls).sum()
            print(f"  {name}={n}", end="")
        print()

        if return_groups:
            return eeg_raw, fnirs_raw, labels, groups
        return eeg_raw, fnirs_raw, labels

    # ── Train / val / test split + scaling ────────────────────

    def prepare(self):
        cfg = self.cfg
        eeg_raw, fnirs_raw, labels, groups = self.load_raw(return_groups=True)

        train_idx, val_idx, test_idx = self.split_indices(labels, groups)
        eeg_train, eeg_val, eeg_test = eeg_raw[train_idx], eeg_raw[val_idx], eeg_raw[test_idx]
        fnirs_train, fnirs_val, fnirs_test = (
            fnirs_raw[train_idx],
            fnirs_raw[val_idx],
            fnirs_raw[test_idx],
        )
        y_train, y_val, y_test = labels[train_idx], labels[val_idx], labels[test_idx]
        dom_train = self.domains[train_idx] if self.domains is not None else None
        dom_val = self.domains[val_idx] if self.domains is not None else None
        dom_test = self.domains[test_idx] if self.domains is not None else None

        if cfg.NORMALIZE:
            print("\n🔧 Normalizing (RobustScaler for EEG, StandardScaler for fNIRS; fit on train only) ...")
            eeg_train   = self.eeg_scaler.fit_transform(eeg_train)
            eeg_val     = self.eeg_scaler.transform(eeg_val)
            eeg_test    = self.eeg_scaler.transform(eeg_test)
            fnirs_train = self.fnirs_scaler.fit_transform(fnirs_train)
            fnirs_val   = self.fnirs_scaler.transform(fnirs_val)
            fnirs_test  = self.fnirs_scaler.transform(fnirs_test)

        if self.eeg_sequence_shape:
            steps, features = self.eeg_sequence_shape
            eeg_train = eeg_train.reshape(len(eeg_train), steps, features)
            eeg_val = eeg_val.reshape(len(eeg_val), steps, features)
            eeg_test = eeg_test.reshape(len(eeg_test), steps, features)

        print(f"\n📊 Split → Train: {len(y_train)} | Val: {len(y_val)} | Test: {len(y_test)}")
        if groups is not None:
            print(
                "   Groups      "
                f"Train: {len(np.unique(groups[train_idx]))} | "
                f"Val: {len(np.unique(groups[val_idx]))} | "
                f"Test: {len(np.unique(groups[test_idx]))}"
            )
        self._print_split_distribution("Train", y_train)
        self._print_split_distribution("Val", y_val)
        self._print_split_distribution("Test", y_test)

        class_weights = None
        if cfg.USE_CLASS_WEIGHTS:
            class_weights = configured_class_weights(y_train, cfg)
            print(f"   Class weights: {class_weights.numpy().round(3)}")

        train_loader = DataLoader(
            CognitiveLoadDataset(eeg_train, fnirs_train, y_train, dom_train),
            batch_size=cfg.BATCH_SIZE, shuffle=True, drop_last=True, num_workers=0
        )
        val_loader = DataLoader(
            CognitiveLoadDataset(eeg_val, fnirs_val, y_val, dom_val),
            batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=0
        )
        test_loader = DataLoader(
            CognitiveLoadDataset(eeg_test, fnirs_test, y_test, dom_test),
            batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=0
        )

        meta = {
            "n_eeg_features"   : self.eeg_sequence_shape or eeg_raw.shape[1],
            "n_eeg_flat_features": eeg_raw.shape[1],
            "n_fnirs_features" : fnirs_raw.shape[1],
            "n_domains"        : self.n_domains,
            "class_weights"    : class_weights,
            "y_test"           : y_test,
            "split_group_columns": self.group_cols,
        }
        return train_loader, val_loader, test_loader, meta

    # ── CV fold generator ─────────────────────────────────────

    def cv_folds(self):
        cfg = self.cfg
        eeg_raw, fnirs_raw, labels, groups = self.load_raw(return_groups=True)

        if groups is not None:
            skf = StratifiedGroupKFold(
                n_splits=cfg.CV_FOLDS,
                shuffle=True,
                random_state=cfg.RANDOM_SEED,
            )
            split_iter = skf.split(eeg_raw, labels, groups)
        else:
            skf = StratifiedKFold(
                n_splits=cfg.CV_FOLDS,
                shuffle=True,
                random_state=cfg.RANDOM_SEED,
            )
            split_iter = skf.split(eeg_raw, labels)

        for fold, (tr_idx, va_idx) in enumerate(split_iter):
            eeg_tr,   fnirs_tr, y_tr = eeg_raw[tr_idx], fnirs_raw[tr_idx], labels[tr_idx]
            eeg_va,   fnirs_va, y_va = eeg_raw[va_idx], fnirs_raw[va_idx], labels[va_idx]
            dom_tr = self.domains[tr_idx] if self.domains is not None else None
            dom_va = self.domains[va_idx] if self.domains is not None else None

            if cfg.NORMALIZE:
                es = RobustScaler(); fs = StandardScaler()
                eeg_tr   = es.fit_transform(eeg_tr);   eeg_va   = es.transform(eeg_va)
                fnirs_tr = fs.fit_transform(fnirs_tr); fnirs_va = fs.transform(fnirs_va)

            if self.eeg_sequence_shape:
                steps, features = self.eeg_sequence_shape
                eeg_tr = eeg_tr.reshape(len(eeg_tr), steps, features)
                eeg_va = eeg_va.reshape(len(eeg_va), steps, features)

            class_weights = configured_class_weights(y_tr, cfg)

            tr_loader = DataLoader(
                CognitiveLoadDataset(eeg_tr, fnirs_tr, y_tr, dom_tr),
                batch_size=cfg.BATCH_SIZE, shuffle=True, drop_last=True, num_workers=0
            )
            va_loader = DataLoader(
                CognitiveLoadDataset(eeg_va, fnirs_va, y_va, dom_va),
                batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=0
            )

            yield fold + 1, tr_loader, va_loader, {
                "n_eeg_features"   : self.eeg_sequence_shape or eeg_raw.shape[1],
                "n_eeg_flat_features": eeg_raw.shape[1],
                "n_fnirs_features" : fnirs_raw.shape[1],
                "n_domains"        : self.n_domains,
                "class_weights"    : class_weights,
                "y_val"            : y_va,
            }
