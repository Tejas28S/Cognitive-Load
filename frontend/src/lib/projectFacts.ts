/**
 * Structural facts about the dataset, features, model and protocol.
 *
 * These are fixed properties of the committed artifacts, not measured results:
 * every value here is transcribed from a file in the repo, and the `source`
 * field on each group names it. Measured *performance* numbers are NOT kept
 * here — those are fetched live from the API so the UI can never drift from
 * the run that produced them.
 *
 * Verified against:
 *   serving/feature_schema.json   — channel/band/descriptor counts, preprocessing
 *   serving/model_config.json     — architecture dimensions
 *   RESULTS_BINARY_IMPROVEMENT.md — cohort, window counts, split membership
 *   PROJECT_STATUS.md             — search history
 */

export const DATASET = {
  source: "RESULTS_BINARY_IMPROVEMENT.md · Dataset description_MATLAB.pdf",
  name: "TU Berlin open EEG + fNIRS n-back",
  citation: "Shin et al., simultaneous EEG–NIRS recordings",
  nSubjects: 26,
  /** 0-back is labelled Low load, 3-back is labelled High load. */
  classNames: ["Low (0-back)", "High (3-back)"] as const,
  windowing: { lengthSeconds: 10, strideSeconds: 5, id: "sliding_10s_5s" },
  nWindows: 4576,
  windowsPerSubject: 176,
  classBalance: { Low: 2288, High: 2288 },
} as const;

export const FEATURES = {
  source: "serving/feature_schema.json",
  eeg: {
    total: 540,
    channels: 30,
    bands: ["delta", "theta", "alpha", "beta", "gamma", "engagement"],
    stats: ["mean", "std", "slope"],
    note: "30 channels × 6 band-power measures × 3 within-window statistics",
  },
  fnirs: {
    total: 936,
    channels: 36,
    chromophores: ["HbO", "HbR"],
    perChromophore: 468,
    descriptors: [
      "mean", "std", "slope", "min", "max", "ptp", "auc",
      "early", "late", "delta", "power", "skew", "kurt",
    ],
    note: "36 channels × 13 response-shape descriptors × 2 chromophores",
  },
  totalPerWindow: 1476,
  preprocessing: [
    "nan_to_num",
    "per-subject z-score",
    "log compression",
    "saved scalers",
  ],
  preprocessingNote:
    "Per-subject normalisation happens BEFORE log compression; the scalers are fitted on training rows only, refitted inside every fold.",
} as const;

export const MODEL = {
  source: "serving/model_config.json",
  family: "Hybrid dual-encoder network with attention fusion",
  projDim: 96,
  encDim: 48,
  nBlocks: 2,
  fusion: "attention",
  fcUnits: [48, 24],
  encoderDropout: 0.5,
  fcDropout: 0.6,
  approxParams: "~246k",
  numClasses: 2,
} as const;

export const PROTOCOL = {
  source: "RESULTS_BINARY_IMPROVEMENT.md · research/campaign.py",
  developmentSubjects: [1, 2, 3, 4, 5, 7, 9, 10, 11, 12, 13, 14, 15, 17, 20, 23, 24, 25],
  lockboxSubjects: [6, 8, 16, 18, 19, 21, 22, 26],
  devFolds: 18,
  headlineFolds: 26,
  checkpointMetric: "validation macro F1",
  acceptanceRule:
    "A candidate is adopted only if it clears the baseline by at least one standard error on the development screen, then survives the full 26-fold LOSO and the lockbox.",
} as const;

export const SEARCH = {
  source: "PROJECT_STATUS.md · outputs/runs/target_campaign_log.md",
  totalConfigs: 58,
  campaigns: [
    { name: "Original campaign", driver: "run_campaign.py", configs: 42 },
    { name: "Target campaign", driver: "target_campaign.py", configs: 15 },
    { name: "Deep-ensemble screen", driver: "ensemble_campaign.py", configs: 1 },
  ],
  adopted: 0,
} as const;

/** Leakage checks asserted before any fold is allowed to train. */
export const LEAKAGE_CHECKS = [
  "Row-index disjointness across train / val / test",
  "Subject disjointness, the held-out subject appears in no other split",
  "Trial disjointness",
  "Sliding-window identity disjointness",
  "No byte-identical feature row across splits",
  "Scalers fitted on training rows only",
  "Per-subject normalisation recomputes identically in isolation",
  "Features finite, labels valid",
] as const;

/**
 * The archived run the pipeline narrative quotes per-fold numbers from.
 *
 * This run is the right one to quote because finalize_winner.py evaluated the
 * untouched baseline and a promoted candidate through the SAME two protocols
 * in the SAME environment, so its baseline folds are a clean, paired reading
 * of the served configuration rather than a cross-run comparison.
 *
 * If the archive is pruned or the id changes, the figures that use it degrade
 * to an explicit "unavailable" note — they never fall back to stale constants.
 */
export const HEADLINE_RUN = {
  runId: "run_20260823_173425_b133",
  tag: "promote-D2-balanced",
  /** The experiment directory holding the served baseline's 26 LOSO folds. */
  baselineExperiment: "BASELINE_loso26",
  candidateExperiment: "CANDIDATE_loso26",
  candidateName: "D2_narrow_patience6",
} as const;
