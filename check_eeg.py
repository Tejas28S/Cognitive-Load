import numpy as np
import pandas as pd


def summarize_skew(df, label, prefixes):
    cols = [c for c in df.columns if c.lower().startswith(prefixes)]
    print(f"\n{label}: found {len(cols)} columns")
    if not cols:
        return

    summary = pd.DataFrame(
        {
            "mean_abs": df[cols].abs().mean(),
            "std": df[cols].std(),
            "p75_abs": df[cols].abs().quantile(0.75),
            "max_abs": df[cols].abs().max(),
            "min": df[cols].min(),
            "max": df[cols].max(),
        }
    )
    summary["std_over_mean_abs"] = summary["std"] / summary["mean_abs"].replace(0, np.nan)
    summary["max_over_p75_abs"] = summary["max_abs"] / summary["p75_abs"].replace(0, np.nan)

    print(f"{df[cols].isna().sum().sum()} NaNs")
    print("\nLowest-variance columns:")
    print(df[cols].var().sort_values().head(10))
    print("\nMost skewed/outlier-heavy columns:")
    print(summary.sort_values("max_over_p75_abs", ascending=False).head(10).round(4))


df = pd.read_csv("data/tuberlin_sliding.csv")
summarize_skew(df, "EEG", ("eeg_",))
summarize_skew(df, "fNIRS", ("hbo_", "hbr_"))
