import argparse

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon


def parse_values(raw):
    return np.asarray([float(x.strip()) for x in raw.split(",") if x.strip()], dtype=float)


def paired_report(name, hybrid, fnirs):
    if len(hybrid) != len(fnirs):
        raise ValueError(f"{name}: hybrid and fNIRS arrays must have the same length")
    if len(hybrid) < 2:
        raise ValueError(f"{name}: need at least two paired folds")

    diff = hybrid - fnirs
    print(f"\n{name}")
    print("=" * 55)
    print(f"Hybrid mean : {hybrid.mean():.4f} +/- {hybrid.std(ddof=0):.4f}")
    print(f"fNIRS mean  : {fnirs.mean():.4f} +/- {fnirs.std(ddof=0):.4f}")
    print(f"Mean diff   : {diff.mean():.4f}")

    t_stat, t_p = ttest_rel(hybrid, fnirs)
    print(f"Paired t-test: statistic={t_stat:.4f}, p={t_p:.4g}")

    try:
        w_stat, w_p = wilcoxon(hybrid, fnirs)
        print(f"Wilcoxon    : statistic={w_stat:.4f}, p={w_p:.4g}")
    except ValueError as exc:
        print(f"Wilcoxon    : skipped ({exc})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", help="CSV with paired fold columns")
    parser.add_argument("--hybrid-acc", help="Comma-separated Hybrid fold accuracies")
    parser.add_argument("--fnirs-acc", help="Comma-separated fNIRS fold accuracies")
    parser.add_argument("--hybrid-f1", help="Comma-separated Hybrid fold macro F1 values")
    parser.add_argument("--fnirs-f1", help="Comma-separated fNIRS fold macro F1 values")
    parser.add_argument("--hybrid-acc-col", default="hybrid_acc")
    parser.add_argument("--fnirs-acc-col", default="fnirs_acc")
    parser.add_argument("--hybrid-f1-col", default="hybrid_f1")
    parser.add_argument("--fnirs-f1-col", default="fnirs_f1")
    args = parser.parse_args()

    if args.csv:
        df = pd.read_csv(args.csv)
        if args.hybrid_acc_col in df and args.fnirs_acc_col in df:
            paired_report("Accuracy", df[args.hybrid_acc_col].to_numpy(), df[args.fnirs_acc_col].to_numpy())
        if args.hybrid_f1_col in df and args.fnirs_f1_col in df:
            paired_report("Macro F1", df[args.hybrid_f1_col].to_numpy(), df[args.fnirs_f1_col].to_numpy())
        return

    if args.hybrid_acc and args.fnirs_acc:
        paired_report("Accuracy", parse_values(args.hybrid_acc), parse_values(args.fnirs_acc))
    if args.hybrid_f1 and args.fnirs_f1:
        paired_report("Macro F1", parse_values(args.hybrid_f1), parse_values(args.fnirs_f1))


if __name__ == "__main__":
    main()
