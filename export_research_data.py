"""
export_research_data.py — consolidate the research-result artifacts from
outputs/ into a single committed file the demo backend can serve.

The /research narrative page (Part 4) reads its numbers from these artifacts so
that, like every other value in the app, they trace to a committed artifact
rather than being hardcoded in the frontend. outputs/ is gitignored; this copies
the relevant JSON into serving/frontend_data/research.json (committed).

Run:  python export_research_data.py
(interpreter: C:/Program Files/Python314/python.exe — the repo .venv is broken)
"""

import datetime as _dt
import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUTS = os.path.join(ROOT, "outputs")
DEST = os.path.join(ROOT, "serving", "frontend_data", "research.json")

# block name -> (artifact filename, one-line provenance shown in the UI figure caption)
BLOCKS = {
    "dann": ("dann_ablation.json", "dann_ablation.py · 26-fold LOSO, paired significance + subject-invariance probe"),
    "dann_lambda_sweep": ("dann_lambda_sweep.json", "dann_ablation.py --lambda-sweep · controlled adversary on/off"),
    "protocol": ("protocol_benchmark.json", "protocol_benchmark.py · same RF across protocol axes"),
    "hbr": ("hbr_ablation.json", "hbr_ablation.py · HbO-only (468) vs HbO+HbR (936), 26-fold LOSO"),
    "hbr_served_split": ("compare_936_vs_live.json", "compare_936_vs_live.py · both bundles through the real backend"),
    "classical": ("classical_sweep.json", "classical_sweep.py · XGBoost / SVM vs Random Forest"),
    "significance": ("significance_tests.json", "part1_rigor.py · locked-model paired tests"),
}


def _load(name):
    path = os.path.join(OUTPUTS, name)
    if not os.path.exists(path):
        print(f"  ! missing {name} — skipped")
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    bundle = {
        "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "sources": {},
    }
    for key, (fname, provenance) in BLOCKS.items():
        data = _load(fname)
        if data is None:
            continue
        bundle[key] = data
        bundle["sources"][key] = {"artifact": fname, "provenance": provenance}
        print(f"  + {key:18s} <- {fname}")

    os.makedirs(os.path.dirname(DEST), exist_ok=True)
    with open(DEST, "w", encoding="utf-8") as f:
        json.dump(bundle, f, indent=2)
    size_kb = os.path.getsize(DEST) / 1024
    print(f"\nWrote {DEST} ({size_kb:.1f} KB, {len(bundle['sources'])} blocks)")


if __name__ == "__main__":
    main()
