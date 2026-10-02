# Demo upload files

Ready-to-upload CSVs for the **Upload Data** page (`/upload`, "Upload Extracted CSV" mode).
Each row is a *real* held-out window from the TU Berlin dataset — randomly sampled
across several subjects (`random.seed(7)`), not cherry-picked for accuracy. This
matches the site's own "no leakage, no cherry-picking" framing, so the numbers
you see live are the same kind of honest numbers reported everywhere else on
the site.

| File | Model to select | Rows | Subjects | What it's for |
|---|---|---|---|---|
| `upload_sample_2class.csv` | **2-class** | 24 | 1, 2, 5, 7, 11, 17 | Main demo: a mixed batch, roughly matches the model's real ~63% LOSO accuracy — good for pointing at the "Model reliability" section and confusion matrix in the generated report. |
| `upload_sample_3class.csv` | **3-class** | 24 | 1, 2, 8, 17, 18, 24 | Same idea for the 3-class (Low/Medium/High) stand-in model. |
| `upload_sample_quick_singlesubject.csv` | **2-class** | 5 | 15 | Fast, low-friction demo if you're short on time — one subject, five windows. |

## How to use tomorrow

1. On the site header, set **MODEL** to match the table above before uploading
   (2-class vs 3-class — the upload endpoint is version-specific).
2. Go to **Assess → Upload Data**, leave it on "Upload Extracted CSV".
3. Drag the file in (or click to browse) and hit **Upload & Predict**.
4. Point out:
   - the **Summary** card's accuracy-vs-uploaded-labels stat (it will land near
     the model's published LOSO accuracy, not 100% — that's honest, not a bug)
   - **Per-row predictions**, click **Report** on any row to open the full
     per-window report (verdict, probability chart, recall-by-class chart,
     confusion matrix — added today)
   - **Export** on either the table or an individual report to show PNG/PDF/JSON
     export works live.

## Regenerating / making more

Source: `serving/demo_samples.json` + `serving/feature_schema.json` (2-class),
`serving_3class/` equivalents (3-class). Each file's `Cognitive_Load` column is
the **numeric** class index (0/1[/2]) — the upload endpoint does `int(labels[i])`,
so a string label name there throws a 500. `subject_id` must be present for
correct per-subject normalization (matches how the model was trained).
