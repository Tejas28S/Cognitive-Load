import { useModelVersion } from "../lib/modelVersion";

// Shown only when the served bundle is NOT the original frozen research
// checkpoint. The evaluation artifacts in such a bundle still describe the
// original model, so without this the page would quietly present published
// numbers next to a different model's live predictions.
export default function ModelProvenanceNote({ style }) {
  const { active } = useModelVersion();
  const p = active?.provenance;
  if (!p) return null;

  return (
    <div className="note" style={style}>
      <strong>Stand-in model.</strong> {p.warning} Trained by{" "}
      <code>{p.trained_by}</code> on {p.n_subjects} subjects; measured on
      held-out subjects {p.validation_subjects?.join(" and ")}:{" "}
      <strong>{(p.heldout_subject_accuracy * 100).toFixed(1)}% accuracy</strong>,{" "}
      macro F1 {p.heldout_subject_macro_f1}. Charts and metrics on this page come
      from the original 3-class evaluation artifacts and describe that model, not
      this one.
    </div>
  );
}
