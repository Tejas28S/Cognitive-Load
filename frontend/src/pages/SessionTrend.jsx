import { useRef, useState } from "react";
import SessionTimeline from "../components/SessionTimeline";
import ExportMenu from "../components/ExportMenu";
import { getSubjects, predictSession } from "../api/client";
import { useApi, Loading, ErrorBox } from "../useApi";
import { useModelVersion } from "../lib/modelVersion";
import { stampFilename } from "../lib/exportData";

export default function SessionTrend() {
  const { version, classNames } = useModelVersion();
  const { data: subjects, loading, error } = useApi(() => getSubjects(version), [version]);
  const [subjectId, setSubjectId] = useState(null);
  const [session, setSession] = useState(null);
  const [busy, setBusy] = useState(false);
  const [sErr, setSErr] = useState(null);
  const chartRef = useRef(null);

  if (loading) return <div className="container"><Loading /></div>;
  if (error) return <div className="container"><ErrorBox error={error} /></div>;

  const current = subjects.find((s) => s.subject_id === subjectId) || subjects[0];
  const activeSubject = current?.subject_id;

  const play = async () => {
    setBusy(true);
    setSErr(null);
    try {
      const r = await predictSession(activeSubject, 0, version);
      setSession(r);
    } catch (e) {
      setSErr(e?.message || "Session prediction failed");
    } finally {
      setBusy(false);
    }
  };

  const correct = session ? session.filter((w) => w.predicted_index === w.true_label).length : 0;
  const acc = session && session.length ? correct / session.length : 0;

  return (
    <div className="container">
      <div className="page-header">
        <div className="eyebrow eyebrow--accent" style={{ marginBottom: 8 }}>Live demo</div>
        <h1>Session Trend</h1>
        <p>
          Play back a subject's trials in sequence and watch the model's
          estimated cognitive load move over time, against the true labels.
        </p>
      </div>

      <div className="card" style={{ marginBottom: 20 }}>
        <div className="controls-row">
          <div>
            <label className="field">Demo session (subject)</label>
            <select
              value={activeSubject}
              onChange={(e) => {
                setSubjectId(Number(e.target.value));
                setSession(null);
              }}
            >
              {subjects.map((s) => (
                <option key={s.subject_id} value={s.subject_id}>
                  Subject {s.subject_id} ({s.num_windows} windows)
                </option>
              ))}
            </select>
          </div>
          <button className="btn" onClick={play} disabled={busy}>
            {busy ? "Playing…" : "Play session"}
          </button>
        </div>
      </div>

      {sErr && <div className="error">{sErr}</div>}

      {session && (
        <div className="card">
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", flexWrap: "wrap", gap: 8 }}>
            <h2 style={{ margin: 0 }}>Predicted load over the session</h2>
            <div style={{ display: "flex", alignItems: "center", gap: 14, flexWrap: "wrap" }}>
              <span className="muted" style={{ fontSize: "0.85rem" }}>
                Session accuracy: <strong>{(acc * 100).toFixed(0)}%</strong> ({correct}/
                {session.length} windows)
              </span>
              <ExportMenu
                pngRef={chartRef}
                csvRows={session.map((w) => ({
                  window_index: w.window_index,
                  timestamp_s: w.timestamp_s,
                  predicted_label: w.predicted_label,
                  true_label: w.true_label_name,
                  correct: w.predicted_index === w.true_label ? 1 : 0,
                  // One probability column per class the served model actually
                  // reports. Hardcoding Low/Medium/High emitted two empty
                  // columns for the binary bundle.
                  ...Object.fromEntries(
                    Object.entries(w.probabilities || {}).map(([k, v]) => [
                      `p_${k.toLowerCase()}`,
                      v,
                    ])
                  ),
                }))}
                jsonData={session}
                filename={stampFilename(["session", `subject-${activeSubject}`])}
              />
            </div>
          </div>
          <div style={{ height: 14 }} />
          <div ref={chartRef}>
            <SessionTimeline classNames={classNames || undefined} session={session} />
          </div>
          <div className="note" style={{ marginTop: 16 }}>
            Read the top two lanes together: where they match, the model agreed
            with the ground truth. The error lane marks every window it got
            wrong, and the confidence trace below shows how sure it was. These
            are demo windows chosen for coherent playback, not a fresh
            recording, and none of this is a diagnosis.
          </div>
        </div>
      )}

      {!session && !sErr && (
        <div className="muted" style={{ padding: "8px 4px" }}>
          Pick a subject and press <strong>Play session</strong>.
        </div>
      )}
    </div>
  );
}
