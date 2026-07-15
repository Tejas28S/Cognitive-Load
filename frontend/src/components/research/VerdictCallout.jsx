// The decision-gate outcome, stamped. This is the project's recurring structural
// motif: every candidate change is judged by the same gate and carries an
// explicit verdict — ADOPTED, NOT ADOPTED, RULED OUT, RETAINED. tone maps the
// stamp to a color: ok (kept/adopted), warn (retained-by-default), stop (declined).
export default function VerdictCallout({ tone = "warn", stamp, title, children }) {
  return (
    <div className={`verdict verdict--${tone}`} role="note">
      <span className="verdict-stamp">{stamp}</span>
      <div>
        {title && <div className="verdict-title">{title}</div>}
        <div className="verdict-body">{children}</div>
      </div>
    </div>
  );
}
