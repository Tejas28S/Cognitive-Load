// The decision-gate outcome. This is the project's recurring structural motif:
// every candidate change is judged by the same gate and carries an explicit
// verdict - ADOPTED, NOT ADOPTED, RULED OUT, RETAINED. tone maps the verdict to
// a color: ok (kept/adopted), warn (retained-by-default), stop (declined).
//
// Layout: the heading leads the card. The verdict word sits directly under it
// as quiet tracked text rather than the boxed badge and coloured left border it
// used to carry - colour is now the text's own, not a bar down the side.
export default function VerdictCallout({ tone = "warn", stamp, title, children }) {
  return (
    <div className={`verdict verdict--${tone}`} role="note">
      {title && <div className="verdict-title">{title}</div>}
      {stamp && <span className="verdict-stamp">{stamp}</span>}
      <div className="verdict-body">{children}</div>
    </div>
  );
}
