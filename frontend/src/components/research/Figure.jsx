// A "plate": a chart/diagram framed with a numbered caption and the committed
// artifact it traces to. The source line reinforces the project's rule that
// every number in the app comes from a committed artifact, not hardcoded copy.
export default function Figure({ num, title, source, children }) {
  return (
    <figure className="figure" style={{ margin: 0 }}>
      <div className="figure-body">{children}</div>
      <figcaption className="figure-cap">
        <span className="cap-text">
          {num != null && <span className="cap-num">Fig. {num}</span>}
          {title}
        </span>
        {source && <span className="cap-src">{source}</span>}
      </figcaption>
    </figure>
  );
}
