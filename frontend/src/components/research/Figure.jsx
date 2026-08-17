import { useRef } from "react";
import ExportMenu from "../ExportMenu";
import { slugify } from "../../lib/exportImage";

// A "plate": a chart/diagram framed with a numbered caption and the committed
// artifact it traces to. The source line reinforces the project's rule that
// every number in the app comes from a committed artifact, not hardcoded copy.
// Each plate can be exported to PNG for slides/posters.
export default function Figure({ num, title, source, children }) {
  const bodyRef = useRef(null);
  const filename = `fig${num != null ? `-${num}` : ""}-${slugify(title)}`;
  return (
    <figure className="figure" style={{ margin: 0 }}>
      <div className="figure-body" ref={bodyRef}>
        {children}
      </div>
      <figcaption className="figure-cap">
        <span className="cap-text">
          {num != null && <span className="cap-num">Fig. {num}</span>}
          {title}
        </span>
        <span className="cap-right">
          {source && <span className="cap-src">{source}</span>}
          <ExportMenu pngRef={bodyRef} filename={filename} label="PNG" />
        </span>
      </figcaption>
    </figure>
  );
}
