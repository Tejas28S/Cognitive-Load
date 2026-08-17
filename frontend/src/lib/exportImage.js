import { toPng } from "html-to-image";

// Render a DOM node (chart/diagram + its computed styles, including CSS vars and
// SVG) to a high-res PNG and trigger a download. Used by the research figures so
// any chart can be dropped straight into slides or a poster. The background
// defaults to the current theme surface so dark-mode exports don't come out with
// black text on white; pass an explicit color to override (e.g. "#ffffff").
export async function downloadNodePng(node, filename = "figure", opts = {}) {
  if (!node) return null;
  const name = filename.endsWith(".png") ? filename : `${filename}.png`;
  const surface =
    getComputedStyle(document.documentElement).getPropertyValue("--surface").trim() ||
    "#ffffff";
  const dataUrl = await toPng(node, {
    pixelRatio: opts.pixelRatio ?? 2,
    cacheBust: true,
    backgroundColor: opts.backgroundColor ?? surface,
    style: { margin: "0" },
    filter: (el) =>
      // Never bake the export controls themselves into the image.
      !(el?.dataset && el.dataset.exportExclude === "true"),
  });
  const a = document.createElement("a");
  a.href = dataUrl;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  return name;
}

export const slugify = (s) =>
  (s || "figure")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 48);
