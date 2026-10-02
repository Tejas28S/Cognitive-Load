import { toPng } from "html-to-image";
import { jsPDF } from "jspdf";

// Render a DOM node to a paginated A4 PDF and trigger a download. Rasterizes
// the node once (same approach as exportImage.js's PNG export, so the two
// stay visually identical) then slices the tall image across as many pages
// as it needs: a report can run longer than one page once the reliability
// and limitations sections are included.
export async function downloadNodePdf(node, filename = "report", opts = {}) {
  if (!node) return null;
  const name = filename.endsWith(".pdf") ? filename : `${filename}.pdf`;
  const surface =
    getComputedStyle(document.documentElement).getPropertyValue("--surface").trim() ||
    "#ffffff";

  // A lower pixel ratio than the PNG export: a printed A4 page only resolves
  // to ~150-200dpi anyway, and re-encoding as JPEG (the source is opaque -
  // backgroundColor is forced below, so there's no transparency to lose)
  // keeps a multi-page report a few hundred KB instead of double-digit MB.
  const dataUrl = await toPng(node, {
    pixelRatio: opts.pixelRatio ?? 1.5,
    cacheBust: true,
    backgroundColor: opts.backgroundColor ?? surface,
    style: { margin: "0" },
    filter: (el) => !(el?.dataset && el.dataset.exportExclude === "true"),
  });
  const img = await loadImage(dataUrl);
  const jpegQuality = opts.jpegQuality ?? 0.92;

  const pdf = new jsPDF({ unit: "mm", format: "a4" });
  const pageWidth = pdf.internal.pageSize.getWidth();
  const pageHeight = pdf.internal.pageSize.getHeight();
  const margin = 10;
  const usableWidth = pageWidth - margin * 2;
  const usableHeight = pageHeight - margin * 2;

  const pxPerMm = img.width / usableWidth;
  const pageHeightPx = usableHeight * pxPerMm;
  const totalPages = Math.max(1, Math.ceil(img.height / pageHeightPx));

  const canvas = document.createElement("canvas");
  canvas.width = img.width;
  const ctx = canvas.getContext("2d");

  for (let page = 0; page < totalPages; page++) {
    const sliceHeightPx = Math.min(pageHeightPx, img.height - page * pageHeightPx);
    canvas.height = sliceHeightPx;
    // JPEG has no alpha channel: paint the page surface color first so a
    // slice that's mostly empty space doesn't composite onto black.
    ctx.fillStyle = surface;
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, -page * pageHeightPx);
    const sliceDataUrl = canvas.toDataURL("image/jpeg", jpegQuality);
    if (page > 0) pdf.addPage();
    pdf.addImage(sliceDataUrl, "JPEG", margin, margin, usableWidth, sliceHeightPx / pxPerMm);
  }

  pdf.save(name);
  return name;
}

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error("Could not rasterize report for PDF export"));
    img.src = src;
  });
}
