import { useEffect, useRef } from "react";

// Animated hero backdrop: drifting brain-signal "nodes" connected by faint
// edges (EEG↔fNIRS motif) on a canvas, layered under the .band-dark gradient.
// Adds a subtle scroll parallax. Honors reduced-motion by rendering a single
// static frame. Sits absolutely behind the hero content.
//
// The edge colour is read from the --ink token rather than hardcoded white:
// the hero band is a light wash now, and white-on-white drew nothing.
export default function AnimatedBand() {
  const canvasRef = useRef(null);
  const rafRef = useRef(0);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    let w = 0;
    let h = 0;
    let dpr = Math.min(window.devicePixelRatio || 1, 2);
    const nodes = [];
    const N = 42;

    const css = () => {
      const read = (name, fallback) =>
        getComputedStyle(document.documentElement).getPropertyValue(name).trim() || fallback;
      return {
        eeg: read("--eeg", "#6366f1"),
        fnirs: read("--fnirs", "#06b6d4"),
        edge: read("--ink-3", "#8a94a1"),
      };
    };
    let colors = css();

    const resize = () => {
      const rect = canvas.getBoundingClientRect();
      w = rect.width;
      h = rect.height;
      dpr = Math.min(window.devicePixelRatio || 1, 2);
      canvas.width = w * dpr;
      canvas.height = h * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };

    const init = () => {
      nodes.length = 0;
      for (let i = 0; i < N; i++) {
        nodes.push({
          x: Math.random() * w,
          y: Math.random() * h,
          vx: (Math.random() - 0.5) * 0.25,
          vy: (Math.random() - 0.5) * 0.25,
          r: 1 + Math.random() * 2,
          eeg: Math.random() > 0.5,
        });
      }
    };

    const draw = () => {
      ctx.clearRect(0, 0, w, h);
      // edges
      for (let i = 0; i < nodes.length; i++) {
        for (let j = i + 1; j < nodes.length; j++) {
          const a = nodes[i];
          const b = nodes[j];
          const dx = a.x - b.x;
          const dy = a.y - b.y;
          const d2 = dx * dx + dy * dy;
          if (d2 < 128 * 128) {
            const alpha = (1 - Math.sqrt(d2) / 128) * 0.34;
            ctx.strokeStyle = colors.edge;
            ctx.globalAlpha = alpha;
            ctx.lineWidth = 0.6;
            ctx.beginPath();
            ctx.moveTo(a.x, a.y);
            ctx.lineTo(b.x, b.y);
            ctx.stroke();
            ctx.globalAlpha = 1;
          }
        }
      }
      // nodes
      for (const n of nodes) {
        ctx.fillStyle = n.eeg ? colors.eeg : colors.fnirs;
        ctx.globalAlpha = 0.55;
        ctx.beginPath();
        ctx.arc(n.x, n.y, n.r, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.globalAlpha = 1;
    };

    const step = () => {
      for (const n of nodes) {
        n.x += n.vx;
        n.y += n.vy;
        if (n.x < 0 || n.x > w) n.vx *= -1;
        if (n.y < 0 || n.y > h) n.vy *= -1;
      }
      draw();
      rafRef.current = requestAnimationFrame(step);
    };

    resize();
    init();
    if (reduced) {
      draw();
    } else {
      rafRef.current = requestAnimationFrame(step);
    }

    const onResize = () => {
      resize();
      init();
      if (reduced) draw();
    };
    const themeObs = new MutationObserver(() => {
      colors = css();
    });
    themeObs.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
    window.addEventListener("resize", onResize);

    return () => {
      cancelAnimationFrame(rafRef.current);
      window.removeEventListener("resize", onResize);
      themeObs.disconnect();
    };
  }, []);

  return <canvas ref={canvasRef} className="animated-band" aria-hidden="true" />;
}
