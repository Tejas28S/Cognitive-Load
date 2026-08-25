import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { motion, useReducedMotion } from "motion/react";

export type Stage = {
  id: string;
  label: string;
  /** Two or three words shown under the label on wide screens. */
  hint?: string;
};

/**
 * The pipeline progress rail: one node per research stage, pinned below the
 * site nav, tracking which stage is currently in view.
 *
 * Scroll-spy uses IntersectionObserver rather than scroll-offset arithmetic so
 * it stays correct when sections have wildly different heights (the LOSO
 * section is ~4x the data-collection one) and when figures load late and
 * reflow the page.
 *
 * The rootMargin crops the observation band to a strip just under the pinned
 * rail: -140px off the top clears the nav + rail, and -55% off the bottom
 * means a section only becomes "active" once it has genuinely taken over the
 * viewport, instead of the next one stealing focus as its first pixel appears.
 */
export default function PipelineStepper({ stages }: { stages: Stage[] }) {
  const [activeIndex, setActiveIndex] = useState(0);
  const railRef = useRef<HTMLOListElement>(null);
  const nodeRefs = useRef<(HTMLLIElement | null)[]>([]);
  const dotRefs = useRef<(HTMLSpanElement | null)[]>([]);
  const [railTop, setRailTop] = useState<number | null>(null);
  const reduced = useReducedMotion();

  // Measure where the node dots actually sit rather than hardcoding an offset.
  // The stack above each dot (index number, gap, padding) changes height with
  // the font scale and the user's zoom, so a fixed `top` drifts off the dots
  // exactly when accessibility settings are in play.
  useLayoutEffect(() => {
    const measure = () => {
      const dot = dotRefs.current[0];
      const rail = railRef.current;
      if (!dot || !rail) return;
      const dotBox = dot.getBoundingClientRect();
      const railBox = rail.getBoundingClientRect();
      setRailTop(dotBox.top - railBox.top + dotBox.height / 2);
    };
    measure();
    window.addEventListener("resize", measure);
    return () => window.removeEventListener("resize", measure);
  }, [stages]);

  useEffect(() => {
    const els = stages
      .map((s) => document.getElementById(s.id))
      .filter((el): el is HTMLElement => Boolean(el));
    if (!els.length) return;

    const observer = new IntersectionObserver(
      (entries) => {
        const visible = entries
          .filter((e) => e.isIntersecting)
          .sort(
            (a, b) => a.boundingClientRect.top - b.boundingClientRect.top
          );
        if (!visible[0]) return;
        const idx = stages.findIndex((s) => s.id === visible[0].target.id);
        if (idx >= 0) setActiveIndex(idx);
      },
      { rootMargin: "-140px 0px -55% 0px", threshold: 0 }
    );

    els.forEach((el) => observer.observe(el));
    return () => observer.disconnect();
  }, [stages]);

  // Keep the active node visible in the mobile strip, which scrolls
  // horizontally and would otherwise leave the user's position off-screen.
  useEffect(() => {
    const node = nodeRefs.current[activeIndex];
    const rail = railRef.current;
    if (!node || !rail) return;
    if (rail.scrollWidth <= rail.clientWidth) return;

    const target =
      node.offsetLeft - rail.clientWidth / 2 + node.clientWidth / 2;
    rail.scrollTo({
      left: Math.max(0, target),
      behavior: reduced ? "auto" : "smooth",
    });
  }, [activeIndex, reduced]);

  const progress =
    stages.length > 1 ? activeIndex / (stages.length - 1) : 0;

  return (
    <div className="sticky top-[63px] z-30 border-y border-line bg-paper/85 backdrop-blur-md backdrop-saturate-150">
      <nav
        aria-label="Research pipeline stages"
        className="mx-auto max-w-(--container-wrap-wide) px-4 sm:px-6"
      >
        <ol
          ref={railRef}
          className="relative flex gap-1 overflow-x-auto py-3 [scrollbar-width:none] [&::-webkit-scrollbar]:hidden lg:grid lg:gap-0 lg:overflow-visible"
          style={{
            gridTemplateColumns: `repeat(${stages.length}, minmax(0, 1fr))`,
          }}
        >
          {/* Connecting rail. Sits behind the nodes, spanning centre-to-centre
              of the first and last node rather than the full width, so the line
              never juts out past the end dots. */}
          <li
            aria-hidden="true"
            className="pointer-events-none absolute hidden lg:block"
            style={{
              top: railTop ?? 0,
              opacity: railTop == null ? 0 : 1,
              left: `${100 / stages.length / 2}%`,
              right: `${100 / stages.length / 2}%`,
            }}
          >
            <div className="h-px w-full bg-line-strong" />
            <motion.div
              className="absolute inset-y-0 left-0 h-px bg-accent origin-left"
              initial={false}
              animate={{ scaleX: progress }}
              transition={
                reduced
                  ? { duration: 0 }
                  : { duration: 0.5, ease: [0.2, 0.65, 0.2, 1] }
              }
              style={{ width: "100%" }}
            />
          </li>

          {stages.map((stage, i) => {
            const state =
              i === activeIndex
                ? "active"
                : i < activeIndex
                  ? "done"
                  : "todo";

            return (
              <li
                key={stage.id}
                ref={(el) => {
                  nodeRefs.current[i] = el;
                }}
                className="relative shrink-0 lg:shrink"
              >
                <a
                  href={`#${stage.id}`}
                  aria-current={state === "active" ? "step" : undefined}
                  className="group flex min-w-[8.5rem] flex-col items-center gap-2 px-2 py-1 text-center lg:min-w-0"
                >
                  <span className="eyebrow-mono text-2xs text-ink-3 tnum">
                    {String(i + 1).padStart(2, "0")}
                  </span>

                  {/* The node. A ring rather than a filled dot for upcoming
                      stages, so "where am I" reads at a glance. */}
                  <span
                    ref={(el) => {
                      dotRefs.current[i] = el;
                    }}
                    className="relative flex h-3.5 w-3.5 items-center justify-center"
                  >
                    <span
                      aria-hidden="true"
                      className={`absolute inset-0 rounded-full bg-accent/20 transition-transform duration-500 ease-signal ${
                        state === "active" ? "scale-[2.2]" : "scale-0"
                      }`}
                    />
                    <span
                      className={`relative h-2.5 w-2.5 rounded-full border-2 transition-colors duration-300 ${
                        state === "todo"
                          ? "border-line-strong bg-paper group-hover:border-ink-3"
                          : "border-accent bg-accent"
                      }`}
                    />
                  </span>

                  <span
                    className={`text-sm leading-tight font-medium transition-colors duration-300 ${
                      state === "active"
                        ? "text-accent"
                        : state === "done"
                          ? "text-ink-2"
                          : "text-ink-3 group-hover:text-ink-2"
                    }`}
                  >
                    {stage.label}
                  </span>

                  {stage.hint && (
                    <span className="hidden text-2xs text-ink-3 xl:block">
                      {stage.hint}
                    </span>
                  )}
                </a>
              </li>
            );
          })}
        </ol>
      </nav>
    </div>
  );
}
