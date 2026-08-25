import type { ElementType, ReactNode } from "react";
import { motion, useReducedMotion } from "motion/react";

type RevealProps = {
  children: ReactNode;
  /** Render as a different element (section, li, figure...). Defaults to div. */
  as?: ElementType;
  /** Seconds of delay — used to stagger siblings by hand where a container
   *  variant isn't practical (e.g. across a CSS grid). */
  delay?: number;
  /** Travel distance in px. Small by default: motion should clarify that a new
   *  block has arrived, not throw it across the viewport. */
  y?: number;
  className?: string;
};

/**
 * Scroll-triggered entrance. Fires once, when ~15% of the block has entered,
 * with a slight upward slide on the project's single easing curve.
 *
 * Honours prefers-reduced-motion by rendering the final state immediately
 * rather than by shortening the animation — a fast animation is still motion.
 */
export default function Reveal({
  children,
  as = "div",
  delay = 0,
  y = 18,
  className,
}: RevealProps) {
  const reduced = useReducedMotion();
  const MotionTag = motion[as as keyof typeof motion] as typeof motion.div;

  if (reduced) {
    const Tag = as;
    return <Tag className={className}>{children}</Tag>;
  }

  return (
    <MotionTag
      className={className}
      initial={{ opacity: 0, y }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, amount: 0.15 }}
      transition={{
        duration: 0.55,
        delay,
        ease: [0.2, 0.65, 0.2, 1],
      }}
    >
      {children}
    </MotionTag>
  );
}

/**
 * Container that staggers its direct <RevealItem> children. Use where the
 * children are siblings in normal flow; use Reveal's `delay` prop where they
 * are laid out by a grid and the DOM order isn't the reading order.
 */
export function RevealGroup({
  children,
  className,
  stagger = 0.08,
}: {
  children: ReactNode;
  className?: string;
  stagger?: number;
}) {
  const reduced = useReducedMotion();
  if (reduced) return <div className={className}>{children}</div>;

  return (
    <motion.div
      className={className}
      initial="hidden"
      whileInView="shown"
      viewport={{ once: true, amount: 0.15 }}
      variants={{
        hidden: {},
        shown: { transition: { staggerChildren: stagger } },
      }}
    >
      {children}
    </motion.div>
  );
}

export function RevealItem({
  children,
  className,
  y = 18,
}: {
  children: ReactNode;
  className?: string;
  y?: number;
}) {
  const reduced = useReducedMotion();
  if (reduced) return <div className={className}>{children}</div>;

  return (
    <motion.div
      className={className}
      variants={{
        hidden: { opacity: 0, y },
        shown: {
          opacity: 1,
          y: 0,
          transition: { duration: 0.55, ease: [0.2, 0.65, 0.2, 1] },
        },
      }}
    >
      {children}
    </motion.div>
  );
}
