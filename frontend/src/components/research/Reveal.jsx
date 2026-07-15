import { useEffect, useRef, useState } from "react";

// Fades + lifts children in once they scroll into view. Under reduced-motion the
// CSS (.reveal rules in index.css) renders them visible with no transition.
export default function Reveal({ children, as: Tag = "div", className = "", ...rest }) {
  const ref = useRef(null);
  const [seen, setSeen] = useState(false);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const obs = new IntersectionObserver(
      (entries) => {
        if (entries[0].isIntersecting) {
          setSeen(true);
          obs.disconnect();
        }
      },
      { threshold: 0.12, rootMargin: "0px 0px -8% 0px" }
    );
    obs.observe(el);
    return () => obs.disconnect();
  }, []);

  return (
    <Tag ref={ref} className={`reveal ${seen ? "in" : ""} ${className}`.trim()} {...rest}>
      {children}
    </Tag>
  );
}
