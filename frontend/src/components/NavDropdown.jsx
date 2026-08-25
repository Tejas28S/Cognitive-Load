import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { NavLink, useLocation } from "react-router-dom";

// A nav-bar dropdown: groups several routes under one trigger so the top-level
// bar stays short. Pattern-matches ExportMenu's open/close mechanics.
//
// The menu is rendered into a portal on document.body, positioned with
// getBoundingClientRect(). .nav-links has overflow-x:auto for horizontal
// scroll-safety on narrow screens, and per the CSS overflow spec that forces
// overflow-y to 'auto' too (a mismatched visible/non-visible pair isn't
// allowed) - so a menu positioned inside it would be clipped invisible even
// while technically open. Escaping to the body sidesteps that entirely.
export default function NavDropdown({ label, items }) {
  const [open, setOpen] = useState(false);
  const [menuPos, setMenuPos] = useState(null);
  const wrapRef = useRef(null);
  const triggerRef = useRef(null);
  const menuRef = useRef(null);
  const location = useLocation();

  const isActive = items.some((it) => location.pathname === it.to);

  const toggle = () => {
    if (!open && triggerRef.current) {
      const r = triggerRef.current.getBoundingClientRect();
      setMenuPos({ top: r.bottom + 6, left: r.left });
    }
    setOpen((o) => !o);
  };

  useEffect(() => {
    if (!open) return;
    const onDown = (e) => {
      const inTrigger = wrapRef.current && wrapRef.current.contains(e.target);
      const inMenu = menuRef.current && menuRef.current.contains(e.target);
      if (!inTrigger && !inMenu) setOpen(false);
    };
    const onKey = (e) => e.key === "Escape" && setOpen(false);
    // Keep the menu glued to the trigger through resizes (the nav itself is
    // sticky-pinned, so plain page scroll doesn't move the trigger).
    const onResize = () => {
      if (triggerRef.current) {
        const r = triggerRef.current.getBoundingClientRect();
        setMenuPos({ top: r.bottom + 6, left: r.left });
      }
    };
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    window.addEventListener("resize", onResize);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
      window.removeEventListener("resize", onResize);
    };
  }, [open]);

  useEffect(() => {
    setOpen(false);
  }, [location.pathname]);

  return (
    <div className="nav-dropdown" ref={wrapRef}>
      <button
        ref={triggerRef}
        type="button"
        className={`nav-dropdown-trigger${isActive ? " active" : ""}`}
        onClick={toggle}
        aria-haspopup="menu"
        aria-expanded={open}
      >
        {label}
        <span className="nav-dropdown-caret" aria-hidden="true">▾</span>
      </button>
      {open && menuPos &&
        createPortal(
          <div
            className="nav-dropdown-menu"
            role="menu"
            ref={menuRef}
            style={{ position: "fixed", top: menuPos.top, left: menuPos.left }}
          >
            {items.map((it) => (
              <NavLink
                key={it.to}
                to={it.to}
                role="menuitem"
                className={({ isActive: linkActive }) =>
                  `nav-dropdown-item${linkActive ? " active" : ""}`
                }
              >
                <span className="nav-dropdown-item-label">{it.label}</span>
                {it.hint && <span className="nav-dropdown-item-hint">{it.hint}</span>}
              </NavLink>
            ))}
          </div>,
          document.body
        )}
    </div>
  );
}
