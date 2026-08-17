// Shimmer skeleton placeholders shown while data loads - a premium upgrade over
// the plain "Loading…" text. `SkeletonCards` mimics the common page layout
// (header + a couple of chart cards) so the swap to real content is calm.
export function Skeleton({ w = "100%", h = 16, r = 6, style }) {
  return <span className="skeleton" style={{ width: w, height: h, borderRadius: r, ...style }} />;
}

export function SkeletonCard({ lines = 3, chart = false }) {
  return (
    <div className="card">
      <Skeleton w="42%" h={22} style={{ display: "block", marginBottom: 14 }} />
      {chart ? (
        <Skeleton w="100%" h={220} r={10} style={{ display: "block" }} />
      ) : (
        Array.from({ length: lines }).map((_, i) => (
          <Skeleton key={i} w={`${90 - i * 12}%`} h={13} style={{ display: "block", marginBottom: 9 }} />
        ))
      )}
    </div>
  );
}

// Page-level loading state: header stub + a grid of chart-card skeletons.
export default function SkeletonPage({ cards = 2 }) {
  return (
    <div className="container" aria-busy="true" aria-live="polite">
      <div style={{ marginBottom: 30 }}>
        <Skeleton w={120} h={12} style={{ display: "block", marginBottom: 12 }} />
        <Skeleton w="55%" h={38} style={{ display: "block", marginBottom: 12 }} />
        <Skeleton w="70%" h={14} style={{ display: "block" }} />
      </div>
      <div className="grid grid-2">
        {Array.from({ length: cards }).map((_, i) => (
          <SkeletonCard key={i} chart />
        ))}
      </div>
    </div>
  );
}
