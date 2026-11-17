import {
  LABELS,
  LABEL_NAMES,
  WINDOW,
  formatTime,
  type Label,
  type Segment,
  type Bin,
} from "@possession/core";

const COLOR: Record<Label, string> = {
  H: "var(--blue)",
  A: "var(--green)",
  D: "var(--amber)",
  X: "var(--unknown)",
};
export function Legend() {
  return (
    <ul className="legend" aria-label="Possession key">
      {LABELS.map((label) => (
        <li key={label}>
          <span className={`swatch ${label}`} aria-hidden="true" />
          {LABEL_NAMES[label]}
        </li>
      ))}
    </ul>
  );
}
export function Gauge({ counts }: { counts: Record<Label, number> }) {
  const radius = 77;
  const circumference = 2 * Math.PI * radius;
  let offset = 0;
  return (
    <div className="gauge">
      <svg
        viewBox="0 0 220 200"
        role="img"
        aria-label={`Barcelona has ${((counts.H / WINDOW.frames) * 100).toFixed(1)} percent of the five-minute clip`}
      >
        <circle
          cx="110"
          cy="105"
          r={radius}
          fill="none"
          stroke="var(--ink)"
          strokeWidth="20"
          strokeDasharray={`${circumference * 0.75} ${circumference}`}
          transform="rotate(135 110 105)"
        />
        {LABELS.map((label) => {
          const length = (counts[label] / WINDOW.frames) * circumference * 0.75;
          const start = offset;
          offset += length;
          return length > 0 ? (
            <circle
              key={label}
              cx="110"
              cy="105"
              r={radius}
              fill="none"
              stroke={COLOR[label]}
              strokeWidth="17"
              strokeDasharray={`${length} ${circumference}`}
              strokeDashoffset={-start}
              transform="rotate(135 110 105)"
            />
          ) : null;
        })}
        <text x="110" y="107" textAnchor="middle" className="gauge-number">
          {((counts.H / WINDOW.frames) * 100).toFixed(1)}%
        </text>
        <text x="110" y="131" textAnchor="middle" className="chart-label">
          Barcelona
        </text>
      </svg>
    </div>
  );
}
export function Pitch() {
  return (
    <div className="pitch-empty">
      <svg viewBox="0 0 560 270" aria-hidden="true">
        <g fill="none" stroke="var(--control-border)" strokeWidth="1.2">
          <path d="M12 12H548V258H12ZM280 12V258M12 70H92V200H12M548 70H468V200H548" />
          <circle cx="280" cy="135" r="43" />
          <circle cx="280" cy="135" r="2" />
        </g>
      </svg>
      <div className="pitch-message">
        <strong>Player positions unavailable</strong>
        <p>
          Possession labels tell us who has the ball, not where the players are.
        </p>
      </div>
    </div>
  );
}
export function Timeline({
  segments,
  frame,
  name,
}: {
  segments: readonly Segment[];
  frame: number;
  name: string;
}) {
  return (
    <div className="timeline-track">
      <svg
        viewBox={`0 0 ${WINDOW.frames} 180`}
        preserveAspectRatio="none"
        role="img"
        aria-label={`${name}: ${segments.length} periods across five minutes`}
      >
        {segments.map((segment) => (
          <rect
            key={segment.start}
            x={segment.start}
            y="0"
            width={segment.end - segment.start}
            height="180"
            fill={COLOR[segment.label]}
            stroke="var(--ink)"
            strokeWidth="6"
          >
            <title>{`${LABEL_NAMES[segment.label]}: ${formatTime(segment.start / 25)} to ${formatTime(segment.end / 25)}`}</title>
          </rect>
        ))}
      </svg>
      <span
        className="timeline-cursor"
        style={{ left: `${(frame / WINDOW.frames) * 100}%` }}
        aria-hidden="true"
      />
    </div>
  );
}
export function ShareChart({
  bins,
  lines = false,
}: {
  bins: Bin[];
  lines?: boolean;
}) {
  const width = 500,
    height = 160,
    left = 34,
    top = 10;
  function pathFor(label: Label, area: boolean) {
    let path = "";
    const index = LABELS.indexOf(label);
    bins.forEach((bin, i) => {
      const n = bin.end - bin.start;
      const previous =
        LABELS.slice(0, index).reduce((sum, l) => sum + bin.counts[l], 0) / n;
      const value = bin.counts[label] / n;
      const y = top + height * (1 - (area ? previous + value : value));
      const x = left + (i / bins.length) * width;
      path += `${i ? "L" : "M"}${x},${y}H${left + ((i + 1) / bins.length) * width}`;
    });
    if (area) {
      for (let i = bins.length - 1; i >= 0; i--) {
        const bin = bins[i];
        const previous =
          LABELS.slice(0, index).reduce((sum, l) => sum + bin.counts[l], 0) /
          (bin.end - bin.start);
        const y = top + height * (1 - previous);
        path += `L${left + ((i + 1) / bins.length) * width},${y}H${left + (i / bins.length) * width}`;
      }
      path += "Z";
    }
    return path;
  }
  return (
    <div className="share-chart">
      <svg
        viewBox="0 0 550 200"
        role="img"
        aria-label={
          lines
            ? "Each team's share in consecutive ten-second intervals, from zero to one hundred percent"
            : "Stacked possession shares for each ten-second interval"
        }
      >
        {[0, 50, 100].map((value) => (
          <g key={value}>
            <path
              d={`M${left} ${top + height * (1 - value / 100)}H${left + width}`}
              stroke="var(--border)"
            />
            <text
              x="25"
              y={top + height * (1 - value / 100) + 4}
              textAnchor="end"
              className="chart-label"
            >
              {value}
            </text>
          </g>
        ))}
        {LABELS.filter((label) =>
          bins.some((bin) => bin.counts[label] > 0),
        ).map((label) => (
          <path
            key={label}
            d={pathFor(label, !lines)}
            fill={lines ? "none" : COLOR[label]}
            stroke={
              lines
                ? label === "A"
                  ? "var(--green-line)"
                  : label === "D"
                    ? "var(--amber-line)"
                    : COLOR[label]
                : "var(--ink)"
            }
            strokeWidth={lines ? 2 : 0.7}
          />
        ))}
        {[0, 150, 300].map((seconds) => (
          <text
            key={seconds}
            x={left + (seconds / 300) * width}
            y="193"
            textAnchor={
              seconds === 0 ? "start" : seconds === 300 ? "end" : "middle"
            }
            className="chart-label"
          >
            {formatTime(seconds)}
          </text>
        ))}
      </svg>
    </div>
  );
}
