export type Label = "H" | "A" | "D" | "X";

export type Segment = {
  start: number;
  end: number;
  label: Label;
};

export type Bin = {
  start: number;
  end: number;
  counts: Record<Label, number>;
};

export const WINDOW = Object.freeze({
  firstFrame: 11185,
  endFrame: 18685,
  fps: 25,
  frames: 7500,
  seconds: 300,
});

export const LABELS = Object.freeze(["H", "A", "D", "X"] as const);

export const LABEL_NAMES: Readonly<Record<Label, string>> = Object.freeze({
  H: "Barcelona",
  A: "Athletic Club",
  D: "Dead ball",
  X: "Unknown",
});

const FULL_MATCH_FRAMES = 199319;
const BIN_FRAMES = 250;

function isLabel(value: unknown): value is Label {
  return value === "H" || value === "A" || value === "D" || value === "X";
}

function emptyCounts(): Record<Label, number> {
  return { H: 0, A: 0, D: 0, X: 0 };
}

export function parsePossessionCsv(text: string): Segment[] {
  const lines = (text.charCodeAt(0) === 0xfeff ? text.slice(1) : text).split(/\r?\n/);
  if (lines[lines.length - 1] === "") lines.pop();
  const header = lines[0]?.split(",").map((column) => column.trim());
  if (header?.length !== 2 || header[0] !== "frame" || header[1] !== "label") {
    throw new Error("CSV columns must be exactly frame,label, in that order.");
  }

  const segments: Segment[] = [];
  let expectedFrame: number | undefined;
  let requiredEnd = 0;
  let coveredFrames = 0;

  for (let row = 1; row < lines.length; row++) {
    const cells = lines[row].split(",");
    if (cells.length !== 2) {
      throw new Error(`CSV row ${row + 1} must have exactly two cells.`);
    }
    const frameText = cells[0].trim();
    if (!/^\d+$/.test(frameText)) {
      throw new Error(`CSV row ${row + 1} has an invalid frame; use nonnegative integers.`);
    }
    const frame = Number(frameText);
    if (!Number.isSafeInteger(frame)) {
      throw new Error(`CSV row ${row + 1} has a frame outside the safe integer range.`);
    }
    const label = cells[1].trim();
    if (!isLabel(label)) {
      throw new Error(`CSV row ${row + 1} has an invalid label; use H, A, D, or X.`);
    }

    if (expectedFrame === undefined) {
      if (frame !== 0 && frame !== WINDOW.firstFrame) {
        throw new Error("CSV must start at frame 0 for a full match or frame 11185 for the public clip.");
      }
      expectedFrame = frame;
      requiredEnd = frame === 0 ? FULL_MATCH_FRAMES : WINDOW.endFrame;
    }
    if (frame !== expectedFrame || frame >= requiredEnd) {
      throw new Error(`CSV row ${row + 1} breaks continuous frame coverage or extends past the selected file range.`);
    }
    expectedFrame++;

    if (frame >= WINDOW.firstFrame && frame < WINDOW.endFrame) {
      const relativeFrame = frame - WINDOW.firstFrame;
      const previous = segments[segments.length - 1];
      if (previous?.label === label) {
        previous.end = relativeFrame + 1;
      } else {
        segments.push({ start: relativeFrame, end: relativeFrame + 1, label });
      }
      coveredFrames++;
    }
  }

  if (expectedFrame !== requiredEnd || coveredFrames !== WINDOW.frames) {
    throw new Error("CSV must contain the complete full match or exactly all 7500 public-clip frames.");
  }
  return segments;
}

export function aggregate(segments: readonly Segment[]): {
  counts: Record<Label, number>;
  possessions: number;
  liveHomeShare: number;
  bins: Bin[];
} {
  const counts = emptyCounts();
  const bins: Bin[] = Array.from({ length: WINDOW.frames / BIN_FRAMES }, (_, index) => ({
    start: index * BIN_FRAMES,
    end: (index + 1) * BIN_FRAMES,
    counts: emptyCounts(),
  }));
  let expectedStart = 0;
  let possessions = 0;
  let previousLabel: Label | undefined;

  for (const segment of segments) {
    if (
      !Number.isInteger(segment.start) ||
      !Number.isInteger(segment.end) ||
      segment.start !== expectedStart ||
      segment.end <= segment.start ||
      segment.end > WINDOW.frames ||
      !isLabel(segment.label)
    ) {
      throw new Error("Segments must continuously cover [0, 7500) with integer frame boundaries and valid labels.");
    }
    counts[segment.label] += segment.end - segment.start;
    if ((segment.label === "H" || segment.label === "A") && segment.label !== previousLabel) {
      possessions++;
    }
    previousLabel = segment.label;
    expectedStart = segment.end;

    let position = segment.start;
    while (position < segment.end) {
      const bin = bins[Math.floor(position / BIN_FRAMES)];
      const end = Math.min(segment.end, bin.end);
      bin.counts[segment.label] += end - position;
      position = end;
    }
  }

  if (expectedStart !== WINDOW.frames) {
    throw new Error("Segments must cover all 7500 public-clip frames.");
  }
  const liveFrames = counts.H + counts.A;
  return { counts, possessions, liveHomeShare: liveFrames === 0 ? 0 : counts.H / liveFrames, bins };
}

export function labelAt(segments: readonly Segment[], frame: number): Label | null {
  if (!Number.isInteger(frame) || frame < 0 || frame >= WINDOW.frames) return null;
  let low = 0;
  let high = segments.length - 1;
  while (low <= high) {
    const middle = Math.floor((low + high) / 2);
    const segment = segments[middle];
    if (frame < segment.start) high = middle - 1;
    else if (frame >= segment.end) low = middle + 1;
    else return segment.label;
  }
  return null;
}

export function formatTime(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) {
    throw new RangeError("Time must be a finite, nonnegative number of seconds.");
  }
  const wholeSeconds = Math.floor(seconds);
  const minutes = Math.floor(wholeSeconds / 60);
  return `${String(minutes).padStart(2, "0")}:${String(wholeSeconds % 60).padStart(2, "0")}`;
}

export const DEMO_SEGMENTS: readonly Segment[] = Object.freeze([
  Object.freeze({ start: 0, end: 175, label: "A" as const }),
  Object.freeze({ start: 175, end: 370, label: "H" as const }),
  Object.freeze({ start: 370, end: 445, label: "A" as const }),
  Object.freeze({ start: 445, end: 1500, label: "H" as const }),
  Object.freeze({ start: 1500, end: 1675, label: "A" as const }),
  Object.freeze({ start: 1675, end: 2505, label: "H" as const }),
  Object.freeze({ start: 2505, end: 2600, label: "A" as const }),
  Object.freeze({ start: 2600, end: 3670, label: "H" as const }),
  Object.freeze({ start: 3670, end: 4710, label: "A" as const }),
  Object.freeze({ start: 4710, end: 5185, label: "D" as const }),
  Object.freeze({ start: 5185, end: 6275, label: "A" as const }),
  Object.freeze({ start: 6275, end: 6645, label: "D" as const }),
  Object.freeze({ start: 6645, end: 6855, label: "A" as const }),
  Object.freeze({ start: 6855, end: 7065, label: "D" as const }),
  Object.freeze({ start: 7065, end: 7500, label: "H" as const }),
]);
