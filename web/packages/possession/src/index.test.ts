import assert from "node:assert/strict";
import test from "node:test";
import {
  aggregate,
  DEMO_SEGMENTS,
  formatTime,
  LABEL_NAMES,
  LABELS,
  labelAt,
  parsePossessionCsv,
  WINDOW,
} from "./index.ts";
import type { Label, Segment } from "./index.ts";

function csvRows(start: number = WINDOW.firstFrame, end: number = WINDOW.endFrame): string[] {
  return Array.from({ length: end - start }, (_, index) => `${start + index},H`);
}

function csv(rows: readonly string[], header = "frame,label"): string {
  return `${header}\n${rows.join("\n")}\n`;
}

test("public CSV uses absolute source frames and returns zero-relative exclusive runs", () => {
  const rows = csvRows();
  rows[0] = `${WINDOW.firstFrame},A`;
  rows[rows.length - 1] = `${WINDOW.endFrame - 1},X`;
  assert.deepEqual(parsePossessionCsv(csv(rows)), [
    { start: 0, end: 1, label: "A" },
    { start: 1, end: 7499, label: "H" },
    { start: 7499, end: 7500, label: "X" },
  ]);
});

test("full match clips exactly [11185, 18685) without including neighboring frames", () => {
  const rows = csvRows(0, 199319);
  rows[WINDOW.firstFrame - 1] = `${WINDOW.firstFrame - 1},D`;
  rows[WINDOW.firstFrame] = `${WINDOW.firstFrame},A`;
  rows[WINDOW.endFrame - 1] = `${WINDOW.endFrame - 1},X`;
  rows[WINDOW.endFrame] = `${WINDOW.endFrame},D`;
  const result = parsePossessionCsv(csv(rows));
  assert.deepEqual(result, [
    { start: 0, end: 1, label: "A" },
    { start: 1, end: 7499, label: "H" },
    { start: 7499, end: 7500, label: "X" },
  ]);
  assert.equal(aggregate(result).counts.H, 7498);
});

test("BOM, CRLF, trimmed columns and cells, and absent final newline are accepted", () => {
  const rows = csvRows();
  rows[0] = ` ${WINDOW.firstFrame} , A `;
  const text = `\ufeff frame , label \r\n${rows.join("\r\n")}`;
  assert.deepEqual(parsePossessionCsv(text), [
    { start: 0, end: 1, label: "A" },
    { start: 1, end: 7500, label: "H" },
  ]);
});

test("rejects missing, duplicate, reordered, empty, and extra public rows", () => {
  const original = csvRows();
  const missing = original.filter((_, index) => index !== 200);
  const duplicate = [...original];
  duplicate.splice(200, 0, original[200]);
  const reordered = [...original];
  [reordered[200], reordered[201]] = [reordered[201], reordered[200]];
  const blank = [...original];
  blank[200] = "";
  const extra = [...original, `${WINDOW.endFrame},H`];
  for (const rows of [missing, duplicate, reordered, blank, extra, original.slice(1), original.slice(0, -1), []]) {
    assert.throws(() => parsePossessionCsv(csv(rows)));
  }
});

test("rejects partial matches, shifted public clips, and incomplete full matches", () => {
  for (const rows of [
    csvRows(0, WINDOW.endFrame),
    csvRows(1, 199319),
    csvRows(0, 199318),
    csvRows(WINDOW.firstFrame - 1, WINDOW.endFrame),
    csvRows(WINDOW.firstFrame + 1, WINDOW.endFrame + 1),
    csvRows(0, 7500),
  ]) {
    assert.throws(() => parsePossessionCsv(csv(rows)));
  }
});

test("rejects malformed frames, including misleading JavaScript number notation", () => {
  for (const value of ["", "11185.0", "1.1185e4", "0x2bb1", "+11185", "-1", "Infinity", "NaN", "11185junk", "11_185", "9007199254740993"]) {
    const rows = csvRows();
    rows[0] = `${value},H`;
    assert.throws(() => parsePossessionCsv(csv(rows)), `Should reject frame ${value}`);
  }
});

test("rejects unsupported columns, labels, extra cells, and empty input", () => {
  for (const header of ["Frame,label", "label,frame", "frame,label,score", "frame", "frame,label,", "frame,team"]) {
    assert.throws(() => parsePossessionCsv(csv(csvRows(), header)));
  }
  for (const row of [`${WINDOW.firstFrame},`, `${WINDOW.firstFrame},h`, `${WINDOW.firstFrame},B`, `${WINDOW.firstFrame},H,1`, `${WINDOW.firstFrame}`]) {
    const rows = csvRows();
    rows[0] = row;
    assert.throws(() => parsePossessionCsv(csv(rows)));
  }
  assert.throws(() => parsePossessionCsv(""));
  assert.throws(() => parsePossessionCsv(`${csv(csvRows())}\n`));
});

test("unknown and dead ball stay distinct in runs, counts, bins, and names", () => {
  const rows = csvRows();
  rows[0] = `${WINDOW.firstFrame},X`;
  rows[1] = `${WINDOW.firstFrame + 1},D`;
  const segments = parsePossessionCsv(csv(rows));
  assert.deepEqual(segments.slice(0, 2), [
    { start: 0, end: 1, label: "X" },
    { start: 1, end: 2, label: "D" },
  ]);
  const result = aggregate(segments);
  assert.deepEqual(result.counts, { H: 7498, A: 0, D: 1, X: 1 });
  assert.deepEqual(result.bins[0].counts, { H: 248, A: 0, D: 1, X: 1 });
  assert.equal(LABEL_NAMES.X, "Unknown");
  assert.equal(LABEL_NAMES.D, "Dead ball");
  assert.deepEqual(LABELS, ["H", "A", "D", "X"]);
});

test("250-frame bins split runs exactly, including exclusive bin boundaries", () => {
  const segments: Segment[] = [
    { start: 0, end: 249, label: "H" },
    { start: 249, end: 251, label: "A" },
    { start: 251, end: 500, label: "D" },
    { start: 500, end: 7500, label: "X" },
  ];
  const result = aggregate(segments);
  assert.equal(result.bins.length, 30);
  assert.deepEqual(result.bins[0], { start: 0, end: 250, counts: { H: 249, A: 1, D: 0, X: 0 } });
  assert.deepEqual(result.bins[1], { start: 250, end: 500, counts: { H: 0, A: 1, D: 249, X: 0 } });
  assert.deepEqual(result.bins[2], { start: 500, end: 750, counts: { H: 0, A: 0, D: 0, X: 250 } });
  assert.equal(result.bins[29].end, WINDOW.frames);
  const totals: Record<Label, number> = { H: 0, A: 0, D: 0, X: 0 };
  for (const bin of result.bins) {
    assert.equal(Object.values(bin.counts).reduce((sum, count) => sum + count, 0), 250);
    for (const label of LABELS) totals[label] += bin.counts[label];
  }
  assert.deepEqual(totals, result.counts);
  assert.equal(Object.values(totals).reduce((sum, count) => sum + count, 0), 7500);
  assert.equal(result.possessions, 2);
  assert.equal(result.liveHomeShare, 249 / 251);
});

test("aggregate rejects incomplete, overlapping, gapped, shifted, and invalid segments", () => {
  const invalid: Segment[][] = [
    [],
    [{ start: 0, end: 7499, label: "H" }],
    [{ start: 1, end: 7500, label: "H" }],
    [{ start: 0, end: 7501, label: "H" }],
    [{ start: 0, end: 0, label: "H" }],
    [{ start: 0, end: 2.5, label: "H" }, { start: 2.5, end: 7500, label: "A" }],
    [{ start: 0, end: 250, label: "H" }, { start: 251, end: 7500, label: "A" }],
    [{ start: 0, end: 250, label: "H" }, { start: 249, end: 7500, label: "A" }],
    [{ start: WINDOW.firstFrame, end: WINDOW.endFrame, label: "H" }],
    [{ start: 0, end: Number.NaN, label: "H" }],
  ];
  for (const segments of invalid) assert.throws(() => aggregate(segments));
  assert.throws(() => aggregate(JSON.parse('[{"start":0,"end":7500,"label":"B"}]')));
});

test("possession counts ignore non-team runs and do not double-count split runs", () => {
  const result = aggregate([
    { start: 0, end: 100, label: "H" },
    { start: 100, end: 200, label: "H" },
    { start: 200, end: 300, label: "D" },
    { start: 300, end: 400, label: "H" },
    { start: 400, end: 500, label: "X" },
    { start: 500, end: 7500, label: "A" },
  ]);
  assert.equal(result.possessions, 3);
  for (const label of ["D", "X"] as const) {
    const inactive = aggregate([{ start: 0, end: 7500, label }]);
    assert.equal(inactive.possessions, 0);
    assert.equal(inactive.liveHomeShare, 0);
  }
});

test("labelAt respects relative frame boundaries without rounding or shifting", () => {
  assert.equal(labelAt(DEMO_SEGMENTS, 0), "A");
  assert.equal(labelAt(DEMO_SEGMENTS, 174), "A");
  assert.equal(labelAt(DEMO_SEGMENTS, 175), "H");
  assert.equal(labelAt(DEMO_SEGMENTS, 4710), "D");
  assert.equal(labelAt(DEMO_SEGMENTS, 7499), "H");
  for (const frame of [-1, 7500, WINDOW.firstFrame, WINDOW.endFrame, 174.5, Number.NaN, Infinity]) {
    assert.equal(labelAt(DEMO_SEGMENTS, frame), null);
  }
  assert.equal(labelAt([], 0), null);
});

test("demo prediction is frozen and preserves the chosen public clip runs", () => {
  assert.equal(Object.isFrozen(DEMO_SEGMENTS), true);
  assert.equal(DEMO_SEGMENTS.every(Object.isFrozen), true);
  assert.equal(Object.isFrozen(WINDOW), true);
  assert.deepEqual(WINDOW, { firstFrame: 11185, endFrame: 18685, fps: 25, frames: 7500, seconds: 300 });
  assert.equal(DEMO_SEGMENTS.length, 15);
  const result = aggregate(DEMO_SEGMENTS);
  assert.deepEqual(result.counts, { H: 3585, A: 2860, D: 1055, X: 0 });
  assert.equal(result.possessions, 12);
  assert.equal(result.liveHomeShare, 3585 / 6445);
});

test("formatTime floors fractional seconds and emits padded minutes and seconds", () => {
  for (const [seconds, expected] of [[0, "00:00"], [9.99, "00:09"], [59.99, "00:59"], [60, "01:00"], [300, "05:00"], [3600, "60:00"]] as const) {
    assert.equal(formatTime(seconds), expected);
  }
  for (const seconds of [-1, Infinity, Number.NaN]) {
    assert.throws(() => formatTime(seconds), RangeError);
  }
});
