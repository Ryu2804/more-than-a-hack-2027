import assert from "node:assert/strict";
import test from "node:test";
import { DEMO_SEGMENTS } from "./index.ts";
import type { Label } from "./index.ts";
import { methodSampleAt, parseMethodReplay } from "./method-replay.ts";

type ReplayFixture = {
  schemaVersion: number;
  match: string;
  methodId: string;
  firstFrame: number;
  fps: number;
  frames: number;
  sampleStep: number;
  pitch: { length: number; width: number };
  teams: number[];
  samples: {
    frame: number;
    players: ([number, number] | null)[];
    ball: number[] | null;
    primary: number[];
    secondary: number[];
    mixed: number[];
    distances: (number | null)[];
    proximityUsed: boolean;
    deadCandidate: boolean;
    deadAccepted: boolean;
    final: Label;
  }[];
  provenance: {
    artifactSha256: string;
    sourceHashes: Record<string, string>;
    fullPredictionVerified: boolean;
    containsGroundTruth: boolean;
  };
};

function fixture(): ReplayFixture {
  const teams = Array.from({ length: 46 }, (_, slot) => slot < 11 ? 0 : slot < 22 ? 1 : -1);
  return {
    schemaVersion: 1,
    match: "BAR-ATH",
    methodId: "farras_wave5_short_gru25_dead45",
    firstFrame: 11185,
    fps: 25,
    frames: 7500,
    sampleStep: 5,
    pitch: { length: 105.3, width: 68 },
    teams,
    samples: Array.from({ length: 1500 }, (_, index) => {
      const frame = index * 5;
      const segment = DEMO_SEGMENTS.find((run) => frame >= run.start && frame < run.end);
      if (!segment) throw new Error(`No frozen segment at frame ${frame}`);
      const dead = segment.label === "D";
      const players: ([number, number] | null)[] = Array.from({ length: 46 }, () => null);
      players[0] = [-0.5, 1.5];
      players[11] = [0.75, 0.25];
      return {
        frame,
        players,
        ball: index % 2 === 0 ? null : [0.5, 0.5],
        primary: dead ? [0.1, 0.2, 0.7] : [0.2, 0.65, 0.15],
        secondary: dead ? [0.2, 0.2, 0.6] : [0.4, 0.5, 0.1],
        mixed: dead ? [0.125, 0.2, 0.675] : [0.25, 0.6125, 0.1375],
        distances: index % 2 === 0 ? [null, null] : [0.25, 12],
        proximityUsed: index % 2 !== 0,
        deadCandidate: dead,
        deadAccepted: dead,
        final: segment.label,
      };
    }),
    provenance: {
      artifactSha256: "cd59fb902882f34d53983cf78478b93a3757bafde65d83e5502f157be214fdb8",
      sourceHashes: { "tests/synthetic-source.ts": "a".repeat(64) },
      fullPredictionVerified: true,
      containsGroundTruth: false,
    },
  };
}

function rejectChange(change: (replay: ReplayFixture) => void, expected: RegExp) {
  const replay = fixture();
  change(replay);
  assert.throws(() => parseMethodReplay(JSON.stringify(replay)), expected);
}

test("accepts the complete relative window without changing coordinates, probabilities, or occlusions", () => {
  const input = fixture();
  const replay = parseMethodReplay(JSON.stringify(input));
  assert.equal(replay.samples.length, 1500);
  assert.equal(replay.samples[1499].frame, 7495);
  assert.deepEqual(replay.samples[0].players[0], [-0.5, 1.5]);
  assert.deepEqual(replay.samples[0].players[11], [0.75, 0.25]);
  assert.equal(replay.samples[0].players[1], null);
  assert.equal(replay.samples[0].players[45], null);
  assert.equal(replay.samples[0].ball, null);
  assert.deepEqual(replay.samples[0].distances, [null, null]);
  assert.deepEqual(replay.samples[1].ball, [0.5, 0.5]);
  assert.deepEqual(replay.samples[1].distances, [0.25, 12]);
  assert.deepEqual(replay.samples[0].primary, [0.2, 0.65, 0.15]);
  assert.deepEqual(replay.samples[0].secondary, [0.4, 0.5, 0.1]);
  assert.deepEqual(replay.samples[0].mixed, [0.25, 0.6125, 0.1375]);
  assert.deepEqual(replay, input);
});

test("lookup holds the previous sample, including sample and frozen-label boundaries", () => {
  const replay = parseMethodReplay(JSON.stringify(fixture()));
  for (const [frame, sampleFrame, final] of [
    [0, 0, "A"], [4, 0, "A"], [5, 5, "A"], [9, 5, "A"],
    [174, 170, "A"], [175, 175, "H"], [4709, 4705, "A"],
    [4710, 4710, "D"], [5184, 5180, "D"], [5185, 5185, "A"],
    [7495, 7495, "H"], [7499, 7495, "H"],
  ] as const) {
    const sample = methodSampleAt(replay, frame);
    assert.equal(sample?.frame, sampleFrame);
    assert.equal(sample?.final, final);
    assert.equal(sample, replay.samples[sampleFrame / 5]);
  }
  for (const frame of [-1, 7500, 11185, 18685, 0.1, 4.9, Number.NaN, Infinity, -Infinity]) {
    assert.equal(methodSampleAt(replay, frame), null);
  }
});

test("rejects malformed JSON, non-object roots, missing fields, and unsupported fields", () => {
  for (const text of ["", "{", "null", "[]", "true", "42", '"replay"']) {
    assert.throws(() => parseMethodReplay(text));
  }
  rejectChange((value) => { Reflect.deleteProperty(value, "samples"); }, /replay.*fields/);
  rejectChange((value) => { Object.assign(value, { groundTruth: [] }); }, /replay.*fields/);
  rejectChange((value) => { Object.assign(value.pitch, { extra: 1 }); }, /pitch.*fields/);
  rejectChange((value) => { Object.assign(value.samples[0], { extra: 1 }); }, /samples\[0\].*fields/);
  rejectChange((value) => { Reflect.deleteProperty(value.samples[0], "ball"); }, /samples\[0\].*fields/);
  rejectChange((value) => { Object.assign(value, { provenance: null }); }, /provenance.*object/);
});

test("rejects shifted windows, altered method identity, and nonliteral metadata", () => {
  for (const [key, invalid] of [
    ["schemaVersion", 2], ["schemaVersion", "1"], ["match", "ATH-BAR"],
    ["methodId", "other"], ["firstFrame", 11186], ["fps", 30],
    ["frames", 7495], ["sampleStep", 10],
  ] as const) {
    rejectChange((value) => { Object.assign(value, { [key]: invalid }); }, new RegExp(key));
  }
  rejectChange((value) => { value.pitch.length = 105; }, /pitch/);
  rejectChange((value) => { value.pitch.width = 68.01; }, /pitch/);
});

test("requires exactly 1500 ordered samples at relative frame multiples of five", () => {
  rejectChange((value) => { value.samples.pop(); }, /samples.*1500/);
  rejectChange((value) => { value.samples.push(value.samples[1499]); }, /samples.*1500/);
  rejectChange((value) => { Object.assign(value, { samples: {} }); }, /samples.*1500/);
  rejectChange((value) => { Object.assign(value.samples, { 12: null }); }, /samples\[12\].*object/);
  rejectChange((value) => { value.samples[1].frame = 0; }, /samples\[1\].frame/);
  rejectChange((value) => { value.samples[1].frame = 10; }, /samples\[1\].frame/);
  rejectChange((value) => { value.samples[1].frame = 5.5; }, /samples\[1\].frame/);
  rejectChange((value) => { value.samples.forEach((sample) => { sample.frame += 11185; }); }, /samples\[0\].frame/);
  rejectChange((value) => {
    [value.samples[10], value.samples[11]] = [value.samples[11], value.samples[10]];
  }, /samples\[10\].frame/);
});

test("requires 46 valid team codes and null positions for every unknown-team slot", () => {
  rejectChange((value) => { value.teams.pop(); }, /teams.*46/);
  rejectChange((value) => { value.teams.push(0); }, /teams.*46/);
  for (const code of [2, -2, 0.5, "0", null, true]) {
    rejectChange((value) => { Object.assign(value.teams, { 0: code }); }, /teams\[0\]/);
  }
  rejectChange((value) => { value.samples[0].players[22] = [0, 0]; }, /unknown team/);
  rejectChange((value) => { value.teams[0] = -1; }, /unknown team/);
});

test("rejects malformed player and ball positions rather than coercing or clamping them", () => {
  rejectChange((value) => { value.samples[0].players.pop(); }, /players.*46/);
  rejectChange((value) => { value.samples[0].players.push(null); }, /players.*46/);
  for (const position of [[], [0], [0, 0, 0], [null, 0], ["0.5", 0], [true, 0], [-0.50001, 0], [0, 1.50001], {}, false]) {
    rejectChange((value) => { Object.assign(value.samples[0].players, { 0: position }); }, /players\[0\]/);
    rejectChange((value) => { Object.assign(value.samples[0], { ball: position }); }, /\.ball/);
  }
  rejectChange((value) => { value.samples[0].players[0] = [Number.NaN, 0]; }, /players\[0\]\[0\]/);
  rejectChange((value) => { value.samples[0].ball = [Infinity, 0]; }, /\.ball\[0\]/);
  const replay = fixture();
  replay.samples[0].ball = [-0.5, 1.5];
  assert.deepEqual(parseMethodReplay(JSON.stringify(replay)).samples[0].ball, [-0.5, 1.5]);
});

test("requires finite normalized H/A/D probabilities in all three stages", () => {
  for (const key of ["primary", "secondary", "mixed"] as const) {
    for (const probabilities of [null, [], [0.5, 0.5], [0.2, 0.3, 0.5, 0], [-0.1, 0.5, 0.6], [1.1, 0, 0], [0.2, 0.3, 0.4], ["0.2", 0.3, 0.5], [null, 0.3, 0.7]]) {
      rejectChange((value) => { Object.assign(value.samples[0], { [key]: probabilities }); }, new RegExp(`\\.${key}`));
    }
    rejectChange((value) => { value.samples[0][key][0] = Number.NaN; }, new RegExp(`\\.${key}`));
  }
  const text = JSON.stringify(fixture());
  assert.throws(() => parseMethodReplay(text.replace('"frame":0', '"frame":NaN')), SyntaxError);
  assert.throws(() => parseMethodReplay(text.replace('"primary":[0.2', '"primary":[1e400')), /primary\[0\]/);
});

test("checks normalized 75/25 weights rather than accepting any normalized mixed distribution", () => {
  rejectChange((value) => { value.samples[0].mixed = [0.3, 0.575, 0.125]; }, /75\/25/);
  rejectChange((value) => { value.samples[0].mixed = [0.25002, 0.61248, 0.1375]; }, /75\/25/);
  const rounded = fixture();
  rounded.samples[0].mixed = [0.250004, 0.612496, 0.1375];
  assert.deepEqual(parseMethodReplay(JSON.stringify(rounded)).samples[0].mixed, [0.250004, 0.612496, 0.1375]);
  const normalized = fixture();
  normalized.samples[0].primary = [0.200004, 0.65, 0.15];
  normalized.samples[0].mixed = [0.250003 / 1.000003, 0.6125 / 1.000003, 0.1375 / 1.000003];
  assert.doesNotThrow(() => parseMethodReplay(JSON.stringify(normalized)));
});

test("accepts nullable metre distances and rejects nonfinite, negative, or implausible values", () => {
  for (const distances of [null, [], [1], [1, 2, 3], [-0.01, 1], [1, 250.01], ["1", 2], [true, 2]]) {
    rejectChange((value) => { Object.assign(value.samples[0], { distances }); }, /distances/);
  }
  const replay = fixture();
  replay.samples[0].distances = [0, 250];
  replay.samples[1].distances = [null, 12];
  const parsed = parseMethodReplay(JSON.stringify(replay));
  assert.deepEqual(parsed.samples[0].distances, [0, 250]);
  assert.deepEqual(parsed.samples[1].distances, [null, 12]);
  const text = JSON.stringify(replay).replace('"distances":[0,250]', '"distances":[1e400,250]');
  assert.throws(() => parseMethodReplay(text), /distances\[0\]/);
});

test("validates decision booleans without inferring an unconfirmed proximity rule", () => {
  for (const key of ["proximityUsed", "deadCandidate", "deadAccepted"] as const) {
    for (const invalid of [null, 0, 1, "false"]) {
      rejectChange((value) => { Object.assign(value.samples[0], { [key]: invalid }); }, new RegExp(key));
    }
  }
  const replay = fixture();
  replay.samples[0].proximityUsed = true;
  replay.samples[1].proximityUsed = false;
  const parsed = parseMethodReplay(JSON.stringify(replay));
  assert.equal(parsed.samples[0].proximityUsed, true);
  assert.equal(parsed.samples[1].proximityUsed, false);
});

test("dead candidate uses a strict 0.45 threshold and need not be an accepted dead interval", () => {
  const replay = fixture();
  replay.samples[0].primary = [0.3, 0.25, 0.45];
  replay.samples[0].secondary = [0.3, 0.25, 0.45];
  replay.samples[0].mixed = [0.3, 0.25, 0.45];
  assert.equal(parseMethodReplay(JSON.stringify(replay)).samples[0].deadCandidate, false);
  replay.samples[0].deadCandidate = true;
  assert.throws(() => parseMethodReplay(JSON.stringify(replay)), /deadCandidate/);
  replay.samples[0].primary = [0.3, 0.249999, 0.450001];
  replay.samples[0].secondary = [0.3, 0.249999, 0.450001];
  replay.samples[0].mixed = [0.3, 0.249999, 0.450001];
  const parsed = parseMethodReplay(JSON.stringify(replay));
  assert.equal(parsed.samples[0].deadCandidate, true);
  assert.equal(parsed.samples[0].deadAccepted, false);
  assert.equal(parsed.samples[0].final, "A");
  replay.samples[0].deadCandidate = false;
  assert.throws(() => parseMethodReplay(JSON.stringify(replay)), /deadCandidate/);
});

test("rejects changed final labels at the first sample, run boundaries, and end of the clip", () => {
  for (const [index, wrong] of [[0, "H"], [35, "A"], [942, "A"], [1037, "D"], [1499, "A"], [0, "X"], [0, "a"], [0, null]] as const) {
    rejectChange((value) => { Object.assign(value.samples[index], { final: wrong }); }, /final.*frozen submission/);
  }
  rejectChange((value) => { value.samples[0].deadAccepted = true; }, /deadAccepted/);
  rejectChange((value) => { value.samples[942].deadAccepted = false; }, /deadAccepted/);
});

test("requires the selected artifact and explicit provenance flags", () => {
  rejectChange((value) => { value.provenance.artifactSha256 = "b".repeat(64); }, /artifactSha256/);
  for (const invalid of [false, "true", null]) {
    rejectChange((value) => { Object.assign(value.provenance, { fullPredictionVerified: invalid }); }, /fullPredictionVerified/);
  }
  for (const invalid of [true, "false", null]) {
    rejectChange((value) => { Object.assign(value.provenance, { containsGroundTruth: invalid }); }, /containsGroundTruth/);
  }
  rejectChange((value) => { Object.assign(value.provenance, { extra: true }); }, /provenance.*fields/);
});

test("source hashes require nonempty relative paths and 64-digit hexadecimal values", () => {
  for (const sourceHashes of [null, [], {}, { "source.ts": "a".repeat(63) }, { "source.ts": "g".repeat(64) }, { "source.ts": 123 }]) {
    rejectChange((value) => { Object.assign(value.provenance, { sourceHashes }); }, /sourceHashes/);
  }
  for (const path of ["", "/private/source.ts", "C:/private/source.ts", "C:\\private\\source.ts", "\\\\host\\source.ts", "../source.ts", "folder/../source.ts", "./source.ts", "folder//source.ts", "folder/", "https://host/source.ts", "folder/\u0000source.ts", "__proto__", "constructor", "prototype", "folder/__proto__/source.ts"]) {
    rejectChange((value) => {
      Object.assign(value.provenance, { sourceHashes: Object.fromEntries([[path, "a".repeat(64)]]) });
    }, /sourceHashes/);
  }
  const replay = fixture();
  Object.assign(replay.provenance, { sourceHashes: { "method/source.py": "ABCDEF0123456789".repeat(4) } });
  assert.deepEqual(parseMethodReplay(JSON.stringify(replay)).provenance.sourceHashes, {
    "method/source.py": "ABCDEF0123456789".repeat(4),
  });
});
