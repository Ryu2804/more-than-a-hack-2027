import { DEMO_SEGMENTS, labelAt } from "./index.ts";
import type { Label } from "./index.ts";

type Position = readonly [number, number];
type Probabilities = readonly [number, number, number];

export type MethodSample = {
  readonly frame: number;
  readonly players: readonly (Position | null)[];
  readonly ball: Position | null;
  readonly primary: Probabilities;
  readonly secondary: Probabilities;
  readonly mixed: Probabilities;
  readonly distances: readonly [number | null, number | null];
  readonly proximityUsed: boolean;
  readonly deadCandidate: boolean;
  readonly deadAccepted: boolean;
  readonly final: Label;
};

export type MethodReplay = {
  readonly schemaVersion: 1;
  readonly match: "BAR-ATH";
  readonly methodId: "farras_wave5_short_gru25_dead45";
  readonly firstFrame: 11185;
  readonly fps: 25;
  readonly frames: 7500;
  readonly sampleStep: 5;
  readonly pitch: { readonly length: 105.3; readonly width: 68 };
  readonly teams: readonly (0 | 1 | -1)[];
  readonly samples: readonly MethodSample[];
  readonly provenance: {
    readonly artifactSha256: "cd59fb902882f34d53983cf78478b93a3757bafde65d83e5502f157be214fdb8";
    readonly sourceHashes: Readonly<Record<string, string>>;
    readonly fullPredictionVerified: true;
    readonly containsGroundTruth: false;
  };
};

const TOLERANCE = 1e-5;
const ARTIFACT_SHA256 = "cd59fb902882f34d53983cf78478b93a3757bafde65d83e5502f157be214fdb8";
const SAMPLE_KEYS = [
  "frame", "players", "ball", "primary", "secondary", "mixed", "distances",
  "proximityUsed", "deadCandidate", "deadAccepted", "final",
];

function invalid(path: string, reason: string): never {
  throw new TypeError(`Invalid method replay: ${path} ${reason}.`);
}

function record(value: unknown, path: string): Record<string, unknown> {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    invalid(path, "must be an object");
  }
  return value as Record<string, unknown>;
}

function keys(value: Record<string, unknown>, expected: readonly string[], path: string): void {
  const actual = Object.keys(value);
  if (actual.length !== expected.length || expected.some((key) => !Object.hasOwn(value, key))) {
    invalid(path, "has missing or unsupported fields");
  }
}

function array(value: unknown, length: number, path: string): unknown[] {
  if (!Array.isArray(value) || value.length !== length) {
    invalid(path, `must contain exactly ${length} entries`);
  }
  return value;
}

function finiteBetween(value: unknown, min: number, max: number, path: string): void {
  if (typeof value !== "number" || !Number.isFinite(value) || value < min || value > max) {
    invalid(path, `must be a finite number in [${min}, ${max}]`);
  }
}

function position(value: unknown, path: string): void {
  if (value === null) return;
  const coordinates = array(value, 2, path);
  for (let index = 0; index < coordinates.length; index++) {
    finiteBetween(coordinates[index], -0.5, 1.5, `${path}[${index}]`);
  }
}

function probabilities(value: unknown, path: string): asserts value is Probabilities {
  const values = array(value, 3, path);
  let sum = 0;
  for (let index = 0; index < values.length; index++) {
    const probability = values[index];
    finiteBetween(probability, 0, 1, `${path}[${index}]`);
    sum += probability as number;
  }
  if (Math.abs(sum - 1) > TOLERANCE) invalid(path, "must sum to one");
}

function validateReplay(value: unknown): asserts value is MethodReplay {
  const replay = record(value, "replay");
  keys(replay, [
    "schemaVersion", "match", "methodId", "firstFrame", "fps", "frames", "sampleStep",
    "pitch", "teams", "samples", "provenance",
  ], "replay");
  const metadata = {
    schemaVersion: 1,
    match: "BAR-ATH",
    methodId: "farras_wave5_short_gru25_dead45",
    firstFrame: 11185,
    fps: 25,
    frames: 7500,
    sampleStep: 5,
  };
  for (const [key, expected] of Object.entries(metadata)) {
    if (replay[key] !== expected) invalid(key, `must equal ${expected}`);
  }

  const pitch = record(replay.pitch, "pitch");
  keys(pitch, ["length", "width"], "pitch");
  if (pitch.length !== 105.3 || pitch.width !== 68) invalid("pitch", "must measure 105.3 by 68 metres");

  const teams = array(replay.teams, 46, "teams");
  for (let index = 0; index < teams.length; index++) {
    if (teams[index] !== 0 && teams[index] !== 1 && teams[index] !== -1) {
      invalid(`teams[${index}]`, "must be 0, 1, or -1");
    }
  }

  const provenance = record(replay.provenance, "provenance");
  keys(provenance, [
    "artifactSha256", "sourceHashes", "fullPredictionVerified", "containsGroundTruth",
  ], "provenance");
  if (provenance.artifactSha256 !== ARTIFACT_SHA256) invalid("provenance.artifactSha256", "does not match the selected artifact");
  if (provenance.fullPredictionVerified !== true) invalid("provenance.fullPredictionVerified", "must be true");
  if (provenance.containsGroundTruth !== false) invalid("provenance.containsGroundTruth", "must be false");
  const sourceHashes = record(provenance.sourceHashes, "provenance.sourceHashes");
  const sources = Object.entries(sourceHashes);
  if (sources.length === 0) invalid("provenance.sourceHashes", "must not be empty");
  for (const [path, hash] of sources) {
    const parts = path.split("/");
    if (/[\\:\u0000-\u001f\u007f]/.test(path) || parts.some((part) =>
      part === "" || part === "." || part === ".." ||
      part === "__proto__" || part === "prototype" || part === "constructor"
    )) {
      invalid("provenance.sourceHashes", "must use repository-relative file paths");
    }
    if (typeof hash !== "string" || !/^[a-fA-F0-9]{64}$/.test(hash)) {
      invalid(`provenance.sourceHashes[${path}]`, "must be a SHA-256 hex string");
    }
  }

  const samples = array(replay.samples, 1500, "samples");
  for (let index = 0; index < samples.length; index++) {
    const path = `samples[${index}]`;
    const sample = record(samples[index], path);
    keys(sample, SAMPLE_KEYS, path);
    const frame = index * 5;
    if (sample.frame !== frame) invalid(`${path}.frame`, `must equal ${frame}`);
    const players = array(sample.players, 46, `${path}.players`);
    for (let slot = 0; slot < players.length; slot++) {
      position(players[slot], `${path}.players[${slot}]`);
      if (teams[slot] === -1 && players[slot] !== null) {
        invalid(`${path}.players[${slot}]`, "must be null for an unknown team");
      }
    }
    position(sample.ball, `${path}.ball`);
    const primary = sample.primary;
    const secondary = sample.secondary;
    const mixed = sample.mixed;
    probabilities(primary, `${path}.primary`);
    probabilities(secondary, `${path}.secondary`);
    probabilities(mixed, `${path}.mixed`);
    const mass = 0.75 * (primary[0] + primary[1] + primary[2]) +
      0.25 * (secondary[0] + secondary[1] + secondary[2]);
    for (let label = 0; label < 3; label++) {
      const expected = (0.75 * primary[label] + 0.25 * secondary[label]) / mass;
      if (Math.abs(mixed[label] - expected) > TOLERANCE) {
        invalid(`${path}.mixed[${label}]`, "must match the normalized 75/25 blend");
      }
    }
    const distances = array(sample.distances, 2, `${path}.distances`);
    for (let team = 0; team < distances.length; team++) {
      if (distances[team] !== null) finiteBetween(distances[team], 0, 250, `${path}.distances[${team}]`);
    }
    for (const key of ["proximityUsed", "deadCandidate", "deadAccepted"]) {
      if (typeof sample[key] !== "boolean") invalid(`${path}.${key}`, "must be a boolean");
    }
    if (sample.deadCandidate !== (mixed[2] > 0.45)) {
      invalid(`${path}.deadCandidate`, "must match mixed dead-ball probability > 0.45");
    }
    if (sample.final !== labelAt(DEMO_SEGMENTS, frame)) {
      invalid(`${path}.final`, "does not match the frozen submission");
    }
    if (sample.deadAccepted !== (sample.final === "D")) {
      invalid(`${path}.deadAccepted`, "must match the final dead-ball label");
    }
  }
}

export function parseMethodReplay(text: string): MethodReplay {
  const replay: unknown = JSON.parse(text);
  validateReplay(replay);
  return replay;
}

export function methodSampleAt(replay: MethodReplay, frame: number): MethodSample | null {
  if (!Number.isInteger(frame) || frame < 0 || frame >= replay.frames) return null;
  return replay.samples[Math.floor(frame / replay.sampleStep)] ?? null;
}
