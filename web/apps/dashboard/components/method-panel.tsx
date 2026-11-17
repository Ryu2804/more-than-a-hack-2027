"use client";

import { memo, useRef, useState, type ChangeEvent } from "react";
import {
  DEMO_SEGMENTS,
  LABEL_NAMES,
  WINDOW,
  formatTime,
  labelAt,
  type Label,
} from "@possession/core";
import {
  methodSampleAt,
  parseMethodReplay,
  type MethodReplay,
  type MethodSample,
} from "@possession/core/method-replay";
import styles from "./method-panel.module.css";

const STATES = ["H", "A", "D"] as const;
const STATE_COLORS = { H: "var(--blue)", A: "var(--green)", D: "var(--amber)" };

function ProbabilityRow({
  name,
  values,
  weight,
}: {
  name: string;
  values: readonly number[];
  weight: string;
}) {
  return (
    <div className={styles.probabilityRow}>
      <div className={styles.probabilityHeading}>
        <strong>{name}</strong>
        <span>{weight}</span>
      </div>
      <div
        className={styles.probabilityBar}
        role="img"
        aria-label={`${name}: ${STATES.map((label, index) => `${LABEL_NAMES[label]} ${(values[index] * 100).toFixed(1)} percent`).join(", ")}`}
      >
        {STATES.map((label, index) => (
          <span
            key={label}
            style={{
              width: `${values[index] * 100}%`,
              background: STATE_COLORS[label],
            }}
          />
        ))}
      </div>
      <div className={styles.probabilityValues}>
        {STATES.map((label, index) => (
          <span key={label}>
            {label === "H" ? "Barcelona" : label === "A" ? "Athletic" : "Dead"}{" "}
            <strong>{(values[index] * 100).toFixed(1)}%</strong>
          </span>
        ))}
      </div>
    </div>
  );
}

const TrackingPitch = memo(function TrackingPitch({
  sample,
  replay,
  rotated,
}: {
  sample: MethodSample;
  replay: MethodReplay;
  rotated: boolean;
}) {
  const width = 512,
    height = (width * replay.pitch.width) / replay.pitch.length;
  const ball = sample.ball;
  const nearest = [0, 1].map((team) => {
    let selected = -1,
      minimum = Infinity;
    if (!ball) return selected;
    sample.players.forEach((point, index) => {
      if (!point || replay.teams[index] !== team) return;
      const distance = Math.hypot(
        (point[0] - ball[0]) * replay.pitch.length,
        (point[1] - ball[1]) * replay.pitch.width,
      );
      if (distance < minimum) {
        minimum = distance;
        selected = index;
      }
    });
    return selected;
  });
  return (
    <svg
      className={styles.pitch}
      viewBox="0 0 560 364"
      role="img"
      aria-label={`Observed tracking positions at ${formatTime(sample.frame / WINDOW.fps)}. ${ball ? "Ball observed." : "Ball not observed; no position is invented."}`}
      data-sample-frame={sample.frame}
    >
      <g transform={rotated ? "translate(560 364) rotate(180)" : undefined}>
        <g transform="translate(24 16)">
          <g fill="none" stroke="var(--control-border)" strokeWidth="1.1">
            <rect width={width} height={height} rx="1" />
            <path
              d={`M${width / 2} 0V${height}M0 ${height / 2 - 98}H80V${height / 2 + 98}H0M${width} ${height / 2 - 98}H${width - 80}V${height / 2 + 98}H${width}`}
            />
            <circle cx={width / 2} cy={height / 2} r="44.5" />
            <circle
              cx={width / 2}
              cy={height / 2}
              r="1.5"
              fill="var(--control-border)"
            />
          </g>
          {ball &&
            nearest.map((index, team) => {
              const point = sample.players[index];
              return point ? (
                <line
                  key={team}
                  x1={ball[0] * width}
                  y1={ball[1] * height}
                  x2={point[0] * width}
                  y2={point[1] * height}
                  stroke={
                    sample.proximityUsed
                      ? "var(--ink)"
                      : "var(--control-border)"
                  }
                  strokeWidth="1.2"
                  strokeDasharray="4 4"
                />
              ) : null;
            })}
          {sample.players.map((point, index) => {
            if (!point) return null;
            const team = replay.teams[index];
            const label = team === 0 ? "H" : "A";
            const selected = sample.final === label;
            return (
              <g
                key={index}
                transform={`translate(${point[0] * width} ${point[1] * height})`}
                data-track-slot={index + 1}
              >
                <title>{`${LABEL_NAMES[label]} tracking slot ${index + 1}. Slot numbers are not shirt numbers.`}</title>
                {selected && (
                  <circle
                    r="10"
                    fill="none"
                    stroke={team === 0 ? "var(--blue)" : "var(--green-line)"}
                    strokeWidth="1.4"
                  />
                )}
                {team === 0 ? (
                  <circle
                    r="5.3"
                    fill="var(--blue)"
                    stroke="var(--ink)"
                    strokeWidth="1.2"
                  />
                ) : (
                  <rect
                    x="-4.8"
                    y="-4.8"
                    width="9.6"
                    height="9.6"
                    rx="1"
                    fill="var(--green)"
                    stroke="var(--ink)"
                    strokeWidth="1.2"
                  />
                )}
              </g>
            );
          })}
          {ball && (
            <g
              transform={`translate(${ball[0] * width} ${ball[1] * height})`}
              data-ball="observed"
            >
              <circle
                r="6.5"
                fill="var(--surface)"
                stroke="var(--ink)"
                strokeWidth="2.2"
              />
              <circle r="2" fill="var(--ink)" />
              <title>Observed ball position</title>
            </g>
          )}
        </g>
      </g>
    </svg>
  );
});

export default function MethodPanel({
  frame,
  syncPending,
  hasVideo,
}: {
  frame: number;
  syncPending: boolean;
  hasVideo: boolean;
}) {
  const [replay, setReplay] = useState<MethodReplay | null>(null);
  const [filename, setFilename] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [rotated, setRotated] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const requestRef = useRef(0);
  const sample = replay ? methodSampleAt(replay, frame) : null;
  const final = labelAt(DEMO_SEGMENTS, frame) ?? "X";
  const sourceFrame = sample ? sample.frame : Math.floor(frame / 5) * 5;
  const historyStart = Math.max(0, Math.min(1460, Math.floor(frame / 5) - 20));
  const history = replay?.samples.slice(historyStart, historyStart + 40);
  const instant: Label | null = sample
    ? STATES[sample.mixed.indexOf(Math.max(...sample.mixed))]
    : null;
  const visible = sample?.players.filter(Boolean).length ?? 0;

  async function chooseData(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;
    const request = ++requestRef.current;
    setLoading(true);
    setError("");
    try {
      if (file.size > 20_000_000)
        throw new Error(
          "Choose the five-minute match data file, smaller than 20 MB.",
        );
      const parsed = parseMethodReplay(await file.text());
      if (request !== requestRef.current) return;
      setReplay(parsed);
      setFilename(file.name);
    } catch (failure) {
      if (request !== requestRef.current) return;
      setReplay(null);
      setFilename("");
      setError(
        failure instanceof Error
          ? failure.message
          : "This match data file could not be read.",
      );
    } finally {
      if (request === requestRef.current) setLoading(false);
    }
  }

  return (
    <section
      className={`panel ${styles.panel}`}
      aria-labelledby="method-title"
      data-method-frame={frame}
    >
      <div className={styles.heading}>
        <div>
          <h2 id="method-title">How we read possession</h2>
          <p>Tracking observations to a team-level decision</p>
        </div>
        <button
          className={styles.loadButton}
          onClick={() => inputRef.current?.click()}
        >
          {replay ? "Replace match data" : "Open match data"}
        </button>
        <input
          ref={inputRef}
          id="method-data-file"
          hidden
          type="file"
          accept=".json,application/json"
          aria-label="Open private method replay data"
          onChange={(event) => void chooseData(event)}
        />
      </div>
      <p className={styles.disclosure}>
        The model reads automatic tracking, not this video. This is a
        synchronized replay of the saved computation, not a new live prediction.
      </p>
      {error && (
        <p className="error-message" role="alert">
          {error}
        </p>
      )}
      {loading && (
        <p className={styles.loading} role="status">
          Reading local positions and model outputs...
        </p>
      )}
      {sample && replay ? (
        <>
          <div className={styles.trackingHeader}>
            <strong>1. Observe movement</strong>
            <span>
              {syncPending
                ? "Waiting for the video to reach this moment"
                : hasVideo
                  ? "Following video time"
                  : "Following the selected time"}
            </span>
          </div>
          <div className={styles.pitchStage}>
            <TrackingPitch sample={sample} replay={replay} rotated={rotated} />
            <div className={styles.pitchStatus}>
              <span>{visible} observed player slots</span>
              <span>{sample.ball ? "Ball observed" : "Ball not observed"}</span>
            </div>
          </div>
          <div className={styles.pitchTools}>
            <ul className={styles.pitchLegend}>
              <li>
                <span className={styles.homeSymbol} />
                Barcelona
              </li>
              <li>
                <span className={styles.awaySymbol} />
                Athletic Club
              </li>
              <li>
                <span className={styles.ballSymbol} />
                Ball
              </li>
            </ul>
            <button
              className="text-button"
              onClick={() => setRotated((value) => !value)}
              aria-pressed={rotated}
            >
              Rotate pitch
            </button>
          </div>
          <div className={styles.distanceRow}>
            <span>
              Nearest Barcelona{" "}
              <strong>
                {sample.distances[0] === null
                  ? "Not observed"
                  : `${sample.distances[0].toFixed(2)} m`}
              </strong>
            </span>
            <span>
              Nearest Athletic{" "}
              <strong>
                {sample.distances[1] === null
                  ? "Not observed"
                  : `${sample.distances[1].toFixed(2)} m`}
              </strong>
            </span>
          </div>
          <p className={styles.note}>
            {sample.proximityUsed
              ? "A very close ball-distance cue contributes at this moment."
              : sample.ball
                ? "Distance alone does not decide possession. The close-distance cue is not active here."
                : "The ball is missing from this observation. The models can still use the surrounding sequence."}{" "}
            Slot markers are not player identities.
          </p>
          <div className={styles.step}>
            <h3>2. Combine two views of the sequence</h3>
            <p>
              Both models use movement over time, including later observations.
              Their saved scores are combined with fixed weights.
            </p>
            <ProbabilityRow
              name="Short-possession model"
              weight="75% weight"
              values={sample.primary}
            />
            <ProbabilityRow
              name="Sequence model"
              weight="25% weight"
              values={sample.secondary}
            />
            <ProbabilityRow
              name="Combined scores"
              weight="75% + 25%"
              values={sample.mixed}
            />
          </div>
          <div className={styles.step}>
            <h3>3. Keep complete possessions</h3>
            <p>
              Whole live-play intervals are decoded together. Brief team changes
              are merged; sustained dead-ball evidence is handled separately.
            </p>
            <div className={styles.decision}>
              <div>
                <span>Instant suggestion</span>
                <strong>{instant ? LABEL_NAMES[instant] : "Unknown"}</strong>
              </div>
              <span aria-hidden="true">→</span>
              <div>
                <span>Saved final result</span>
                <strong data-final-label={final}>{LABEL_NAMES[final]}</strong>
              </div>
            </div>
            <p className={styles.note}>
              {sample.deadAccepted
                ? "This moment belongs to a sustained dead-ball period."
                : sample.deadCandidate
                  ? "The dead-ball score crosses the threshold here, but the complete duration rule does not keep this as dead ball."
                  : "This moment is not in a retained dead-ball period."}{" "}
              An instant suggestion can differ from the final result because the
              surrounding sequence matters.
            </p>
            <div
              className={styles.context}
              aria-label="Saved final decisions across eight seconds around the selected time"
            >
              {history?.map((point) => (
                <span
                  key={point.frame}
                  data-current={
                    sourceFrame === point.frame ? "true" : undefined
                  }
                  style={{
                    background:
                      point.final === "X"
                        ? "var(--unknown)"
                        : STATE_COLORS[point.final],
                  }}
                  title={`${formatTime(point.frame / 25)}: ${LABEL_NAMES[point.final]}`}
                />
              ))}
            </div>
            <div className={styles.contextAxis}>
              <span>{formatTime(historyStart / 5)}</span>
              <span>Surrounding saved decisions</span>
              <span>{formatTime((historyStart + 40) / 5)}</span>
            </div>
          </div>
          <div className={styles.resultFooter}>
            <span className="possession-state">
              <span className={`swatch ${final}`} aria-hidden="true" />
              {final === "H" || final === "A"
                ? `${LABEL_NAMES[final]} in possession`
                : LABEL_NAMES[final]}
            </span>
            <output>
              {formatTime(frame / 25)}.
              {String((frame % 25) * 4).padStart(2, "0")}
            </output>
          </div>
          <p className={styles.note}>
            Observations update 5 times per second. Missing positions remain
            missing; no movement is invented between samples. The output
            identifies a team, not an individual ball holder.
          </p>
          <div className={styles.loadedFile}>
            <span>{filename}</span>
            <button
              className="text-button"
              onClick={() => {
                ++requestRef.current;
                setReplay(null);
                setFilename("");
                setError("");
                setLoading(false);
              }}
            >
              Remove match data
            </button>
          </div>
        </>
      ) : (
        <>
          <div className={styles.emptyPitch}>
            <svg viewBox="0 0 560 280" aria-hidden="true">
              <g fill="none" stroke="var(--control-border)">
                <path d="M20 16H540V264H20ZM280 16V264M20 65H100V215H20M540 65H460V215H540" />
                <circle cx="280" cy="140" r="44" />
              </g>
            </svg>
            <div>
              <h3>Replay the actual model inputs</h3>
              <p>
                Open <strong>method-replay.private.json</strong> beside your
                video to animate observed players, the ball and the two model
                scores.
              </p>
            </div>
          </div>
          <ol className={styles.methodOutline}>
            <li>
              <strong>Observe movement</strong>
              <p>
                Read automatic player and ball positions. Missing observations
                stay missing.
              </p>
            </li>
            <li>
              <strong>Combine two sequence models</strong>
              <p>
                A short-possession model contributes 75%; a second sequence
                model contributes 25%.
              </p>
            </li>
            <li>
              <strong>Keep complete possessions</strong>
              <p>
                Use context across play, separate sustained dead ball and avoid
                spurious brief switches.
              </p>
            </li>
          </ol>
          <div className={styles.resultFooter}>
            <span className="possession-state">
              <span className={`swatch ${final}`} aria-hidden="true" />
              Saved result: {LABEL_NAMES[final]}
            </span>
            <output>{formatTime(frame / 25)}</output>
          </div>
          <p className={styles.note}>
            Position data is licensed for participants, so it is opened locally
            rather than hosted. The model predicts the team in possession, not
            the identity of a player. Video remains a review aid.
          </p>
        </>
      )}
    </section>
  );
}
