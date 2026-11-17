"use client";

import Link from "next/link";
import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ChangeEvent,
} from "react";
import {
  aggregate,
  DEMO_SEGMENTS,
  formatTime,
  labelAt,
  LABELS,
  LABEL_NAMES,
  parsePossessionCsv,
  WINDOW,
  type Segment,
} from "@possession/core";
import { Gauge, Legend, Pitch, ShareChart, Timeline } from "./charts";
import MethodPanel from "./method-panel";

const REFERENCE_URL =
  "https://raw.githubusercontent.com/Ryu2804/more-than-a-hack-2027/3d113d003fc5ab9f7a696635f4f0fe85570f616c/evaluation/gt/BAR-ATH_possession_gt_public.csv";
const LAST_SECOND = (WINDOW.frames - 1) / WINDOW.fps;
type MediaState =
  | "empty"
  | "loading"
  | "ready"
  | "seeking"
  | "buffering"
  | "error"
  | "ended";
type LocalMedia = { url: string; name: string; mode: "clip" | "original" };
const SUMMARY = aggregate(DEMO_SEGMENTS);

async function readPublicReference(text: string): Promise<Segment[]> {
  const segments = parsePossessionCsv(text);
  const bytes = new TextEncoder().encode(JSON.stringify(segments));
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  const identity = Array.from(new Uint8Array(digest), byte =>
    byte.toString(16).padStart(2, "0"),
  ).join("");
  if (identity !== "f740890e254bc46eecbe213c982f705132a2aff5d50ddd63925c078ebcda849f") {
    throw new Error("These labels do not match the published five-minute reference. Choose the original reference file.");
  }
  return segments;
}

function Mark() {
  return (
    <span className="brand-mark" aria-hidden="true">
      <i />
      <i />
      <i />
    </span>
  );
}
function TransportIcon({
  kind,
}: {
  kind: "previous" | "next" | "play" | "pause";
}) {
  return (
    <svg
      width="18"
      height="18"
      viewBox="0 0 20 20"
      aria-hidden="true"
      fill="currentColor"
    >
      {kind === "play" ? (
        <path d="M5 2L18 10 5 18Z" />
      ) : kind === "pause" ? (
        <path d="M4 3H8V17H4ZM12 3H16V17H12Z" />
      ) : kind === "previous" ? (
        <path d="M3 3H5V17H3ZM17 3L6 10 17 17Z" />
      ) : (
        <path d="M15 3H17V17H15ZM3 3L14 10 3 17Z" />
      )}
    </svg>
  );
}

export default function Dashboard({ view }: { view: "overview" | "replay" }) {
  const [reference, setReference] = useState<Segment[] | null>(null);
  const [referenceState, setReferenceState] = useState<
    "loading" | "ready" | "error"
  >("loading");
  const [referenceError, setReferenceError] = useState("");
  const [referenceAttempt, setReferenceAttempt] = useState(0);
  const [seconds, setSeconds] = useState(0);
  const [seekTarget, setSeekTarget] = useState<number | null>(null);
  const [media, setMedia] = useState<LocalMedia | null>(null);
  const [mediaState, setMediaState] = useState<MediaState>("empty");
  const [mediaError, setMediaError] = useState("");
  const [mode, setMode] = useState<LocalMedia["mode"]>("clip");
  const [playing, setPlaying] = useState(false);
  const [dark, setDark] = useState(false);
  const videoRef = useRef<HTMLVideoElement>(null);
  const dialogRef = useRef<HTMLDialogElement>(null);
  const referenceRequest = useRef(0);
  const referenceController = useRef<AbortController | null>(null);
  // Browser seeks can report one microsecond below an exact frame boundary.
  const frame = Math.min(
    WINDOW.frames - 1,
    Math.max(0, Math.floor(seconds * WINDOW.fps + 0.0001)),
  );
  const predicted = labelAt(DEMO_SEGMENTS, frame);
  const actual = reference ? labelAt(reference, frame) : null;
  const referenceSummary = useMemo(
    () => (reference ? aggregate(reference) : null),
    [reference],
  );
  const offset =
    media?.mode === "original" ? WINDOW.firstFrame / WINDOW.fps : 0;

  useEffect(() => {
    try {
      setDark(localStorage.getItem("possession-theme") === "dark");
    } catch {
      /* Storage is optional in private browsing. */
    }
  }, []);
  useEffect(() => {
    document.documentElement.dataset.theme = dark ? "dark" : "light";
  }, [dark]);
  useEffect(() => {
    const controller = new AbortController();
    referenceController.current = controller;
    const request = ++referenceRequest.current;
    setReferenceState("loading");
    setReferenceError("");
    async function load() {
      try {
        const response = await fetch(REFERENCE_URL, {
          signal: controller.signal,
          cache: "force-cache",
        });
        if (!response.ok) throw new Error("The public source did not respond.");
        const text = await response.text();
        if (text.length > 5_000_000)
          throw new Error("The reference file is too large.");
        const segments = await readPublicReference(text);
        if (request !== referenceRequest.current) return;
        setReference(segments);
        setReferenceState("ready");
      } catch {
        if (controller.signal.aborted || request !== referenceRequest.current)
          return;
        setReferenceState("error");
        setReferenceError(
          "The public reference could not load. Retry or open the reference file from your device.",
        );
      }
    }
    void load();
    return () => controller.abort();
  }, [referenceAttempt]);
  useEffect(() => {
    if (!media) return;
    return () => URL.revokeObjectURL(media.url);
  }, [media]);

  const updateFromVideo = useCallback(
    (time: number) => {
      const clipTime = time - offset;
      const video = videoRef.current;
      if (!video || !Number.isFinite(time)) return;
      if (clipTime < -0.02) {
        video.currentTime = offset;
        return;
      }
      if (clipTime >= LAST_SECOND - 0.0001) {
        video.pause();
        if (Math.abs(clipTime - LAST_SECOND) > 0.0001) {
          video.currentTime = offset + LAST_SECOND;
        }
        setSeconds(LAST_SECOND);
        setMediaState("ended");
        return;
      }
      setSeconds(Math.max(0, Math.min(LAST_SECOND, clipTime)));
    },
    [offset],
  );

  useEffect(() => {
    const video = videoRef.current;
    if (!video || !media || !playing || !video.requestVideoFrameCallback)
      return;
    let handle = 0;
    const onFrame: VideoFrameRequestCallback = (_now, metadata) => {
      if (!video.seeking) updateFromVideo(metadata.mediaTime);
      handle = video.requestVideoFrameCallback(onFrame);
    };
    handle = video.requestVideoFrameCallback(onFrame);
    return () => video.cancelVideoFrameCallback(handle);
  }, [media, playing, updateFromVideo]);

  function seek(value: number) {
    const target = Math.max(0, Math.min(LAST_SECOND, value));
    setSeekTarget(target);
    const video = videoRef.current;
    if (video && mediaState !== "error" && mediaState !== "loading") {
      setMediaState("seeking");
      video.currentTime = offset + target;
    } else {
      setSeconds(target);
      setSeekTarget(null);
    }
  }
  function openSetup() {
    dialogRef.current?.showModal();
  }
  function toggleTheme() {
    const next = !dark;
    setDark(next);
    try {
      localStorage.setItem("possession-theme", next ? "dark" : "light");
    } catch {
      /* The toggle still works without storage. */
    }
  }
  function chooseVideo(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;
    if (!/\.(mp4|m4v|mov|webm|mkv|avi)$/i.test(file.name)) {
      setMediaError(
        "Choose a video file. An MP4 with H.264 video is recommended.",
      );
      return;
    }
    videoRef.current?.pause();
    setPlaying(false);
    setMediaError("");
    setMediaState("loading");
    setSeconds(0);
    setMedia({ url: URL.createObjectURL(file), name: file.name, mode });
  }
  async function validateVideo() {
    const video = videoRef.current;
    if (!video || !media) return;
    const valid =
      Number.isFinite(video.duration) &&
      (media.mode === "clip"
        ? Math.abs(video.duration - WINDOW.seconds) <= 0.12
        : video.duration >= offset + WINDOW.seconds);
    if (!valid) {
      setMediaState("error");
      setMediaError(
        media.mode === "clip"
          ? "This is not a five-minute clip. Choose the matching clip, or select the original tactical recording option."
          : "This recording does not cover the required five-minute window. Choose the original tactical recording.",
      );
      return;
    }
    video.currentTime = offset;
    setMediaState("ready");
    setMediaError("");
    dialogRef.current?.close();
    try {
      await video.play();
    } catch (error) {
      if (videoRef.current !== video || (error instanceof DOMException && error.name === "AbortError")) return;
      setMediaError("Automatic playback could not start. Press Play video to continue.");
    }
  }
  async function togglePlayback() {
    const video = videoRef.current;
    if (!video) return;
    if (!video.paused) {
      video.pause();
      return;
    }
    if (mediaState === "ended") video.currentTime = offset;
    try {
      await video.play();
    } catch {
      setMediaError(
        "Playback could not start. Try again, or choose an MP4 with H.264 video.",
      );
    }
  }
  async function chooseReference(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;
    referenceController.current?.abort();
    const request = ++referenceRequest.current;
    setReferenceState("loading");
    setReferenceError("");
    try {
      if (file.size > 5_000_000)
        throw new Error("Choose the public reference file, smaller than 5 MB.");
      const segments = await readPublicReference(await file.text());
      if (request !== referenceRequest.current) return;
      setReference(segments);
      setReferenceState("ready");
    } catch (error) {
      if (request !== referenceRequest.current) return;
      setReference(null);
      setReferenceState("error");
      setReferenceError(
        error instanceof Error
          ? error.message
          : "The reference file could not be read.",
      );
    }
  }
  const videoUsable =
    media && !["empty", "loading", "error"].includes(mediaState);
  const status =
    mediaState === "empty"
      ? "No video selected"
      : mediaState === "loading"
        ? "Opening local video..."
        : mediaState === "seeking"
          ? "Seeking video; labels show the selected time"
          : mediaState === "buffering"
            ? "Waiting for video..."
            : mediaState === "error"
              ? "Video could not open"
              : mediaState === "ended"
                ? "End of the five-minute clip"
                : playing
                  ? "Playing local video"
                  : "Video paused";

  return (
    <div className="app-shell">
      <aside className="nav-rail" aria-label="Match navigation">
        <Link href="/" className="brand-link" aria-label="Possession overview">
          <Mark />
        </Link>
        <nav>
          <Link
            href="/"
            aria-current={view === "overview" ? "page" : undefined}
            aria-label="Overview"
            title="Overview"
          >
            <svg
              viewBox="0 0 20 20"
              width="20"
              aria-hidden="true"
              fill="currentColor"
            >
              <path d="M2 2H8V8H2ZM12 2H18V8H12ZM2 12H8V18H2ZM12 12H18V18H12Z" />
            </svg>
            <span>Overview</span>
          </Link>
          <Link
            href="/replay"
            aria-current={view === "replay" ? "page" : undefined}
            aria-label="Video review"
            title="Video review"
          >
            <TransportIcon kind="play" />
            <span>Review</span>
          </Link>
          <Link
            href="/evidence"
            aria-label="Results explained"
            title="Results explained"
          >
            <svg
              viewBox="0 0 20 20"
              width="20"
              aria-hidden="true"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.5"
            >
              <path d="M4 2H16V18H4ZM7 6H13M7 10H13M7 14H11" />
            </svg>
            <span>Results</span>
          </Link>
        </nav>
        <button
          className="theme-button"
          onClick={toggleTheme}
          aria-label={dark ? "Switch to light theme" : "Switch to dark theme"}
          title={dark ? "Light theme" : "Dark theme"}
        >
          <svg
            viewBox="0 0 20 20"
            width="20"
            aria-hidden="true"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.5"
          >
            <circle cx="10" cy="10" r="6" />
            <path d="M10 4V16" />
          </svg>
          <span>{dark ? "Light" : "Dark"}</span>
        </button>
      </aside>
      <main id="main" className="main-content">
        <header className="page-header">
          <div>
            <div className="title-row">
              <h1>
                {view === "overview" ? "Match possession" : "Review the clip"}
              </h1>
              <span className="window-label">5-minute sample</span>
            </div>
            <p>
              Barcelona <span className="versus">vs</span> Athletic Club{" "}
              <span className="header-detail">/ Public reference window</span>
            </p>
          </div>
          <div className="header-stats">
            <div>
              <strong>5 min</strong>
              <span>Same window, both timelines</span>
            </div>
            <div>
              <strong>
                77.2<span className="score-unit"> /100</span>
              </strong>
              <span>Published possession score</span>
            </div>
          </div>
        </header>
        <div className="match-bar">
          <div>
            <strong>Barcelona vs Athletic Club</strong>
            <span>Precomputed results / 00:00 to 05:00</span>
          </div>
          <div className="bar-actions">
            {view === "overview" ? (
              <Link className="button primary" href="/replay">
                Compare with video
              </Link>
            ) : (
              <button className="primary" onClick={openSetup}>
                Open local video
              </button>
            )}
            <Link className="button secondary" href="/evidence">
              Results explained
            </Link>
          </div>
        </div>
        {view === "overview" ? (
          <div className="overview-grid">
            <section className="panel">
              <div className="panel-heading">
                <h2>Who has the ball?</h2>
                <p>Model results across the five-minute clip</p>
              </div>
              <div className="possession-breakdown">
                <Gauge counts={SUMMARY.counts} />
                <dl className="share-list">
                  {LABELS.map((label) => (
                    <div key={label}>
                      <dt>
                        <span
                          className={`swatch ${label}`}
                          aria-hidden="true"
                        />
                        {LABEL_NAMES[label]}
                      </dt>
                      <dd>
                        {(
                          (SUMMARY.counts[label] / WINDOW.frames) *
                          100
                        ).toFixed(1)}
                        %
                      </dd>
                    </div>
                  ))}
                </dl>
              </div>
              <p className="footnote">
                Includes time when the ball is out of play.
              </p>
            </section>
            <section className="panel">
              <div className="panel-heading">
                <h2>Player positions</h2>
                <p>Position data is not part of this result</p>
              </div>
              <Pitch />
              <p className="footnote">
                No player or ball locations have been inferred from possession
                labels.
              </p>
            </section>
            <section className="panel">
              <div className="panel-heading">
                <h2>How possession changes</h2>
                <p>Time held by each team, in 10-second intervals</p>
              </div>
              <div className="mini-stats">
                <div>
                  <strong>{(SUMMARY.liveHomeShare * 100).toFixed(1)}%</strong>
                  <span>Barcelona, while the ball is in play</span>
                </div>
                <div>
                  <strong>{SUMMARY.possessions}</strong>
                  <span>Model possessions</span>
                </div>
                <div>
                  <strong>{referenceSummary?.possessions ?? "..."}</strong>
                  <span>Reference possessions</span>
                </div>
              </div>
              <ShareChart bins={SUMMARY.bins} />
              <p className="footnote">
                Time is measured from the start of the clip, not the match
                clock.
              </p>
            </section>
            <section className="panel">
              <div className="panel-heading">
                <h2>Possession per 10 seconds</h2>
                <p>The share of each interval, not model confidence</p>
              </div>
              <ShareChart bins={SUMMARY.bins} lines />
              <Legend />
              <p className="footnote">
                Each step is measured from the saved result. No smoothing is
                applied.
              </p>
            </section>
          </div>
        ) : (
          <div className="replay-grid">
            <section className="panel video-panel">
              <div className="panel-heading">
                <h2>Match video</h2>
                <p>Only the five-minute reference window is shown</p>
              </div>
              <div className={`video-stage ${media ? "has-video" : ""}`}>
                {media ? (
                  <video
                    key={media.url}
                    ref={videoRef}
                    src={media.url}
                    playsInline
                    muted
                    preload="metadata"
                    aria-label="Local tactical match video"
                    onLoadedMetadata={validateVideo}
                    onTimeUpdate={(event) => {
                      const video = event.currentTarget;
                      if (!video.seeking && (!video.requestVideoFrameCallback || video.paused)) {
                        updateFromVideo(video.currentTime);
                      }
                    }}
                    onSeeking={() => {
                      if (mediaState !== "error") setMediaState("seeking");
                    }}
                    onSeeked={(event) => {
                      setSeekTarget(null);
                      if (mediaState !== "error") {
                        updateFromVideo(event.currentTarget.currentTime);
                        if (event.currentTarget.currentTime - offset < LAST_SECOND - 0.0001) {
                          setMediaState("ready");
                        }
                      }
                    }}
                    onPlay={() => setPlaying(true)}
                    onPause={(event) => {
                      setPlaying(false);
                      if (!event.currentTarget.seeking) {
                        updateFromVideo(event.currentTarget.currentTime);
                      }
                    }}
                    onWaiting={() => {
                      if (mediaState !== "error") setMediaState("buffering");
                    }}
                    onPlaying={() => setMediaState("ready")}
                    onEnded={() => {
                      setSeconds(LAST_SECOND);
                      setMediaState("ended");
                      setPlaying(false);
                    }}
                    onError={() => {
                      setMediaState("error");
                      setPlaying(false);
                      setMediaError(
                        "Your browser could not decode this file. Choose an MP4 with H.264 video.",
                      );
                    }}
                  />
                ) : (
                  <div className="video-empty">
                    <span className="video-placeholder-mark" aria-hidden="true">
                      <TransportIcon kind="play" />
                    </span>
                    <h3>Bring your own match video</h3>
                    <p>
                      Open the five-minute tactical clip from your device.
                      <br />
                      The recording stays in your browser.
                    </p>
                    <button className="primary" onClick={openSetup}>
                      Choose local video
                    </button>
                  </div>
                )}
              </div>
              <div className="transport">
                <div className="transport-buttons">
                  <button
                    className="icon-button"
                    aria-label="Previous video frame"
                    disabled={(seekTarget === null ? frame : Math.round(seekTarget * 25)) === 0}
                    onClick={() => seek((seekTarget ?? frame / 25) - 1 / 25)}
                  >
                    <TransportIcon kind="previous" />
                  </button>
                  <button
                    className="icon-button primary"
                    aria-label={playing ? "Pause video" : "Play video"}
                    disabled={!videoUsable}
                    onClick={() => void togglePlayback()}
                  >
                    <TransportIcon kind={playing ? "pause" : "play"} />
                  </button>
                  <button
                    className="icon-button"
                    aria-label="Next video frame"
                    disabled={(seekTarget === null ? frame : Math.round(seekTarget * 25)) === WINDOW.frames - 1}
                    onClick={() => seek((seekTarget ?? frame / 25) + 1 / 25)}
                  >
                    <TransportIcon kind="next" />
                  </button>
                </div>
                <p className="video-status" role="status">
                  {status}
                </p>
              </div>
              {mediaError && (
                <p className="error-message" role="alert">
                  {mediaError}
                </p>
              )}
              {media && (
                <div className="file-caption">
                  <span title={media.name}>{media.name}</span>
                  <button
                    className="text-button"
                    onClick={() => {
                      videoRef.current?.pause();
                      setMedia(null);
                      setPlaying(false);
                      setMediaState("empty");
                      setMediaError("");
                    }}
                  >
                    Remove video
                  </button>
                </div>
              )}
              <p className="footnote">
                Use the tactical recording for this match. Broadcast footage is
                not aligned.
              </p>
            </section>
            <MethodPanel
              frame={frame}
              syncPending={mediaState === "seeking" || mediaState === "buffering"}
              hasVideo={Boolean(media) && mediaState !== "loading" && mediaState !== "error"}
            />
          </div>
        )}
        <section
          className="panel comparison"
          aria-labelledby="comparison-title"
        >
          <div className="comparison-header">
            <div>
              <h2 id="comparison-title">Possession comparison</h2>
              <p>
                Same five minutes.{" "}
                {referenceSummary
                  ? `${referenceSummary.possessions} reference possessions, ${SUMMARY.possessions} model possessions.`
                  : `${SUMMARY.possessions} model possessions; reference loading separately.`}
              </p>
            </div>
            <output className="time-output" htmlFor="clip-time">
              {formatTime(frame / 25)}
              <span> / 05:00</span>
            </output>
          </div>
          <div className="timeline-row">
            <strong>Reference labels</strong>
            {reference && referenceState === "ready" ? (
              <Timeline
                segments={reference}
                frame={frame}
                name="Reference labels"
              />
            ) : (
              <div className="reference-placeholder" role="status">
                {referenceState === "loading"
                  ? "Loading public reference labels..."
                  : "Reference labels are not loaded"}
              </div>
            )}
          </div>
          <div className="timeline-row">
            <strong>Model result</strong>
            <Timeline
              segments={DEMO_SEGMENTS}
              frame={frame}
              name="Model result"
            />
          </div>
          <div className="time-axis">
            <span>00:00</span>
            <span>02:30</span>
            <span>05:00</span>
          </div>
          <label className="sr-only" htmlFor="clip-time">
            Inspect time in the five-minute clip
          </label>
          <input
            className="time-slider"
            id="clip-time"
            type="range"
            min="0"
            max={LAST_SECOND}
            step="0.04"
            value={seekTarget ?? frame / 25}
            aria-valuetext={seekTarget === null ? `${formatTime(frame / 25)}, ${predicted ? LABEL_NAMES[predicted] : "unknown"}, reference ${actual ? LABEL_NAMES[actual] : "not loaded"}` : `Seeking to ${formatTime(seekTarget)}`}
            onChange={(event) => seek(Number(event.target.value))}
          />
          <div className="comparison-footer">
            <Legend />
            <button className="text-button" onClick={openSetup}>
              Video and reference settings
            </button>
          </div>
          {referenceState === "error" && (
            <div className="reference-error" role="alert">
              <p>{referenceError}</p>
              <button
                onClick={() => setReferenceAttempt((attempt) => attempt + 1)}
              >
                Retry public reference
              </button>
            </div>
          )}
          <p className="footnote">
            Move the slider to inspect a moment. Arrow keys move one video
            frame. Label files are read for display only.
          </p>
        </section>
        <section
          className="panel scores"
          aria-label="Published results on the public clip"
        >
          <div>
            <h2>Published result on this sample</h2>
            <p>Saved evaluation scores, not recalculated by this viewer</p>
          </div>
          <dl>
            <div>
              <dt>Possession score /100</dt>
              <dd>77.2</dd>
            </div>
            <div>
              <dt>Moment score /100</dt>
              <dd>90.2</dd>
            </div>
            <div>
              <dt>Reference / model possessions</dt>
              <dd>
                17 <span>/ 12</span>
              </dd>
            </div>
          </dl>
          <p className="footnote">
            Scores are not percentages of accuracy. Full-match results are not
            known. <Link href="/evidence">Read the evaluation limits</Link>.
          </p>
        </section>
        <footer className="page-footer">
          <span>Precomputed possession review / Public five-minute sample</span>
          <p>
            The sample was seen during development. It is not a fresh blind
            test. <Link href="/evidence">Sources and limitations</Link>
          </p>
        </footer>
      </main>
      <dialog
        ref={dialogRef}
        className="setup-dialog"
        aria-labelledby="setup-title"
      >
        <div className="dialog-header">
          <h2 id="setup-title">Video and reference settings</h2>
          <button
            className="icon-button"
            aria-label="Close video settings"
            onClick={() => dialogRef.current?.close()}
          >
            &times;
          </button>
        </div>
        <p>
          Video and local label files stay on your device. Nothing is uploaded.
        </p>
        <div className="settings-section">
          <h3>Local tactical video</h3>
          {view === "replay" ? (
            <>
              <label className="field-label" htmlFor="video-mode">
                Which file are you opening?
              </label>
              <select
                id="video-mode"
                value={mode}
                onChange={(event) =>
                  setMode(event.target.value as LocalMedia["mode"])
                }
              >
                <option value="clip">Five-minute reference clip</option>
                <option value="original">
                  Original tactical recording (window only)
                </option>
              </select>
              <p className="footnote">
                {mode === "clip"
                  ? "A 5:00 clip beginning at 07:27.40 in the original recording."
                  : "Only 07:27.40 to 12:27.40 of the original tactical recording will play. Other moments are not shown."}
              </p>
              <label className="file-picker">
                <span>Choose video from device</span>
                <input
                  id="video-file"
                  type="file"
                  accept="video/*,.mkv,.avi"
                  onChange={chooseVideo}
                />
              </label>
              <p className="footnote">
                The app cannot verify a video's identity from its duration.
                Choose the matching tactical recording, not broadcast footage.
              </p>
              {mediaError && (
                <p className="error-message" role="alert">
                  {mediaError}
                </p>
              )}
              {mediaState === "loading" && (
                <p role="status">Opening the local video...</p>
              )}
            </>
          ) : (
            <p>
              <Link href="/replay">Open video review</Link> to choose and play
              your local recording.
            </p>
          )}
        </div>
        <div className="settings-section">
          <h3>Reference labels</h3>
          <p>
            The public labels load directly from the{" "}
            <a href={REFERENCE_URL} target="_blank" rel="noreferrer">
              published source
            </a>
            . If access fails, open the same reference file locally.
          </p>
          <label className="file-picker">
            <span>Open local reference labels</span>
            <input
              id="reference-file"
              type="file"
              accept=".csv,text/csv"
              onChange={(event) => void chooseReference(event)}
            />
          </label>
          {referenceState === "ready" && (
            <p className="success-message" role="status">
              Reference labels loaded for the same five-minute window.
            </p>
          )}
          {referenceError && (
            <p className="error-message" role="alert">
              {referenceError}
            </p>
          )}
        </div>
        <div className="dialog-footer">
          <button
            className="primary"
            onClick={() => dialogRef.current?.close()}
          >
            Done
          </button>
        </div>
      </dialog>
    </div>
  );
}
