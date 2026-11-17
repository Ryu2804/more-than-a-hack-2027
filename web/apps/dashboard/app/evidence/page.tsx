import Link from "next/link";
import styles from "./evidence.module.css";

const METHOD_SOURCE =
  "https://github.com/Ryu2804/more-than-a-hack-2027/blob/3d113d003fc5ab9f7a696635f4f0fe85570f616c/docs/FINAL_METHOD.md";

export default function EvidencePage() {
  return (
    <main id="main" className={styles.page}>
      <div className={styles.container}>
        <header className={styles.header}>
          <div>
            <h1>What these results mean</h1>
            <p>How to read the scores, and where the evidence stops.</p>
          </div>
          <nav className={styles.navigation} aria-label="Results navigation">
            <Link className={styles.secondaryLink} href="/">
              Back to overview
            </Link>
            <Link className={styles.primaryLink} href="/replay">
              Open replay
            </Link>
          </nav>
        </header>

        <div className={styles.grid}>
          <article className={styles.card} aria-labelledby="reading-title">
            <h2 id="reading-title">Two ways to read the same five minutes</h2>
            <p className={styles.intro}>
              A model can get the team right at many moments while still missing
              where one possession ends and another begins.
            </p>
            <div className={styles.definitions}>
              <section aria-labelledby="possession-definition">
                <h3 id="possession-definition">Whole possessions</h3>
                <p>
                  The possession score compares complete periods of team
                  possession with the reference labels. It checks their overlap
                  in time, including the start and end boundaries. Missing,
                  splitting, or joining possessions can lower this score.
                </p>
              </section>
              <section aria-labelledby="moment-definition">
                <h3 id="moment-definition">Each moment</h3>
                <p>
                  The moment score compares the label at each instant, then
                  averages the results for Barcelona, Athletic Club, and dead
                  ball. It measures a different kind of agreement from the
                  whole-possession score.
                </p>
              </section>
            </div>
            <p className={styles.scoreNote}>
              These are scores out of 100, not percentages of accuracy.
            </p>
            <section className={styles.states} aria-labelledby="states-title">
              <h3 id="states-title">Labels in the viewer</h3>
              <dl className={styles.stateList}>
                <div>
                  <dt className={styles.home}>H</dt>
                  <dd>Barcelona</dd>
                </div>
                <div>
                  <dt className={styles.away}>A</dt>
                  <dd>Athletic Club</dd>
                </div>
                <div>
                  <dt className={styles.dead}>D</dt>
                  <dd>Dead ball</dd>
                </div>
                <div>
                  <dt className={styles.unknown}>X</dt>
                  <dd>Unknown</dd>
                </div>
              </dl>
              <p className={styles.smallText}>
                Unknown means no possession label is available. Possession labels
                alone do not show player positions.
              </p>
            </section>
          </article>

          <section className={styles.card} aria-labelledby="public-title">
            <div className={styles.sectionHeader}>
              <h2 id="public-title">Public five-minute clip</h2>
              <p>Barcelona vs Athletic Club. Both scores use the same clip.</p>
            </div>
            <dl className={styles.scoreList}>
              <div className={styles.scoreRow}>
                <dt>Possession score /100</dt>
                <dd>77.2</dd>
              </div>
              <div className={styles.scoreRow}>
                <dt>Moment score /100</dt>
                <dd>90.2</dd>
              </div>
            </dl>
            <div className={styles.counts}>
              <p>
                <strong>17</strong> reference possessions
              </p>
              <p>
                <strong>12</strong> model possessions
              </p>
            </div>
            <p className={styles.smallText}>
              Counts include Barcelona and Athletic Club possessions within this
              clip only, not dead-ball periods. They are not full-match totals.
            </p>
            <a className={styles.sourceLink} href="#sources">
              Published score source
            </a>
          </section>

          <article className={styles.card} aria-labelledby="limits-title">
            <h2 id="limits-title">What has not been established</h2>
            <dl className={styles.limitList}>
              <div>
                <dt>Full-match score: unknown</dt>
                <dd>
                  There is no reported private or full-match score for this
                  model. The five-minute scores cannot stand in for one.
                </dd>
              </div>
              <div>
                <dt>This clip was seen during development</dt>
                <dd>
                  The public reference results were already known. This is not
                  a fresh blind test, and the scores do not guarantee performance
                  on new matches.
                </dd>
              </div>
              <div>
                <dt>Reproduction from the raw data is incomplete</dt>
                <dd>
                  The published predictions can be exported, but recreating the
                  identical submission from the original raw dataset is not yet
                  fully reproducible.
                </dd>
              </div>
            </dl>
          </article>

          <article className={styles.card} aria-labelledby="external-title">
            <h2 id="external-title">Separate development results</h2>
            <dl className={styles.scoreList}>
              <div className={styles.scoreRow}>
                <dt>Other matches /100</dt>
                <dd>67.0</dd>
              </div>
            </dl>
            <p>
              This possession score comes from separate external development
              data, not the public five-minute clip and not the competition
              leaderboard.
            </p>
            <p>
              These matches were used during development and method selection.
              They had already been seen in earlier research, so this is not a
              final blind score or a guarantee of generalization.
            </p>
            <a className={styles.sourceLink} href="#sources">
              Development result source
            </a>
          </article>
        </div>

        <footer className={styles.footer}>
          <section className={styles.viewerNote} aria-labelledby="viewer-title">
            <h2 id="viewer-title">A viewer for saved results</h2>
            <p>
              This viewer shows precomputed predictions. It does not run live
              inference or create new predictions from your video. A video you
              choose stays on your device and is not uploaded. No raw licensed
              match media is hosted here.
            </p>
          </section>
          <section id="sources" className={styles.sources} aria-labelledby="sources-title">
            <h2 id="sources-title">Source and technical details</h2>
            <p>
              Scores and research limitations follow the published method report
              at a fixed source revision. Some comparison-data reuse permissions
              remain unclear; the report records those limits.
            </p>
            <cite>
              <a className={styles.sourceLink} href={METHOD_SOURCE}>
                Read the published method report on GitHub
              </a>
            </cite>
          </section>
        </footer>
      </div>
    </main>
  );
}
