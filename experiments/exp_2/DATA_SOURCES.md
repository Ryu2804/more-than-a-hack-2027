# exp_2 data sources and the extra-data licence gate

Checked 2026-11-08 via the GitHub API, before any exp_2 fit.

| source | used | licence | labels / provenance | decision |
|---|---|---|---|---|
| IDSSE / DFL open tracking (7 matches `J03WMX J03WN1 J03WOH J03WOY J03WPY J03WQQ J03WR9`) | yes (via exp_1's copies, hash-checked) | CC BY 4.0 (dataset publication) | provider frame labels: `BallStatus` (alive/dead) and `BallPossession` (home/away) per frame | proxy for every arm, as in exp_0/exp_1 |
| Metrica Sports sample data (`metrica-sports/sample-data`) | **no** | none: the repository has no licence file; the README only asks users to "be responsible" and acknowledge the source | events + tracking for 3 games; possession would need event-derived rules | **skipped** — licence unclear (PROMPT rule: unclear licence → skip) |
| SkillCorner open data (`SkillCorner/opendata`) | **no** | MIT (repository `LICENSE`, 2020) | broadcast tracking (10 fps, players extrapolated outside the camera view) with a per-frame `possession` field (player / group) produced by SkillCorner; method not documented; ball-in-play only implied by phases-of-play files | **not used in exp_2** — see below |

## Why `exp2_aug_multi` is not run (decided before any fit)

The licence gate passes for SkillCorner, but its labels are not comparable with IDSSE's: SkillCorner's
`possession` is a player-level field from broadcast video (method undocumented) and dead ball is only
implied by gaps between phases of play, whereas IDSSE labels are the provider's team possession and
ball status per frame. A leave-one-source-out score would mix transfer with label-definition mismatch,
so it would not measure what H1 asks. Harmonising the two label definitions (and resampling 10 fps
broadcast data to the 25 fps degrader) is its own study; it is listed as the next step in `REPORT.md`.
ryu's branch already trains on Metrica + SkillCorner with event-derived labels, and its shipped model
is evaluated here zero-shot on both views (`ryu_final`).

## BAR-ATH

Read only by notebook 05 through exp_1's loader (`exp1_target.load_target_match`); aggregates only.
No human-produced data about BAR-ATH is used. The public ground truth is never opened by exp_2.
