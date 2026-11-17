# Ball Possession (Penguasaan Bola): design handoff

## Status

Archived handoff from the FigCraft design phase. The design files and previews in this folder still show the Indonesian UI of that phase. The web application itself is implemented in English; see the [web README](../README.md) for implementation status, browser verification and deployment. Technical documents and research code outside `web/` were not changed.

The visual direction follows the [brief](../football_match_ball_possession_dashboard_ui_ux_spec.md): light workspace, Inter, canvas `#F6F7F9`, white cards, Barcelona blue, Athletic Club green, amber for dead ball. ENERGY 2 / RHYTHM 2 / MOTION 1.

## Scope of the views

All summaries and charts use **the 5 minutes that have public reference labels**, not the whole match.

- Boundary reference: [public metadata](../../evaluation/gt/BAR-ATH_possession_gt_public_meta.json).
- Tactical video source: frames 11185 to 18684, inclusive.
- Half-open interval: `[11185, 18685)`, 7,500 frames at 25 fps.
- Part of the original file: `07:27.40` to `12:27.40`, end exclusive.
- Displayed time: `00:00` to `05:00`, counted from the start of the clip.
- With a trimmed clip: `source_frame = 11185 + clip_frame`.
- With the original match file: limit playback/seek to `[447.4, 747.4)` seconds, and do not present any other part as the public evaluation window.
- The broadcast camera has no verified time mapping; its time is not the tactical time.

The details the implementation needs live in this metadata and in [games.py](../../games.py), not in interface text.

## Fair comparison

The [public reference labels](../../evaluation/gt/BAR-ATH_possession_gt_public.csv) and the [model output](../../submission/artifacts/farras_wave5.csv) are cut to the identical interval. The two rows in Replay and MobileReplay share start point, width, colours and time scale:

1. **Reference labels**: 21 label runs, including 17 Barcelona/Athletic Club possessions.
2. **Model output**: 15 label runs, including 12 Barcelona/Athletic Club possessions.

Segments are drawn directly from each source's run-length encoding. There is no smoothing, label correction, time shift, training, model selection or score recomputation. The public GT is read only for the visual comparison; the private GT is not used.

The model output here is the **frozen campaign candidate** (`farras_wave5.csv`). The submitted file (`carried_by_admin.csv`) is the same method retrained from scratch and agrees with it on 99.52% of graded frames ([FINAL_METHOD.md](../../docs/FINAL_METHOD.md)).

See [replay.json](replay.json) for the interval and source contract and [evidence.json](evidence.json) for the displayed distribution. Predicted counts in the clip: Barcelona 3,585, Athletic Club 2,860, dead ball 1,055, unknown 0. Percentages use 7,500 as denominator. `55.6%` uses Barcelona + Athletic Club as denominator and is labelled "Barcelona while the ball is live".

The area/line summary uses 30 intervals of 250 frames (10 seconds) each. Values are actual label frequencies drawn as steps, not probabilities or splines.

## Language and scores

The final method name and research details are in the [report source](../../docs/final_method.json); they are not shown in the UI.

Published scores are only rescaled for display, not recomputed:

| UI label | Source | Display |
| --- | --- | --- |
| Possession / 100 (Penguasaan bola) | Public spellF1 0.7724137931034483 | 77.2 |
| Every moment / 100 (Tiap momen) | Public macroF1 0.9020586271571484 (frozen candidate) | 90.2 |
| Other matches / 100 (Pertandingan lain) | External-development spellF1 0.670099305497763 | 67.0 |

These are **scores out of 100, not accuracy percentages**. The other-matches value appears only on the explanation page, separate from the clip assessment. The Replay card shows the two public scores and the reference/model possession counts.

The limits of the assessment are explained in plain language: the clip was seen during development, new testing is still needed, and the full-match result is unknown. (At design time the from-scratch rebuild was not yet complete; it has since been done by the E06 notebook.)

## Local video and sample image

The clip `cuplikan-acuan-5-menit.mp4` was made locally, **outside the repo**: 300 seconds, 7,500 frames, 1920 × 1080. Its first and last frames decode. Its first frame is closest to source frame 11185 when compared with its neighbours; compression differences remain.

Do not copy or publish the clip, the original video or frames from the bundle to GitHub, Vercel or public documents. [RULES.md](../../RULES.md) forbids redistributing the data and clipped video. The web implementation uses local files and revokes the object URL when the file is replaced.

The poster in FigCraft is still a crop of an image supplied by the user, credited to [Hadi Sotudeh / LinkedIn](https://lnkd.in/p/gDmRew2e). The "Sample image" label stays visible. The poster is not evidence that the video is playing or matches the indicated moment; permission for production reuse has not been verified. No GT, player positions or reference model scores were copied from the screenshot.

## Screens and interaction

Ten frames remain: Desktop, Tablet, Mobile, Evidence, MobileEvidence, States, Replay, MobileReplay, MediaSetup, MobileMediaSetup. The file stores 6 flows and 39 prototype connections. The transport icons are sample buttons, not an active player.

The desktop overview grid stays 2 × 2, gap 20, cards 638 × 356. Mobile uses one column; the summary and comparison stay readable. Minimum touch target 44 px, minimum font 11 px, focus ring 2 px, and every colour is paired with a team/ball-state name. Text over the poster sits on a blurred dark panel.

For the implementation: one clip-relative cursor updates the video and both label rows together. Results from an earlier moment must not appear to match after a seek. Show no-video-selected, buffering, seek, decode error and end-of-clip clearly. A video failure must not remove the label comparison. Do not derive player positions from possession labels.

## Next implementation

The implementation part of this handoff is historical. The monorepo, application, lockfile, production build and local video playback now exist. Deployment status and application verification follow the [web README](../README.md), not the static design checks in this folder.
