# Possession web

English match-review application, implemented from the approved FigCraft design.

**Production:** https://possession-match-review.vercel.app

- `/`: possession overview, measured shares and a reference/model comparison.
- `/replay`: local video on the left, synchronized 2D method replay on the right, frame stepping and aligned timelines.
- `/evidence`: published scores, sources and evaluation limitations.

This is a viewer for the selected saved result, not a live prediction service.
The Python research code and technical report remain unchanged.

## Run locally

Use Node.js 24 and npm. From `web/`:

```bash
npm ci
npm run dev
```

Open http://127.0.0.1:3000. For the production server:

```bash
npm test
npm run build
npm start
```

## Monorepo

```text
web/
  apps/dashboard/          @possession/dashboard: Next.js, React, TypeScript
  packages/possession/     @possession/core: label parsing and aggregation
  packages/design-tokens/ @possession/design-tokens: shared theme variables
  design/                 archived FigCraft sources and previews
```

The npm lockfile is shared. No nested Git repository or Python/GPU runtime is required.

## Data and video

Every view uses source frames `[11185, 18685)`: 7,500 frames at 25 frames per second.
Displayed time starts at `00:00` and ends at `05:00`. The last selectable frame is at `04:59.96`.
The bundled model segments match the same window of `../submission/artifacts/farras_wave5.csv` exactly.

Public reference labels are fetched in the browser from a pinned public repository revision.
They are not bundled into the app. A local reference file can be opened if that source is unavailable.
Both sources must match the published window's normalized label identity; predictions cannot be passed off as reference labels.
Files must have continuous absolute `frame,label` rows for either the entire match or the exact public window.
The viewer does not train, tune, correct labels or calculate a new benchmark score.

In **Video review**, select **Open local video**:

- **Five-minute reference clip:** the matching 300-second tactical clip, starting at original recording time `07:27.40`.
- **Original tactical recording:** playback is limited to `07:27.40` through the last frame before `12:27.40`.

After the selected recording passes the duration check, playback starts automatically with sound muted.
If the browser blocks automatic playback, the viewer shows a message and the Play button remains available.
The app cannot automatically read a local recording when the website opens; a file must first be selected.

The private clip already created for this project is outside the repository, under
`PossessionAI/private-media/cuplikan-acuan-5-menit.mp4` in the local FigCraft workspace.
Duration checks cannot establish a video's identity. Choose the matching tactical recording; broadcast footage is not aligned.
MP4 with H.264 video is recommended. File decoding depends on browser codec support.
Local media uses a browser object URL, revoked when replaced or removed.

Do not commit or deploy recordings, video frames, raw tracking, private labels, checkpoints or the annotation harness.
See [dataset rules](../RULES.md#the-data). The third-party reference poster is not used by the application.
The overview does not infer positions from labels. In replay, real observed positions and saved intermediate model scores can be opened from a private local file.
The method identifies the team in possession, not an individual ball holder.

## Synchronized method replay

Open `/replay`, choose the matching video, then **Open match data** and select
`method-replay.private.json`. The local file is beside the private clip in
`PossessionAI/private-media/` in the FigCraft workspace. Neither file is deployed.

The right-hand panel shows:

- Actual observed tracking positions at the model's five-samples-per-second rate. Missing players or ball stay missing; markers are anonymous tracking slots, not shirt numbers.
- The saved short-possession and sequence-model outputs, their fixed 75%/25% mixture, and the close-distance cue when used.
- The instant highest score beside the saved final decision, with surrounding decisions and retained dead-ball state.

The final output is fixed and offline. Both sequence models can use future context.
The video is a synchronized review aid, not an input to these models or a new live inference stream.
Movement is driven by the video clock without synthetic interpolation or free-running decorative animation.
Seek waits for the video; pause freezes the same observations. Rotating the pitch changes only its display orientation.

To recreate the private replay file from existing research caches, run from `web/`:

```powershell
python -B tools/prepare_method_replay.py --repo "D:/private/frozen-wave5-workspace" --output "C:/Users/farras/Documents/FigCraft/PossessionAI/private-media/method-replay.private.json"
```

This optional exporter requires the original frozen workspace with its `experiments/exp_farras/wave3`, `wave4` and `wave5` source files, preregistration and ignored caches. The consolidated repository uses E04–E06 instead; point `--repo` to the preserved original workspace. Renaming files or substituting newly retrained checkpoints does not satisfy its frozen hashes. The web application itself needs neither that workspace nor Python.

The exporter refuses repository-local destinations and existing files unless `--force` is supplied.
It uses observation-only keys from the target archive, the selected saved component outputs,
the production fusion/decoder, and training-only statistics. It verifies the complete 199,319-label
prediction against the frozen artifact before exporting 1,500 samples for the public window.
It does not train or read target ground truth. This is cached-stage verification, not a claim of complete raw-dataset reproduction.
The primary cache has no preregistered hash; current source hashes are recorded and complete decoded-label equality is required.
Local-file validation checks shape, frame bounds, probability weights and frozen labels, not independent cryptographic authenticity of user-edited input.

## Results and limitations

The public diagnostic scores are **77.2 /100** for possessions and **90.2 /100** for individual moments.
These are published scores, not percentages of accuracy. The same window contains 17 reference possessions and 12 model possessions.
The separate development score of **67.0 /100** appears only on the explanation page.
It is not a competition leaderboard result or a public-clip score.

The public sample was seen during development, external evaluation is not a fresh blind holdout,
the private/full-match score is unknown, and raw-data-to-submission reproduction remains incomplete.
Canonical facts: [final method](../docs/FINAL_METHOD.md), [machine-readable provenance](../docs/final_method.json).

## Vercel deployment

Project: `eycoos-projects/possession-match-review`. The deployment is not connected to GitHub.
The existing CLI link is local and ignored. From `web/`:

```bash
npx vercel deploy --prod --yes --scope eycoos-projects
```

The project's source root is the uploaded `web/` directory, with Root Directory `apps/dashboard`,
Next.js preset, Node.js 24, and files outside Root Directory enabled for workspace packages.
`apps/dashboard/vercel.json` installs with `cd ../.. && npm ci` and builds with `cd ../.. && npm run build`.
The output is `.next`. `.vercelignore` excludes design assets, local credentials, caches and recordings.
For Git deployment from this repository, set Root Directory to `web/apps/dashboard` and enable access to files outside that directory. Keep Node.js 24, the Next.js preset, the install/build commands above and output `.next`. Pushing source does not connect the existing Vercel project to GitHub; its current deployment remains CLI-managed.

## Verification

- 29 TypeScript regression tests passed, including method-data bounds, occlusions, probability mixing and frozen-label consistency.
- Python exporter checks: 8 passed; the directory-symlink test was skipped because this Windows environment did not permit creating the test symlink.
- The actual private replay parsed successfully: 1,500 samples, including 177 missing-ball observations. Full cached decoding matched all 199,319 frozen prediction labels.
- Method replay browser checks covered video-driven sample changes, observed/missing ball rendering, rapid keyboard seeks, frame stepping, the final frame, pause stability, pitch rotation, invalid-file rejection and local-data removal.
- The loaded method panel reflowed at widths 360, 390, 834 and 1440 in the browser, including the existing light/dark themes. No local-file POST requests were observed.
- Production build and TypeScript checks passed locally; the Vercel production deployment is Ready and returns HTTP 200.
- Browser checks covered overview/review/results navigation, local playback, pause, seek, frame stepping, original-recording bounds, dialog Escape/focus restoration, theme toggling and public-source retry.
- Muted autoplay was observed after file selection without pressing Play, for both the five-minute clip and original recording at its restricted start. Manual pause worked; a recording failing the duration check remained paused.
- A valid but incorrect reference file was rejected; the original public reference restored both timelines.
- Overview widths 360, 390, 834, 1024 and 1440 had no horizontal overflow. Replay and results were also inspected on mobile; a separate 200% text-size reflow check passed at width 720.
- Live playback used a local blob URL and generated no POST requests. Removing media cancels its local blob reads; those expected aborts are not application exceptions.
- The corrected production page hydrated without JavaScript errors.

Design delivery gate: **Hard Gate PASS** (working controls, keyboard/recovery states, verified contrast and layouts);
**Purpose PASS** (established team colors, readable Inter, equal timeline scales);
**Liveliness PASS** (ENERGY 2 / RHYTHM 2; motion only follows observed video/track time, never a decorative loop);
**Craftsmanship PASS** (real frozen results and intermediate scores, honest occlusions, local media and observed browser behavior).

## Design archive

The [editable FigCraft file](design/possession-ai.figcraft), [desktop](design/desktop.png),
[mobile](design/mobile.png), [replay](design/replay.png), [mobile replay](design/replay-mobile.png),
[results explanation](design/method.png), [state specimens](design/states.png),
[tokens](design/tokens.json) and [handoff](design/HANDOFF.md) (translated to English) preserve the earlier design phase, whose mockups are in Indonesian.
Its static [design verification](design/verification.json) is not the application's runtime test report.
The live English application is the current implementation.
