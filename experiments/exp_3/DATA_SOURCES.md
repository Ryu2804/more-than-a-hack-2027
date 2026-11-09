# exp_3 data sources

Checked 2026-11-08 through the Figshare API, before any exp_3 fit.

| source | used for | licence | provenance | location |
|---|---|---|---|---|
| IDSSE / DFL open tracking, 7 matches (`J03WMX J03WN1 J03WOH J03WOY J03WPY J03WQQ J03WR9`) | proxy inputs and provider labels (`BallStatus`, `BallPossession`) | CC BY 4.0 | via exp_1's frozen copies (`exp_1/cache/proxy`, hash-checked by exp_1's judge) and exp_2's token caches, read in place | `../exp_1/cache`, `../exp_2/cache` |
| IDSSE event data `DFL_03_02_events_raw_*.xml`, the same 7 matches | `L_ctrl` rule (c), only if the event sync passes (PREREG §2.3); here it did not, so the events are **not used** in the labels | CC BY 4.0 | Figshare article 28196177, DOI [10.6084/m9.figshare.28196177](https://doi.org/10.6084/m9.figshare.28196177), "An integrated dataset of spatiotemporal and event data in elite soccer" | `external/idsse_events/` (gitignored) |
| IDSSE match information `DFL_02_01_matchinformation_*.xml` | club id → home/guest, person id → track slot (event sync) | CC BY 4.0 | same Figshare article; byte-identical to exp_0's copies | `external/idsse_meta/` (gitignored) |

Event files: download URLs, sizes and MD5s are listed in `external/figshare_files.json` (gitignored,
re-created by the command below). Every file's MD5 matched Figshare's `supplied_md5`.

| file | bytes | Figshare download |
|---|---|---|
| `DFL_03_02_events_raw_DFL-COM-000001_DFL-MAT-J03WMX.xml` | 776 578 | `https://ndownloader.figshare.com/files/51643493` |
| `DFL_03_02_events_raw_DFL-COM-000001_DFL-MAT-J03WN1.xml` | 612 114 | `https://ndownloader.figshare.com/files/51643496` |
| `DFL_03_02_events_raw_DFL-COM-000002_DFL-MAT-J03WOH.xml` | 635 747 | `https://ndownloader.figshare.com/files/51643499` |
| `DFL_03_02_events_raw_DFL-COM-000002_DFL-MAT-J03WOY.xml` | 683 018 | `https://ndownloader.figshare.com/files/51643502` |
| `DFL_03_02_events_raw_DFL-COM-000002_DFL-MAT-J03WPY.xml` | 698 898 | `https://ndownloader.figshare.com/files/51643505` |
| `DFL_03_02_events_raw_DFL-COM-000002_DFL-MAT-J03WQQ.xml` | 732 186 | `https://ndownloader.figshare.com/files/51643508` |
| `DFL_03_02_events_raw_DFL-COM-000002_DFL-MAT-J03WR9.xml` | 681 050 | `https://ndownloader.figshare.com/files/51643511` |

The match information files are the 7 `DFL_02_01_matchinformation_*` entries of the same listing.

## Re-download

```bash
cd exp_jeremy/exp_3
mkdir -p external/idsse_events external/idsse_meta
curl -s "https://api.figshare.com/v2/articles/28196177/files?page_size=100" -o external/figshare_files.json
python3 - <<'EOF'
import json, hashlib, subprocess
for f in json.load(open('external/figshare_files.json')):
    folder = 'idsse_events' if 'events_raw' in f['name'] else 'idsse_meta' if 'matchinformation' in f['name'] else None
    if folder:
        path = f'external/{folder}/{f["name"]}'
        subprocess.run(['curl', '-sL', '-o', path, f['download_url']], check=True)
        assert hashlib.md5(open(path, 'rb').read()).hexdigest() == f['supplied_md5'], path
EOF
```

## Event schema actually used (parsed with lxml; floodlight / kloppy / databallpy not installed)

- `Event@EventTime`: ISO 8601 with milliseconds and a time-zone offset. Only shots carry
  `Event@CalculatedFrame`; it is used solely to measure sync quality, never as a label.
- Half kick-offs: the earliest `KickOff` per `GameSection` (`firstHalf` / `secondHalf`).
- Controlled on-ball events, with their attributes and conditions:

  | element | team attribute | condition |
  |---|---|---|
  | `Play` | `Team` | `Evaluation` ∈ {successfullyCompleted, successful} |
  | `ShotAtGoal` | `Team` | none |
  | `BallClaiming` | `Team` | `Type` ∈ {InterceptedBall, BallClaimed, BallHeld} |
  | `TacklingGame` | `WinnerTeam` | `WinnerResult` ∈ {ballClaimed, ballControlRetained, dribbledAround} |

- `Delete` events carry no target id (537 of them over the 7 files) and are ignored.

## Not used

Metrica (no licence) and SkillCorner (label mismatch), for the same reasons as in exp_2
(`../exp_2/DATA_SOURCES.md`).

## BAR-ATH

Read only by notebook 05, through exp_1's loader, with the broadcast ATD and homography passed to
the pinned exp_0 fusion function. Aggregates only. No human-produced data about BAR-ATH is used,
and the public ground truth is never opened.
