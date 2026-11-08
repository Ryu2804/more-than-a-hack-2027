# More than a Hack 2027 — Literature & Methods Brief for Frame-Level Possession

Nov 8, 2026 ·

## Bottom line

No published method solves our exact setting, so we should combine three proven ideas rather than copy one paper. The setting is uncorrected tracking, the ball missing on 41% of graded frames, no player identities, and a segment-level metric.

**The main weakness of the literature for us.** Almost every possession paper assumes corrected tracking with a reliable ball. Their headline accuracies (83–92%) will not transfer \[High\]. A 2024 validation study found rule-based pass detection fell from F = 0.92 to 0.71 once event and positional data were independent ([Bischofberger et al.](https://doi.org/10.1371/journal.pone.0298107)). Reported numbers are also frame-level, and frame accuracy says little about spellF1 \[High\].

**Priority order, by expected gain per hour of work:**

1. **Segment-level decoder** (hidden semi-Markov model or penalised Viterbi with a switch cost and minimum durations). The four published baselines differ only in smoothing, yet spellF1 moves fivefold \[High\].
2. **Labelled matches from other games for validation and training.** The open DFL/IDSSE dataset gives 7 full Bundesliga matches with per-frame ball status and team possession, CC-BY 4.0. This replaces the forbidden tuning on the public 5 minutes \[High that the data exists; Medium that its possession definition matches ours\].
3. **Dedicated dead-ball and restart detector** built from Vidal-Codina-style restart triggers, collective speed and Smart Tagging set-piece anchors \[Medium\].
4. **Ball-free possession model** for the long ball gaps (BallRadar / Borghesi-style permutation-invariant network). Highest ceiling, highest cost \[Medium\].

Deprioritise video ball detection and action spotting: with 9 days left, their payoff is uncertain \[Medium\].

## How the challenge maps to the literature

Each difficulty in the challenge has a matching research area. The "What our EDA measured" column comes from our own notebook (01\_eda\_possession.ipynb), not from the papers.

| Challenge sub-problem | What our EDA measured | Research area that addresses it |
| --- | --- | --- |
| Ranking metric: possessions matched by temporal IoU 0.3–0.7 | Raw nearest-team flips 29.6 times per minute; one spurious flip costs about 0.09 spellF1 at the public window's scale | Temporal action segmentation (segmental F1, over-segmentation), hidden semi-Markov models, change-point detection |
| "Controlled" possession; loose balls stay with the last team | 6 of 17 public possessions last 3.6 s or less | Rule-based possession models (Link & Hoernig; Vidal-Codina), learned models (Borghesi) |
| Ball missing on 41% of graded frames | Gaps of 5–60 s; nearest team differs across 40–47% of gaps | Ball and possessor inference from player trajectories (BallRadar, TranSPORTmer, Peral et al., Wei et al.) |
| Dead ball (D) separates possessions | Baseline labels 61% of true-D frames as H; collective speed bottoms at 0.6–1.5 m/s before restarts | Restart detection from player configurations; dead/alive classifiers; video action spotting |
| No player identities; 46 fragmented slots | Median tracklet 13.6 s | Permutation-invariant set models; formation-based role assignment |
| No labels on this match | Public window is unrepresentative (ball visible 88% vs 58% elsewhere) | Labelled matches from other games; weak supervision from noisy rules |

## Methods ranked by fit

Ranked by how directly each method moves spellF1 on our data, discounted by effort. Confidence is in how well the method fits our setting, not in the paper's own results.

| Rank | Method / source | Sub-problem it solves | Main gap for our data | Code / data | Fit confidence |
| --- | --- | --- | --- | --- | --- |
| 1 | Hidden semi-Markov model or penalised Viterbi decoding ([Yu 2010](https://doi.org/10.1016/j.artint.2009.11.011); PELT via [ruptures](https://arxiv.org/abs/1801.00718)) | Over-segmentation | Only as good as the per-frame probabilities it decodes | Standard; small numpy implementation | High |
| 2 | [IDSSE / DFL open data](https://doi.org/10.1038/s41597-025-04505-y) (Bassek et al. 2025) | Labels for validation and training | Clean optical tracking; DFL's possession definition, not Metrica's | Public, CC-BY 4.0; loads via kloppy, floodlight, databallpy | High (data) / Medium (definition) |
| 3 | [Borghesi et al. 2023](https://doi.org/10.1016/j.eswa.2023.119780), TCN for dead / home / away | Exactly our three labels from tracking | Clean Serie A data; frame accuracy, not segment metric | No code found | Medium |
| 4 | [Vidal-Codina et al. 2022](https://doi.org/10.1007/s12283-022-00381-6) possession and set-piece model | Controlled possession, restarts and restart team | Assumes the ball is tracked; thresholds tuned on clean 25 Hz data | No code; rules described in paper | High (rules) |
| 5 | [BallRadar](https://arxiv.org/abs/2306.08206) (Kim et al., KDD 2023) | Possessor and ball inferred from players only | Built on complete per-player trajectories; our slots fragment | [Public code](https://github.com/pientist/ballradar), trains on Metrica sample data | Medium |
| 6 | [TranSPORTmer](https://arxiv.org/abs/2410.17785) (Capellera et al., ACCV 2024) | Per-frame state: pass, possession, uncontrolled, out-of-play | Uses offense/defense agent types (a possession leak if copied); proprietary data | No code stated | Medium |
| 7 | [Linked HMM weak supervision](https://doi.org/10.1609/aaai.v34i04.6009) (Safranchik et al., AAAI 2020) | Labels on our match without human labelling | Built for text tagging; needs adapting | Paper only (we did not check for code) | Medium |
| 8 | [MS-TCN](https://arxiv.org/abs/1903.01945) smoothing loss / [ASRF](https://arxiv.org/abs/2007.06866) boundary branch | Over-segmentation inside a deep model | Needs labelled sequences (external matches) | Public code (MS-TCN) | Medium |
| 9 | [Peral et al. 2025](https://arxiv.org/abs/2512.19528), multi-modal tactical-camera model | Ball state and possessor with no ball input | Preprint; proprietary data | Not stated | Low |
| 10 | [T-DEED](https://github.com/arturxe2/T-DEED) action spotting on the broadcast video | Ball out of play, throw-ins, restarts | Replays, compute, mapping back to tactical frames | Public code | Low |
| 11 | [WASB](https://github.com/nttcom/WASB-SBDT) ball detector on the tactical video | Filling ball gaps | Our gaps look like stoppages, not detector misses | Public code and weights | Low |

## Area 1 — Segment-level decoding against over-segmentation

spellF1 is the segmental F1 that action-segmentation research has optimised for years, and that field's main finding is that over-segmentation, not frame accuracy, caps segment F1 \[High\].

- [MS-TCN](https://arxiv.org/abs/1903.01945) (Farha & Gall, CVPR 2019) adds a smoothing loss that penalises over-segmentation errors.
- [ASRF](https://arxiv.org/abs/2007.06866) (Ishikawa et al., WACV 2021) adds a boundary-regression branch and reports up to 16.1% better segmental F1. Its benchmark metric is F1 at IoU 10/25/50%, the same family as our spellF1.
- [Hidden semi-Markov models](https://doi.org/10.1016/j.artint.2009.11.011) (Yu 2010) give each state an explicit duration distribution, decoded with a Viterbi algorithm. [Offline change-point detection](https://arxiv.org/abs/1801.00718) (Truong et al. 2020) frames the same problem as cost plus a penalty per change; PELT solves it exactly with a linear penalty, in the ruptures package.

**What to build.** Decode per-frame probabilities p(H, A, D | features) with a cost per switch:

```latex
\hat{y} = \arg\min_{y} \sum_t -\log p(y_t \mid x_t) + \lambda_{HA} N_{H\leftrightarrow A}(y) + \lambda_{AD} N_{\text{alive}\leftrightarrow D}(y)
```

- Use separate penalties for team switches and for alive/dead switches. A false team flip splits a possession; a missed D merges two.
- Add a minimum duration per state, or a full duration distribution (HSMM), learned from external labelled matches. Do not hard-code a minimum above about 1 s: 6 of the 17 public possessions last 3.6 s or less.
- Restrict transitions: after D, the next alive state should start with the restarting team (from Area 5).
- Use X deliberately. The scorer fills X with the previous label, so X inside a long ball gap means "no switch". At a confident transition, commit to the new label.

**How to tune it.** Pick the penalties and durations by spellF1 on the external labelled matches (Area 2), scored with our copy of the official evaluation/possession\_metrics.py. Never tune them on the public 5 minutes \[High: rule and EVALUATION.md\].

## Area 2 — Labelled data from other games

The open DFL/IDSSE matches are the single most useful external resource: they give per-frame H/A/D labels on 7 full matches, enough to tune and validate without touching the public window \[High\].

| Dataset | What it gives us | Labels for H / A / D | Caveat | Fit |
| --- | --- | --- | --- | --- |
| [IDSSE](https://doi.org/10.1038/s41597-025-04505-y) (Bassek et al., Scientific Data 2025; [mirror](https://huggingface.co/datasets/pysport/idsse-data)) | 7 Bundesliga matches (1st and 2nd division), official position and event data, CC-BY 4.0 | Yes: per-frame ball status and team possession from the provider; floodlight's DFL reader returns both, and [databallpy](https://databallpy.readthedocs.io/en/latest/features/team_and_player_possession.html) exposes them | Clean optical tracking; DFL's possession definition. databallpy's own docs show event-derived and provider possession disagreeing in places | High |
| [Metrica sample data](https://github.com/metrica-sports/sample-data) | 3 anonymised matches, synchronised tracking and events; Game 3 is in the same EPTS FIFA format as our ATD | Derivable from events (set pieces, ball out, team of each action) | Only 3 matches; event-derived labels, not corrected-tracking labels | Medium (same provider, same format) |
| [SkillCorner open data](https://github.com/SkillCorner/opendata) | Broadcast tracking at 10 fps, per-frame possession (player and team) | Team yes; dead ball not given directly | Broadcast-derived, so noisier and closer to our conditions; 10 fps | Medium |
| [PFF FC World Cup 2022](https://www.blog.fc.pff.com/blog/pff-fc-release-2022-world-cup-data) | Tracking and event data for the tournament | Derivable from events | Registration required; heavier to process | Low (time) |

**Make the clean data look like ours.** Train and validate on degraded copies, or the model will rely on a ball that is missing 41% of the time:

- Drop the ball in long blocks: in our match 99.7% of missing ball frames sit in gaps over 5 s, mean 17 s. Bias the drops toward stoppages and crowded areas, where our ball disappears.
- Break each player trajectory into tracklets (our median is 13.6 s), shuffle slot order, drop 0–3 players and duplicate one occasionally.
- Remove identities so the model only sees team membership, as in our ATD.

These are statistics of our input, not of our labels, so they should be allowed. This is still worth confirming with the organisers (see the rules section) \[Medium\].

**Validation protocol.** Score each decoder setting on all 7 IDSSE matches with the official scorer and bootstrap the result. The organisers say spellF1 differences below about 0.04 are noise on one full match \[High\].

## Area 3 — Possession models from tracking

The rule-based papers give us defensible definitions of "controlled" possession, and Borghesi et al. is the closest prior work to our exact three labels \[High\].

**Rule-based (use for definitions and features):**

- [Link & Hoernig 2017](https://doi.org/10.1371/journal.pone.0179953) (PLOS ONE). Team possession is an unbroken sequence of individual possessions; when the ball is out of play, nobody has possession. That matches our D. Their model reached F = 0.83 for individual possession on 60 Bundesliga matches.
- [Hoernig et al. 2016](https://doi.org/10.1007/978-3-319-24560-7_13). A player keeps possession while he could still control the ball within about 1 s, estimated from ball position and velocity. Accuracy fell about 4 points on uncorrected data. This is a usable rule for "a loose ball stays with the last team".
- [Vidal-Codina et al. 2022](https://doi.org/10.1007/s12283-022-00381-6) (Sports Engineering; FIFA and MIT authors). A possession zone of 0.5–1 m around each player, a duel zone of 1 m with at least two opponents, and a rule that a possession gain needs a change in ball direction or speed. That last rule directly handles "a deflection without control does not count". They report over 90% detection for most event types across three tracking providers.
- [Khaustov & Mozgovoy 2020](https://doi.org/10.3390/app10228046). A simple rule-based baseline for possession, passes and shots; useful as a sanity check, not a target.

**Learned (closest to our task):**

- [Borghesi et al. 2023](https://doi.org/10.1016/j.eswa.2023.119780) (Expert Systems with Applications). Classifies every frame as dead, home or away from tracking, like our H/A/D. A TCN encodes each object's trajectory; permutation-invariant layers (sum, self-attention or 2D convolution) combine them, so no identities are needed. They report 86.2% possession accuracy and 89.2% dead/alive accuracy, beating rule-based methods by 7.3 and 33.2 points. Data were Serie A 2019–20 tracking, per the first author's thesis.
- Two design choices worth copying: predict dead vs alive first, then the team; and encode each object separately before pooling.

**Caveat.** All of these use clean tracking with a ball present most of the time, and report frame-level scores. A model at 86% frame accuracy can still flicker enough to score poorly on spellF1, so pair any of them with the Area 1 decoder \[High\].

## Area 4 — Inferring possession when the ball is missing

This is where the score is decided: 41% of graded frames have no ball, and possession changes across 40–47% of the long gaps, so carrying the last label forward is wrong almost half the time \[Medium: measured with the noisy nearest-team signal\].

- [BallRadar](https://arxiv.org/abs/2306.08206) (Kim et al., KDD 2023). Predicts which player has the ball at each frame, then the ball trajectory, from player trajectories only. Set Transformers make it permutation-invariant; a hierarchical Bi-LSTM handles time. [Code is public](https://github.com/pientist/ballradar), with a team-level variant, and it trains on the Metrica sample data. Gap: it was built on complete per-player trajectories, and our 46 fragmented slots would need adapting \[Medium\].
- [TranSPORTmer](https://arxiv.org/abs/2410.17785) (Capellera et al., ACCV 2024; Kognia, Barcelona). Classifies each frame as pass, possession, uncontrolled or out-of-play. With the ball fully hidden, state accuracy was 80.8%, versus about 87–90% with the ball. Trained on 283 LaLiga 2022–23 matches at 6.25 Hz. Gap: agents are typed offense or defense, which would leak the answer if copied; we would type them home/away. No code release is mentioned \[Medium\].
- [Peral et al. 2025](https://arxiv.org/abs/2512.19528) (same group, preprint). Uses tactical-camera footage: player trajectories, player types and image crops around each player. It predicts ball trajectory, ball state and possessor with no ball input, and masks image crops during pre-training so the model does not lean on them. Closest to our setup, since we also have the tactical video; data and code are not stated \[Low: preprint\].
- [Wei et al. 2015](https://openaccess.thecvf.com/content_iccv_2015_workshops/w21/html/Wei_Predicting_Ball_Ownership_ICCV_2015_paper.html) (basketball, ICCV workshops). Re-ordering players by role gives a compact feature that predicts the ball owner, and can fuse a noisy ball detector. [Bialkowski et al. 2014](https://doi.org/10.1109/ICDM.2014.133) assign a formation role to each player at every frame. Both are ways to build identity-free features \[Medium\].

**A cheaper version to try first.** Compute per-team features every frame: centroid velocity toward the opponent goal, compactness, defensive-line height, pressing distance, and players near the last-seen ball. Add context from both edges of each gap (who had the ball before and after). Train a gradient-boosted model on degraded IDSSE matches and feed its probabilities to the decoder. Move to a BallRadar-style network only if this plateaus \[Medium\].

## Area 5 — Dead-ball and restart detection

A dedicated D detector matters because D is what separates two possessions by the same team: in our metric test, never predicting D dropped spellF1 to 0.81 even with perfect team labels \[High, public-window scale\].

**Restart triggers from player positions** ([Vidal-Codina et al. 2022](https://doi.org/10.1007/s12283-022-00381-6)). Each dead-ball interval is checked for a player configuration, confirmed by a pattern on the first in-play frame:

| Restart | Trigger in the tracking |
| --- | --- |
| Kick-off | All players in their own half, one near the centre mark |
| Penalty | One player on the goal line between the posts, one opponent near the spot, everyone else outside the box |
| Goal kick | At least one player inside their own goal area |
| Corner | At least one player near a corner mark |
| Throw-in | At least one player beyond the touchline minus a tolerance |
| Free kick | Default when no other pattern is confirmed |

The player closest to the ball in the pattern zone is the executor, which also gives us the **restart team**. That fills the gap left by Smart Tagging, which tags every throw-in and kick-off with Team = N/A \[High\].

**Signals our EDA already found:** collective speed bottoms at 0.6–1.5 m/s before set pieces; the visible ball is outside the pitch on 4.2% of visible frames; tracked-object counts drop during goal celebrations; frames 110,808–113,741 are all D before the second-half kick-off.

**Learned option.** [Borghesi et al.](https://doi.org/10.1016/j.eswa.2023.119780) reached 89.2% dead/alive accuracy with a learned first stage, 33.2 points above rule-based methods. Training one on degraded IDSSE matches, where ball status is labelled, is feasible \[Medium\].

**Video option (lower priority).** SoccerNet-v2 annotates Ball out of play, Throw-in, Clearance (goal kick), Corner, Kick-off and free kicks on broadcast video ([Deliège et al.](https://arxiv.org/abs/2011.13367)). [T-DEED](https://github.com/arturxe2/T-DEED) won the 2024 SoccerNet Ball Action Spotting challenge and its code is public. Running it on our broadcast video and mapping back with the offsets our EDA found (+4,014 frames first half, +26,920 second half) could anchor restarts. Replays must be excluded, and the compute cost is real \[Low for 9 days\].

## Area 6 — Weak supervision on the competition match

Weak supervision lets us learn from the BAR-ATH match itself without any human labels, which RULES.md explicitly allows for algorithmic pseudo-labels \[High\].

[Linked HMMs](https://doi.org/10.1609/aaai.v34i04.6009) (Safranchik, Luo & Bach, AAAI 2020) combine many noisy rules of unknown accuracy into probabilistic sequence labels, with no observed labels. Besides rules that vote on a tag, they add "linking rules" that vote on whether neighbouring elements share a tag. On text-tagging benchmarks this beat models that treat elements independently by about 7 F1 points.

**How it maps to our data:**

- Tag-voting rules: nearest team when the ball is visible and within 1 m; the team of offensive Smart Tagging phases (offensive codes only, since our EDA found the defensive codes are exact mirrors); the restart executor's team; D votes from low collective speed, ball out of bounds and set-piece anchors.
- Linking rules: "same state" while the ball stays within reach of one team; "same state" through a low-speed stretch with no ball.
- Output: per-frame probabilities, decoded by Area 1, or used to train a sequence model on this match.

**Risk.** Smart Tagging is generated from the same tracking we use, so several rules share errors. Label models assume rules err independently, so they may be overconfident here \[Medium\]. Measure :each tracking-based rule's accuracy on the degraded IDSSE matches, where the truth is known (Smart Tagging does not exist there).

## Area 7 — Video-based options (lower priority)

Video methods have the highest ceiling but the worst cost-to-gain ratio for a 9-day window, so try them only after Areas 1–5 work \[Medium\].

- **Broadcast tracking fusion (do this first).** Our EDA synced the broadcast ATD to tactical frames algorithmically. It recovers 12.1% of the missing-ball frames, raising graded ball coverage from 59.3% to 64.2%. Cheap, because the offsets are already known.
- **Ball re-detection on the tactical video.** [WASB](https://github.com/nttcom/WASB-SBDT) (Tarashima et al., BMVC 2023) is a sports ball detection and tracking baseline with public code, evaluated on soccer among five sports. Our EDA suggests limited gain: the ball rarely vanishes near the image border, and players move at half their usual speed when it is missing, so many gaps are stoppages rather than detector misses. Running it on 199,319 full-HD frames also needs GPU hours \[Low\].
- **Action spotting for restarts.** T-DEED on the broadcast video (Area 5). Useful as an independent D anchor, but replays and compute make it a stretch goal \[Low\].
- **Image crops around players.** Peral et al. (Area 4) show crops help possessor identification from tactical footage. No public code, so this is a research project, not a 9-day task \[Low\].

The terms forbid uploading the videos or data to any third-party hosted LLM. Running open models on our own hardware is fine; confirm with the organisers before using rented cloud GPUs (rules section).

## Recommended pipeline for the 9 days to the deadline

Build the validation harness before any model, because without it we cannot tell a real gain from noise; the deadline is 17 November 2026 \[High\].

1. **Validation harness (days 1–2).** Load the 7 IDSSE matches as H/A/D per frame. Degrade the tracking to look like our ATD (Area 2). Score with our copy of evaluation/possession\_metrics.py, with bootstrap intervals.
2. **Per-frame model v1 (days 2–3).** Features: ball–player geometry when the ball is visible; team-shape features on every frame; offensive Smart Tagging prior (BAR-ATH only); restart triggers. A gradient-boosted classifier for p(H, A, D), trained on degraded IDSSE plus the Metrica sample games.
3. **Decoder (days 3–4).** Penalised Viterbi or HSMM with separate team-switch and alive/dead penalties and minimum durations (Area 1). Tune on IDSSE spellF1 only.
4. **Dead-ball module (days 4–5).** Restart triggers, collective speed and out-of-bounds into the D probability; restart team from the executor (Area 5).
5. **Broadcast ball fusion (day 5).** Add the synced broadcast ball where tactical has none.
6. **Stretch (days 6–7).** BallRadar-style set model for ball gaps (Area 4), or a linked-HMM label model on BAR-ATH (Area 6). Keep a change only if IDSSE spellF1 improves beyond the bootstrap interval.
7. **Freeze and ship (days 8–9).** Use the public 5 minutes only to confirm frames, labels and file format. Run validate\_submission.py on the exact file, regenerate it from a clean clone to check it is byte-identical, and write the technical document.

**Model-selection rule:** a change that helps on the public window but not on IDSSE does not ship. The published baselines show the public window picks the wrong smoothing window \[High\].

## Rule-compliance notes and questions for the organisers

Everything recommended here fits RULES.md as written, with three points worth confirming in a GitHub issue before we build on them \[Medium\].

**Clearly allowed:**

- Labelled data from other games (IDSSE, Metrica sample, SkillCorner, PFF), in any amount, if the technical document says what was labelled, by whom and how much. IDSSE is CC-BY 4.0, so cite it.
- Algorithmic pseudo-labels on BAR-ATH, as long as no person decides any label and no rule or constant is tied to specific frames.
- Watching the video and inspecting the data to design features.

**Not allowed:** tuning anything on the public 5 minutes; hand-correcting tracking or labels; any human-produced data about BAR-ATH (event data, corrected tracking).

**Questions to ask:**

- [ ] Is it acceptable to fit the noise simulator (ball-gap and tracklet-length distributions) to BAR-ATH's raw input statistics? No labels are involved.
- [ ] The terms forbid uploading match videos or data to third-party hosted LLMs. Does that also cover running non-LLM models (ball detector, action spotter) on a rented cloud GPU?
- [ ] Is Metrica's own sample data acceptable as training data, and does its event-based possession roughly match the corrected-tracking definition behind the ground truth?

## References

Pages opened or searched on 8 November 2026; the challenge facts come from the repository's CHALLENGE, DATA, EVALUATION, RULES and FAQ files and our EDA notebook.

**Possession and events from tracking**

- Vidal-Codina, Evans, El Fakir & Billingham (2022). [Automatic event detection in football using tracking data](https://doi.org/10.1007/s12283-022-00381-6). Sports Engineering 25:18. [arXiv](https://arxiv.org/abs/2202.00804)
- Link & Hoernig (2017). [Individual ball possession in soccer](https://doi.org/10.1371/journal.pone.0179953). PLOS ONE 12(7).
- Hoernig, Link, Herrmann, Radig & Lames (2016). [Detection of individual ball possession in soccer](https://doi.org/10.1007/978-3-319-24560-7_13). IACSS 2015, AISC.
- Borghesi, Costa, Morra & Lamberti (2023). [Using Temporal Convolutional Networks to estimate ball possession in soccer games](https://doi.org/10.1016/j.eswa.2023.119780). Expert Systems with Applications 223. [Open copy](https://iris.polito.it/handle/11583/2976470)
- Khaustov & Mozgovoy (2020). [Recognizing events in spatiotemporal soccer data](https://doi.org/10.3390/app10228046). Applied Sciences 10(22).
- Bischofberger, Baca & Schikuta (2024). [Event detection in football: Improving the reliability of match analysis](https://doi.org/10.1371/journal.pone.0298107). PLOS ONE 19(4).

**Ball and possessor inference without the ball**

- Kim, Choi, Kim, Yoon & Ko (2023). [Ball trajectory inference from multi-agent sports contexts using Set Transformer and hierarchical Bi-LSTM](https://arxiv.org/abs/2306.08206). KDD 2023. [Code](https://github.com/pientist/ballradar)
- Capellera, Ferraz, Rubio, Agudo & Moreno-Noguer (2024). [TranSPORTmer: a holistic approach to trajectory understanding in multi-agent sports](https://arxiv.org/abs/2410.17785). ACCV 2024.
- Peral, Capellera, Ferraz, Rubio & Agudo (2025). [Multi-modal soccer scene analysis with masked pre-training](https://arxiv.org/abs/2512.19528). arXiv preprint.
- Wei, Sha, Lucey, Carr, Sridharan & Matthews (2015). [Predicting ball ownership in basketball from a monocular view using only player trajectories](https://openaccess.thecvf.com/content_iccv_2015_workshops/w21/html/Wei_Predicting_Ball_Ownership_ICCV_2015_paper.html). ICCV Workshops.
- Bialkowski, Lucey, Carr, Yue, Sridharan & Matthews (2014). [Large-scale analysis of soccer matches using spatiotemporal tracking data](https://doi.org/10.1109/ICDM.2014.133). ICDM 2014.

**Temporal segmentation and decoding**

- Yu (2010). [Hidden semi-Markov models](https://doi.org/10.1016/j.artint.2009.11.011). Artificial Intelligence 174.
- Truong, Oudre & Vayatis (2020). [Selective review of offline change point detection methods](https://arxiv.org/abs/1801.00718). Signal Processing 167 (ruptures package).
- Farha & Gall (2019). [MS-TCN: multi-stage temporal convolutional network for action segmentation](https://arxiv.org/abs/1903.01945). CVPR 2019.
- Ishikawa, Kasai, Aoki & Kataoka (2021). [Alleviating over-segmentation errors by detecting action boundaries](https://arxiv.org/abs/2007.06866). WACV 2021.
- Safranchik, Luo & Bach (2020). [Weakly supervised sequence tagging from noisy rules](https://doi.org/10.1609/aaai.v34i04.6009). AAAI 2020.

**Video**

- Deliège et al. (2021). [SoccerNet-v2](https://arxiv.org/abs/2011.13367). CVPR Workshops.
- Xarles, Escalera, Moeslund & Clapés (2024). [T-DEED](https://github.com/arturxe2/T-DEED). CVsports 2024.
- Tarashima, Haq, Wang & Tagawa (2023). [WASB: widely applicable strong baseline for sports ball detection and tracking](https://github.com/nttcom/WASB-SBDT). BMVC 2023.

**Datasets and tools**

- Bassek, Rein, Weber & Memmert (2025). [An integrated dataset of spatiotemporal and event data in elite soccer](https://doi.org/10.1038/s41597-025-04505-y). Scientific Data 12:195. [Hugging Face mirror](https://huggingface.co/datasets/pysport/idsse-data)
- [Metrica Sports sample data](https://github.com/metrica-sports/sample-data)
- [SkillCorner open data](https://github.com/SkillCorner/opendata)
- [PFF FC World Cup 2022 dataset](https://www.blog.fc.pff.com/blog/pff-fc-release-2022-world-cup-data)
- [databallpy: team and player possession](https://databallpy.readthedocs.io/en/latest/features/team_and_player_possession.html)
