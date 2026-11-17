# Pinned Ryu references

Source commit `5eaf3e595e91fa0502aac48037d0cae8c356ec6f`, branch Ryu in this repository:

- `features.py`: `experiments/features.py`.
- `train_model.py`, `decoder.py`: `experiments/015_final_pipeline/`.
- `skillcorner_games.py`, `common.py`: `experiments/`.

Files in this directory are unchanged source snapshots. Campaign modules are separate adaptations. Original source: https://github.com/Ryu2804/more-than-a-hack-2027/tree/5eaf3e595e91fa0502aac48037d0cae8c356ec6f . Repository license applies to copied code.

Shipped `weights_nometrica` are retained only in ignored local cache. They were trained on all7 IDSSE and20 SkillCorner matches and cannot serve as held-out predictions on those datasets. Source notes report public spellF1 about0.747 for the full-mix player-set variant; the exact no-Metrica checkpoint and cached-input adapter must be evaluated separately, not assumed identical.

SkillCorner open data: https://github.com/SkillCorner/opendata , MIT license, credit SkillCorner. Provider phases-of-play supply external proxy labels; raw tracking and frame labels are not committed.

IDSSE: Bassek et al., Scientific Data12,195 (2025), https://doi.org/10.1038/s41597-025-04505-y ; data DOI https://doi.org/10.6084/m9.figshare.28196177 ; CC BY4.0. Prepared data and degrader provenance inherited unchanged from wave3.

Metrica sample data has no explicit license in the audited source. This campaign does not use Metrica in new-model training. Existing local Metrica-derived inputs serve cross-provider diagnostics and the previously trained Farras anchor only; that inherited anchor dependency is disclosed rather than called license-clean.
