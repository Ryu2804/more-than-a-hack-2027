import numpy as np

from support import base


IOU_THRESHOLDS = (0.5, 0.75, 0.9)
DURATION_BUCKETS = ('lt_4s', '4_to_lt_15s', 'gte_15s')


def _known_runs(gt_codes, blocks25):
    runs = []
    for start, stop in np.asarray(blocks25, dtype=np.int64):
        start, stop = int(start), int(stop)
        runs.extend((start + first, start + last)
                    for first, last in base.pm.valid_runs(gt_codes[start:stop]))
    return runs


def _duration_bucket(spell):
    duration = spell[1] - spell[0]
    if duration < 4 * base.pm.FPS:
        return 'lt_4s'
    if duration < 15 * base.pm.FPS:
        return '4_to_lt_15s'
    return 'gte_15s'


def _unambiguous_matches(overlaps, n_gt, threshold):
    gt_candidates = [set() for _ in range(n_gt)]
    pred_candidates = {}
    matched_gt = set()
    matched_pred = set()
    pairs = []

    for iou, gt_index, pred_index in overlaps:
        if iou < threshold:
            break
        gt_candidates[gt_index].add(pred_index)
        pred_candidates.setdefault(pred_index, set()).add(gt_index)
        if gt_index in matched_gt or pred_index in matched_pred:
            continue
        matched_gt.add(gt_index)
        matched_pred.add(pred_index)
        pairs.append((gt_index, pred_index))

    return [(gt_index, pred_index) for gt_index, pred_index in pairs
            if len(gt_candidates[gt_index]) == 1
            and len(pred_candidates[pred_index]) == 1]


def _boundary_errors(gt_spells, pred_spells, pairs):
    start_errors = [pred_spells[pred_index][0] - gt_spells[gt_index][0]
                    for gt_index, pred_index in pairs]
    stop_errors = [pred_spells[pred_index][1] - gt_spells[gt_index][1]
                   for gt_index, pred_index in pairs]
    count = len(pairs)

    def mean_or_none(values):
        return float(np.mean(values)) if values else None

    return {
        'unambiguous_match_count': count,
        'mean_signed_start_error_frames': mean_or_none(start_errors),
        'mean_signed_end_error_frames': mean_or_none(stop_errors),
        'mean_abs_start_error_frames': mean_or_none([abs(value) for value in start_errors]),
        'mean_abs_end_error_frames': mean_or_none([abs(value) for value in stop_errors]),
        'mean_abs_start_error_seconds': mean_or_none(
            [abs(value) / base.pm.FPS for value in start_errors]),
        'mean_abs_end_error_seconds': mean_or_none(
            [abs(value) / base.pm.FPS for value in stop_errors]),
    }


def summarize(codes, data):
    """Summarize duration-stratified GT spell recall using official scorer conventions."""
    gt_raw = np.asarray(data['y25'])
    gt_codes = np.where(gt_raw < 0, base.pm.X, gt_raw).astype(np.int8, copy=False)
    prediction = np.asarray(base.expand(codes, data), dtype=np.int8)
    if prediction.shape != gt_codes.shape:
        raise ValueError('expanded prediction and y25 must have identical shapes')

    runs = _known_runs(gt_codes, data['blocks25'])
    filled_prediction = base.pm.fill_abstentions(prediction, runs)
    gt_spells = base.pm.possession_spells(gt_codes, runs)
    pred_spells = base.pm.possession_spells(filled_prediction, runs)
    overlaps = base.pm.spell_overlaps(gt_spells, pred_spells)
    gt_bucket_names = [_duration_bucket(spell) for spell in gt_spells]
    bucket_indices = {
        name: [index for index, bucket in enumerate(gt_bucket_names) if bucket == name]
        for name in DURATION_BUCKETS
    }
    duration_buckets = {
        name: {'gt_count': len(indices), 'at_iou': {}}
        for name, indices in bucket_indices.items()
    }

    for threshold in IOU_THRESHOLDS:
        threshold_key = str(threshold)
        matched = base.pm.matched_gt_flags(overlaps, len(gt_spells), threshold)
        pairs = _unambiguous_matches(overlaps, len(gt_spells), threshold)
        pairs_by_bucket = {name: [] for name in DURATION_BUCKETS}
        for gt_index, pred_index in pairs:
            pairs_by_bucket[gt_bucket_names[gt_index]].append((gt_index, pred_index))
        for name, indices in bucket_indices.items():
            matched_count = sum(bool(matched[index]) for index in indices)
            duration_buckets[name]['at_iou'][threshold_key] = {
                'matched_count': int(matched_count),
                'matched_recall': (float(matched_count / len(indices)) if indices else None),
                'boundary_timing_error': _boundary_errors(
                    gt_spells, pred_spells, pairs_by_bucket[name]),
            }

    return {
        'fps': int(base.pm.FPS),
        'n_gt_spells': len(gt_spells),
        'n_pred_spells': len(pred_spells),
        'duration_buckets': duration_buckets,
    }
