import numpy as np

import core


CONFIGS = [
    {'name': f'max{max_s}_confidence{int(confidence * 100)}',
     'max_s': max_s, 'confidence': confidence}
    for max_s in (1, 2, 3)
    for confidence in (0.60, 0.70)
]


def guard(codes, observations, prediction, max_s, confidence):
    original = np.asarray(codes)
    output = original.astype(np.int8, copy=True)
    visibility = observations['X'][:, 3]
    p_team = prediction['p_team']
    replacements = []
    for half_start, half_end in observations['blocks']:
        half = original[half_start:half_end]
        if len(half) == 0:
            continue
        boundaries = np.r_[half_start,
                           half_start + np.flatnonzero(half[1:] != half[:-1]) + 1,
                           half_end]
        for index in range(1, len(boundaries) - 2):
            start, end = boundaries[index:index + 2]
            team = original[start]
            if team not in (0, 1) or end - start > max_s * 5:
                continue
            other = 1 - team
            if (original[boundaries[index - 1]] != other
                    or original[boundaries[index + 1]] != other):
                continue
            if np.mean(visibility[start:end]) >= 0.5:
                continue
            team_probability = p_team[start:end] if team == 0 else 1 - p_team[start:end]
            if np.mean(team_probability) < confidence:
                replacements.append((start, end, other))
    for start, end, other in replacements:
        output[start:end] = other
    return output


def predict(bundle, observations, stats, config):
    def postprocess(codes, current_observations, prediction):
        return guard(codes, current_observations, prediction,
                     config['max_s'], config['confidence'])

    return core.decode(bundle['short'], observations, stats, postprocess=postprocess)
