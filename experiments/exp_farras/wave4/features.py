import numpy as np

L, W, FPS = 105.3, 68.0, 5
PLAYER_FEATS = [
    "depth a", "depth b", "len a", "len b", "wid a", "wid b",
    "compact a", "compact b", "speed a", "speed b", "d depth", "d len",
    "d compact", "d speed", "min dist corner", "min dist centre",
    "n beyond touchline", "n own goal area a", "n own goal area b",
    "frac own half a", "frac own half b", "speed all", "frac moving all",
]
BALL_FEATS = [
    "control c", "ball obs", "log ball speed", "ball inside dist",
    "log nearest dist",
]
PP_FEATS = [
    "same team", "attacking x", "y", "attacking vx", "vy", "speed",
    "ball distance", "ball visible", "closing speed",
    "nearest opponent distance",
]
PP_F10 = 10
PP_K48 = 48
CONTROL_FEATS = [
    "home distance", "away distance", "team margin", "ball visible",
    "ball boundary distance", "gap age", "gap remaining",
]

_PLAYER_CHUNK = 512
_PP_CHUNK = 128
_ROLLING_WIDTH = 10
_ROLLING_MIN_PERIODS = 2


def _blocks(data, n):
    raw = data.get("blocks")
    if raw is None:
        return [(0, n)] if n else []
    bounds = np.asarray(raw)
    if bounds.size == 0:
        if n:
            raise ValueError("blocks do not cover player samples")
        return []
    if bounds.ndim != 2 or bounds.shape[1] != 2:
        raise ValueError("blocks must contain start/end pairs")
    blocks = [(int(a), int(b)) for a, b in bounds]
    cursor = 0
    for start, end in blocks:
        if start != cursor or end <= start or end > n:
            raise ValueError("blocks must be ordered, contiguous, and nonempty")
        cursor = end
    if cursor != n:
        raise ValueError("blocks do not cover player samples")
    return blocks


def _segments(data, blocks, n):
    supplied = data.get("half")
    if supplied is None:
        halves = np.empty(n, dtype=np.int8)
        segments = []
        for index, (start, end) in enumerate(blocks):
            half = 1 if index == 0 else 2
            halves[start:end] = half
            segments.append((start, end, half))
        return halves, segments

    halves = np.asarray(supplied)
    if halves.shape != (n,) or not np.isin(halves, (1, 2)).all():
        raise ValueError("half must contain one 1/2 value per sample")
    halves = halves.astype(np.int8, copy=False)
    segments = []
    for start, end in blocks:
        edges = np.flatnonzero(halves[start + 1:end] != halves[start:end - 1]) + start + 1
        bounds = np.r_[start, edges, end]
        segments.extend((int(a), int(b), int(halves[a])) for a, b in zip(bounds[:-1], bounds[1:]))
    return halves, segments


def _roll_segment(values, width, method):
    values = np.asarray(values, dtype=np.float32)
    length = values.shape[0]
    out = np.full(length, np.nan, dtype=np.float32)
    if length == 0:
        return out
    left = width // 2
    right = width - left - 1
    if method == "mean":
        valid = np.isfinite(values)
        sums = np.empty(length + 1, dtype=np.float64)
        counts = np.empty(length + 1, dtype=np.int32)
        sums[0] = 0.0
        counts[0] = 0
        np.cumsum(np.where(valid, values, 0.0), dtype=np.float64, out=sums[1:])
        np.cumsum(valid, dtype=np.int32, out=counts[1:])
        indices = np.arange(length)
        starts = np.maximum(indices - left, 0)
        ends = np.minimum(indices + right + 1, length)
        totals = sums[ends] - sums[starts]
        seen = counts[ends] - counts[starts]
        usable = seen >= _ROLLING_MIN_PERIODS
        out[usable] = (totals[usable] / seen[usable]).astype(np.float32)
        return out

    for index in range(length):
        start = max(0, index - left)
        end = min(length, index + right + 1)
        window = values[start:end]
        finite = np.isfinite(window)
        if finite.sum() >= _ROLLING_MIN_PERIODS:
            out[index] = np.median(window[finite])
    return out


def _roll(values, segments, width=_ROLLING_WIDTH, method="mean"):
    values = np.asarray(values, dtype=np.float32)
    result = np.full(values.shape, np.nan, dtype=np.float32)
    for start, end, _ in segments:
        result[start:end] = _roll_segment(values[start:end], width, method)
    return result


def _infer_home_directions(x, present, teams, halves):
    base_plus_cost = 0.0
    base_minus_cost = 0.0
    goal_evidence = False
    centroid_score = 0.0

    for half in (1, 2):
        rows = halves == half
        if not rows.any():
            continue
        parity = 1 if half == 1 else -1
        home_mask = present[rows] & (teams[None, :] == 0)
        home_counts = home_mask.sum(axis=0)
        home_sums = np.where(home_mask, x[rows], 0.0).sum(axis=0, dtype=np.float64)
        home_means = np.divide(
            home_sums, home_counts, out=np.full(home_counts.shape, np.nan), where=home_counts > 0
        )
        home_observations = int(home_counts.sum())
        if home_observations:
            centroid = float(home_sums.sum() / home_observations)
            centroid_score += (L / 2.0 - centroid) * parity * home_observations

        away_mask = present[rows] & (teams[None, :] == 1)
        away_counts = away_mask.sum(axis=0)
        away_sums = np.where(away_mask, x[rows], 0.0).sum(axis=0, dtype=np.float64)
        away_means = np.divide(
            away_sums, away_counts, out=np.full(away_counts.shape, np.nan), where=away_counts > 0
        )
        home_goalkeepers = home_means[home_counts >= 200]
        away_goalkeepers = away_means[away_counts >= 200]
        if not home_goalkeepers.size or not away_goalkeepers.size:
            continue

        home_left = float(home_goalkeepers.min())
        home_right = L - float(home_goalkeepers.max())
        away_left = float(away_goalkeepers.min())
        away_right = L - float(away_goalkeepers.max())
        home_attacks_plus = home_left + away_right
        home_attacks_minus = home_right + away_left
        if parity == 1:
            base_plus_cost += home_attacks_plus
            base_minus_cost += home_attacks_minus
        else:
            base_plus_cost += home_attacks_minus
            base_minus_cost += home_attacks_plus
        goal_evidence = True

    if goal_evidence and base_plus_cost != base_minus_cost:
        base_direction = 1 if base_plus_cost < base_minus_cost else -1
    elif centroid_score != 0.0:
        base_direction = 1 if centroid_score > 0.0 else -1
    else:
        base_direction = 0

    directions = np.zeros(halves.shape, dtype=np.int8)
    for half in (1, 2):
        parity = 1 if half == 1 else -1
        directions[halves == half] = base_direction * parity
    return directions


def _player_speeds(x, y, present, segments):
    n, k = x.shape
    speed = np.full((n, k), np.nan, dtype=np.float32)
    for start, end, _ in segments:
        for first in range(start + 1, end, _PLAYER_CHUNK):
            last = min(first + _PLAYER_CHUNK, end)
            dx = (x[first:last] - x[first - 1:last - 1]) * FPS
            dy = (y[first:last] - y[first - 1:last - 1]) * FPS
            step_speed = np.hypot(dx, dy)
            valid = present[first:last] & present[first - 1:last - 1] & (step_speed <= 12.0)
            speed[first:last] = np.where(valid, step_speed, np.nan)
    return speed


def _team_stats(x, y, present, speed, teams, team):
    n, k = x.shape
    stats = {name: np.full(n, np.nan, dtype=np.float32) for name in ("mx", "my", "sx", "sy", "compact", "speed")}
    team_slots = teams == team
    for start in range(0, n, _PLAYER_CHUNK):
        end = min(start + _PLAYER_CHUNK, n)
        mask = present[start:end] & team_slots[None, :]
        count = mask.sum(axis=1)
        safe_x = np.where(mask, x[start:end], 0.0)
        safe_y = np.where(mask, y[start:end], 0.0)
        denominator = np.maximum(count, 1)
        mx = safe_x.sum(axis=1) / denominator
        my = safe_y.sum(axis=1) / denominator
        sx = np.sqrt(np.maximum((safe_x * safe_x).sum(axis=1) / denominator - mx * mx, 0.0))
        sy = np.sqrt(np.maximum((safe_y * safe_y).sum(axis=1) / denominator - my * my, 0.0))
        compact = np.hypot(
            np.where(mask, x[start:end] - mx[:, None], 0.0),
            np.where(mask, y[start:end] - my[:, None], 0.0),
        ).sum(axis=1) / denominator
        speed_mask = mask & np.isfinite(speed[start:end])
        speed_count = speed_mask.sum(axis=1)
        mean_speed = np.where(speed_mask, speed[start:end], 0.0).sum(axis=1) / np.maximum(speed_count, 1)
        enough = count >= 5
        for name, values in (("mx", mx), ("my", my), ("sx", sx), ("sy", sy), ("compact", compact)):
            stats[name][start:end] = np.where(enough, values, np.nan)
        stats["speed"][start:end] = np.where(enough & (speed_count > 0), mean_speed, np.nan)
    return stats


def _pool_stats(x, y, present, speed):
    n, k = x.shape
    corner = np.full(n, np.nan, dtype=np.float32)
    centre = np.full(n, np.nan, dtype=np.float32)
    outside = np.zeros(n, dtype=np.float32)
    speed_all = np.full(n, np.nan, dtype=np.float32)
    moving_all = np.full(n, np.nan, dtype=np.float32)
    if k == 0:
        return {"corner": corner, "centre": centre, "outside": outside,
                "speed": speed_all, "moving": moving_all}

    corners = ((0.0, 0.0), (0.0, W), (L, 0.0), (L, W))
    for start in range(0, n, _PLAYER_CHUNK):
        end = min(start + _PLAYER_CHUNK, n)
        mask = present[start:end]
        any_player = mask.any(axis=1)
        nearest_corner = np.full(end - start, np.inf, dtype=np.float32)
        for cx, cy in corners:
            distance = np.hypot(x[start:end] - cx, y[start:end] - cy)
            distance[~mask] = np.inf
            nearest_corner = np.minimum(nearest_corner, distance.min(axis=1))
        centre_distance = np.hypot(x[start:end] - L / 2.0, y[start:end] - W / 2.0)
        centre_distance[~mask] = np.inf
        nearest_centre = centre_distance.min(axis=1)
        corner[start:end] = np.where(any_player, np.clip(nearest_corner, 0.0, 40.0), np.nan)
        centre[start:end] = np.where(any_player, np.clip(nearest_centre, 0.0, 40.0), np.nan)
        outside[start:end] = (mask & ((y[start:end] < -0.3) | (y[start:end] > W + 0.3))).sum(axis=1)

        speed_mask = mask & np.isfinite(speed[start:end])
        speed_count = speed_mask.sum(axis=1)
        speed_sum = np.where(speed_mask, speed[start:end], 0.0).sum(axis=1)
        moving_sum = (speed_mask & (speed[start:end] > 1.5)).sum(axis=1)
        speed_all[start:end] = np.divide(
            speed_sum, speed_count, out=np.full(end - start, np.nan, dtype=np.float32), where=speed_count > 0
        )
        moving_all[start:end] = np.divide(
            moving_sum, speed_count, out=np.full(end - start, np.nan, dtype=np.float32), where=speed_count > 0
        )
    return {"corner": corner, "centre": centre, "outside": outside,
            "speed": speed_all, "moving": moving_all}


def _team_area_features(x, y, present, teams, directions):
    n, k = x.shape
    result = {}
    for team in (0, 1):
        own_area = np.full(n, np.nan, dtype=np.float32)
        own_half = np.full(n, np.nan, dtype=np.float32)
        slots = teams == team
        for start in range(0, n, _PLAYER_CHUNK):
            end = min(start + _PLAYER_CHUNK, n)
            direction = directions[start:end] * (1 if team == 0 else -1)
            mask = present[start:end] & slots[None, :]
            count = mask.sum(axis=1)
            own_x = np.where(
                direction[:, None] == 1,
                x[start:end] < 5.5,
                x[start:end] > L - 5.5,
            )
            goal_count = (mask & own_x & (np.abs(y[start:end] - W / 2.0) < 9.15)).sum(axis=1)
            in_own_half = np.where(
                direction[:, None] == 1,
                x[start:end] < L / 2.0,
                x[start:end] > L / 2.0,
            )
            fraction = np.divide(
                (mask & in_own_half).sum(axis=1), count,
                out=np.full(end - start, np.nan, dtype=np.float32), where=count > 0,
            )
            known = direction != 0
            own_area[start:end] = np.where(known, goal_count, np.nan)
            own_half[start:end] = np.where(known, fraction, np.nan)
        result[team] = own_area, own_half
    return result


def _pooled_player_features(x, y, present, teams, speed, directions, segments):
    n = len(x)
    stats = {}
    for team in (0, 1):
        stats[team] = {
            name: _roll(values, segments)
            for name, values in _team_stats(x, y, present, speed, teams, team).items()
        }

    pool = _pool_stats(x, y, present, speed)
    pool = {
        "corner": _roll(np.clip(pool["corner"], 0.0, 40.0), segments),
        "centre": _roll(np.clip(pool["centre"], 0.0, 40.0), segments),
        "outside": _roll(pool["outside"], segments),
        "speed": _roll(pool["speed"], segments),
        "moving": _roll(pool["moving"], segments),
    }
    areas = _team_area_features(x, y, present, teams, directions)
    areas = {
        team: (_roll(area, segments), _roll(fraction, segments))
        for team, (area, fraction) in areas.items()
    }

    views = []
    valid_views = []
    team_counts = np.stack([(present & (teams[None, :] == team)).sum(axis=1) for team in (0, 1)])
    enough_players = (team_counts[0] >= 5) & (team_counts[1] >= 5)
    for view_team in (0, 1):
        direction = directions * (1 if view_team == 0 else -1)
        known = direction != 0

        def depth(team):
            raw = stats[team]["mx"]
            return np.where(known, np.where(direction == 1, raw, L - raw), np.nan)

        depth_a = depth(view_team)
        depth_b = depth(1 - view_team)
        values = np.stack([
            depth_a,
            depth_b,
            stats[view_team]["sx"],
            stats[1 - view_team]["sx"],
            stats[view_team]["sy"],
            stats[1 - view_team]["sy"],
            stats[view_team]["compact"],
            stats[1 - view_team]["compact"],
            stats[view_team]["speed"],
            stats[1 - view_team]["speed"],
            depth_a - depth_b,
            stats[view_team]["sx"] - stats[1 - view_team]["sx"],
            stats[view_team]["compact"] - stats[1 - view_team]["compact"],
            stats[view_team]["speed"] - stats[1 - view_team]["speed"],
            pool["corner"],
            pool["centre"],
            pool["outside"],
            areas[view_team][0],
            areas[1 - view_team][0],
            areas[view_team][1],
            areas[1 - view_team][1],
            pool["speed"],
            pool["moving"],
        ], axis=1).astype(np.float32, copy=False)
        valid = np.isfinite(values).all(axis=1) & enough_players
        views.append(values)
        valid_views.append(valid)

    player = np.stack(views).astype(np.float32, copy=False)
    player = np.nan_to_num(player, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return player, np.stack(valid_views).astype(bool, copy=False)


def _ball_speeds(bx, by, ball_visible, segments):
    n = len(bx)
    speed = np.full(n, np.nan, dtype=np.float32)
    for start, end, _ in segments:
        for first in range(start + 1, end, _PLAYER_CHUNK):
            last = min(first + _PLAYER_CHUNK, end)
            step = np.hypot(bx[first:last] - bx[first - 1:last - 1],
                            by[first:last] - by[first - 1:last - 1]) * FPS
            valid = ball_visible[first:last] & ball_visible[first - 1:last - 1] & (step <= 40.0)
            speed[first:last] = np.where(valid, step, np.nan)
    return speed


def _ball_features(x, y, present, teams, bx, by, ball_visible, segments):
    n, k = x.shape
    control = np.zeros(n, dtype=np.float32)
    observed = np.zeros(n, dtype=bool)
    nearest = np.full(n, np.inf, dtype=np.float32)
    for start in range(0, n, _PLAYER_CHUNK):
        end = min(start + _PLAYER_CHUNK, n)
        if k == 0:
            continue
        distance = np.hypot(x[start:end] - bx[start:end, None], y[start:end] - by[start:end, None])
        distance[~present[start:end]] = np.inf
        distance[~ball_visible[start:end]] = np.inf
        nearest[start:end] = distance.min(axis=1)
        weight = np.exp(-distance / 3.0)
        home_weight = np.where(teams[None, :] == 0, weight, 0.0).sum(axis=1)
        away_weight = np.where(teams[None, :] == 1, weight, 0.0).sum(axis=1)
        total = home_weight + away_weight
        observed[start:end] = ball_visible[start:end] & (total >= 0.05)
        control[start:end] = np.divide(
            home_weight - away_weight, total + 0.05,
            out=np.zeros(end - start, dtype=np.float32), where=total >= 0.05,
        )

    ball_speed = _roll(_ball_speeds(bx, by, ball_visible, segments), segments,
                       width=5, method="median")
    inside = np.minimum.reduce([bx, L - bx, by, W - by])
    log_speed = np.where(observed, np.log1p(np.maximum(ball_speed, 0.0)), 0.0)
    boundary = np.where(ball_visible, np.clip(inside, -5.0, 15.0) / 15.0, 0.0)
    log_nearest = np.where(ball_visible, np.log1p(np.clip(nearest, 0.0, 30.0)), 0.0)
    channels = np.stack([
        np.where(observed, control, 0.0),
        observed.astype(np.float32),
        log_speed,
        boundary,
        log_nearest,
    ], axis=1).astype(np.float32, copy=False)
    channels = np.nan_to_num(channels, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    views = np.stack((channels, channels.copy()), axis=0)
    views[1, :, 0] *= -1.0
    return views.astype(np.float32, copy=False)


def _player_set(x, y, present, teams, bx, by, ball_visible, directions, segments):
    n, k = x.shape
    pp = np.zeros((2, n, PP_K48, PP_F10), dtype=np.float16)
    ppm = np.zeros((2, n, PP_K48), dtype=bool)
    if n == 0 or k == 0:
        return pp, ppm

    for start, end, _ in segments:
        for first in range(start, end, _PP_CHUNK):
            last = min(first + _PP_CHUNK, end)
            frame_count = last - first
            present_chunk = present[first:last]
            counts = present_chunk.sum(axis=1)
            if np.any(counts > PP_K48):
                raise ValueError("more than 48 players present in one sample")

            indices = np.zeros((frame_count, PP_K48), dtype=np.intp)
            if k <= PP_K48:
                indices[:, :k] = np.arange(k, dtype=np.intp)
                active = np.zeros((frame_count, PP_K48), dtype=bool)
                active[:, :k] = present_chunk
            else:
                indices[:, :PP_K48] = np.argsort(~present_chunk, axis=1, kind="stable")[:, :PP_K48]
                active = np.arange(PP_K48)[None, :] < counts[:, None]
                active = np.broadcast_to(active, (frame_count, PP_K48))

            px = np.take_along_axis(x[first:last], indices, axis=1)
            py = np.take_along_axis(y[first:last], indices, axis=1)
            player_team = teams[indices]
            previous_rows = np.maximum(np.arange(first, last) - 1, start)
            previous_present = np.take_along_axis(present[previous_rows], indices, axis=1)
            previous_x = np.take_along_axis(x[previous_rows], indices, axis=1)
            previous_y = np.take_along_axis(y[previous_rows], indices, axis=1)
            has_previous = (np.arange(first, last) > start)[:, None] & previous_present & active
            vx = (px - previous_x) * FPS
            vy = (py - previous_y) * FPS
            speed = np.hypot(vx, vy)
            valid_velocity = has_previous & (speed < 12.0)
            vx = np.where(valid_velocity, vx, 0.0)
            vy = np.where(valid_velocity, vy, 0.0)
            speed = np.where(valid_velocity, speed, 0.0)

            current_ball_visible = ball_visible[first:last]
            current_distance = np.hypot(px - bx[first:last, None], py - by[first:last, None])
            current_distance = np.where(active & current_ball_visible[:, None], current_distance, np.nan)
            previous_ball_visible = ball_visible[previous_rows] & (np.arange(first, last) > start)
            previous_distance = np.hypot(
                previous_x - bx[previous_rows, None], previous_y - by[previous_rows, None]
            )
            closing_valid = active & previous_present & current_ball_visible[:, None] & previous_ball_visible[:, None]
            closing = np.where(closing_valid, (previous_distance - current_distance) * FPS, 0.0)

            dx = px[:, :, None] - px[:, None, :]
            dy = py[:, :, None] - py[:, None, :]
            pair_distance = np.hypot(dx, dy)
            opponents = active[:, :, None] & active[:, None, :] & (player_team[:, :, None] != player_team[:, None, :])
            pair_distance[~opponents] = np.inf
            nearest_opponent = pair_distance.min(axis=2)
            nearest_opponent = np.where(
                np.isfinite(nearest_opponent), np.minimum(nearest_opponent, 30.0), 30.0
            )

            for view, team in enumerate((0, 1)):
                direction = directions[first:last] * (1 if team == 0 else -1)
                oriented_x = np.where(direction[:, None] == 1, px, L - px)
                oriented_x = np.where(direction[:, None] != 0, oriented_x, np.nan)
                oriented_vx = np.where(
                    direction[:, None] != 0,
                    np.where(direction[:, None] == 1, vx, -vx),
                    0.0,
                )
                distance_valid = active & current_ball_visible[:, None] & np.isfinite(current_distance)
                features = np.stack([
                    (player_team == team).astype(np.float32),
                    oriented_x / L,
                    py / W,
                    np.where(valid_velocity, oriented_vx, 0.0) / 8.0,
                    np.where(valid_velocity, vy, 0.0) / 8.0,
                    speed / 8.0,
                    np.where(distance_valid, np.minimum(current_distance, 30.0) / 30.0, 0.0),
                    distance_valid.astype(np.float32),
                    np.clip(closing, -12.0, 12.0) / 8.0,
                    np.minimum(nearest_opponent, 30.0) / 30.0,
                ], axis=-1)
                features = np.nan_to_num(features, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
                features *= active[..., None]
                pp[view, first:last] = features.astype(np.float16, copy=False)
                ppm[view, first:last] = active
    return pp, ppm


def _control_features(prepared):
    selected = np.asarray(prepared[:, (0, 1, 2, 3, 7, 14, 15)], dtype=np.float32)
    selected /= np.asarray((30.0, 30.0, 10.0, 1.0, 34.0, 60.0, 60.0), dtype=np.float32)
    mirrored = selected.copy()
    mirrored[:, 0] = selected[:, 1]
    mirrored[:, 1] = selected[:, 0]
    mirrored[:, 2] = -selected[:, 2]
    control = np.stack((selected, mirrored)).astype(np.float32, copy=False)
    return np.nan_to_num(control, copy=False, nan=0.0, posinf=0.0, neginf=0.0)


def build(data):
    players = np.asarray(data["players"])
    ball = np.asarray(data["ball"])
    teams = np.asarray(data["teams"])
    if players.ndim != 3 or players.shape[2] != 2:
        raise ValueError("players must have shape (N, K, 2)")
    n, k, _ = players.shape
    if ball.shape != (n, 2) or teams.shape != (k,):
        raise ValueError("ball and teams must match player samples and slots")
    if not np.isin(teams, (-1, 0, 1)).all():
        raise ValueError("teams must use 0, 1, or -1")
    if float(data.get("fps", FPS)) != FPS:
        raise ValueError("features require 5 fps input")
    prepared = np.asarray(data["X"])
    if prepared.shape != (n, 16):
        raise ValueError("prepared X must have shape (N, 16)")

    blocks = _blocks(data, n)
    halves, segments = _segments(data, blocks, n)
    x = players[..., 0].astype(np.float32, copy=False) * L
    y = players[..., 1].astype(np.float32, copy=False) * W
    raw_present = np.isfinite(x) | np.isfinite(y)
    present = np.isfinite(x) & np.isfinite(y)
    if np.any(raw_present & (teams[None, :] < 0)):
        raise ValueError("present player has unknown team")
    bx = ball[:, 0].astype(np.float32, copy=False) * L
    by = ball[:, 1].astype(np.float32, copy=False) * W
    ball_visible = np.isfinite(bx) & np.isfinite(by)

    directions = _infer_home_directions(x, present, teams, halves)
    speed = _player_speeds(x, y, present, segments)
    player, valid = _pooled_player_features(
        x, y, present, teams, speed, directions, segments
    )
    ball_features = _ball_features(x, y, present, teams, bx, by, ball_visible, segments)
    pp, ppm = _player_set(
        x, y, present, teams, bx, by, ball_visible, directions, segments
    )
    control = _control_features(prepared)
    return {
        "player": player,
        "ball": ball_features,
        "valid": valid,
        "pp": pp,
        "ppm": ppm,
        "control": control,
        "blocks": [(start, end) for start, end, _ in segments],
    }
