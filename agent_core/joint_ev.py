"""Teammate joint loop with science-gain and horizon-visibility integration.

Anchor count, fibre projection, durations, overhead, defer weighting and required
bonus are retained. Terrain height uncertainty is integrated in science gain.
"""
import math
from .spatial_search import FieldSearch
from .base_planner import (JOINT_DURATIONS, JOINT_OVERHEAD_SECONDS, PLAN_FACTOR_SAFETY,
    DEFER_MIN_NIGHTS, DEFER_POWER, REQUIRED_BONUS, ANCHOR_POOL, JOINT_ANCHORS)
from .geometry import (lunar_factor, wrap180, SIDEREAL_DEG_PER_SECOND, shift_altaz,
    altaz_to_radec, max_hour_angle_deg)

def joint_search_ev(self, candidates, altaz, visible, moon, lst, seconds_left, fibers, night_index=0):
    """Pick (pointing, fibre assignment, exposure length) together.

    For every target near an anchor, its gain is tabulated for each exposure length:
    expected positive improvement of public best_score, plus the original required
    bonus once the target crosses its completion threshold. Other urgency and
    defer factors retain the teammate's engineering priorities. For
    a candidate pointing each fibre then takes, per length, the best target on its
    glass, and the pointing/length pair with the highest total gain per second wins.
    Returns (alt, az, {fibre: (gain, target, margin)}, seconds) or None when nothing has
    a positive rate (the caller then uses the older search, which also knows how to
    take a probing exposure)."""
    state, scoring, grid = self.state, self.state.scoring, self.grid
    band_scale = self._band_scale(night_index)
    durations = sorted({int(max(state.min_exposure, min(state.max_exposure,
                                                        round(base * state.duration_scale / 30.0) * 30)))
                        for base in JOINT_DURATIONS})
    durations = [d for d in durations if d <= seconds_left]
    if not durations:
        return None
    count = len(durations)
    per_second = [1.0 / (d + JOINT_OVERHEAD_SECONDS) for d in durations]
    gains_cache: dict[int, list] = {}

    def gains(j: int) -> list:
        cached = gains_cache.get(j)
        if cached is not None:
            return cached
        alt, az = altaz(j)
        damp = (0.6 ** state.misses[j]) * (0.7 ** state.attempts[j]) * self._environment_factor(alt, az, posterior=False)
        model = scoring.quality_model(alt, lunar_factor(moon, state.ra[j], state.dec[j], scoring.lunar_model)) or 0.0
        k = (state.flux[j] * model * state.scale * PLAN_FACTOR_SAFETY) / scoring.f0t0
        ha = wrap180(lst - state.ra[j])
        up = (state.hmax[j] - ha) / SIDEREAL_DEG_PER_SECOND if state.hmax[j] < 180 else 1e9
        f = state.factor[j]
        weight = state.weight[j]
        needs = state.required[j] and f < 0.5
        if state.last_night[j] - night_index + 1 >= DEFER_MIN_NIGHTS:
            best_alt = min(89.0, 90.0 - abs(state.dec[j] - state.lat))
            best_model = scoring.quality_model(best_alt, 1.0) or 0.0
            here = scoring.quality_model(alt, 1.0) or 0.0
            if best_model > 0.0:
                damp *= min(1.0, here / best_model) ** DEFER_POWER
        ceiling = max(1e-6, min(1.0, k * min(state.max_exposure, up)))
        row = []
        for d in durations:
            if d > up or damp <= 0.0 or k <= 0.0:
                row.append(0.0)
                continue
            reach = min(1.0, k * d)
            # Only the science term is replaced; all urgency and required terms remain.
            value, completion_probability = self._exposure_gain(j, d, k, model, band_scale)
            if needs:
                value += REQUIRED_BONUS * completion_probability
            row.append(value * damp)
        gains_cache[j] = row
        return row

    # anchors: the targets that pay best per second on their own, urgency included
    seeds = []
    for checked, (_priority, i) in enumerate(candidates):
        if checked >= ANCHOR_POOL:
            break
        row = gains(i)
        rate = max(row[ti] * per_second[ti] for ti in range(count))
        if rate > 0.0:
            # urgency only for targets running out of nights; no bonus for setting ones
            nights_left = max(1, state.last_night[i] - night_index + 1)
            seeds.append((rate * (1.0 + 2.0 / nights_left), i))
    if not seeds:
        return None
    seeds.sort(key=lambda s: -s[0])
    n_anchors = 1 if state.fast_level >= 1 else JOINT_ANCHORS

    side, pitch = grid.side, grid.pitch
    half_fov, half_glass, half_side, middle = grid.fov / 2.0, grid.glass / 2.0, grid.side / 2.0, (grid.side - 1) / 2.0
    to_deg = 180.0 / math.pi
    best = None  # (rate, c_alt, c_az, duration index, cells, margins)
    spatial = FieldSearch(state, grid, altaz, visible, fibers, self.neighbour_radius) \
        if getattr(state, 'neighbour_search_enabled', False) else None
    for _, anchor in seeds[:n_anchors]:
        a_alt, a_az = altaz(anchor)
        if spatial is not None:
            near_rows = spatial.near_rows(anchor, self.edge_margin)
        else:
            near = [j for j in state.neighbours(state.ra[anchor], state.dec[anchor], self.neighbour_radius) if j in visible]
            near_rows = []
            for j in near:
                row = gains(j)
                if max(row) > 0.0:
                    alt, az = altaz(j)
                    alt, az = math.radians(alt), math.radians(az)
                    ca = math.cos(alt)
                    near_rows.append((j, row, ca * math.cos(az), ca * math.sin(az), math.sin(alt),
                                      self.edge_margin * (1 + 1.5 * state.misses[j])))
        for fiber in fibers:
            d_north, d_east = grid.fiber_center(fiber)
            c_alt, c_az = shift_altaz(a_alt, a_az, -d_north, -d_east)
            if not (state.min_alt + 1.5 <= c_alt <= 89.0):
                continue
            c_alt = round(c_alt, 4)
            c_az = round(c_az, 4) % 360.0
            c_ra, c_dec = altaz_to_radec(c_alt, c_az, lst, state.lat)
            c_hmax = max_hour_angle_deg(c_dec, state.lat, state.min_alt + 0.3)
            centre_up = (c_hmax - wrap180(lst - c_ra)) / SIDEREAL_DEG_PER_SECOND if c_hmax < 180 else 1e9
            cells: dict[int, tuple[list, list]] = {}   # fibre -> (best gain per length, target per length)
            margins: dict[int, float] = {}
            # same gnomonic projection and fibre lookup as geometry.tangent_offsets /
            # FiberGrid.classify, with the centre's basis vectors computed once
            calt, caz = math.radians(c_alt), math.radians(c_az)
            s_alt, c_cos, s_az, c_az_cos = math.sin(calt), math.cos(calt), math.sin(caz), math.cos(caz)
            cx, cy, cz = c_cos * c_az_cos, c_cos * s_az, s_alt
            nx, ny, nz = -s_alt * c_az_cos, -s_alt * s_az, c_cos
            ex, ey = -s_az, c_az_cos
            for j, row, tx, ty, tz, keep_clear in near_rows:
                depth = tx * cx + ty * cy + tz * cz
                if depth <= 0.0 or (spatial is not None and depth < spatial.cos_radius):
                    continue
                d_n = to_deg * (tx * nx + ty * ny + tz * nz) / depth
                d_e = to_deg * (tx * ex + ty * ey) / depth
                if abs(d_n) > half_fov or abs(d_e) > half_fov:
                    continue
                r_i = min(max(int(math.floor(d_n / pitch + half_side)), 0), side - 1)
                c_i = min(max(int(math.floor(d_e / pitch + half_side)), 0), side - 1)
                margin = half_glass - max(abs(d_n - (r_i - middle) * pitch), abs(d_e - (c_i - middle) * pitch))
                if margin < 0.0:
                    continue
                if row is None:
                    row = gains(j)
                    if max(row) <= 0.0:
                        continue
                fib = r_i * side + c_i
                edge = 1.0 if margin >= keep_clear else 0.4
                cell = cells.get(fib)
                if cell is None:
                    cell = cells[fib] = ([0.0] * count, [-1] * count)
                values, owners = cell
                for ti in range(count):
                    v = row[ti] * edge
                    if v > values[ti]:
                        values[ti] = v
                        owners[ti] = j
                margins[j] = margin
            if not cells:
                continue
            for ti in range(count):
                if durations[ti] > centre_up:
                    break
                total = 0.0
                for values, _owners in cells.values():
                    total += values[ti]
                rate = total * per_second[ti]
                if best is None or rate > best[0]:
                    best = (rate, c_alt, c_az, ti, cells, margins)
    if best is None or best[0] <= 0.0:
        return None
    _, c_alt, c_az, ti, cells, margins = best
    chosen = {fib: (values[ti], owners[ti], margins[owners[ti]])
              for fib, (values, owners) in cells.items() if owners[ti] >= 0}
    return (c_alt, c_az, chosen, durations[ti]) if chosen else None
