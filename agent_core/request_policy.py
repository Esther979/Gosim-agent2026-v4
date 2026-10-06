"""Bounded request pursuit using public windows and planning estimates.

Feasibility is optimistic, not a proof about hidden future weather. The final
credit is an engineering progress incentive capped once per request, not a
calibrated probability of the whole request succeeding.
"""
from datetime import timedelta

from .geometry import parse_utc, local_sidereal_deg, radec_to_altaz, wrap180, SIDEREAL_DEG_PER_SECOND


class RequestPolicy:
    def __init__(self, state):
        self.state = state
        self.active = ()
        self.status = {}
        self._possible_cache = {}

    def possible(self, i, now, deadline, threshold):
        bucket = int((now - self.state.survey_start).total_seconds() // 1440)
        key = (i, bucket, deadline, threshold, round(self.state.scale, 1))
        if key not in self._possible_cache:
            if len(self._possible_cache) > 2000:
                self._possible_cache.clear()
            self._possible_cache[key] = self._possible(i, now, deadline, threshold)
        return self._possible_cache[key]

    def _possible(self, i, now, deadline, threshold):
        state = self.state
        h = state.hmax[i]
        if h <= 0:
            return False
        # Optimistic scale; used only to avoid pursuing geometrically/depth-wise
        # implausible bundles, never described as knowledge of future weather.
        scale = max(1.0, state.scale)
        for start, end in state.nights:
            left, right = max(start, now), min(end, deadline, state.survey_end)
            if (right - left).total_seconds() < state.min_exposure:
                continue
            lst = local_sidereal_deg(left, state.lon)
            ha = wrap180(lst - state.ra[i])
            # Try the window start, a rise-boundary interior and the transit.
            offsets = {0.0, max(0.0, (-h + 0.1 - ha) / SIDEREAL_DEG_PER_SECOND),
                       max(0.0, -ha / SIDEREAL_DEG_PER_SECOND)}
            for seconds in offsets:
                moment = left + timedelta(seconds=seconds)
                if moment >= right:
                    continue
                here = local_sidereal_deg(moment, state.lon)
                angle = wrap180(here - state.ra[i])
                if not -h <= angle <= h:
                    continue
                up = (h - angle) / SIDEREAL_DEG_PER_SECOND if h < 180 else 1e9
                duration = min(state.max_exposure, (right - moment).total_seconds(), up)
                if duration < state.min_exposure:
                    continue
                alt, _az = radec_to_altaz(state.ra[i], state.dec[i], here, state.lat)
                # Best public geometry over this short future opportunity;
                # removing lunar degradation makes this an optimistic screen.
                end_alt, _ = radec_to_altaz(state.ra[i], state.dec[i],
                    local_sidereal_deg(moment + timedelta(seconds=duration), state.lon), state.lat)
                best_alt = max(alt, end_alt)
                if angle <= 0 <= angle + duration * SIDEREAL_DEG_PER_SECOND:
                    best_alt = 90.0 - abs(state.dec[i] - state.lat)
                factor = state.flux[i] * duration * state.scoring.quality_model(best_alt, 1.0) * scale / state.scoring.f0t0
                if factor >= threshold:
                    return True
        return False

    def update(self, requests, now, enabled=True, pursue=True):
        chosen = []
        for source in requests:
            try:
                req = dict(source)
                start = parse_utc(req['issued_at_utc'])
                deadline = parse_utc(req['deadline_utc'])
                done = set(map(str, req.get('completed_target_ids') or []))
                deficit = int(req.get('minimum_completed', 0)) - int(req.get('completed_count', len(done)))
                threshold = float(req['completion_factor_threshold'])
            except (KeyError, TypeError, ValueError):
                continue
            if not start <= now < deadline or deficit <= 0 or not pursue:
                continue
            missing = [state_id for state_id in map(str, req.get('target_ids') or []) if state_id not in done]
            possible = [tid for tid in missing if tid in self.state.index_of
                        and (not enabled or self.possible(self.state.index_of[tid], now, deadline, threshold))]
            rid = str(req.get('request_id', ''))
            self.status[rid] = dict(deficit=deficit, possible=len(possible), pursued=len(possible) >= deficit)
            if len(possible) >= deficit:
                req.update(target_ids=possible, completed_target_ids=[], completed_count=0,
                           minimum_completed=deficit, remaining_count=deficit)
                chosen.append(req)
        self.active = tuple(chosen)
        return self.active

    def credit(self, factors, hits, now, duration, urgency, margin=1.25):
        credit = 0.0
        state = self.state
        end = now + timedelta(seconds=duration)
        for req in self.active:
            if end > parse_utc(req['deadline_utc']):
                continue
            reward = max(0.0, float(req.get('completion_reward', 0.0)))
            deficit = max(1, int(req['remaining_count']))
            threshold = min(1.0, float(req['completion_factor_threshold']) * margin)
            progress = sum(hits.get(state.index_of[tid], 1.0) for tid in req['target_ids']
                           if tid in state.index_of and factors.get(state.index_of[tid], 0.0) >= threshold)
            credit += min(reward, max(0.0, urgency) * reward * progress / deficit)
        return credit
