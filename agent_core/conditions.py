"""Causal, API-free observation-condition estimates from public messages and scores.

The estimator supports a portable supervised reconstruction model, or a rule
model that learns a log-space residual over weather-category priors. An
exposure supplies at most one update (its fibres are correlated). Missed fibres
are geometry evidence, zero-score *hits* are availability evidence, and saturated
scores are censored rather than treated as exact quality measurements. No card,
weather-truth, or future bulletin is read by this module. Offline supervision
and model fitting live outside the submitted python/ directory.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from statistics import median

from .geometry import parse_utc, wrap180
from .reconstruction import ReconstructionModel, features, future_features

COMPASS = {"N": 0.0, "NE": 45.0, "E": 90.0, "SE": 135.0,
           "S": 180.0, "SW": 225.0, "W": 270.0, "NW": 315.0}
WEATHER = {"overcast", "haze", "cold_snap", "rain", "storm"}
DEFAULTS = {
    "base_scale": 0.7,
    "event_factors": {"overcast": 0.65, "haze": 0.7, "cold_snap": 0.75,
                      "rain": 0.15, "storm": 0.1},
    "blend": 0.5,
    "learning_rate": 0.25,
    "forecast_strength": 0.15,
    "residual_ttl_seconds": 7200.0,
    "zero_ttl_seconds": 900.0,
    "instrument_prior": 0.95,
    "use_for_ranking": True,
    "use_for_exposure": True,
    "use_for_program": True,
    "supervised_blend": 0.8,
}


@dataclass(frozen=True)
class ConditionEstimate:
    # Quality proxy and a heuristic usability weight are kept separate. Neither
    # usable_probability nor relative_uncertainty is statistically calibrated.
    quality_scale: float
    usable_probability: float
    band_scale: float
    relative_uncertainty: float
    active_events: tuple
    forecast_events: tuple
    mean_scale: float = None

    @property
    def effective_scale(self):
        return (self.quality_scale * self.usable_probability
                if self.mean_scale is None else self.mean_scale)


def event_keys(notices, alt, az):
    """Coarse applicability, NOT a claim to know the hidden event boundary."""
    keys = set()
    for notice in notices:
        kind, direction = notice.get("event_kind"), notice.get("direction")
        if kind not in WEATHER:
            continue
        if direction == "ALL" or (direction in COMPASS and alt < 75.0
                and abs(wrap180(az - COMPASS[direction])) <= 67.5):
            keys.add((kind, direction))
    return tuple(sorted(keys))


def quality_sample(score, weight, flux, seconds, model, declared, scoring):
    """Return a proxy scale from an unsaturated hit, or None.

    Matching vs mismatching program multipliers are not identifiable from a
    score alone. Use the original agent's band-consistency approximation and
    train/predict with exactly the same convention; never call this truth.
    """
    if min(score, weight, flux, seconds, model) <= 0:
        return None
    multiplier = scoring.program_multipliers.get(declared, 1.0)
    factor_if_match = score / (weight * multiplier)
    ratio_match = factor_if_match * scoring.f0t0 / (flux * seconds * model)
    band = scoring.program_band(ratio_match * model / 0.95)
    factor = factor_if_match if band == declared else score / (weight * scoring.mismatch_multiplier)
    # A capped observation supplies a lower bound, not a point measurement.
    if factor >= 0.97:
        return None
    ratio = factor * scoring.f0t0 / (flux * seconds * model)
    return min(5.0, max(0.01, ratio))


class ConditionsEstimator:
    def __init__(self, state, parameters=None):
        self.state = state
        self.params = {**DEFAULTS, **(parameters or {})}
        self.params["event_factors"] = {**DEFAULTS["event_factors"],
                                       **self.params.get("event_factors", {})}
        for name in ("base_scale", "residual_ttl_seconds",
                     "zero_ttl_seconds", "instrument_prior"):
            if not math.isfinite(float(self.params[name])) or self.params[name] <= 0:
                raise ValueError("invalid condition parameter: " + name)
        for name in ("blend", "learning_rate", "forecast_strength", "supervised_blend"):
            if not 0 <= float(self.params[name]) <= 1:
                raise ValueError("invalid condition parameter: " + name)
        if any(not math.isfinite(float(x)) or x <= 0 for x in self.params["event_factors"].values()):
            raise ValueError("event factors must be finite and positive")
        for name in ("use_for_ranking", "use_for_exposure", "use_for_program"):
            if not isinstance(self.params[name], bool):
                raise ValueError("condition switch must be boolean: " + name)
        self.now = None
        self.notices = []
        self.forecasts = []
        self.forecast_seen = set()
        self.pending = None
        self.residual = 0.0
        self.last_sample_time = None
        self.context_samples = {}
        self.zeros = deque(maxlen=64)
        self.updates = 0
        self.zero_hits = 0
        self.predictions = 0
        self._cache = {}
        self._feature_cache = {}
        self.reconstruction = (ReconstructionModel(self.params["supervised_model"])
                               if self.params.get("supervised_model") else None)
        self.future_reconstruction = (ReconstructionModel(self.params["future_model"])
                                      if self.params.get("future_model") else None)

    def _prior(self, keys):
        # Multiple notices for the same event class may describe several sectors.
        return self.params["base_scale"] * math.prod(
            self.params["event_factors"].get(kind, 1.0) for kind in {k for k, _ in keys})

    def update(self, payload):
        now = parse_utc(payload["now_utc"])
        # Consume last_result against the context saved BEFORE this exposure.
        self._learn(payload.get("last_result"), now)
        result = payload.get("last_result") or {}
        if result.get("action") == "report" and result.get("correct"):
            self.context_samples.clear()
            self.zeros.clear()
            self.residual = 0.
            self.last_sample_time = None
        self.now = now
        bulletin = payload.get("latest_bulletin") or {}
        if bulletin and parse_utc(bulletin["issued_at_utc"]) <= now:
            self.notices = bulletin.get("notices") or []
        forecast_records = [m for m in payload.get("new_messages", []) if m.get("record_type") == "forecast"]
        if payload.get("latest_forecast"):
            forecast_records.append(payload["latest_forecast"])
        for record in forecast_records:
            issued = record.get("issued_at_utc")
            if not issued or parse_utc(issued) > now or issued in self.forecast_seen:
                continue
            self.forecast_seen.add(issued)
            self.forecasts.extend(record.get("notices") or [])
        # A record may cover multiple nights. Keep it until its last night passes.
        night = self.state.current_night(now)
        night_date = None
        if night is not None:
            from datetime import timedelta
            night_date = (night[1] - timedelta(hours=12)).date().isoformat()
        self.tonight = [n for n in self.forecasts if night_date in (n.get("nights") or [])]
        if night_date:
            self.forecasts = [n for n in self.forecasts if any(d >= night_date for d in n.get("nights", []))]
        self.zeros = deque((z for z in self.zeros if (now - z[0]).total_seconds() <
                            4 * self.params["zero_ttl_seconds"]), maxlen=64)
        self._cache.clear()
        self._feature_cache.clear()

    def _learn(self, result, now):
        pending, self.pending = self.pending, None
        if not pending or not result or result.get("action") != "observe":
            return
        if result.get("observe_index") != pending["index"]:
            return
        duration = min(pending["duration"], max(1.0, (now - pending["start"]).total_seconds()))
        grouped = {}
        residuals = []
        for hit in result.get("hits", []):
            info = pending["targets"].get(hit["target_id"])
            if info is None:
                continue
            i, prediction, keys = info
            score = float(hit.get("score", 0))
            if score <= 0:
                self.zero_hits += 1
                self.zeros.append((now, prediction.az, prediction.alt))
                continue
            scale = quality_sample(score, self.state.weight[i], self.state.flux[i],
                                   duration, prediction.model, pending["program"], self.state.scoring)
            if scale is None:
                continue
            grouped.setdefault(keys, []).append(scale)
            residuals.append(math.log(scale / self._prior(keys)))
        if residuals:
            innovation = max(-0.7, min(0.7, median(residuals) - self.residual))
            self.residual += self.params["learning_rate"] * innovation
            self.last_sample_time = now
            self.updates += 1
        for keys, scales in grouped.items():
            history = self.context_samples.setdefault(keys, deque(maxlen=12))
            history.append((now, median(scales)))

    def predict(self, alt, az, duration_seconds=900):
        keys = event_keys(self.notices, alt, az)
        future = event_keys(getattr(self, "tonight", []), alt, az)
        # Numeric horizon and rough sector are sufficient for caching per decision.
        cache_key = ((keys, future, az, alt, int(duration_seconds)) if self.reconstruction is not None
                     else (keys, future, int(az // 15), int(alt // 10), int(duration_seconds)))
        if cache_key in self._cache:
            return self._cache[cache_key]
        if self.reconstruction is not None:
            output = self.reconstruction.predict(features(self, keys, future, alt, az, duration_seconds))
            blend = self.params["supervised_blend"]
            mean = (1-blend)*self.state.scale + blend*max(0.0, min(2.5, output["quality"]))
            band = (1-blend)*self.state.scale/.95 + blend*max(0.0, min(3.0, output["band"]))
            usable = min(1.0, max(0.05, output["usable"]))
            estimate = ConditionEstimate(mean/usable, usable, band,
                self.reconstruction.rmse/max(.05, mean), keys, future, mean_scale=mean)
            self.predictions += 1
            self._cache[cache_key] = estimate
            return estimate
        age = ((self.now - self.last_sample_time).total_seconds() if self.last_sample_time else float("inf"))
        decay = math.exp(-age / self.params["residual_ttl_seconds"])
        predicted = self._prior(keys) * math.exp(self.residual * decay)
        recent = [v for t, v in self.context_samples.get(keys, [])
                  if (self.now - t).total_seconds() <= self.params["residual_ttl_seconds"]]
        if recent:
            strength = min(0.8, len(recent) / (len(recent) + 2.0))
            predicted = math.exp((1 - strength) * math.log(predicted) + strength * math.log(median(recent)))
        # Forecasts disclose a night, not continuous closure. The small horizon
        # weight is a prior, calibrated on training cards, not a claimed probability.
        unseen = {kind for kind, _ in future} - {kind for kind, _ in keys}
        forecast_weight = self.params["forecast_strength"] * min(1.0, duration_seconds / 3600.0)
        if unseen:
            forecast_factor = min(self.params["event_factors"].get(k, 1.0) for k in unseen)
            predicted *= 1 - forecast_weight * (1 - forecast_factor)
        blend = self.params["blend"]
        quality = (1 - blend) * self.state.scale + blend * predicted
        # Zero-score hits reduce local usability temporarily, never permanently.
        recent_zero = [math.exp(-(self.now - t).total_seconds() / self.params["zero_ttl_seconds"])
                       for t, za, al in self.zeros
                       if abs(wrap180(az - za)) <= 15.0 and abs(alt - al) <= 15.0]
        usable = 1 - blend * 0.65 * max(recent_zero, default=0.0)
        # Band quality excludes directional effects. Efficiency is not separately
        # identifiable; expose the explicit prior, rather than claiming true weather.
        all_keys = tuple(k for k in keys if k[1] == "ALL")
        ratio = self._prior(all_keys) / max(1e-9, self._prior(keys))
        band_scale = quality * ratio / self.params["instrument_prior"]
        estimate = ConditionEstimate(max(0.02, quality), usable, max(0.02, band_scale),
                                     0.15 + 0.35 / (1 + len(recent)) + 0.2 * bool(unseen), keys, future)
        self.predictions += 1
        self._cache[cache_key] = estimate
        return estimate

    def feature_vector(self, alt, az, duration):
        return features(self, event_keys(self.notices, alt, az),
                        event_keys(getattr(self, "tonight", []), alt, az), alt, az, duration)

    def future_feature_vector(self, alt, az, duration, lead):
        return future_features(self, event_keys(self.notices, alt, az),
            event_keys(getattr(self, "tonight", []), alt, az), alt, az, duration, lead)

    def predict_future(self, alt, az, duration, lead):
        """Exposure beginning `lead` seconds from now (same night, at most 2 h).

        Conditions and history remain those known now; no hypothetical feedback
        is inserted. The offline future model learns persistence and transitions.
        """
        if lead <= 0 or self.future_reconstruction is None:
            return self.predict(alt, az, duration)
        lead = min(7200., lead)
        key = ("future", alt, az, duration, lead)
        if key in self._cache:
            return self._cache[key]
        output = self.future_reconstruction.predict(self.future_feature_vector(alt, az, duration, lead))
        blend = self.params["supervised_blend"]
        mean = (1-blend)*self.state.scale + blend*max(0., min(2.5, output["quality"]))
        band = (1-blend)*self.state.scale/.95 + blend*max(0., min(3., output["band"]))
        usable = max(.05, min(1., output["usable"]))
        estimate = ConditionEstimate(mean/usable, usable, band,
            self.future_reconstruction.rmse/max(.05, mean), event_keys(self.notices, alt, az),
            event_keys(getattr(self, "tonight", []), alt, az), mean_scale=mean)
        self._cache[key] = estimate
        return estimate

    def remember(self, now, index, targets, program, duration):
        self.pending = {"start": now, "index": index, "program": program,
                        "duration": duration, "targets": {
                            target: (self.state.index_of[target], prediction,
                                     event_keys(self.notices, prediction.alt, prediction.az))
                            for target, prediction in targets.items()}}

    def summary(self):
        return {"quality_updates": self.updates, "zero_hits": self.zero_hits,
                "predictions": self.predictions, "residual": round(self.residual, 6)}
