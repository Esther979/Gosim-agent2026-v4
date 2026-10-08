"""Public-only features and portable linear inference for offline supervised models.

No training data, target/card IDs, absolute dates, truth readers or dependencies
are present here. Labels and fitting live outside the submitted Python project.
"""
import math
from statistics import median

KINDS = ("overcast", "haze", "cold_snap", "rain", "storm")
FEATURES = (
    "scale", "context_proxy", "context_freshness", "scale_freshness",
    "local_zero", "any_zero", "duration", "altitude", "altitude_squared",
    "scale_duration", "zero_duration",
) + tuple(scope + "_" + kind for scope in ("all", "sector", "forecast") for kind in KINDS) + tuple(
    "scale_" + scope + "_" + kind for scope in ("all", "sector") for kind in KINDS)
FUTURE_FEATURES = FEATURES + ("lead_hours", "retention", "retained_scale",
    "retained_context", "retained_zero") + tuple(
    "retained_" + scope + "_" + kind for scope in ("all", "sector") for kind in KINDS)


def features(estimator, keys, future, alt, az, duration):
    # Import here avoids a module cycle; event applicability is shared with the
    # rule estimator so training and deployment use exactly the same geometry.
    from .geometry import wrap180
    state, now = estimator.state, estimator.now
    scale = min(2.5, max(0.0, state.scale))
    # These histories and their ages are constant within a decision. Reuse the
    # exact values across targets, exposure lengths and hypothetical branches.
    # update() invalidates the cache on every new public observation/message.
    cache = estimator._feature_cache
    context_key = ("context", keys, scale)
    if context_key not in cache:
        history = estimator.context_samples.get(keys, [])
        recent = [(t, v) for t, v in history if (now-t).total_seconds() <= 7200]
        context = median([v for _, v in recent]) if recent else scale
        freshness = math.exp(-(now-recent[-1][0]).total_seconds()/1800) if recent else 0.0
        cache[context_key] = context, freshness
    context, freshness = cache[context_key]
    if "ages" not in cache:
        last = estimator.last_sample_time
        global_freshness = math.exp(-(now-last).total_seconds()/1800) if last else 0.0
        zeros = [(math.exp(-(now-t).total_seconds()/900), za, al) for t, za, al in estimator.zeros]
        cache["ages"] = global_freshness, zeros, max((z for z, _, _ in zeros), default=0.0)
    global_freshness, zeros, any_zero = cache["ages"]
    local_zero = max((z for z, za, al in zeros
                      if abs(wrap180(az-za)) <= 30 and abs(alt-al) <= 20), default=0.0)
    horizon = min(1.0, max(0.0, duration/3600))
    all_events = {k for k, direction in keys if direction == "ALL"}
    sector_events = {k for k, direction in keys if direction != "ALL"}
    forecast_events = {k for k, _ in future}
    events = [float(k in group) for group in (all_events, sector_events) for k in KINDS]
    values = [scale, context, freshness, scale*global_freshness, local_zero, any_zero,
              horizon, alt/90, (alt/90)**2, scale*horizon, local_zero*horizon]
    values.extend(events)
    values.extend(float(k in forecast_events)*horizon for k in KINDS)
    values.extend(scale*x for x in events)
    return values


def future_features(estimator, keys, future, alt, az, duration, lead):
    values = features(estimator, keys, future, alt, az, duration)
    retention = math.exp(-max(0., lead)/1800)
    return values + [lead/3600, retention, values[0]*retention, values[1]*retention,
                     values[4]*retention] + [values[j]*retention for j in range(11, 21)]


class ReconstructionModel:
    def __init__(self, record):
        if record.get("features") not in (list(FEATURES), list(FUTURE_FEATURES)):
            raise ValueError("incompatible reconstruction feature schema")
        self.center, self.scale = record["center"], record["scale"]
        self.coefficients = record["coefficients"]
        self.rmse = float(record["training_quality_rmse"])
        size = len(record["features"])
        if len(self.center) != size or len(self.scale) != size:
            raise ValueError("invalid reconstruction normalization")
        if any(not math.isfinite(x) for x in self.center+self.scale) or any(x <= 0 for x in self.scale):
            raise ValueError("non-finite reconstruction normalization")
        for name in ("quality", "band", "usable"):
            if len(self.coefficients[name]) != size+1 or any(not math.isfinite(x) for x in self.coefficients[name]):
                raise ValueError("invalid reconstruction coefficients: " + name)

    def predict(self, values):
        if len(values) != len(self.center):
            raise ValueError("reconstruction input feature count mismatch")
        vector = [1.0] + [(v-c)/s for v, c, s in zip(values, self.center, self.scale)]
        return {name: sum(w*x for w, x in zip(weights, vector))
                for name, weights in self.coefficients.items()}
