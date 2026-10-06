"""Expected positive improvement of the actual per-target best-score ledger.

Forecasts condition on the existing quality-scale point estimate. Program bands
are marginalized, inside max(), and pointing probabilities may be supplied by
an action-specific finite posterior design. This does not enable the uncalibrated
physical factor posterior or certify the pointing quadrature error.
"""
from .program_calibration import PROGRAMS


def expected_increment(weight, best_score, factor, match_probability, multiplier,
                       mismatch, hit_probability=1.0):
    f = max(0.0, min(1.0, factor))
    p = max(0.0, min(1.0, match_probability))
    h = max(0.0, min(1.0, hit_probability))
    return h * (p * max(0.0, weight * f * multiplier - best_score)
                + (1.0 - p) * max(0.0, weight * f * mismatch - best_score))


def program_values(state, factors, models, hit_probabilities=None, *, calibration=True):
    """Return program -> {i: expected science increment}, in score units."""
    scoring = state.scoring
    hits = hit_probabilities or {}
    result = {program: {} for program in PROGRAMS}
    for i, factor in factors.items():
        model = models[i]
        if calibration:
            probabilities = state.program_calibration.probabilities(model)
        else:
            # Retain the existing legacy band-scale approximation when the
            # independent program estimator is explicitly disabled.
            band = scoring.program_band(model * state.scale / 0.95)
            probabilities = {p: float(p == band) for p in PROGRAMS}
        for program in PROGRAMS:
            result[program][i] = expected_increment(
                state.weight[i], state.best_score[i], factor,
                probabilities.get(program, 0.0),
                scoring.program_multipliers.get(program, 1.0),
                scoring.mismatch_multiplier, hits.get(i, 1.0))
    return result


def best_program(state, factors, models, hit_probabilities=None, *, calibration=True):
    rows = program_values(state, factors, models, hit_probabilities, calibration=calibration)
    totals = {program: sum(values.values()) for program, values in rows.items()}
    selected = state.force_program or max(PROGRAMS, key=totals.get)
    return selected, totals, rows[selected]
