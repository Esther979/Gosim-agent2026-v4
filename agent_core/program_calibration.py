"""Online program-band inference from public score feedback (v4.0.4).

Band quality excludes instrument efficiency and directional events. Science scale
does not, so it is deliberately NOT an input to this estimator. Score feedback
alone cannot identify all weather factors. We retain a distribution over the
effective site factor transparency * sky_quality / seeing, using capped-score
evidence, rather than pretending to recover exact weather.
"""
from __future__ import annotations

import math
from datetime import timedelta

from .geometry import Moon, local_sidereal_deg, lunar_factor, radec_to_altaz

PROGRAMS = ("DARK", "BRIGHT", "BACKUP")


def exposure_models(state, indices, start, duration, night_start):
    """Public geometry averaged in <=120s pieces, split at slot boundaries.

The scorer uses the same public geometry and piece boundaries. Site weather is
unknown; this is only lunar / (q0 * airmass**beta), not an oracle quality.
"""
    totals = {i: 0.0 for i in indices}
    elapsed = 0.0
    while elapsed < duration:
        moment = start + timedelta(seconds=elapsed)
        to_slot = state.slot_seconds - (moment - night_start).total_seconds() % state.slot_seconds
        seconds = min(120.0, to_slot, duration - elapsed)
        midpoint = moment + timedelta(seconds=seconds / 2)
        lst = local_sidereal_deg(midpoint, state.lon)
        moon = Moon(midpoint, lst, state.lat)
        for i in totals:
            alt, _ = radec_to_altaz(state.ra[i], state.dec[i], lst, state.lat)
            lunar = lunar_factor(moon, state.ra[i], state.dec[i], state.scoring.lunar_model)
            totals[i] += seconds * state.scoring.quality_model(alt, lunar)
        elapsed += seconds
    return {i: value / duration for i, value in totals.items()}


class ProgramCalibration:
    """Small grid posterior with bounded, exposure-level evidence and forgetting.

score/weight > mismatch proves a match because g <= 1. Exact mismatch is
weaker evidence: normally a saturated mismatch, but it can also be an
unsaturated match. Other unsaturated scores and zero/missing hits carry no
identifiable band evidence and are ignored. Correlated fibres never count as
independent weather measurements.
"""

    def __init__(self, scoring):
        self.scoring = scoring
        self.grid = [0.02 * i for i in range(1, 101)]
        # Broad dimensionless engineering prior, not fitted to a card's truth.
        raw = [math.exp(-0.5 * ((x - 0.7) / 0.3) ** 2) for x in self.grid]
        self.prior = [p / sum(raw) for p in raw]
        self.weights = list(self.prior)
        self.hours = None
        self.weather_context = None
        self.evidence_exposures = 0
        self.evidence_hits = 0
        self.last_evidence = 0

    def advance(self, hours, notices=None):
        if self.hours is not None:
            # Inter-night gaps reset naturally; instrument drops never drive this.
            retain = math.exp(-max(0.0, hours - self.hours) / 2.0)
            self.weights = [retain * p + (1 - retain) * q
                            for p, q in zip(self.weights, self.prior)]
        self.hours = hours
        if notices is not None:
            context = frozenset(key for key in notices if key in
                                {"rain|ALL", "storm|ALL", "overcast|ALL", "haze|ALL"})
            if self.weather_context is not None and context != self.weather_context:
                self.weights = list(self.prior)
            self.weather_context = context

    def observe(self, declared, rows, hours):
        """rows: (science_weight, received_score, exposure_geometry_model)."""
        self.advance(hours)
        multiplier = self.scoring.program_multipliers[declared]
        mismatch = self.scoring.mismatch_multiplier
        evidence = []
        for weight, score, model in rows:
            if weight <= 0 or score <= 0 or model <= 0:
                continue
            ratio = score / weight
            tolerance = max(2e-5, 1e-6 / weight)  # scorer rounds scores to six decimals
            if multiplier <= mismatch + tolerance or ratio > multiplier + tolerance:
                continue
            if ratio > mismatch + tolerance:
                evidence.append((model, True, 1.0))
            elif abs(ratio - mismatch) <= tolerance:
                evidence.append((model, False, 0.5))
        self.last_evidence = len(evidence)
        if not evidence:
            return
        self.evidence_exposures += 1
        self.evidence_hits += len(evidence)
        likelihood = []
        for scale in self.grid:
            loss = sum(strength for model, matched, strength in evidence
                       if (self.scoring.program_band(model * scale) == declared) != matched)
            likelihood.append(math.exp(-4.0 * loss / len(evidence)))
        updated = [p * l for p, l in zip(self.weights, likelihood)]
        normalizer = sum(updated)
        self.weights = [p / normalizer for p in updated]

    def probabilities(self, model):
        result = dict.fromkeys(PROGRAMS, 0.0)
        for scale, probability in zip(self.grid, self.weights):
            result[self.scoring.program_band(model * scale)] += probability
        return result

    def choose(self, rows):
        """Maximize predicted weighted science score, not raw match count.

        rows: (predicted weight*g, public exposure_geometry_model).
        """
        expected = dict.fromkeys(PROGRAMS, 0.0)
        mismatch = self.scoring.mismatch_multiplier
        for base_score, model in rows:
            probabilities = self.probabilities(model)
            for program in PROGRAMS:
                expected[program] += base_score * (
                    mismatch + probabilities[program]
                    * (self.scoring.program_multipliers[program] - mismatch))
        return max(PROGRAMS, key=expected.get), expected

    def snapshot(self):
        mean = sum(x * p for x, p in zip(self.grid, self.weights))
        return {"band_scale_mean": round(mean, 4),
                "evidence_exposures": self.evidence_exposures,
                "evidence_hits": self.evidence_hits,
                "last_evidence_hits": self.last_evidence}
