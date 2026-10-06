"""Survey state: the target catalogue, learned sky-quality scale, and per-target
progress. Built once from `initialize`, then updated from every `decision_request`'s
messages and `last_result`. Holds no hidden data -- only what the public protocol
hands us, plus what we infer from our own hits (never from a file).

Mirrors the structure-of-parallel-arrays design (ids/ra/dec/flux/weight/required/...,
all indexed by the same integer i) used by this project's companion TypeScript example,
so both examples solve the same problem the same way and can be compared directly.
"""
from __future__ import annotations

import bisect
import math
from collections import deque
from typing import NamedTuple, Optional

from .geometry import FiberGrid, max_hour_angle_deg, parse_utc, radec_to_altaz, wrap180
from .scoring import ScoringModel

ALT_MARGIN_DEG = 0.6
SKY_MEMORY_HOURS = 2.0
SIDEREAL_DEG_PER_SECOND = 360.98564736629 / 86400.0

# Terrain: the opening bulletin names only a rough compass direction for each ridge; the
# guide says it does not give the sector edges or the altitude it reaches. Until our own
# hits say otherwise, a ridge may extend up to RIDGE_MAX_TOP_DEG high and
# RIDGE_MAX_HALF_DEG either side of the named direction (the compass rounding alone can
# put the true sector centre 22.5 deg away). A hit that scored zero while others in the
# same exposure scored says the ridge reaches at least that high and that far; a hit that
# scored says it does not cover that point. Both bounds tighten from there.
RIDGE_MAX_TOP_DEG = 55.0
RIDGE_MAX_HALF_DEG = 60.0
RIDGE_EVIDENCE_MARGIN_DEG = 0.5     # a zero hit is read slightly lower/nearer than computed, to be safe
COMPASS_AZ = {"N": 0.0, "NE": 45.0, "E": 90.0, "SE": 135.0, "S": 180.0, "SW": 225.0, "W": 270.0, "NW": 315.0}


class PendingPrediction(NamedTuple):
    model: float          # lunar/airmass quality model used at planning time
    band_model: float      # model / 0.95, used for program-band back-estimation
    alt: float
    az: float
    clean: bool            # true when no all-sky notice / directional block applied at plan time
    lst: float = float("nan")   # local sidereal time (deg) at the start of the exposure


def _mod(a: float, n: float) -> float:
    m = a % n
    return m + n if m < 0 else m


class Ridge:
    """What we know about one announced terrain obstruction: a sector of azimuths around
    the named direction, blocked from the horizon up to some top altitude. The top T and
    the half-width H are unknown; every (T, H) pair on a grid carries a weight, flat at
    first. Each hit is evidence: a hit that scored zero at (alt, d) was probably covered,
    so pairs that cover it (T >= alt and H >= d) gain against the rest; a hit that scored
    was almost surely not covered, so pairs that cover it lose. Evidence is soft -- an
    unannounced closure can zero a field too -- so one wrong reading is outweighed by the
    next few right ones. The chance a point is clear is the weight of pairs that do not
    cover it, read from a suffix-sum table so the planner's hot loop pays O(1); the table
    is rebuilt once per exposure."""

    TOP_STEP = 1.0
    HALF_STEP = 2.0
    # log likelihood ratios (covering pairs vs the rest) of one unit of evidence
    ZERO_IN_MIXED_FIELD = math.log(0.95 / 0.05)   # zero beside scoring fibres: a directional block
    CLEAR = math.log(0.02 / 0.98)                 # scored: not covered (sampling aside)
    EVIDENCE_PER_EXPOSURE = 4.0                   # one exposure's hits share this much weight

    def __init__(self, direction: str):
        self.direction = direction
        self.centre = COMPASS_AZ[direction]
        self.n_top = int(RIDGE_MAX_TOP_DEG / self.TOP_STEP) + 1
        self.n_half = int(RIDGE_MAX_HALF_DEG / self.HALF_STEP) + 1
        self._diff = [[0.0] * (self.n_half + 1) for _ in range(self.n_top + 1)]   # 2-D difference array of log weights
        self._prior = [0.0] * self.n_top
        self._dirty = False
        self.zero_hits = 0
        self.clear_hits = 0
        self._rebuild()

    def set_prior(self, top_deg: float, width_deg: float = 8.0, outside: float = 0.2) -> None:
        """Someone said the ridge reaches about `top_deg`: favour tops within `width_deg` of
        it. A soft prior only -- evidence accumulates on top of it as before."""
        self._prior = [0.0 if abs(t * self.TOP_STEP - top_deg) <= width_deg else math.log(outside)
                       for t in range(self.n_top)]
        self._dirty = True

    def _cell(self, alt: float, d: float) -> tuple[int, int]:
        """Grid indices of the first (T, H) that covers (alt, d)."""
        t = max(0, min(self.n_top, math.ceil(alt / self.TOP_STEP)))
        h = max(0, min(self.n_half, math.ceil(d / self.HALF_STEP)))
        return t, h

    def _add_quadrant(self, t: int, h: int, value: float) -> None:
        """Add `value` to the log weight of every pair with T >= t and H >= h."""
        if t < self.n_top and h < self.n_half:
            self._diff[t][h] += value
        self._dirty = True

    def _rebuild(self) -> None:
        n_top, n_half = self.n_top, self.n_half
        logw = [[0.0] * n_half for _ in range(n_top)]
        running = [0.0] * (n_half + 1)
        peak = -1e300
        for t in range(n_top):
            acc = 0.0
            row, drow, prior = logw[t], self._diff[t], self._prior[t]
            for h in range(n_half):
                acc += drow[h]
                running[h] += acc
                value = running[h] + prior
                row[h] = value
                if value > peak:
                    peak = value
        suffix = [[0.0] * (n_half + 1) for _ in range(n_top + 1)]
        for t in range(n_top - 1, -1, -1):
            srow, below, row = suffix[t], suffix[t + 1], logw[t]
            acc = 0.0
            for h in range(n_half - 1, -1, -1):
                acc += math.exp(row[h] - peak)
                srow[h] = acc + below[h]
        self._suffix = suffix
        self.total = suffix[0][0]
        self._dirty = False

    def flush(self) -> None:
        if self._dirty:
            self._rebuild()

    def clear_probability(self, alt: float, az: float) -> float:
        if self._dirty:
            self._rebuild()
        d = abs(wrap180(az - self.centre))
        t, h = self._cell(alt, d)
        return 1.0 - self._suffix[t][h] / self.total if self.total else 1.0

    def note_zero(self, alt: float, az: float, weight: float = 1.0) -> bool:
        """A target at (alt, az) scored nothing while other fibres of the same exposure scored:
        probably covered. False when no ridge in this direction could reach that point."""
        d = abs(wrap180(az - self.centre))
        if d > RIDGE_MAX_HALF_DEG or alt > RIDGE_MAX_TOP_DEG:
            return False
        self.zero_hits += 1
        t, h = self._cell(alt - RIDGE_EVIDENCE_MARGIN_DEG, d - RIDGE_EVIDENCE_MARGIN_DEG)
        self._add_quadrant(t, h, weight * self.ZERO_IN_MIXED_FIELD)
        return True

    def note_clear(self, alt: float, az: float, weight: float = 1.0) -> None:
        """A target at (alt, az) scored: pairs that would cover it lose."""
        d = abs(wrap180(az - self.centre))
        if d > RIDGE_MAX_HALF_DEG or alt > RIDGE_MAX_TOP_DEG:
            return
        self.clear_hits += 1
        t, h = self._cell(alt, d)
        self._add_quadrant(t, h, weight * self.CLEAR)

    def bounds(self, mass: float = 0.9) -> tuple[float, float, float, float]:
        """Top and half-width ranges holding `mass` of the weight (marginals)."""
        if self._dirty:
            self._rebuild()
        s = self._suffix
        tops = [s[t][0] - s[t + 1][0] for t in range(self.n_top)]
        halves = [s[0][h] - s[0][h + 1] for h in range(self.n_half)]

        def span(values: list, step: float) -> tuple[float, float]:
            total = sum(values) or 1.0
            lo, acc = 0, 0.0
            while lo < len(values) - 1 and acc + values[lo] < (1.0 - mass) / 2 * total:
                acc += values[lo]
                lo += 1
            hi, acc = len(values) - 1, 0.0
            while hi > lo and acc + values[hi] < (1.0 - mass) / 2 * total:
                acc += values[hi]
                hi -= 1
            return lo * step, hi * step

        t0, t1 = span(tops, self.TOP_STEP)
        h0, h1 = span(halves, self.HALF_STEP)
        return t0, t1, h0, h1

    def summary(self) -> str:
        t0, t1, h0, h1 = self.bounds()
        return f"{self.direction} top {t0:.0f}-{t1:.0f} half {h0:.0f}-{h1:.0f} (zero {self.zero_hits} clear {self.clear_hits})"


class SurveyState:
    def __init__(self, init_payload: dict):
        site = init_payload["site"]
        survey = init_payload["survey"]
        instrument = init_payload["instrument"]
        limits = init_payload.get("limits", {})

        self.lat = float(site["latitude_deg"])
        self.lon = float(site["longitude_deg"])
        self.min_alt = float(site.get("minimum_altitude_deg", 30.0))
        self.sun_altitude_limit_deg = float(site.get("sun_altitude_limit_deg", -18.0))

        self.survey_start = parse_utc(survey["start_utc"])
        self.survey_end = parse_utc(survey["end_utc"])
        self.slot_seconds = int(survey.get("slot_seconds", 900))
        self.nights = [(parse_utc(n["observing_start_utc"]), parse_utc(n["observing_end_utc"]))
                       for n in survey.get("nights", [])]

        self.fiber_grid = FiberGrid(instrument)
        exposure = instrument.get("exposure", {})
        self.min_exposure = int(exposure.get("min_duration_seconds", 60))
        self.max_exposure = int(exposure.get("max_duration_seconds", 3600))

        self.scoring = ScoringModel(init_payload.get("scoring", {}), site)

        reporting = init_payload.get("scoring", {}).get("reporting", {})
        self.max_consecutive_reports = int(reporting.get("max_consecutive_reports", limits.get("max_consecutive_reports", 32)))
        self.false_report_free_allowance = int(reporting.get("false_report_free_allowance", 0))
        self.response_max_bytes = int(limits.get("response_max_bytes", 524288))
        self.wallclock_limit = float(limits.get("global_wallclock_seconds", 900) or 900)

        # Parallel arrays, one slot per target, in catalogue order.
        self.ids: list[str] = []
        self.ra: list[float] = []
        self.dec: list[float] = []
        self.flux: list[float] = []
        self.weight: list[float] = []
        self.required: list[bool] = []
        self.target_class: list[str] = []
        self.index_of: dict[str, int] = {}

        columns = init_payload.get("targets", {}).get("columns", [])
        col = {name: idx for idx, name in enumerate(columns)}
        for row in init_payload.get("targets", {}).get("rows", []):
            target_id = str(row[col["target_id"]])
            self.index_of[target_id] = len(self.ids)
            self.ids.append(target_id)
            self.ra.append(float(row[col["ra_deg"]]))
            self.dec.append(float(row[col["dec_deg"]]))
            self.flux.append(float(row[col["feature_flux"]]))
            self.weight.append(float(row[col["science_weight"]]))
            self.required.append(bool(row[col["required"]]))
            self.target_class.append(str(row[col["target_class"]]) if "target_class" in col else "")

        n = len(self.ids)
        self.hmax = [max_hour_angle_deg(self.dec[i], self.lat, self.min_alt + ALT_MARGIN_DEG) for i in range(n)]
        self.factor = [0.0] * n
        self.best_score = [0.0] * n
        self.misses = [0] * n
        self.attempts = [0] * n
        self.active = [i for i in range(n) if self.hmax[i] > 0.0]

        self._cells: dict[int, list[tuple[float, int]]] = {}
        self._build_index()
        self.first_night, self.last_night = self._build_windows()

        self.scale = 1.0
        self.prior_scale = 1.0
        self._samples: deque = deque(maxlen=24)           # (hours, ratio)
        self._all_ratios: deque = deque(maxlen=400)        # ratio
        self.clean_history: list[tuple[float, int, float, bool]] = []  # (hours, night, ratio, is a lower bound)
        self.pending_night = -1
        self._band_checks: deque = deque(maxlen=600)       # (program, matched, model, night)
        self.force_program: Optional[str] = None
        self.pending: dict[str, PendingPrediction] = {}
        self.pending_program = "BACKUP"
        self.pending_band_scale: Optional[float] = None   # sky-only band factor the program was declared with
        self.pending_duration = 0
        self.blocked: list[tuple[float, float]] = []       # (az, alt) where a hit scored zero
        self.notices: set[str] = set()                      # "kind|direction"
        self.terrain: set[str] = set()
        self.ridges: dict[str, Ridge] = {}                 # learned extent of each announced obstruction
        self.duration_scale = 1.0
        self.fast_level = 0

        # Per-observe-action ledger: (observe_action_index, target_id, factor), mirroring
        # the backend's own BestLedger so a Hard-mode state_resync can be answered exactly
        # (see _resync) instead of only from the resync message's best_scores.
        self.ledger: list[tuple[int, str, float]] = []
        self.pending_action_index: Optional[int] = None

    # -- spatial index -------------------------------------------------------

    def _build_index(self) -> None:
        for i in self.active:
            key = math.floor(self.dec[i])
            self._cells.setdefault(key, []).append((self.ra[i], i))
        for band in self._cells.values():
            band.sort(key=lambda pair: pair[0])
        self._cell_ra = {key: [ra for ra, _ in band] for key, band in self._cells.items()}

    def neighbours(self, ra: float, dec: float, radius: float):
        """Indices within `radius` degrees of (ra, dec), using the 1-degree declination-band index."""
        found: list[int] = []
        cos_dec = max(0.05, math.cos(math.radians(min(89.0, abs(dec) + radius))))
        width = radius / cos_dec
        lo_key, hi_key = math.floor(dec - radius), math.floor(dec + radius)
        for key in range(lo_key, hi_key + 1):
            band = self._cells.get(key)
            if not band:
                continue
            spans: list[tuple[float, float]]
            lo, hi = ra - width, ra + width
            if lo < 0:
                spans = [(0.0, hi), (lo + 360.0, 360.0)]
            elif hi >= 360:
                spans = [(lo, 360.0), (0.0, hi - 360.0)]
            else:
                spans = [(lo, hi)]
            keys = self._cell_ra[key]
            for low, high in spans:
                start = bisect.bisect_left(keys, low)
                end = bisect.bisect_right(keys, high)
                for k in range(start, end):
                    found.append(band[k][1])
        return found

    def _build_windows(self):
        """First/last night index on which each target has >=20 minutes above the limit."""
        need = 20 * 60 * SIDEREAL_DEG_PER_SECOND
        spans = []
        for start, end in self.nights:
            from .geometry import local_sidereal_deg
            l0 = local_sidereal_deg(start, self.lon)
            span = (end - start).total_seconds() * SIDEREAL_DEG_PER_SECOND
            spans.append((l0, span))
        n = len(self.ra)
        first_night = [len(self.nights)] * n
        last_night = [-1] * n
        for i in self.active:
            h = self.hmax[i]
            for k, (l0, span) in enumerate(spans):
                if h >= 180.0:
                    overlap = span
                else:
                    a = _mod(self.ra[i] - h - l0, 360.0)
                    overlap = max(0.0, min(span, a + 2 * h) - a) + max(0.0, min(span, a - 360.0 + 2 * h))
                if overlap >= need:
                    if first_night[i] > k:
                        first_night[i] = k
                    last_night[i] = k
        return first_night, last_night

    # -- messages and results -------------------------------------------------

    def on_messages(self, messages: list[dict], latest_bulletin: Optional[dict]) -> None:
        for message in messages:
            if message.get("record_type") == "bulletin" and message.get("initial"):
                for notice in message.get("notices", []):
                    if notice.get("event_kind") == "terrain_obstruction":
                        direction = notice.get("direction")
                        self.terrain.add(direction)
                        if direction in COMPASS_AZ and direction not in self.ridges:
                            self.ridges[direction] = Ridge(direction)
            elif message.get("record_type") == "state_resync":
                self._resync(message)
        notices = (latest_bulletin or {}).get("notices", [])
        self.notices = {f"{n.get('event_kind')}|{n.get('direction')}" for n in notices
                        if n.get("event_kind") != "terrain_obstruction"}

    def _resync(self, message: dict) -> None:
        """Hard-mode state_resync (participant guide, Appendix A / section 8): a prior
        window of `observe` actions was invalidated. The message itself only gives
        `best_scores` (score, not factor) for targets with any surviving valid hit --
        the guide is explicit that it does not return each target's completion factor,
        and that an agent that needs it exactly should combine `invalidated_window`
        with its own saved valid-exposure history.

        We can do exactly that: every entry in `self.ledger` already holds the EXACT
        factor for one past observe action (on_result backs it out of the real score
        the backend returned, the public weight, and whichever of the two public
        program multipliers it matches -- not an estimate). Dropping the ledger
        entries inside the invalidated action-index window and taking, per target, the
        max factor among what is left reproduces the backend's own ledger exactly --
        this is a reconstruction, not an approximation from best_score.

        The only remaining uncertainty: a target with no ledger entry at all (e.g. this
        process restarted mid-run and lost its in-memory history) falls back to the old
        best_score/top_multiplier estimate below, same as before this change.
        """
        window = message.get("invalidated_window") or {}
        start, end = window.get("action_index_start"), window.get("action_index_end_exclusive")
        if start is not None and end is not None:
            self.ledger = [entry for entry in self.ledger if not (start <= entry[0] < end)]
        else:
            self.ledger = []  # no window given: nothing in the ledger can be trusted
        exact: dict[str, float] = {}
        for _, target_id, factor in self.ledger:
            if factor > exact.get(target_id, 0.0):
                exact[target_id] = factor

        best_scores = message.get("best_scores") or []
        best: dict[str, float] = {}
        if best_scores and isinstance(best_scores[0], dict):
            for row in best_scores:
                best[row.get("target_id")] = float(row.get("best_score", 0.0))
        else:
            for target_id, score in zip(message.get("observed_target_ids", []), best_scores):
                best[target_id] = float(score)
        top_multiplier = max(self.scoring.program_multipliers.values()) if self.scoring.program_multipliers else 1.2

        for i in range(len(self.ids)):
            target_id = self.ids[i]
            self.best_score[i] = best.get(target_id, 0.0)
            if target_id in exact:
                self.factor[i] = exact[target_id]
                continue
            score = best.get(target_id, 0.0)
            self.factor[i] = min(1.0, score / (self.weight[i] * top_multiplier)) if score > 0 and self.weight[i] > 0 else 0.0
        self.active = [i for i in range(len(self.ids)) if self.hmax[i] > 0.0]
        self.pending.clear()
        self.pending_action_index = None

    def site_closed(self) -> bool:
        for key in self.notices:
            kind, _, direction = key.partition("|")
            if kind in ("rain", "storm") and direction == "ALL":
                return True
        return False

    def all_sky_notice(self) -> bool:
        """An ALL-direction weather notice is active. An earthquake notice is not weather:
        it lowers instrument efficiency for many nights in a row, and the fault detector
        needs samples from exactly those nights, so it does not make a sample unclean."""
        return any(key.partition("|")[2] == "ALL" and key.partition("|")[0] != "earthquake"
                   for key in self.notices)

    def on_result(self, last_result: Optional[dict], hours: float) -> None:
        action_index = self.pending_action_index
        self.pending_action_index = None
        if not last_result or last_result.get("action") != "observe" or not self.pending:
            self.pending.clear()
            return
        hits = {h.get("target_id"): float(h.get("score", 0.0)) for h in last_result.get("hits", [])}
        any_positive = any(score > 0 for score in hits.values())
        scoring = self.scoring
        multipliers = scoring.program_multipliers
        mismatch = scoring.mismatch_multiplier
        declared_multiplier = multipliers.get(self.pending_program, 1.0)
        f0t0 = scoring.f0t0

        # Zero scores next to positive ones in the same exposure are directional blocks, not
        # weather. They teach the ridge model -- unless a rocket closure is active, which
        # also zeroes a sector for an hour and would be mistaken for terrain.
        # Only a zero standing NEXT TO a score in the same exposure is evidence about a ridge.
        # A field that came back entirely dark is almost always a closure, and the bulletins do
        # not announce every one of them (a third of the closed slots of practice card L1 carry
        # no all-sky notice), so reading those as terrain poisons the model: on the three v5
        # cards of the 10-06 evaluation it put ridges where there were none. A rocket closure
        # zeroes a sector for an hour and would be read as terrain too.
        rocket = any(key.partition("|")[0] == "rocket_launch" for key in self.notices)
        teach_ridges = bool(self.ridges) and not rocket and any_positive
        evidence_weight = min(1.0, Ridge.EVIDENCE_PER_EXPOSURE / max(1, len(hits))) if teach_ridges else 0.0
        for target_id, prediction in self.pending.items():
            i = self.index_of.get(target_id)
            if i is None:
                continue
            if target_id not in hits:
                self.misses[i] += 1
                continue
            score = hits[target_id]
            self.best_score[i] = max(self.best_score[i], score)
            explained = teach_ridges and self._teach_ridges(i, prediction, score > 0.0, evidence_weight)
            if score <= 0.0:
                if any_positive and not explained:
                    self.blocked.append((prediction.az, prediction.alt))
                continue
            weight = self.weight[i] if self.weight[i] > 0 else 1e-9
            multiplier_seen = score / weight
            if prediction.clean:
                if abs(multiplier_seen - declared_multiplier) < 2e-4:
                    self._band_checks.append((self.pending_program, True, prediction.model, self.pending_night))
                elif abs(multiplier_seen - mismatch) < 2e-4:
                    self._band_checks.append((self.pending_program, False, prediction.model, self.pending_night))
            factor_if_match = score / (weight * declared_multiplier) if declared_multiplier > 0 else 0.0
            factor_if_miss = score / (weight * mismatch) if mismatch > 0 else 0.0
            ratio_match = (factor_if_match * f0t0) / (self.flux[i] * self.pending_duration * prediction.model) \
                if self.flux[i] > 0 and self.pending_duration > 0 and prediction.model > 0 else 0.0
            # Did this hit earn the declared program's multiplier? The score alone cannot say
            # (score = weight x factor x multiplier). The band is a pure sky quantity, so it
            # is predicted from the sky-only level the planner declared the program with.
            # Predicting it from ratio_match instead (the fallback, used until that level
            # exists) is wrong whenever instrument efficiency is down: the ratio contains
            # the efficiency, the band comes out too low, a matched hit is read as a mismatch
            # and its factor is overstated by the full multiplier -- during a fault or after
            # an earthquake that marked required targets at 0.46 as done at 0.56.
            if self.pending_band_scale is not None:
                band = scoring.program_band(self.pending_band_scale * prediction.model)
            else:
                band = scoring.program_band(ratio_match * prediction.band_model)
            matched = band == self.pending_program
            if abs(multiplier_seen - declared_multiplier) < 2e-4:
                matched = True                      # saturated in the declared band, no doubt
            if abs(multiplier_seen - mismatch) < 2e-4:
                # score == weight x mismatch multiplier exactly: a saturated hit in another
                # band. Without this, a BACKUP/BRIGHT declaration reads it as an unsaturated
                # match (factor 1/1.06 or 1/1.12) with a tiny ratio, which drags the learned
                # sky scale down, which makes the planner declare BACKUP again, and so on.
                matched = False
            factor = factor_if_match if matched else factor_if_miss
            factor = min(1.0, factor)
            self.factor[i] = max(self.factor[i], factor)
            if action_index is not None:
                self.ledger.append((action_index, target_id, factor))
            if self.required[i] and self.factor[i] < scoring.required_threshold:
                self.attempts[i] += 1
            if factor < 0.97 and self.flux[i] > 0 and self.pending_duration > 0 and prediction.model > 0:
                ratio = (factor * f0t0) / (self.flux[i] * self.pending_duration * prediction.model)
                self._samples.append((hours, ratio))
                self._all_ratios.append(ratio)
            # Samples for the fault detector. A score of exactly weight x a program multiplier
            # is a SATURATED hit: its ratio is only a lower bound (the exposure was longer
            # than needed), and it is kept as such -- dropping saturated hits would leave
            # only the worst moments of a night. This also catches the saturated hit in the
            # wrong band, which the inference above can misread as an unsaturated match
            # (factor 1/1.06 or 1/1.12) with a far too low ratio.
            if prediction.clean and self.flux[i] > 0 and self.pending_duration > 0 and prediction.model > 0:
                saturated = factor >= 0.97 or abs(multiplier_seen - mismatch) < 2e-4 \
                    or abs(multiplier_seen - declared_multiplier) < 2e-4
                level = (1.0 if saturated else factor) * f0t0 / (self.flux[i] * self.pending_duration * prediction.model)
                self.clean_history.append((hours, self.pending_night, level, saturated))
        self.pending.clear()
        for ridge in self.ridges.values():
            ridge.flush()
        self.update_scale(hours)

    def _teach_ridges(self, i: int, prediction: PendingPrediction, scored: bool, weight: float) -> bool:
        """A hit scored (the target was clear of every ridge at some moment of the exposure,
        so its highest point is clear) or scored zero (it was covered throughout, so its
        lowest point is covered). The exposure's start position is the prediction; the end
        position follows from the sidereal time elapsed."""
        start = (prediction.alt, prediction.az)
        positions = [start]
        if prediction.lst == prediction.lst and self.pending_duration > 0:   # lst known (not NaN)
            lst_end = prediction.lst + self.pending_duration * SIDEREAL_DEG_PER_SECOND
            positions.append(radec_to_altaz(self.ra[i], self.dec[i], lst_end, self.lat))
        if scored:
            alt, az = max(positions)
            for ridge in self.ridges.values():
                ridge.note_clear(alt, az, weight)
            return True
        alt, az = min(positions)
        # attribute the zero to the ridge whose sector the target is nearest to
        ridge = min(self.ridges.values(), key=lambda r: abs(wrap180(az - r.centre)))
        return ridge.note_zero(alt, az, weight)

    def terrain_clear_probability(self, alt: float, az: float) -> float:
        """1.0 when no announced ridge can cover (alt, az); 0.0 when one surely does."""
        p = 1.0
        for ridge in self.ridges.values():
            p *= ridge.clear_probability(alt, az)
        return p

    def has_recent_sample(self, hours: float) -> bool:
        return any(when >= hours - SKY_MEMORY_HOURS for when, _ in self._samples)

    def update_scale(self, hours: float) -> None:
        if len(self._all_ratios) >= 8:
            ordered = sorted(self._all_ratios)
            self.prior_scale = ordered[len(ordered) // 2]
        recent = sorted(ratio for when, ratio in self._samples if when >= hours - SKY_MEMORY_HOURS)
        self.scale = max(0.05, recent[len(recent) // 2]) if len(recent) >= 4 else self.prior_scale

    # -- fault diagnostics ------------------------------------------------------

    def night_ratios(self, night: int, since_hours: Optional[float] = None) -> list[tuple[float, bool]]:
        """Clean quality ratios (observed / modelled) collected on one night, optionally
        only those measured at or after `since_hours`, as (ratio, is_lower_bound)."""
        return [(ratio, bound) for when, n, ratio, bound in self.clean_history
                if n == night and (since_hours is None or when >= since_hours)]

    def night_samples(self, night: int) -> list[tuple[float, float, bool]]:
        """(survey hours, ratio, is_lower_bound) for every clean sample of one night."""
        return [(when, ratio, bound) for when, n, ratio, bound in self.clean_history if n == night]

    def night_band_checks(self, night: int) -> list[tuple[str, bool, float]]:
        """(declared program, scored as that program, quality model) for every clean hit
        of one night whose score revealed the program band it really fell in."""
        return [(c[0], c[1], c[2]) for c in self._band_checks if c[3] == night]

    def forget_quality_history(self) -> None:
        self.clean_history = []
        self._band_checks.clear()
        self._samples.clear()
        self._all_ratios.clear()
        self.prior_scale = 1.0

    # -- night lookup -------------------------------------------------------------

    def current_night(self, now):
        for index, (start, end) in enumerate(self.nights):
            if start <= now < end:
                return index, start, end
        return None

    def next_night_start(self, now):
        for start, _end in self.nights:
            if start > now:
                return start
        return None


if __name__ == "__main__":
    # Self-test of the ridge model: a SE ridge reaching 33 deg over azimuths 110-140.
    r = Ridge("SE")
    assert r.clear_probability(60.0, 135.0) == 1.0 and r.clear_probability(20.0, 300.0) == 1.0
    p0 = r.clear_probability(30.0, 130.0)
    assert 0.0 < p0 < 1.0, p0                       # unknown yet: worth a probe where valuable
    assert r.note_zero(30.0, 130.0, 2.0)            # covered at 30 deg, 5 deg off centre (two fibres' worth)
    assert r.clear_probability(28.0, 132.0) < 0.1
    r.note_clear(34.0, 135.0, 2.0)                  # scored at 34 deg dead centre: top below 34
    assert r.clear_probability(34.0, 135.0) > 0.95 and r.clear_probability(28.0, 132.0) < 0.1
    for k in range(3):
        r.note_clear(20.0, 170.0)                   # scored low down 35 deg off centre: sector narrower
    assert r.clear_probability(20.0, 171.0) > 0.9 and r.clear_probability(20.0, 125.0) < 0.25
    assert not r.note_zero(10.0, 20.0)              # far outside anything a SE ridge could cover
    p_mid = r.clear_probability(40.0, 135.0)
    r.note_zero(45.0, 130.0, 0.3)                   # a weak, isolated zero high up
    assert r.clear_probability(40.0, 135.0) < p_mid  # it counts a little ...
    for k in range(2):
        r.note_clear(40.0, 135.0)
    assert r.clear_probability(40.0, 135.0) > 0.9   # ... and is outweighed by two scoring hits
    print("ridge self-test:", r.summary())
    r3 = Ridge("SW"); p_before = r3.clear_probability(47.0, 225.0)
    r3.set_prior(37.0)                              # a note says the SW ridge is about 37 deg
    assert r3.clear_probability(47.0, 225.0) > max(0.9, p_before) and r3.clear_probability(30.0, 225.0) < 0.3
    r3.note_clear(33.0, 225.0)                      # evidence still beats the prior
    assert r3.clear_probability(33.0, 225.0) > 0.9
    import time
    r2 = Ridge("SW"); t = time.perf_counter()
    for k in range(20000):
        r2.clear_probability(30.0 + (k % 30), 200.0 + (k % 50))
    lookup = (time.perf_counter() - t) / 20000 * 1e6
    t = time.perf_counter()
    for k in range(100):
        r2.note_zero(30.0, 225.0, 0.04); r2.flush()
    print(f"ridge self-test ok: {r3.summary()}; lookup {lookup:.2f} us, rebuild {(time.perf_counter() - t) / 100 * 1e3:.2f} ms")
