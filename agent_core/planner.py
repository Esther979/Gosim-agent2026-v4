"""Decision logic: pick a pointing, fill the 16 fibres, choose exposure length and
program -- or wait / report / finish.

1. Rank visible, not-yet-done targets. Required targets that are not done yet get a
   bonus (missing one costs real points at the end). Targets that set soon, or that
   have few nights left, rank higher.
2. For the best few "anchor" candidates, try each fibre as the pointing centre; fill
   every fibre with the best-value neighbour that lands on its glass; keep the best
   pointing.
3. Pick the exposure length with the best expected score per second, and the program
   (DARK / BRIGHT / BACKUP) most assigned targets will match.

Everything here uses only the public catalogue, the public scoring config, and the
agent's own past hits (SurveyState) -- never hidden weather truth. The model has two jobs,
both with bounded outputs. It reads every weekly forecast (`_on_forecast`) and turns the
event list into an expected usable fraction per night, which decides what cannot wait
for a better night. And it arbitrates borderline instrument-fault alarms (`_arbitrate`).
Clear-cut fault reports and the pointing search itself never go through the model. A failed call means neutral advice (= the rule-based path) for that
night; the next night's call still runs normally.

This mirrors the anchor-search algorithm of this project's companion TypeScript
example target-for-target, so both examples solve the problem the same way.
"""
from __future__ import annotations

import math
import os
import time
from datetime import timedelta

from .geometry import (
    Moon,
    SIDEREAL_DEG_PER_SECOND,
    altaz_to_radec,
    format_utc,
    local_sidereal_deg,
    lunar_factor,
    max_hour_angle_deg,
    parse_utc,
    radec_to_altaz,
    shift_altaz,
    tangent_offsets,
    wrap180,
)
from .llm_client import LLMClient
from .memory import TraceLog
from .state import PendingPrediction

REQUIRED_BONUS = 60.0
DONE_FACTOR = 0.95
PLAN_FACTOR_SAFETY = 0.9
EDGE_MARGIN_FRACTION = 0.08 / 0.6325   # keep-clear band at a fibre's edge, as a fraction of its width
DURATIONS = (300, 450, 600, 900, 1200, 1500, 1800, 2400, 3000, 3600)
MIN_VISIBLE_SECONDS = 600
NEIGHBOUR_RADIUS_FRACTION = 0.83      # of the field side: far enough to fill a field centred on any fibre
MAX_CENTRE_FIBRES = 25                # fibres tried as the pointing centre per anchor
FAST_CENTRE_FIBRES = 4
ANCHORS = 6
# Joint search (see _joint_search): pointing, fibre assignment and exposure length are
# chosen together, by expected score per second. One exposure length is shared by every
# fibre, so the fibres must be filled with targets that need about that long -- picking the
# targets first and the length afterwards leaves bright targets saturated and waiting while
# the faint ones are still short.
JOINT_DURATIONS = (180, 240, 300, 450, 600, 750, 900, 1200, 1500, 1800, 2400, 3000, 3600)
JOINT_OVERHEAD_SECONDS = 20.0     # charged to every exposure so that very short ones are not favoured
JOINT_ANCHORS = 24
POOL_REFRESH_HOURS = 0.4          # how often the whole catalogue is rescanned for candidates
# A target that will still be observable on later nights is worth less right now the further
# it is below the altitude it reaches at transit: the same exposure will buy more of it then.
# Its gain is scaled by (quality here / quality at transit) ** DEFER_POWER. Without this the
# search keeps pointing low (median 34-47 deg on the official cards) at targets that pass
# near the zenith a few hours or weeks later.
DEFER_POWER = 5.0
DEFER_MIN_NIGHTS = 2
BAND_CHECKS = 80                  # most recent band checks used to read the sky level for the program declaration
BAND_MIN_CHECKS = 12
ANCHOR_POOL = 300
# Region pressure (see _update_pressure). The sky is cut into RA strips; a strip whose
# remaining work exceeds its share of the remaining observing time has pressure > 1, and a
# target's worth is multiplied by pressure ** PRESSURE_GAMMA. Without it the search only
# asks "what pays best right now" and leaves a strip that is setting for the season half
# done (official card A: a third of the catalogue was visible for 323 hours and got 167),
# while spending those hours on strips that stay up for months. The exponent is kept low:
# measured on the synthetic official cards 0.5 gains about as much on the cards that have
# such a strip as 1.0 does and loses less on those that do not; 1.5 and 2.0 lose overall.
RA_BIN_DEG = 15.0
PRESSURE_ALT_DEG = 40.0            # a strip counts as available while it stands above this
PRESSURE_GAMMA = 0.5
PRESSURE_RANGE = (0.5, 2.5)
CLOSED_KINDS = {"rain", "storm"}
# A directional weather notice covers a sector whose real extent and strength the notice does
# not state. One conservative factor for all of them: the truth tables of the public practice
# cards put the combined effect of such an event (transparency x sky / seeing) between 0.36
# and 0.55. Learning it per kind from our own hits was tried and reverted -- with all-sky
# notices on nearly every night the clean control sample comes from the good nights, so the
# estimate charges the night's weather to the sector and collapses to the floor (0.15 on all
# three v5 cards of the 10-06 evaluation, which cost about 8000 points of required targets).
NOTICE_FACTOR = 0.35
TERRAIN_REQUIRED_FLOOR = 0.25   # of its value, for a required target not yet past the threshold
BLOCKING_KINDS = {"terrain_obstruction", "rocket_launch"}
DIRECTION_AZ = {"N": 0.0, "NE": 45.0, "E": 90.0, "SE": 135.0, "S": 180.0,
                "SW": 225.0, "W": 270.0, "NW": 315.0}

# Instrument-fault detector (see "instrument fault reporting" below). Each entry is
# (a night counts as low when its median ratio is <= this fraction of the reference,
#  consecutive low nights needed before reporting).
FAULT_BASE = (0.80, 2)             # no false report since the last correct one
FAULT_AFTER_FALSE = (0.70, 3)      # after a false report, still inside the free allowance
FAULT_AFTER_ALLOWANCE = (0.60, 3)  # the next false report would cost points
FAULT_STRONG = 0.70                # counted nights averaging at or below this: report without asking
FAULT_MIN_SAMPLES = 8              # a night with fewer clean samples is not judged
FAULT_REF_MIN_NIGHTS = 4           # quiet nights needed before a low night counts as suspicious
FAULT_REF_NIGHTS = 15              # quiet nights behind the reference level: a fault is a drop against what
                                   # is typical, not against a run of unusually fine nights (five fine nights
                                   # in a row once pushed the reference up by half and bought two false reports)
FAULT_DARK_CHECKS_WANTED = 6       # force DARK on a suspicious night until this many DARK band checks exist
FAULT_SKY_CHECKS_MIN = 6           # fewer band checks than this give no sky estimate for the night
FAULT_SKY_GRID = [0.5 + 0.025 * k for k in range(41)]   # candidate sky levels, relative to the quiet reference
FAULT_SKY_OK = 0.9                 # logged as sky_ok when the night's sky estimate is at least this
# Per-night factor by which an earthquake's efficiency loss shrinks. It is fitted from the
# agent's own nights but may not exceed the upper bound: a sequence that "recovers" more
# slowly than that is not recovering, it has a fault sitting on top of the earthquake.
FAULT_RECOVERY_RANGE = (0.6, 0.90)
FAULT_MAX_FORCED_PER_NIGHT = 4
# How many nights in a row the model may hold back a borderline alarm. It is asked again
# every night with the new night's data. Its "wait" verdicts have been right every time
# so far (weather spells of 3-4 nights at 70-80%), while the rule taking over after two
# of them produced false reports; a clear-cut drop never reaches the model at all.
FAULT_MAX_LLM_WAITS = 5

# Fast fault detection, within the night. A fault is a step: the instrument's efficiency
# drops at one moment (to about a quarter on the cards of 2026-10-06, measured over 300+
# onsets) and stays down until reported. Weather does not step like that in samples that
# carry no weather notice. So the clean quality ratios of the last FAST_WINDOW_EXPOSURES
# exposures are compared with those of the FAST_REF_EXPOSURES exposures before them -- the
# reference may lie on the previous night, so a fault that began in daytime is a step too.
# A report costs no time, its answer comes at once, and a false one is free while the
# allowance lasts (it starts over at every correct report), so a free probe is spent on a
# clear step without waiting for the night to end -- one per stretch between correct reports,
# the other free one is left to the nightly detector. Paid reports stay with that detector. The idea follows the organisers' python-pro example (hourly E = quality /
# band, free probes before paid ones, no probing just after an earthquake notice).
FAST_WINDOW_EXPOSURES = 3         # the last this many exposures of the night, every one of them low ...
FAST_MIN_SPAN_HOURS = 0.25        # ... spanning at least this long: unannounced weather dips for a
                                  # slot or two, a fault is a plateau. No upper limit -- under a fault
                                  # the learned scale drops and exposures stretch to half an hour.
FAST_REF_EXPOSURES = 6
FAST_MIN_SAMPLES = 8              # clean samples needed on each side of the step
FAST_STEP = 0.55                  # every window exposure below this share of the reference level
FAST_FLOOR = 0.10                 # a window this far down is the dome shutting part-way, not the instrument
# Weather lowers the sky, and with it the program band the hits really score in; a fault
# lowers only the instrument. So when the band checks of the window put the sky clearly
# below where those of the reference put it, the step is weather and is not probed. This is
# the organisers' python-pro idea (E = quality level / band level) used as a veto: without
# enough band checks on both sides the step is probed as before.
FAST_BAND_VETO = 0.8
FAST_BAND_MIN_CHECKS = 3
FAST_RECOVER = 0.85               # after a false probe: re-armed once the level is back to this share
FAST_MIN_SPACING_HOURS = 0.5
FAST_QUAKE_HOLD_HOURS = 12.0      # an earthquake lowers efficiency too, in a step, and a report does not repair it


FAULT_SYSTEM_PROMPT = (
    "You arbitrate a borderline instrument-fault alarm for a robotic telescope survey. "
    "For each recent night you get: ratio = that night's measured efficiency relative to "
    "what a healthy instrument should deliver (already corrected for the measured sky "
    "level and, during an earthquake sequence, for the expected night-by-night recovery); "
    "c = number of samples; sky = measured sky level (1.0 normal); quake = an earthquake "
    "notice was active; notice_kinds = public weather/event notices that night; "
    "profile = the night in two-hour blocks as [hours after night start, ratio, samples]. "
    "The last nights have ratio between 0.70 and 0.80: too low to ignore, too high to be "
    "sure. Say whether this should be reported as a fault or held back one more night; "
    "you will be asked again tomorrow with the new night.\n"
    "How the candidates look: an INSTRUMENT FAULT is one abrupt step at a single moment "
    "(inside a night's profile: normal blocks, then low blocks) and afterwards the ratio "
    "stays flat at the same reduced level, block after block and night after night, with "
    "no notice explaining it. WEATHER the sky estimate missed wanders between blocks and "
    "nights, tends to come with overcast/haze/rain/cold_snap notices on or next to those "
    "nights, and passes within a night or two. A SECOND EARTHQUAKE inside a quake "
    "sequence drops the ratio and then lets it climb back night after night.\n"
    "Costs: a correct report repairs the fault at once. A false report uses up a limited "
    "free allowance and makes the rule stricter afterwards, which delays catching a real "
    "fault later. Waiting one night while a fault of this size is real costs only about "
    "a quarter of one night. If the picture is not clearly a fault, wait.\n"
    'Reply with exactly one JSON object and nothing else: {"verdict": "report" | '
    '"wait_one_night", "why": "<=100 chars"}. Answer quickly.'
)

def _median(values: list) -> float:
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def _censored_median(samples: list) -> float:
    """Median of (value, is_lower_bound) samples by the Kaplan-Meier product-limit
    estimate: a lower bound stays "at risk" up to its value and is then withdrawn without
    counting as an observation. If the estimate never falls to one half, more than half of
    the night consists of lower bounds -- exposures sized so that the target just finished.
    Their bounds then sit a fixed margin under the true level, so the median of all values
    is returned: it is comparable from night to night, which is what the detector needs
    (the largest bound, the formally correct "at least", jumps around with a single
    sample)."""
    ordered = sorted(samples, key=lambda s: (s[0], s[1]))
    surviving, at_risk = 1.0, len(ordered)
    for value, bound in ordered:
        if not bound:
            surviving *= 1.0 - 1.0 / at_risk
            if surviving <= 0.5:
                return value
        at_risk -= 1
    return ordered[len(ordered) // 2][0]

# Time-limited observation requests (participant guide section 8 / appendix B). There is
# no penalty for letting one expire -- `observation_requests.miss_penalty` is fixed at 0 --
# so the only thing worth doing here is not leaving a reachable `completion_reward` on the
# table, and only when doing so is close to free. Several more aggressive designs were
# tried and rejected: adding a priority bonus into the normal anchor-search's ranking
# (_value/achievable), and a separate dedicated exposure pointed straight at an urgent
# target, both distorted which pointing got chosen and for how long, for a score loss
# repeatedly far bigger than any request reward across these practice cards -- a public
# weight scale of ~0.3-1.7 and a 50-60 point REQUIRED_BONUS leave no room for also
# carrying a "maybe worth 100" incentive without it taking over. What is left is a
# tie-break, applied only inside _finish_plan's own duration search over a pointing/fibre
# assignment chosen with ZERO knowledge of requests: among durations within
# REQUEST_RATE_TOLERANCE of the best expected-score-per-second rate, prefer one that also
# clears a needed request target's completion_factor_threshold. It can only ever trade a
# small, bounded amount of rate (never redirect the pointing itself, never reach for a
# target that is not already going to be exposed anyway) for a chance at the reward.
REQUEST_RATE_TOLERANCE = 0.9

# Active pursuit of a request (see _pick_request_target). Only exposures that lie wholly
# inside the request window count, each needed target must reach the threshold in ONE
# exposure, and nothing is lost when a request expires -- so a request is pursued only
# while enough of its targets are cheaply reachable, and dropped the moment they are not.
# A pursued target becomes the pointing's anchor; the other fibres are filled as usual, so
# the exposure still earns ordinary science score.
REQUEST_MARGIN = 1.15         # aim this far above the completion threshold
REQUEST_MAX_SECONDS = 1800    # a target needing a longer exposure than this is not worth it
REQUEST_MAX_TRIES = 2         # per target, per request


def _az_distance(a: float, b: float) -> float:
    return abs(wrap180(a - b))


PACE_PROBE = 40                     # planned decisions run at full depth before the pace is set
# Besides the fair clock (agent CPU, what `wallclock` in the payload reports), the platform
# stops a run outright when its REAL duration -- engine time included -- passes a hard cap.
# The cap is not in the protocol; the one observed is four times the fair-clock limit. The
# engine's own time per decision is measured here as the gap between our answer and the
# next request, so a scenario the engine simulates slowly (many directional events, many
# fibres) is noticed, and optional model calls are the first thing dropped to stay inside.
# The platform states the real time left under its per-card cap in every request
# (`wallclock.wall_remaining_seconds`, participant guide); that is used whenever present. Without
# it the cap is not derived from the fair-clock limit -- the platform changed it once on
# 2026-10-06 (3600 s -> 7200 s), and a cap guessed too low throttles a run with budget to
# spare -- but the smallest cap ever observed is assumed. Either way the guard may only halve
# the planning budget, never zero it: dropping the optional model calls is the cheaper lever.
REAL_WALL_ASSUMED_CAP = 3600.0
REAL_WALL_SAFETY = 0.9
REAL_WALL_MIN_SHARE = 0.5
# Of the card's fair-clock limit. The platform does not charge time spent waiting on the
# model, but until a call has proved that, _pace must hold the budget back out of the
# planning time (see `reserve`), so a budget as large as the limit leaves no planning time
# at all: that alone crippled the search on all three v5 cards of the 10-06 evaluation.
LLM_BUDGET_FRACTION = 500.0 / 900.0
PACE_MAX_RESERVE_SHARE = 0.4          # of the clock still remaining, whatever the model budget is
# A forecast question that also carries duty-log notes makes the model decode text first;
# it needs several thousand reasoning tokens and one to two minutes (72-120+ s measured on k3).
DUTY_LOG_CALL_TIMEOUT = 150.0
DUTY_LOG_MAX_TOKENS = 8000
# The model reads every weekly forecast once (see _on_forecast) and turns it into an
# expected usable fraction for each night it covers. A share of the budget is kept back for
# fault arbitration; a forecast that arrives when less than that is left is read by rule.
OUTLOOK_RESERVE_SECONDS = 60.0
OUTLOOK_PACING_SLACK_SECONDS = 45.0  # how far ahead of an even spend over the survey a forecast call may be
OUTLOOK_MIN_WALLCLOCK_FRACTION = 150.0 / 900.0
OUTLOOK_LOST_NIGHT = 0.3            # a night expected to be less usable than this is not counted on
CLOSING_KINDS = ("rain|ALL", "storm|ALL")   # the only prior the rule has; everything else is learned

# Free text in messages (an observation request's `reason`, say) can carry what the structured
# records do not: informal hand-over notes between observers. They are read with the next
# forecast, never in a call of their own. What the model may take from them is bounded:
# a night it calls bad but no forecast event covers is still planned at DUTY_LOG_MIN_USABLE
# of its worth, and a ridge height it reads is only a prior the ridge model's own evidence
# overrides (see state.Ridge.set_prior).
DUTY_LOG_MIN_CHARS = 60             # shorter strings are labels, not notes
DUTY_LOG_MAX_CHARS = 1500           # per note handed to the model
DUTY_LOG_KEEP_FIRST = 2             # earliest notes often define the conventions later ones use
DUTY_LOG_KEEP_RECENT = 8
DUTY_LOG_NIGHTS_AHEAD = 7           # nights the notes are asked about beyond the forecast's coverage
DUTY_LOG_MIN_USABLE = 0.25
RIDGE_PRIOR_MIN_DEG, RIDGE_PRIOR_MAX_DEG = 10.0, 55.0

DUTY_LOG_PROMPT_ADDENDUM = (
    "\n`duty_log` holds informal notes passed between the observers on duty, in whatever "
    "language and style they wrote them; earlier notes may define code words, simple ciphers "
    "or shorthand that later notes use, so read them in order and decode. A later correction "
    "supersedes what it corrects. Ignore hearsay the writer doubts and anything about other "
    "sites. Use them for two things only: (1) nights in `nights` that no forecast event covers "
    "-- estimate `usable` for them from the notes (1.0 when the notes say nothing about a "
    "night); (2) `terrain`: if the notes give how high a ridge or obstruction in some compass "
    "direction reaches, return {\"SE\": altitude_deg, ...} converting zenith distances to "
    "altitude above the horizon; otherwise {}."
)

OUTLOOK_SYSTEM_PROMPT = (
    "You plan a robotic telescope survey. A weekly forecast has just arrived: a list of "
    "events, each with a kind, a compass direction (or ALL for the whole sky) and the "
    "nights it is expected on. For every night in `nights`, estimate `usable`: the "
    "fraction of that night (0.0-1.0) in which exposures will return useful data. The "
    "scheduler uses it to decide what cannot wait: targets and time-limited requests "
    "whose remaining nights are mostly unusable are observed now, even at low altitude, "
    "instead of being left for a better moment that will not come.\n"
    "Evidence: `experience` gives, for each event kind and scope already met in this "
    "survey, how many nights it was forecast for and the usable fraction those nights "
    "really had (`quiet` = nights with no forecast event). Trust it over intuition once a "
    "kind has a few nights; for kinds not yet met, reason from what the event is -- rain "
    "or a storm over the whole sky closes the dome, a directional event blocks part of "
    "the sky, haze and cold snaps degrade rather than close, a rocket launch is a short "
    "local closure. Several events on one night combine. Nights without events are quiet. "
    "`rule_estimate` is what a fixed table would say; depart from it when the evidence "
    "or the combination of events justifies it.\n"
    'Reply with exactly one JSON object and nothing else: {"usable": {"YYYY-MM-DD": '
    'number, ...}, "notes": "<=120 chars"} with one entry per night in `nights`. '
    "Answer quickly."
)


class Planner:
    def __init__(self, state, log=lambda text: None):
        from .weather_adapter import build_conditions
        self.conditions_mode, self.conditions = build_conditions(state)
        self.incremental_scores = os.environ.get('AGENT_SCORE_GAIN', '0') == '1'
        # looked up for every candidate on every decision: compute it once
        self._top_multiplier = max(state.scoring.program_multipliers.values(), default=1.2)
        self._pool_stride = 1                         # >1 only when even the lightest search would overrun the clock
        self._pool_phase = 0
        self.state = state
        self.log = log
        self.grid = state.fiber_grid
        # Everything about the fibre layout is derived from the instrument the card
        # describes (3x3 wide fibres up to 10x10 small ones), nothing is hard-coded.
        self.edge_margin = EDGE_MARGIN_FRACTION * self.grid.glass
        self.neighbour_radius = NEIGHBOUR_RADIUS_FRACTION * self.grid.fov
        by_centrality = sorted(range(self.grid.n), key=lambda f: (sum(c * c for c in self.grid.fiber_center(f)), f))
        self.fast_fibres = tuple(sorted(by_centrality[:FAST_CENTRE_FIBRES]))
        step = max(1, -(-self.grid.side // 5))   # ceil(side / 5): every fibre up to 5x5, a spread sample beyond
        sample = [f for f in range(self.grid.n)
                  if (f // self.grid.side) % step == 0 and (f % self.grid.side) % step == 0]
        self.centre_fibres = tuple(sample if len(sample) <= MAX_CENTRE_FIBRES else by_centrality[:MAX_CENTRE_FIBRES])
        self.llm = LLMClient(log=log)
        self.llm.total_budget_seconds = min(self.llm.total_budget_seconds,
                                            LLM_BUDGET_FRACTION * state.wallclock_limit)
        log(f"planner: wall-clock limit {state.wallclock_limit:.0f}s, llm budget {self.llm.total_budget_seconds:.0f}s; "
            f"fibres {self.grid.side}x{self.grid.side}, centres tried {len(self.centre_fibres)}")
        self.trace = TraceLog(log=log)
        # Forecast outlook (see _on_forecast): expected usable fraction of every night, 1.0
        # until a forecast says otherwise, and its running sum for "usable nights left".
        self._night_dates = [(start - timedelta(hours=12)).date().isoformat() for start, _end in state.nights]
        self._night_of_date = {date: n for n, date in enumerate(self._night_dates)}
        self._usable = [1.0] * len(state.nights)
        self._usable_prefix = list(range(len(state.nights) + 1))
        self._forecast_keys: dict = {}                # night index -> {"kind|ALL" or "kind|DIR"} forecast for it
        self._kind_stats: dict = {}                   # key -> [sum of usable fractions actually seen, nights]
        self._duty_log_first: list[dict] = []        # earliest free-text notes seen (conventions)
        self._duty_log_recent: list[dict] = []       # notes not yet shown to the model
        self._duty_log_total = 0
        self._duty_log_unread = 0                    # notes (first or recent) the model has not seen
        self._night_good: dict = {}                   # night index -> seconds of exposures that returned data
        self._last_observe: tuple | None = None       # (night index, seconds) of the previous action, if an exposure
        self._defer_power = DEFER_POWER
        self.outlook_calls = 0

        self.observe_count = 0
        self.reports = 0
        self.correct_reports = 0
        self.false_reports = 0                       # since the last correct report
        self._fast_false = 0                         # of those, the ones the fast detector made
        self.fault_nights: list[dict] = []           # one record per closed night
        self._fault_floor = 0                        # records before this index no longer count towards a streak
        self._fault_open_night: int | None = None
        self._quiet_fallback: float | None = None    # quiet reference carried across a correct report
        # Earthquake sequence: efficiency is modelled as quiet * (1 - loss), the loss shrinking
        # by a constant factor per night; all of it estimated from the agent's own nights.
        self._quake_quiet: float | None = None       # quiet reference frozen when the sequence began
        self._quake_first: tuple | None = None       # (night index, loss) the recovery curve starts from
        self._quake_recovery = FAULT_RECOVERY_RANGE[1]
        self._night_kinds: dict[str, set] = {}       # night_id -> notice kinds seen in its bulletins
        self._quake_onset: dict[str, float] = {}     # night_id -> survey hours of its first earthquake bulletin
        self._report_due: str | None = None
        self._report_streak = 0
        self._awaiting_report_result = False
        self._fast_pending = False                   # the report in flight came from the fast detector
        self._fast_block = None                      # level of a step that a fast probe found to be no fault
        self._fast_ref_level = None
        self._last_report_hours = float("-inf")
        self._quake_seen = False                     # an earthquake notice is in the current bulletin
        self._quake_onset_hours = float("-inf")      # when the current earthquake notice first appeared
        self.fast_reports = 0
        self._fast_veto_logged = None
        self._llm_waits = 0
        self._forced_tonight = 0
        self.night_index_seen: int | None = None
        self.consecutive_reports = 0
        self._last_forecast_notices: list = []
        self.total_assigned = 0
        self.total_hit = 0
        self._current_action_index = None
        self._request_thresholds_now: dict = {}
        self._active_requests: list = []
        self._pace_cost: dict = {0: None, 1: None, 2: None}   # measured seconds per planned decision, per level
        self._pace_samples: dict = {0: [], 1: [], 2: []}
        self._real_start = time.monotonic()
        self._real_last_return: float | None = None
        self._engine_gaps: list[float] = []            # real seconds between our answer and the next request
        self._real_slack = float("inf")                # projected real seconds to spare under the hard cap
        self._pool: list | None = None                # candidate pool, see plan()
        # quality model of every target at its own transit altitude with no moon: the best
        # its geometry ever allows, used to judge what a target could reach at its best
        self._model_best = [state.scoring.quality_model(min(89.0, 90.0 - abs(dec - state.lat)), 1.0) or 0.0
                            for dec in state.dec]
        # altitude from hour angle: sin(alt) = sin(lat) sin(dec) + cos(lat) cos(dec) cos(HA)
        lat = math.radians(state.lat)
        self._sin_sin = [math.sin(lat) * math.sin(math.radians(dec)) for dec in state.dec]
        self._cos_cos = [math.cos(lat) * math.cos(math.radians(dec)) for dec in state.dec]
        # sidereal time at the start of every night and how far it turns during the night
        self._night_lst0 = [local_sidereal_deg(start, state.lon) for start, _end in state.nights]
        self._night_turn = [(end - start).total_seconds() * SIDEREAL_DEG_PER_SECOND for start, end in state.nights]
        bins = int(round(360.0 / RA_BIN_DEG))
        self._bin_of = [int(ra // RA_BIN_DEG) % bins for ra in state.ra]
        reach: dict = {}
        sin_alt = math.sin(math.radians(PRESSURE_ALT_DEG))
        for i in state.active:
            if self._cos_cos[i] > 1e-9:
                cos_h = (sin_alt - self._sin_sin[i]) / self._cos_cos[i]
                if cos_h < 1.0:
                    reach.setdefault(self._bin_of[i], []).append(math.degrees(math.acos(max(-1.0, cos_h))))
        self._bin_reach = [(_median(reach[b]) if b in reach else 0.0) for b in range(bins)]
        self._pressure = [1.0] * bins
        self._band_scale_last = 1.0
        self._band_scale_night = -1
        self._pool_night = -1
        self._pool_hours = 0.0
        self._pace_prev: tuple | None = None
        self._pace_planned = 0
        self._pace_night_total: float | None = None
        # The fair clock charges agent CPU, not time spent waiting on the model: measured on
        # every platform card so far (a1 of the 10-06 evaluation: 791 s waiting, 229 s charged
        # of 900). The first model call of a run confirms it, and _pace switches if it does not.
        self._llm_charged = False
        self._last_forced: tuple | None = None        # (request_id, target index) of the previous action, if forced
        self._request_hold_until = float("-inf")      # survey hours: no forced exposure before this
        self._hours_now = 0.0
        self._request_tries: dict = {}                # (request_id, target index) -> forced exposures so far
        self._request_status: dict = {}               # request_id -> last logged "pursue"/"drop"
        self._forced_request: tuple | None = None     # (target index, threshold, seconds to deadline, request_id)
        self.request_exposures = 0

        log(f"planner: {len(state.ids)} targets ({sum(state.required)} required), "
            f"{len(state.nights)} nights, llm model={self.llm.model} base_url={self.llm.base_url}")

    # -- top-level decision ----------------------------------------------------

    def decide(self, payload: dict) -> dict:
        entered = time.monotonic()
        if self._real_last_return is not None:
            self._engine_gaps.append(entered - self._real_last_return)
            if len(self._engine_gaps) > 400:
                del self._engine_gaps[:200]
        try:
            return self._decide(payload)
        finally:
            self._real_last_return = time.monotonic()

    def _decide(self, payload: dict) -> dict:
        state = self.state
        now = parse_utc(payload["now_utc"])
        hours = (now - state.survey_start).total_seconds() / 3600.0

        for message in list(payload.get("new_messages", [])) + [payload.get("latest_bulletin") or {}]:
            if message.get("record_type") == "bulletin" or "slot_id" in message:
                self._note_bulletin(message)
        for message in payload.get("new_messages", []):
            self._collect_duty_log(message)
        for message in payload.get("new_messages", []):
            if message.get("record_type") == "state_resync":
                self._pool = None                     # invalidated observations bring targets back
            if message.get("record_type") == "forecast":
                self._last_forecast_notices = message.get("notices", [])
                self._on_forecast(message, payload)
            elif message.get("record_type") == "observation_request":
                self.log(f"planner: observation request {message.get('request_id')} issued, "
                         f"{len(message.get('target_ids', []))} targets by {message.get('deadline_utc')}")
            elif message.get("record_type") == "observation_request_result":
                self.log(f"planner: observation request {message.get('request_id')} "
                         f"{message.get('status')} (reward {message.get('score_delta', 0.0)})")
        state.on_messages(payload.get("new_messages", []), payload.get("latest_bulletin"))
        quake_now = any(key.partition("|")[0] == "earthquake" for key in state.notices)
        if quake_now and not self._quake_seen:
            self._quake_onset_hours = hours
        self._quake_seen = quake_now
        state.on_result(payload.get("last_result"), hours)
        if state.offset_changed:
            state.offset_changed = False
            off = state.pointing
            self.log(f"planner: pointing offset alt {off.offset[0]:+.3f} az {off.offset[1]:+.3f} deg "
                     f"(explains {off.explained[0]} more fibre outcomes than none, {off.explained[1]} tied cells, "
                     f"{off.misses} misses so far)")
        if self.conditions is not None:
            self.conditions.update(payload)
        last_result = payload.get("last_result")
        if last_result and last_result.get("action") == "observe":
            self.total_assigned += int(last_result.get("assigned_count", 0))
            self.total_hit += int(last_result.get("hit_count", 0))
        self._pace(payload, now)
        self._current_action_index = payload.get("observe_action_index")
        self._active_requests = payload.get("active_requests") or []
        self._hours_now = hours
        if self._last_observe is not None:
            observed_night, seconds = self._last_observe
            self._last_observe = None
            if last_result and last_result.get("action") == "observe" and \
                    any(float(h.get("score", 0.0)) > 0.0 for h in last_result.get("hits", [])):
                self._night_good[observed_night] = self._night_good.get(observed_night, 0) + seconds
        if self._last_forced is not None:
            request_id, forced_index = self._last_forced
            self._last_forced = None
            if last_result and last_result.get("action") == "observe":
                score = next((float(h.get("score", 0.0)) for h in last_result.get("hits", [])
                              if h.get("target_id") == state.ids[forced_index]), None)
                if score is not None and score <= 0.0:
                    # the sky was shut, not the target's fault: give the try back and hold off a while
                    key = (request_id, forced_index)
                    self._request_tries[key] = max(0, self._request_tries.get(key, 1) - 1)
                    self._request_hold_until = hours + 0.5
        self._request_thresholds_now = self._request_thresholds(payload.get("active_requests") or [])

        night = state.current_night(now)
        report = self._maybe_report(night[0] if night else None, hours, payload)
        if report is not None:
            return report
        if night is None:
            nxt = state.next_night_start(now)
            if nxt is None:
                return {"action": "finish", "reason": "no observing night left"}
            return {"action": "wait", "until_utc": format_utc(nxt), "reason": "daytime: sleep until the next night"}
        night_index, night_start, night_end = night

        if self.night_index_seen != night_index:
            self.night_index_seen = night_index
            # how much the coming nights can be counted on sets how patiently tonight waits
            # for targets to climb: with the next three nights lost, almost not at all
            self._update_pressure(now, night_index)
            ahead = self._usable[night_index + 1:night_index + 4]
            self._defer_power = DEFER_POWER * (0.3 + 0.7 * (sum(ahead) / len(ahead) if ahead else 1.0))

        if (night_end - now).total_seconds() < state.min_exposure:
            nxt = state.next_night_start(now)
            if nxt is None:
                return {"action": "finish", "reason": "survey over"}
            return {"action": "wait", "until_utc": format_utc(nxt), "reason": "night ending"}

        if state.site_closed():
            return {"action": "wait", "duration_seconds": self._to_next_slot(now, night_start),
                    "reason": "bulletin: rain/storm over the whole sky"}

        self._force_dark_if_suspicious(night_index)
        self._pace_prev = self._pace_prev[:2] + (True,) + self._pace_prev[3:]
        action = self.plan(now, night_end, night_index, hours)
        if action is not None and state.force_program:
            self._forced_tonight += 1
        if action is None:
            return {"action": "wait", "duration_seconds": self._to_next_slot(now, night_start),
                    "reason": "nothing useful is up"}
        self.observe_count += 1
        self._last_observe = (night_index, int(action["duration_seconds"]))
        action["reason"] = f"{len(action['assignments'])} fibres, program {action['program']}"
        return action

    def on_finish(self, payload: dict) -> None:
        if self.conditions is not None:
            self.log(f"conditions: {self.conditions.summary()}")
        self.trace.write({"event": "finish", **payload})
        self.trace.close()
        self.log(f"planner: finished termination_reason={payload.get('termination_reason')} "
                 f"observes={self.observe_count} request_exposures={self.request_exposures} reports={self.reports} "
                 f"(correct {self.correct_reports}, fast {self.fast_reports}) llm_calls={self.llm.calls_made}")

    def note_action(self, action: dict) -> None:
        """Called by agent.py right after an action is validated, so the consecutive-report
        counter (enforced by validation.py) stays correct even when a fallback replaced it."""
        self.consecutive_reports = self.consecutive_reports + 1 if action.get("action") == "report" else 0

    def _to_next_slot(self, now, night_start) -> int:
        slot = self.state.slot_seconds
        into = (now - night_start).total_seconds() % slot
        return int(max(60, min(3600, slot - into if into else slot)))

    def _pace(self, payload: dict, now) -> None:
        """Choose how much search to do per decision (0 = full, 1 = one anchor, 2 = one
        anchor and only the central fibres) from what decisions have actually cost so far.
        The cost is read off the engine's own clock (`wallclock.elapsed_seconds`), so it is
        in whatever units the limit is charged in; the first PACE_PROBE planned decisions
        run at full depth to measure it. Time spent waiting on the model is set aside only
        if the engine turns out to charge for it."""
        state = self.state
        clock = payload.get("wallclock") or {}
        remaining = float(clock.get("remaining_seconds", 1e9))
        elapsed = float(clock.get("elapsed_seconds", 0.0))
        previous = self._pace_prev      # (elapsed, level, went through plan(), model seconds mark)
        if previous is not None:
            delta = max(0.0, elapsed - previous[0])
            model_seconds = self.llm.spent_seconds - previous[3]
            if model_seconds > 2.0:
                self._llm_charged = delta > 0.5 * model_seconds
            elif previous[2]:
                # median of everything seen at this level so far: one slow decision (start-up,
                # a night boundary) must not set the pace
                samples = self._pace_samples[previous[1]]
                samples.append(delta)
                if len(samples) > 400:
                    del samples[:200]
                self._pace_cost[previous[1]] = _median(samples)
                self._pace_planned += 1
        night_seconds = sum(max(0.0, (end - max(start, now)).total_seconds()) for start, end in state.nights if end > now)
        if self._pace_night_total is None:
            self._pace_night_total = night_seconds
        # bright targets go first, so early exposures are short; later ones are not
        per_decision_sim = 700.0 if self._pace_planned < PACE_PROBE else \
            max(700.0, (self._pace_night_total - night_seconds) / self._pace_planned)
        decisions_left = max(1.0, night_seconds / per_decision_sim)
        held_back = self.llm.budget_left_seconds() if self._llm_charged and not self.llm.disabled else 0.0
        reserve = 60.0 + min(held_back, PACE_MAX_RESERVE_SHARE * remaining)
        usable = max(0.0, remaining - reserve) * 0.85
        full = self._pace_cost[0]
        level = 0
        # Real wall clock: what the engine itself takes per decision, plus what we take, must
        # also fit under the hard cap. When it does not, first the model calls go (see
        # real_slack_allows), then the search depth, through the same `usable` as above.
        engine_per = _median(self._engine_gaps[-200:]) if len(self._engine_gaps) >= PACE_PROBE else 0.0
        real_elapsed = time.monotonic() - self._real_start
        if "wall_remaining_seconds" in clock:
            # the platform states the real time left under its cap (participant guide, wallclock)
            real_cap = real_elapsed + float(clock["wall_remaining_seconds"])
            real_left = REAL_WALL_SAFETY * float(clock["wall_remaining_seconds"]) - decisions_left * engine_per
        else:
            real_cap = REAL_WALL_SAFETY * max(REAL_WALL_ASSUMED_CAP, state.wallclock_limit)
            real_left = real_cap - real_elapsed - decisions_left * engine_per
        self._real_slack = real_left - decisions_left * (full or 0.0)
        if real_left < usable:
            if self._real_log_due(real_left):
                self.log(f"planner: real wall clock {real_elapsed:.0f}s of {real_cap:.0f}s, engine {engine_per * 1000:.0f} ms "
                         f"per decision, {decisions_left:.0f} decisions left: {max(0.0, real_left):.0f}s usable for planning")
            usable = max(usable * REAL_WALL_MIN_SHARE, max(0.0, real_left))
        if self._pace_planned >= PACE_PROBE and full is not None:
            costs = (full, self._pace_cost[1] or full / 4.0, self._pace_cost[2] or full / 8.0)
            level = next((lv for lv in (0, 1, 2) if costs[lv] * decisions_left <= usable), 2)
            # Last resort. The levels only thin out the pointing search; scanning the
            # candidates costs the same at every level and grows with the catalogue. If
            # even level 2 would overrun, look at only every n-th candidate per decision
            # (a different n-th each time), so the survey slows down instead of being cut off.
            stride = 1
            if level == 2 and usable > 0.0 and costs[2] * decisions_left > usable:
                stride = min(8, int(costs[2] * decisions_left / usable) + 1)
            if stride != self._pool_stride:
                self.log(f"planner: candidate scan stride {stride} ({costs[2] * 1000:.0f} ms per light decision, "
                         f"{decisions_left:.0f} decisions left, {usable:.0f}s usable)")
                self._pool_stride = stride
        if level != state.fast_level:
            self.log(f"planner: pace level {level} ({(full or 0) * 1000:.0f} ms per full decision, "
                     f"{decisions_left:.0f} decisions left, {usable:.0f}s usable)")
            state.fast_level = level
        self._pace_prev = (elapsed, level, False, self.llm.spent_seconds)

    def _terrain_floor(self, i: int) -> float:
        """How much of a still-unfinished required target's value survives a likely ridge."""
        state = self.state
        return TERRAIN_REQUIRED_FLOOR if state.required[i] \
            and state.factor[i] < state.scoring.required_threshold else 0.0

    def _real_log_due(self, real_left: float) -> bool:
        bucket = int(max(0.0, real_left) // 100.0)
        if bucket == getattr(self, "_real_log_bucket", None):
            return False
        self._real_log_bucket = bucket
        return True

    def real_slack_allows(self, seconds: float) -> bool:
        """Is there room under the real-time hard cap for an optional wait of `seconds`?"""
        return self._real_slack >= seconds

    # -- LLM: reading the weekly forecast -----------------------------------------------
    #
    # A forecast lists events (kind, direction or ALL, nights) for the week ahead. What the
    # scheduler needs from it is one number per night: how much of that night can be counted
    # on. The model turns the event list into those numbers, using both what the events
    # mean and what the same kinds of event have actually done on this card so far; the
    # numbers then decide which targets and requests are "now or never" (see _joint_search,
    # _request_reachable_later, and the nightly defer power in decide). Without a model
    # answer a fixed rule supplies them, so the look-ahead never switches off.

    def _rule_usable(self, night: int) -> float:
        """What the events forecast for a night are worth without a model: the usable
        fraction each kind has really shown on this card (relative to quiet nights) once
        it has been met three times, and until then 0.1 for rain or a storm over the
        whole sky and 1.0 for everything else. The worst event of the night counts."""
        keys = self._forecast_keys.get(night)
        if not keys:
            return 1.0
        quiet = self._kind_stats.get("quiet")
        quiet_level = quiet[0] / quiet[1] if quiet and quiet[1] >= 3 and quiet[0] > 0 else None
        worst = 1.0
        for key in keys:
            seen = self._kind_stats.get(key)
            if seen and seen[1] >= 3:
                value = seen[0] / seen[1]
                if quiet_level:
                    value = min(1.0, value / quiet_level)
            else:
                value = 0.1 if key in CLOSING_KINDS else 1.0
            worst = min(worst, value)
        return worst

    @staticmethod
    def _sanitize_outlook(answer, dates: list, rule: dict) -> tuple[dict, int]:
        """Model reply -> {date: usable in [0, 1]} for exactly the nights asked about.
        Any night the reply leaves out or garbles keeps the rule's value. Returns the
        result and how many nights the model actually supplied."""
        supplied = answer.get("usable") if isinstance(answer, dict) else None
        result, taken = {}, 0
        for date in dates:
            value = supplied.get(date) if isinstance(supplied, dict) else None
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value != value:
                result[date] = rule[date]
            else:
                result[date] = min(1.0, max(0.0, float(value)))
                taken += 1
        return result, taken

    def _collect_duty_log(self, message: dict) -> None:
        """Keep any long free-text field of a message as a note for the model to read with
        the next forecast. Structured fields are short; a note is not."""
        for key, value in message.items():
            if not isinstance(value, str) or len(value) < DUTY_LOG_MIN_CHARS or key in ("record_type", "schema_version"):
                continue
            note = {"at": message.get("issued_at_utc"), "field": key, "text": value[:DUTY_LOG_MAX_CHARS]}
            self._duty_log_total += 1
            self._duty_log_unread += 1
            if len(self._duty_log_first) < DUTY_LOG_KEEP_FIRST:
                self._duty_log_first.append(note)
            else:
                self._duty_log_recent.append(note)
                if len(self._duty_log_recent) > DUTY_LOG_KEEP_RECENT:
                    del self._duty_log_recent[0]
            if self._duty_log_total <= 3 or self._duty_log_total % 10 == 0:
                self.log(f"planner: duty-log note #{self._duty_log_total} ({len(value)} chars in '{key}')")

    def _apply_terrain_prior(self, answer) -> None:
        terrain = answer.get("terrain") if isinstance(answer, dict) else None
        if not isinstance(terrain, dict):
            return
        for direction, value in terrain.items():
            ridge = self.state.ridges.get(str(direction).upper())
            if ridge is None or isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            if RIDGE_PRIOR_MIN_DEG <= float(value) <= RIDGE_PRIOR_MAX_DEG:
                ridge.set_prior(float(value))
                self.log(f"planner: terrain prior from duty log {direction} ~{float(value):.0f} deg -> {ridge.summary()}")

    def _on_forecast(self, forecast: dict, payload: dict) -> None:
        state = self.state
        try:
            first = parse_utc(forecast["coverage_start_utc"])
            last = parse_utc(forecast["coverage_end_utc"])
        except (KeyError, TypeError, ValueError):
            return
        covered = [n for n, (start, _end) in enumerate(state.nights) if first <= start < last]
        for n in covered:
            self._forecast_keys[n] = set()             # a newer forecast replaces an older one for the same night
        for notice in forecast.get("notices") or []:
            kind, direction = notice.get("event_kind"), notice.get("direction")
            if not kind:
                continue
            key = f"{kind}|{'ALL' if direction == 'ALL' else 'DIR'}"
            for date in notice.get("nights") or []:
                n = self._night_of_date.get(date)
                if n is not None:
                    self._forecast_keys.setdefault(n, set()).add(key)
                    if n not in covered:
                        covered.append(n)
        covered.sort()
        if not covered:
            return
        notes = self._duty_log_first + self._duty_log_recent if self._duty_log_unread else []
        from_notes: set[int] = set()
        if notes:
            for n in range(covered[0], min(len(state.nights), covered[0] + DUTY_LOG_NIGHTS_AHEAD)):
                if n not in covered:
                    covered.append(n)
                    from_notes.add(n)
            covered.sort()
        dates = [self._night_dates[n] for n in covered]
        rule = {self._night_dates[n]: round(self._rule_usable(n), 2) for n in covered}
        eventful = any(self._forecast_keys.get(n) for n in covered) or bool(notes)
        left = float((payload.get("wallclock") or {}).get("remaining_seconds", 0))
        status, taken, usable = "rule", 0, dict(rule)
        if not eventful:
            status = "quiet"                           # nothing to interpret
        elif self.llm.disabled:
            status = "disabled"
        elif self.llm.budget_left_seconds() < OUTLOOK_RESERVE_SECONDS + \
                (DUTY_LOG_CALL_TIMEOUT if notes else self.llm.call_timeout_seconds) \
                or left < OUTLOOK_MIN_WALLCLOCK_FRACTION * state.wallclock_limit:
            status = "skipped"
        elif not self.real_slack_allows(3.0 * (DUTY_LOG_CALL_TIMEOUT if notes else self.llm.call_timeout_seconds)):
            status = "skipped (real wall clock)"       # the engine is slow on this card: keep the time
        elif self.llm.spent_seconds > OUTLOOK_PACING_SLACK_SECONDS + \
                (self.llm.total_budget_seconds - OUTLOOK_RESERVE_SECONDS) * (covered[0] + 7) / max(7, len(state.nights)):
            status = "paced"                           # ahead of the budget's pace: keep some for later forecasts
        else:
            experience = {key: {"nights": count, "usable_fraction": round(total / count, 2)}
                          for key, (total, count) in sorted(self._kind_stats.items()) if count}
            request = {
                "nights": dates,
                "forecast_notices": forecast.get("notices") or [],
                "experience": experience,
                "rule_estimate": rule,
                "nights_left_in_survey": len(state.nights) - covered[0],
            }
            prompt = OUTLOOK_SYSTEM_PROMPT
            if notes:
                request["duty_log"] = notes
                request["nights_without_forecast_events"] = [self._night_dates[n] for n in covered
                                                             if not self._forecast_keys.get(n)]
                prompt += DUTY_LOG_PROMPT_ADDENDUM
            if notes:
                answer = self.llm.ask_json(prompt, request, left, timeout=DUTY_LOG_CALL_TIMEOUT,
                                           max_tokens=DUTY_LOG_MAX_TOKENS)
            else:
                answer = self.llm.ask_json(prompt, request, left)
            self.outlook_calls += 1
            usable, taken = self._sanitize_outlook(answer, dates, rule)
            if notes:
                for n in from_notes:                   # the notes alone may not write a night off
                    date = self._night_dates[n]
                    usable[date] = max(DUTY_LOG_MIN_USABLE, usable[date])
                if taken:
                    self._duty_log_recent = []         # read; the earliest notes stay as the key
                    self._duty_log_unread = 0
                self._apply_terrain_prior(answer)
            status = "ok" if taken else "fell back"
        for n in covered:
            self._usable[n] = usable[self._night_dates[n]]
        total, prefix = 0.0, [0.0]
        for value in self._usable:
            total += value
            prefix.append(total)
        self._usable_prefix = prefix
        if eventful:
            self.log(f"planner: forecast outlook {status} " + " ".join(
                f"{self._night_dates[n][5:]}:{self._usable[n]:.2f}" + (f"(rule {rule[self._night_dates[n]]:.2f})"
                                                                         if abs(self._usable[n] - rule[self._night_dates[n]]) >= 0.05 else "")
                for n in covered))
        self.trace.write({"event": "forecast_outlook", "status": status, "usable": usable, "rule": rule})

    def _close_outlook_night(self, night: int) -> None:
        """Book what a finished night was really worth under the event kinds forecast for
        it, so that later forecasts are read against this card's own record."""
        start, end = self.state.nights[night]
        fraction = min(1.0, self._night_good.pop(night, 0) / max(1.0, (end - start).total_seconds()))
        for key in (self._forecast_keys.get(night) or {"quiet"}):
            seen = self._kind_stats.setdefault(key, [0.0, 0])
            seen[0] += fraction
            seen[1] += 1

    # -- instrument fault reporting -------------------------------------------------
    #
    # A fault lowers instrument efficiency from some moment until it is reported; nothing
    # announces it. The detector works night by night on the median "clean ratio"
    # (observed / modelled quality of exposures with no weather notice in the way):
    #   * the median is first divided by that night's sky level, estimated from the program
    #     bands the exposures really scored in (the band ignores instrument efficiency), so
    #     ordinary night-to-night weather does not look like a fault;
    #   * outside an earthquake sequence, the result is compared with the median of up to
    #     FAULT_REF_NIGHTS earlier quiet nights;
    #   * inside one, efficiency recovers night by night, so it is compared with the level
    #     the fitted recovery predicts for tonight -- a night clearly under that curve
    #     cannot be the earthquake.
    # Enough consecutive low nights -> report at the very next decision, day or night.
    # A clear-cut drop is reported by rule. A borderline one (between FAULT_STRONG and the
    # base threshold) is put to the model together with each night's two-hour profile; the
    # model may report at once or postpone at most FAULT_MAX_LLM_WAITS times, and without
    # a model verdict the rule waits for one extra low night. After a false report the
    # rule tightens; once the free allowance is spent and a paid report failed too, it
    # stops reporting.

    def _night_id(self, night_index: int) -> tuple[str, str]:
        date = (self.state.nights[night_index][0] - timedelta(hours=12)).date().isoformat()
        return "N" + date.replace("-", ""), date

    def _note_bulletin(self, bulletin: dict) -> None:
        night_id = bulletin.get("night_id")
        if not night_id:
            return
        kinds = self._night_kinds.setdefault(night_id, set())
        for notice in bulletin.get("notices") or []:
            kind = notice.get("event_kind")
            if not kind or kind == "terrain_obstruction":
                continue
            kinds.add(kind)
            if kind == "earthquake" and night_id not in self._quake_onset:
                try:
                    issued = parse_utc(bulletin["issued_at_utc"])
                    self._quake_onset[night_id] = (issued - self.state.survey_start).total_seconds() / 3600.0
                except (KeyError, TypeError, ValueError):
                    self._quake_onset[night_id] = float("-inf")

    def _fault_mode(self) -> tuple[float, int]:
        # A false fast probe says the step detector met weather; it says nothing against the
        # nightly evidence, so only this detector's own false reports raise its bar -- unless
        # the free allowance is gone, when any false report from now on costs points.
        if self.false_reports >= self.state.false_report_free_allowance:
            return FAULT_AFTER_ALLOWANCE
        if self.false_reports - self._fast_false > 0:
            return FAULT_AFTER_FALSE
        return FAULT_BASE

    def _quiet_pool_size(self) -> int:
        return sum(1 for r in self.fault_nights if r["level"] is not None and not r["quake"] and r["accepted"])

    def _quiet_ref(self):
        """Median sky-corrected level of the last few quiet (no earthquake) nights that
        were not suspected. Right after a correct report the night records start over; the
        healthy level from before the fault stands in until new quiet nights exist, so an
        earthquake that follows straight away still has a reference."""
        pool = [r["level"] for r in self.fault_nights
                if r["level"] is not None and not r["quake"] and r["accepted"]]
        return _median(pool[-FAULT_REF_NIGHTS:]) if pool else self._quiet_fallback

    def _sky_level(self, checks: list, quiet: float):
        """Tonight's sky relative to the quiet reference, from the program bands the
        exposures really scored in. The band is a pure sky quantity (instrument efficiency
        is not in it), so this separates "the sky got worse" from "the instrument got
        worse". Every candidate level is scored by how many band checks it explains; of the
        best-explaining levels the one nearest 1.0 is returned, so without evidence to the
        contrary the sky counts as normal. None when there are too few checks."""
        if len(checks) < FAULT_SKY_CHECKS_MIN:
            return None
        band = self.state.scoring.program_band
        best, best_levels = -1, []
        for level in FAULT_SKY_GRID:
            scale = level * quiet / 0.95
            agree = sum(1 for program, matched, model in checks if (band(scale * model) == program) == matched)
            if agree > best:
                best, best_levels = agree, [level]
            elif agree == best:
                best_levels.append(level)
        return min(best_levels, key=lambda level: abs(level - 1.0))

    def _night_profile(self, night_index: int, expected: float) -> list:
        """The night in two-hour blocks: [hours after night start, level / expected, samples].
        An instrument fault is one step that then stays flat; weather wanders."""
        start_hours = (self.state.nights[night_index][0] - self.state.survey_start).total_seconds() / 3600.0
        blocks: dict[int, list] = {}
        for when, ratio, bound in self.state.night_samples(night_index):
            blocks.setdefault(int(max(0.0, when - start_hours) // 2), []).append((ratio, bound))
        return [[2 * block, round(_censored_median(samples) / expected, 2), len(samples)]
                for block, samples in sorted(blocks.items()) if len(samples) >= 5]

    def _expected_level(self, night_index: int, quake: bool, quiet):
        """The sky-corrected level a healthy instrument should reach tonight, or None when
        tonight cannot be judged (no reference yet, or the first night of a sequence)."""
        if not quake:
            return quiet
        if self._quake_quiet is None or self._quake_first is None:
            return None
        # Always measured from the start of the curve, never re-anchored on last night: a
        # fault that arrives during the sequence must not be absorbed one night at a time.
        first_night, loss = self._quake_first
        loss = max(0.0, loss) * self._quake_recovery ** max(0, night_index - first_night)
        return self._quake_quiet * (1.0 - loss)

    @staticmethod
    def _fault_streak(records: list, threshold: float) -> list:
        """Ratios of the trailing run of low nights (the ratio is already corrected for the
        sky). Nights that were not judged (too few samples, no reference) neither extend
        nor break the run; a judged night above the threshold ends it."""
        run: list[float] = []
        for record in reversed(records):
            ratio = record["ratio"]
            if ratio is None:
                continue
            if ratio > threshold:
                break
            run.append(ratio)
        return run

    @staticmethod
    def _fault_tier(run: list, mode: tuple) -> str:
        """"none" (not enough low nights), "strong" (report by rule) or "mid" (borderline:
        only in the base mode, and only when the counted nights AVERAGE above FAULT_STRONG --
        one deep night next to one just over the line is a fault, not a borderline case)."""
        threshold, needed = mode
        if len(run) < needed:
            return "none"
        if mode == FAULT_BASE and sum(run[:needed]) / needed > FAULT_STRONG:
            return "mid"
        return "strong"

    def _maybe_report(self, night_index, hours: float, payload: dict):
        """Called at every decision. Settles the previous report, closes a night that has
        just ended, and returns a `report` action when one is due."""
        state = self.state
        last_result = payload.get("last_result") or {}
        if self._awaiting_report_result and last_result.get("action") == "report":
            self._awaiting_report_result = False
            self._settle_report(bool(last_result.get("correct")), payload)
        if self._fault_open_night is not None and night_index != self._fault_open_night:
            self._close_fault_night(self._fault_open_night, payload)
            self._close_outlook_night(self._fault_open_night)
            self._fault_open_night = None
        if night_index is not None and self._fault_open_night is None:
            self._fault_open_night = night_index
            self._forced_tonight = 0
        if self._report_due is None and night_index is not None and not self._awaiting_report_result:
            reason = self._fast_fault_check(hours)
            if reason:
                self._report_due = reason
                self._fast_pending = True
                self.fast_reports += 1
        if self._report_due is None:
            return None
        reason, self._report_due = self._report_due, None
        self.reports += 1
        self._awaiting_report_result = True
        self._last_report_hours = hours
        self.log(f"planner: reporting instrument fault at {payload.get('now_utc')} ({reason})")
        return {"action": "report", "reason": reason,
                "decision_source": "rule+llm" if "model agrees" in reason else "rule"}

    def _fast_fault_check(self, hours: float):
        """A reason string when the clean quality level has stepped down to a plateau low
        enough to spend a free report on it, else None. See FAST_* above."""
        state = self.state
        if self._fast_false > 0 or self.false_reports >= state.false_report_free_allowance:
            # one free probe per stretch between correct reports: the other free one stays with
            # the nightly detector, which a wasted allowance would otherwise slow down for days
            return None
        if hours - self._last_report_hours < FAST_MIN_SPACING_HOURS:
            return None
        if hours - self._quake_onset_hours < FAST_QUAKE_HOLD_HOURS:
            return None
        history = state.clean_history
        if len(history) < 2 * FAST_MIN_SAMPLES:
            return None
        # group the clean samples by exposure: the hits of one exposure share its time
        exposures = []                                 # (when, night, [(ratio, bound)])
        for when, night, ratio, bound in history[-600:]:
            if exposures and exposures[-1][0] == when:
                exposures[-1][2].append((ratio, bound))
            else:
                exposures.append((when, night, [(ratio, bound)]))
        last_when, last_night = exposures[-1][0], exposures[-1][1]
        split = len(exposures) - FAST_WINDOW_EXPOSURES
        if split < 1:
            return None
        window, before = exposures[split:], exposures[max(0, split - FAST_REF_EXPOSURES):split]
        if any(e[1] != last_night for e in window) or last_when - window[0][0] < FAST_MIN_SPAN_HOURS:
            return None
        window_samples = [x for e in window for x in e[2]]
        reference = [x for e in before for x in e[2]]
        if len(window_samples) < FAST_MIN_SAMPLES or len(reference) < FAST_MIN_SAMPLES:
            return None
        now_level = _censored_median(window_samples)
        if self._fast_block is not None:
            if now_level >= FAST_RECOVER * self._fast_block:
                self.log(f"planner: fast fault probe re-armed (level {now_level:.2f} back near {self._fast_block:.2f})")
                self._fast_block = None
            return None
        ref_level = _censored_median(reference)
        if ref_level <= 0.0:
            return None
        per_exposure = [_censored_median(e[2]) for e in window]
        if max(per_exposure) >= FAST_STEP * ref_level or now_level < FAST_FLOOR * ref_level:
            return None
        ref_start, ref_end = before[0][0] - 1.0, before[-1][0]
        sky_now = self._band_level([(p, m, model) for when, p, m, model in state.band_events if ref_end < when <= last_when])
        sky_ref = self._band_level([(p, m, model) for when, p, m, model in state.band_events if ref_start <= when <= ref_end])
        if sky_now is not None and sky_ref is not None and sky_now < FAST_BAND_VETO * sky_ref:
            if self._fast_veto_logged != round(last_when, 2):
                self._fast_veto_logged = round(last_when, 2)
                self.log(f"planner: fast fault probe withheld: the sky band fell too ({sky_now:.2f} vs {sky_ref:.2f}), weather")
            return None
        self._fast_ref_level = ref_level
        return (f"fast: clean level {now_level:.2f} for {last_when - window[0][0]:.2f} h over {len(window)} exposures, "
                f"{now_level / ref_level:.0%} of the {ref_level:.2f} before; free reports left "
                f"{state.false_report_free_allowance - self.false_reports}")

    def _band_level(self, checks: list):
        """Sky level (the factor on a target's quality model) that best explains a set of
        band checks (declared program, scored in it, quality model); the middle of the
        best-explaining range. None with fewer than FAST_BAND_MIN_CHECKS checks."""
        if len(checks) < FAST_BAND_MIN_CHECKS:
            return None
        band = self.state.scoring.program_band
        best, values = -1, []
        for step in range(66):
            value = 0.30 + 0.02 * step
            agree = sum(1 for program, matched, model in checks if (band(value * model) == program) == matched)
            if agree > best:
                best, values = agree, [value]
            elif agree == best:
                values.append(value)
        return values[len(values) // 2]

    def _settle_report(self, correct: bool, payload: dict) -> None:
        fast, self._fast_pending = self._fast_pending, False
        self.log(f"planner: report at {payload.get('now_utc')} was {'correct' if correct else 'FALSE'}"
                 + (" (fast)" if fast else ""))
        self._llm_waits = 0
        if correct:
            self._fast_block = None
            self.correct_reports += 1
            self.false_reports = 0
            self._fast_false = 0
            self._quiet_fallback = self._quiet_ref() or self._quake_quiet or self._quiet_fallback
            self.state.forget_quality_history()   # the instrument is repaired: learn the level again
            self.fault_nights = []
            self._fault_floor = 0
            self._quake_first = None              # restart the recovery curve on the next judged night
            return
        self.false_reports += 1
        if fast:
            self._fast_false += 1
            # the step was weather or something else a report cannot repair: wait until the
            # level comes back before probing again; the nightly records are not touched
            self._fast_block = self._fast_ref_level
            return
        # The low level was real and not a fault: accept it as the new normal.
        counted = 0
        for record in reversed(self.fault_nights[self._fault_floor:]):
            if counted >= self._report_streak:
                break
            if record["ratio"] is not None:
                record["accepted"] = True
                counted += 1
        last = self.fault_nights[-1] if self.fault_nights else None
        if last and last["quake"] and last["level"] is not None and self._quake_quiet:
            self._quake_first = (last["index"], 1.0 - last["level"] / self._quake_quiet)
        self._fault_floor = len(self.fault_nights)

    def _close_fault_night(self, night_index: int, payload: dict) -> None:
        state = self.state
        night_id, date = self._night_id(night_index)
        kinds = self._night_kinds.pop(night_id, set())
        quake = "earthquake" in kinds
        quiet = self._quiet_ref()
        record = {"index": night_index, "date": date, "m": None, "c": 0, "sky": None, "level": None,
                  "ref": None, "ratio": None, "quake": quake, "sky_ok": True, "kinds": sorted(kinds),
                  "accepted": True, "profile": []}

        if quake and self._quake_quiet is None:
            self._quake_quiet = quiet                  # sequence begins: freeze the quiet reference
            self._quake_first = None
            self._quake_recovery = FAULT_RECOVERY_RANGE[1]
        elif not quake:
            self._quake_quiet = self._quake_first = None
        first_of_sequence = quake and self._quake_first is None
        since = self._quake_onset.pop(night_id, None) if first_of_sequence else None
        ratios = state.night_ratios(night_index, since)
        record["c"] = len(ratios)
        sky_base = self._quake_quiet if quake else quiet

        if len(ratios) >= FAULT_MIN_SAMPLES:
            record["m"] = _censored_median(ratios)
            sky = self._sky_level(state.night_band_checks(night_index), sky_base) if sky_base else None
            record["sky"] = sky
            record["sky_ok"] = sky is None or sky >= FAULT_SKY_OK
            record["level"] = record["m"] / (sky or 1.0)
            record["ref"] = self._expected_level(night_index, quake, quiet)
            if record["ref"]:
                record["ratio"] = record["level"] / record["ref"]
        if record["ref"]:
            record["profile"] = self._night_profile(night_index, record["ref"] * (record["sky"] or 1.0))
        low = record["ratio"] is not None and record["ratio"] <= FAULT_BASE[0]
        if low and not quake and self.correct_reports == 0 and self._quiet_pool_size() < FAULT_REF_MIN_NIGHTS:
            # one or two nights are no reference: a single misjudged sky level on either of
            # them moves it by a third. Such a night is booked as ordinary, not suspected.
            # (After a repair the healthy level from before the fault stands in instead.)
            low = False
            record["ratio"] = None
        record["accepted"] = not low
        if quake and record["level"] is not None and self._quake_quiet and not low:
            loss = 1.0 - record["level"] / self._quake_quiet
            if self._quake_first is None:
                self._quake_first = (night_index, loss)
            elif night_index > self._quake_first[0] and self._quake_first[1] > 0.05 and loss > 0.0:
                factor = (loss / self._quake_first[1]) ** (1.0 / (night_index - self._quake_first[0]))
                self._quake_recovery = min(FAULT_RECOVERY_RANGE[1], max(FAULT_RECOVERY_RANGE[0], factor))
        self.fault_nights.append(record)

        mode = self._fault_mode()
        run = self._fault_streak(self.fault_nights[self._fault_floor:], mode[0])
        fmt = lambda v: "n/a" if v is None else f"{v:.2f}"  # noqa: E731
        self.log(f"planner: night {date} fault-check m={fmt(record['m'])} ref={fmt(record['ref'])} "
                 f"quake={quake} sky_ok={record['sky_ok']} streak={len(run)} "
                 f"c={record['c']} sky={fmt(record['sky'])} ratio={fmt(record['ratio'])}"
                 + (f" recovery={self._quake_recovery:.2f}" if quake else ""))
        self.trace.write({"event": "fault_check", **record, "streak": len(run)})
        if night_index < 3 or night_index % 10 == 0 or night_index == len(state.nights) - 1:
            if state.ridges:
                self.log("planner: terrain " + "; ".join(r.summary() for r in state.ridges.values()))

        tier = self._fault_tier(run, mode)
        if not run:
            self._llm_waits = 0
        if tier == "none" or record["ratio"] is None:
            return                                # no alarm, or tonight added no evidence to an old one
        if self.false_reports > state.false_report_free_allowance:
            # The free false reports are used up and one paid attempt has already failed:
            # whatever this is, the detector cannot tell it from a fault. Stop paying.
            self.log(f"planner: night {date} fault alarm suppressed "
                     f"({self.false_reports} false reports since the last correct one)")
            return
        self._report_streak = len(run)
        summary = f"{len(run)} low nights, latest {run[0]:.0%} of reference"
        if tier == "strong":
            self._report_due = f"rule: {summary}"
            return
        # Borderline. The rule waits for one more low night than a clear-cut drop needs and
        # then reports. The model is asked every night and has a veto, not a trigger: it
        # can hold the alarm back, at most FAULT_MAX_LLM_WAITS nights running, after which
        # the rule reports whatever the model says; its "report" does not bring the report
        # forward.
        needed = mode[1]
        if len(run) >= needed + FAULT_MAX_LLM_WAITS:
            self._report_due = f"rule (borderline, {len(run)} nights): {summary}"
            return
        verdict = None
        if not self.llm.disabled and self._llm_waits < FAULT_MAX_LLM_WAITS:
            verdict = self._arbitrate(payload)
        if verdict == "wait_one_night":
            self._llm_waits += 1
        elif verdict is None and self._llm_waits > 0 and not self.llm.disabled:
            # the model has been holding this alarm back and could not be reached tonight:
            # one failed call is no reason to overrule it, but it does use up one of its waits
            self._llm_waits += 1
            self.log(f"planner: night {date} borderline alarm kept on hold (model unreachable, "
                     f"{self._llm_waits} of {FAULT_MAX_LLM_WAITS} waits used)")
        elif len(run) > needed:
            # the rule's own condition (one low night more than a clear-cut drop needs) is
            # met and the model does not object
            agreed = "model agrees" if verdict == "report" else "no model verdict"
            self._report_due = f"rule (borderline, {agreed}, {len(run)} nights): {summary}"
        elif verdict == "report":
            # The model can hold an alarm back but not bring one forward: of its five
            # "report" verdicts on a two-night borderline drop four were weather. The rule
            # still wants its extra low night, and the model is asked again then.
            self.log(f"planner: night {date} model would report; the rule waits for one more low night")

    def _arbitrate(self, payload: dict):
        """Borderline alarm: one model call decides between reporting now and waiting one
        more night. Returns "report", "wait_one_night", or None when the call failed (the
        rule then waits for one extra low night)."""
        if not self.real_slack_allows(self.llm.call_timeout_seconds):
            self.log("planner: fault arbitration skipped (real wall clock); the rule decides")
            return None
        rounded = lambda v: None if v is None else round(v, 3)  # noqa: E731
        nights = [{"date": r["date"], "ratio": rounded(r["ratio"]), "c": r["c"], "sky": rounded(r["sky"]),
                   "quake": r["quake"], "notice_kinds": r["kinds"], "profile": r["profile"]}
                  for r in self.fault_nights[-6:]]
        answer = self.llm.ask_json(
            FAULT_SYSTEM_PROMPT,
            {"last_nights": nights, "false_reports_since_last_correct": self.false_reports,
             "free_false_reports_allowed": self.state.false_report_free_allowance,
             "times_already_postponed": self._llm_waits},
            float((payload.get("wallclock") or {}).get("remaining_seconds", 0)),
        )
        verdict = answer.get("verdict") if isinstance(answer, dict) else None
        if verdict not in ("report", "wait_one_night"):
            self.log("planner: fault arbitration unavailable; the rule decides")
            return None
        self.log(f"planner: fault arbitration verdict={verdict} why={str(answer.get('why') or '')[:100]!r}")
        return verdict

    def _force_dark_if_suspicious(self, night_index: int) -> None:
        """While tonight's running median looks low, declare DARK on a few exposures: a
        DARK declaration that scores as DARK shows the sky is fine (so the instrument is
        not), one that does not shows the sky is to blame."""
        state = self.state
        state.force_program = None
        if self._forced_tonight >= FAULT_MAX_FORCED_PER_NIGHT:
            return
        night_id, _ = self._night_id(night_index)
        quake = "earthquake" in self._night_kinds.get(night_id, ())
        if quake and self._quake_first is None:
            return
        expected = self._expected_level(night_index, quake, self._quiet_ref())
        if not expected:
            return
        ratios = state.night_ratios(night_index)
        if len(ratios) < FAULT_MIN_SAMPLES or _censored_median(ratios) > self._fault_mode()[0] * expected:
            return
        dark = sum(1 for program, _matched, _model in state.night_band_checks(night_index) if program == "DARK")
        if dark < FAULT_DARK_CHECKS_WANTED:
            state.force_program = "DARK"

    # -- planning value / achievability -----------------------------------------

    def _direction_factor(self, alt: float, az: float, terrain_floor: float = 0.0) -> float:
        state = self.state
        # Terrain: the chance the point is clear of every announced ridge, from what our own
        # hits have shown so far (state.Ridge). Unknown at first, so low pointings towards a
        # ridge are tried where they are worth it, and each result tightens the bounds.
        # `terrain_floor` keeps a target the card insists on in play even where a ridge is
        # likely: a wrongly learned ridge would otherwise cost 50 points per target with no
        # way back, while probing one costs a fraction of one exposure.
        factor = max(state.terrain_clear_probability(alt, az), terrain_floor)
        if factor <= 0.0:
            return 0.0
        for direction in state.terrain:
            if direction not in DIRECTION_AZ and alt < 50.0:
                return 0.0                    # unknown direction label: stay out low down
        for key in state.notices:
            kind, _, direction = key.partition("|")
            if direction not in DIRECTION_AZ:
                continue
            near = _az_distance(az, DIRECTION_AZ[direction]) <= 67.5
            if kind in BLOCKING_KINDS and near and alt < 62.0:
                return 0.0
            if near and alt < 75.0:
                factor = min(factor, NOTICE_FACTOR)
        for blocked_az, blocked_alt in state.blocked[-40:]:
            if _az_distance(az, blocked_az) <= 12.0 and alt <= blocked_alt + 3.0:
                factor = min(factor, 0.2)
        return factor

    def _request_thresholds(self, active_requests: list) -> dict:
        """target_index -> smallest still-needed completion_factor_threshold, for every
        target that is part of some still-open request (`remaining_count>0`), not
        already counted (`completed_target_ids`), and not already past that threshold.
        Used only as a read-only tie-break in _finish_plan's duration search -- it never
        feeds back into _value/achievable, so it cannot change which pointing gets
        chosen, only (within REQUEST_RATE_TOLERANCE) how long an already-chosen exposure
        runs once request targets happen to already be among its assigned fibres."""
        state = self.state
        thresholds: dict[int, float] = {}
        for request in active_requests:
            if int(request.get("remaining_count", 0)) <= 0:
                continue
            threshold = float(request.get("completion_factor_threshold", 1.0))
            completed = set(request.get("completed_target_ids") or [])
            for target_id in request.get("target_ids", []):
                if target_id in completed:
                    continue
                i = state.index_of.get(target_id)
                if i is None or state.factor[i] >= threshold:
                    continue
                if i not in thresholds or threshold < thresholds[i]:
                    thresholds[i] = threshold
        return thresholds

    def _request_reach(self, i: int, lst: float, moon, time_left: float, threshold: float):
        """(seconds needed, seconds the target stays up) if target i can clear `threshold`
        with margin in one exposure starting now, else None."""
        state, scoring = self.state, self.state.scoring
        h = state.hmax[i]
        ha = wrap180(lst - state.ra[i])
        if h <= 0.0 or not (-h <= ha <= h):
            return None
        up = (h - ha) / SIDEREAL_DEG_PER_SECOND if h < 180 else 1e9
        alt, az = radec_to_altaz(state.ra[i], state.dec[i], lst, state.lat)
        if self._direction_factor(alt, az) < 1.0:
            return None
        model = scoring.quality_model(alt, lunar_factor(moon, state.ra[i], state.dec[i], scoring.lunar_model)) or 0.0
        k = (state.flux[i] * model * self._quality_scale(alt, az) * PLAN_FACTOR_SAFETY) / scoring.f0t0
        if k <= 0.0:
            return None
        needed = threshold * REQUEST_MARGIN / k
        if needed > min(REQUEST_MAX_SECONDS, up, time_left, state.max_exposure):
            return None
        return needed, up

    def _request_reachable_later(self, i: int, lst: float, threshold: float, deadline, night_index: int,
                                 seconds_left: float) -> bool:
        """Optimistic check: could target i still clear the threshold later in the window
        (later tonight while it is still rising, or on a later night before the deadline)?"""
        state, scoring = self.state, self.state.scoring
        h = state.hmax[i]
        if h <= 0.0:
            return False
        best_alt = min(89.0, 90.0 - abs(state.dec[i] - state.lat))
        k = (state.flux[i] * (scoring.quality_model(best_alt, 1.0) or 0.0) * state.scale * PLAN_FACTOR_SAFETY) / scoring.f0t0
        if k <= 0.0 or threshold * REQUEST_MARGIN / k > REQUEST_MAX_SECONDS:
            return False
        ha = wrap180(lst - state.ra[i])
        if ha < 0.0 and (-ha - min(h, -ha)) / SIDEREAL_DEG_PER_SECOND + REQUEST_MAX_SECONDS < seconds_left:
            return True                              # still east of the meridian with time left tonight
        return any(start < deadline and state.first_night[i] <= n <= state.last_night[i]
                   and self._usable[n] >= OUTLOOK_LOST_NIGHT        # a night forecast lost is not "later"
                   for n, (start, _end) in enumerate(state.nights) if n > night_index)

    def _pick_request_target(self, now, lst: float, moon, seconds_left: float, night_index: int):
        """The request target to anchor the next exposure on, or None. A request is pursued
        only while (targets reachable now) + (targets reachable later) covers what it still
        needs; among the targets reachable now the cheapest goes first."""
        state = self.state
        best = None  # (seconds needed, target index, threshold, seconds to deadline, request_id)
        for request in self._active_requests:
            need = int(request.get("remaining_count", 0))
            if need <= 0:
                continue
            request_id = request.get("request_id")
            try:
                deadline = parse_utc(request["deadline_utc"])
                threshold = float(request["completion_factor_threshold"])
                reward = float(request.get("completion_reward", 0.0))
            except (KeyError, TypeError, ValueError):
                continue
            if reward <= 0.0:
                continue                              # nothing to gain: the message is only a note
            deadline_left = (deadline - now).total_seconds()
            completed = set(request.get("completed_target_ids") or [])
            now_ok, later = [], 0
            for target_id in request.get("target_ids", []):
                i = state.index_of.get(target_id)
                if i is None or target_id in completed or \
                        self._request_tries.get((request_id, i), 0) >= REQUEST_MAX_TRIES:
                    continue
                reach = self._request_reach(i, lst, moon, min(seconds_left, deadline_left), threshold)
                if reach is not None:
                    now_ok.append((reach[0], i))
                elif self._request_reachable_later(i, lst, threshold, deadline, night_index, seconds_left):
                    later += 1
            status = "pursue" if len(now_ok) + later >= need else "drop"
            if self._request_status.get(request_id) != status:
                self._request_status[request_id] = status
                self.log(f"planner: request {request_id} {status}: needs {need} more, "
                         f"{len(now_ok)} reachable now, {later} later")
            if status == "pursue" and now_ok:
                needed, i = min(now_ok)
                if best is None or needed < best[0]:
                    best = (needed, i, threshold, deadline_left, request_id)
        return best

    def _value(self, i: int) -> float:
        """Planning value of fully completing target i from here (ignores how much
        exposure is achievable tonight)."""
        state = self.state
        f = state.factor[i]
        damp = 0.6 ** state.misses[i]
        threshold = state.scoring.required_threshold
        if self.incremental_scores:
            top = self._top_multiplier
            gain = max(0., state.weight[i]*top - state.best_score[i])
            if state.required[i] and f < threshold:
                gain += REQUIRED_BONUS
            return gain*damp if gain > .02*state.weight[i] else 0.
        if state.required[i]:
            if f >= threshold:
                return state.weight[i] * max(0.0, 1.0 - f * f) * damp
            return (state.weight[i] * (1.0 - f * f) + REQUIRED_BONUS * (1.0 if f < 0.5 else 0.35)) * damp
        return 0.0 if f >= DONE_FACTOR else state.weight[i] * (1.0 - f * f) * damp

    # -- main planning pass -------------------------------------------------------

    def _quality_scale(self, alt, az):
        # Use a representative 900s weather factor for the joint duration scan;
        # exact per-duration inference was slower and did not improve four-card score.
        if self.conditions is None:
            return self.state.scale
        return self.conditions.predict(alt, az, 900).effective_scale

    def plan(self, now, night_end, night_index: int, hours: float, allow_request: bool = True):
        state = self.state
        state.update_scale(hours)
        lst = local_sidereal_deg(now, state.lon)
        horizon = min(night_end, state.survey_end)
        seconds_left = (horizon - now).total_seconds()
        if seconds_left < state.min_exposure:
            return None
        min_visible = min(MIN_VISIBLE_SECONDS, seconds_left) * SIDEREAL_DEG_PER_SECOND

        # The whole catalogue is scanned only every POOL_REFRESH_HOURS: in between, the
        # candidates can only come from the targets that were up, or about to rise, at
        # the last scan.
        if self._pool is None or self._pool_night != night_index or not (0.0 <= hours - self._pool_hours <= POOL_REFRESH_HOURS):
            still_active, pool = [], []
            rise_margin = (POOL_REFRESH_HOURS * 3600.0 + 600.0) * SIDEREAL_DEG_PER_SECOND
            for i in state.active:
                if self._value(i) <= 0.0:
                    continue
                still_active.append(i)
                h = state.hmax[i]
                if -h - rise_margin <= wrap180(lst - state.ra[i]) <= h:
                    pool.append(i)
            state.active = still_active
            self._pool, self._pool_night, self._pool_hours = pool, night_index, hours
        candidates: list[tuple[float, int]] = []
        best_scale = max(state.scale, state.prior_scale) * PLAN_FACTOR_SAFETY / state.scoring.f0t0
        longest = float(state.max_exposure)
        self._pool_phase += 1
        scan = self._pool if self._pool_stride <= 1 else self._pool[self._pool_phase % self._pool_stride::self._pool_stride]
        for i in scan:
            v = self._value(i)
            if v <= 0.0:
                continue
            ha = wrap180(lst - state.ra[i])
            h = state.hmax[i]
            if -h <= ha <= h - min_visible:
                nights_left = max(1, state.last_night[i] - night_index + 1)
                # Rank by what the target can pay per second at its best, not by how much of
                # it is left: the seeds of the joint search are taken from the head of this
                # list, and a list headed by untouched faint targets -- many of which cannot
                # be improved at all -- keeps the cheap, finishable ones from ever seeding a
                # pointing. A target that even at its best cannot beat what it already has
                # is dropped.
                k_best = state.flux[i] * self._model_best[i] * best_scale
                if k_best <= 0.0:
                    continue
                f = state.factor[i]
                cap = min(1.0, k_best * longest)
                if state.required[i] and f < 0.5:
                    worth = v
                elif (cap - f <= 0.02 if not self.incremental_scores else
                      state.weight[i]*self._top_multiplier*cap
                      - state.best_score[i] <= .02*state.weight[i]):
                    continue
                else:
                    worth = (state.weight[i]*self._top_multiplier*cap
                             - state.best_score[i] if self.incremental_scores else state.weight[i]*(cap-f))
                seconds = max(180.0, min(longest, cap / k_best))
                worth *= self._pressure[self._bin_of[i]] ** PRESSURE_GAMMA
                candidates.append((worth / seconds * (1.0 + 2.0 / nights_left), i))
        if not candidates:
            return None
        candidates.sort(key=lambda t: -t[0])

        moon = Moon(now + timedelta(seconds=450), lst, state.lat)
        altaz_cache: dict[int, tuple[float, float]] = {}

        def altaz(i: int) -> tuple[float, float]:
            cached = altaz_cache.get(i)
            if cached is None:
                cached = radec_to_altaz(state.ra[i], state.dec[i], lst, state.lat)
                altaz_cache[i] = cached
            return cached

        visible = {i for _, i in candidates}
        achievable_cache: dict[int, float] = {}
        scoring = state.scoring

        def achievable(i: int) -> float:
            cached = achievable_cache.get(i)
            if cached is not None:
                return cached
            alt, az = altaz(i)
            lunar = lunar_factor(moon, state.ra[i], state.dec[i], scoring.lunar_model)
            model = scoring.quality_model(alt, lunar) or 0.0
            k = (state.flux[i] * model * self._quality_scale(alt, az) * PLAN_FACTOR_SAFETY) / scoring.f0t0
            ha = wrap180(lst - state.ra[i])
            up = (state.hmax[i] - ha) / SIDEREAL_DEG_PER_SECOND if state.hmax[i] < 180 else 1e9
            reach = min(1.0, k * min(state.max_exposure, up, seconds_left))
            f = state.factor[i]
            gain = state.weight[i] * max(0.0, reach * reach - f * f)
            if state.required[i] and f < 0.5 and reach >= 0.5:
                gain += REQUIRED_BONUS
            damp = (0.6 ** state.misses[i]) * (0.7 ** state.attempts[i])
            result = gain * damp * self._direction_factor(alt, az, self._terrain_floor(i))
            achievable_cache[i] = result
            return result

        anchors: list[tuple[float, int]] = []
        request = self._pick_request_target(now, lst, moon, seconds_left, night_index) \
            if allow_request and self._hours_now >= self._request_hold_until else None
        forced = request[1] if request is not None else None
        if forced is not None:
            anchors = [(1.0, forced)]
        else:
            for checked, (priority, i) in enumerate(candidates):
                if checked >= ANCHOR_POOL and len(anchors) >= 3 * ANCHORS:
                    break
                weighted = achievable(i) * priority / max(1e-9, self._value(i))
                if weighted > 0:
                    anchors.append((weighted, i))
        if not anchors:
            return None
        anchors.sort(key=lambda t: -t[0])

        fibers = self.centre_fibres if state.fast_level < 2 else self.fast_fibres
        if forced is None:
            joint = self._joint_search(candidates, altaz, visible, moon, lst, seconds_left, fibers, night_index)
            if joint is not None:
                c_alt, c_az, chosen, duration = joint
                return self._finish_plan(now, lst, c_alt, c_az, chosen, seconds_left, moon, altaz, hours,
                                         night_index, fixed_duration=duration)
        n_anchors = 1 if state.fast_level >= 1 else ANCHORS
        best = None  # (total, c_alt, c_az, chosen)
        tried = 0
        for _, anchor in anchors:
            if tried >= n_anchors and best is not None:
                break
            if tried >= n_anchors + 8:
                break
            tried += 1
            a_alt, a_az = altaz(anchor)
            near = [j for j in state.neighbours(state.ra[anchor], state.dec[anchor], self.neighbour_radius) if j in visible]
            near_values = {j: achievable(j) for j in near}
            for fiber in fibers:
                d_north, d_east = self.grid.fiber_center(fiber)
                c_alt, c_az = shift_altaz(a_alt, a_az, -d_north, -d_east)
                if not (state.min_alt + 1.5 <= c_alt <= 89.0):
                    continue
                c_alt = round(c_alt, 4)
                c_az = round(c_az, 4) % 360.0
                chosen: dict[int, tuple[float, int, float]] = {}  # fiber -> (score, j, margin)
                for j, v in near_values.items():
                    if v <= 0.0:
                        continue
                    alt, az = altaz(j)
                    offsets = tangent_offsets(alt, az, c_alt, c_az)
                    if offsets is None:
                        continue
                    fib, margin = self.grid.classify(*offsets)
                    if fib is None:
                        continue
                    score = v * (1.0 if margin >= self.edge_margin * (1 + 1.5 * state.misses[j]) else 0.4)
                    existing = chosen.get(fib)
                    if existing is None or score > existing[0]:
                        chosen[fib] = (score, j, margin)
                if forced is not None:
                    # the request target keeps the fibre it was centred on, whatever else
                    # would have scored higher in that cell
                    offsets = tangent_offsets(a_alt, a_az, c_alt, c_az)
                    fib, margin = self.grid.classify(*offsets) if offsets is not None else (None, 0.0)
                    if fib is None or margin < self.edge_margin:
                        continue
                    chosen[fib] = (0.0, forced, margin)
                if not chosen:
                    continue
                total = sum(score for score, _, _ in chosen.values())
                if best is None or total > best[0]:
                    best = (total, c_alt, c_az, chosen)
        if best is None:
            return self.plan(now, night_end, night_index, hours, allow_request=False) if forced is not None else None
        _, c_alt, c_az, chosen = best
        self._forced_request = (forced, request[2], request[3], request[4]) if forced is not None else None
        action = self._finish_plan(now, lst, c_alt, c_az, chosen, seconds_left, moon, altaz, hours, night_index)
        self._forced_request = None
        if forced is not None:
            if action is None:
                return self.plan(now, night_end, night_index, hours, allow_request=False)
            key = (request[4], forced)
            self._request_tries[key] = self._request_tries.get(key, 0) + 1
            self._last_forced = key
            self.request_exposures += 1
        return action

    def _update_pressure(self, now, night_index: int) -> None:
        """Once a night: how oversubscribed is each RA strip from here to the end of the
        survey? Work = exposure-hours its unfinished targets still need at their best,
        with the fibres filled as well as they have been so far. Time = the hours the strip
        stands above PRESSURE_ALT_DEG on the remaining (usable) nights, each hour shared
        among the strips up at that moment in proportion to their work. Pressure is work
        over that share: above 1 the strip cannot be finished at the present pace."""
        state = self.state
        bins = len(self._pressure)
        fill = (self.total_assigned / max(1, self.observe_count)) if self.observe_count >= 20 else 0.6 * min(self.grid.n, 16)
        k_scale = max(0.05, state.prior_scale) * PLAN_FACTOR_SAFETY / state.scoring.f0t0
        work = [0.0] * bins
        for i in state.active:
            f = state.factor[i]
            if f >= DONE_FACTOR:
                continue
            k = state.flux[i] * self._model_best[i] * k_scale
            if k > 0.0:
                work[self._bin_of[i]] += min(state.max_exposure, 1.0 / k) * max(0.0, 1.0 - f)
        work = [w / max(1.0, fill) / 3600.0 for w in work]
        share = [0.0] * bins
        centres = [(b + 0.5) * RA_BIN_DEG for b in range(bins)]
        active_bins = [b for b in range(bins) if work[b] > 0.0 and self._bin_reach[b] > 0.0]
        for n in range(night_index, len(state.nights)):
            start, end = state.nights[n]
            begin = max(start, now)
            seconds = (end - begin).total_seconds()
            if seconds <= 0.0:
                continue
            lst0 = local_sidereal_deg(begin, state.lon)
            samples = max(1, int(seconds // 1800.0))
            dt = seconds / samples / 3600.0 * (self._usable[n] if n > night_index else 1.0)
            for s in range(samples):
                lst = lst0 + (s + 0.5) * (seconds / samples) * SIDEREAL_DEG_PER_SECOND
                up = [b for b in active_bins if abs(wrap180(lst - centres[b])) <= self._bin_reach[b]]
                total = sum(work[b] for b in up)
                if total > 0.0:
                    for b in up:
                        share[b] += dt / total
        low, high = PRESSURE_RANGE
        self._pressure = [1.0 if work[b] <= 0.0 else (high if share[b] <= 0.0 else min(high, max(low, 1.0 / share[b])))
                          for b in range(bins)]
        if night_index % 15 == 0:
            self.log("planner: region pressure " + " ".join(f"{int(b * RA_BIN_DEG)}:{self._pressure[b]:.1f}"
                                                              for b in range(bins) if work[b] > 0.0))

    def _best_hour_angle_left(self, j: int, ha_now: float, night_index: int, seconds_left: float,
                              later_nights: bool) -> float:
        """Smallest |hour angle| (degrees) target j can still be observed at: during the rest
        of tonight, or -- when later nights can be counted on -- on any night up to its last.
        From night to night the hour angle a target has at dusk only grows (about a degree a
        night), so the nights between tomorrow and the last need not be walked: a target
        that is already west of the meridian at dusk tomorrow will never be better than
        tomorrow at dusk, unless the survey is long enough for it to come round again; one
        that is still east of the meridian at dawn gets better until its last night."""
        turn_left = seconds_left * SIDEREAL_DEG_PER_SECOND
        if ha_now <= 0.0 <= ha_now + turn_left:
            return 0.0
        tonight = min(abs(ha_now), abs(ha_now + turn_left))
        tomorrow = night_index + 1
        last = min(self.state.last_night[j], len(self._night_lst0) - 1)
        if not later_nights or tomorrow > last:
            return tonight
        ra = self.state.ra[j]
        start = wrap180(self._night_lst0[tomorrow] - ra)
        end = start + self._night_turn[tomorrow]
        if start <= 0.0 <= end:
            return 0.0
        last_start = wrap180(self._night_lst0[last] - ra)
        last_end = last_start + self._night_turn[last]
        came_round = last_start < start - 5.0          # passed 180 degrees and wrapped
        if start > 0.0:                                 # setting at dusk
            later = start if not came_round else (0.0 if last_end >= 0.0 else min(start, -last_end))
        else:                                           # still rising at dawn
            later = 0.0 if (last_end >= 0.0 or came_round) else -last_end
        return min(tonight, later)

    def _joint_search(self, candidates, altaz, visible, moon, lst, seconds_left, fibers, night_index=0):
        """Pick (pointing, fibre assignment, exposure length) together.

        For every target near an anchor, its gain is tabulated for each exposure length:
        weight x (reach^2 - done^2) -- squared, as in the rest of the planner, because only a
        target's best single exposure counts, so half a target now is worth much less than
        half its score -- plus the required-target bonus once it crosses the threshold. For
        a candidate pointing each fibre then takes, per length, the best target on its
        glass, and the pointing/length pair with the highest total gain per second wins.
        Returns (alt, az, {fibre: (gain, target, margin)}, seconds) or None when nothing has
        a positive rate (the caller then uses the older search, which also knows how to
        take a probing exposure)."""
        state, scoring, grid = self.state, self.state.scoring, self.grid
        durations = sorted({int(max(state.min_exposure, min(state.max_exposure,
                                                            round(base * state.duration_scale / 30.0) * 30)))
                            for base in JOINT_DURATIONS})
        durations = [d for d in durations if d <= seconds_left]
        if not durations:
            return None
        count = len(durations)
        # A scheduling preference, never added to physical observation duration.
        per_second = [1.0 / (d + JOINT_OVERHEAD_SECONDS) for d in durations]
        gains_cache: dict[int, list] = {}
        prefix, defer_power = self._usable_prefix, self._defer_power
        usable_before = prefix[night_index]          # tonight counts in full: it is being used right now
        usable_before -= 1.0 - self._usable[night_index]
        sky_band_scale = self._band_scale(night_index) if self.incremental_scores else None

        def gains(j: int) -> list:
            cached = gains_cache.get(j)
            if cached is not None:
                return cached
            alt, az = altaz(j)
            damp = (0.6 ** state.misses[j]) * (0.7 ** state.attempts[j]) \
                * self._direction_factor(alt, az, self._terrain_floor(j))
            model = scoring.quality_model(alt, lunar_factor(moon, state.ra[j], state.dec[j], scoring.lunar_model)) or 0.0
            k = (state.flux[j] * model * self._quality_scale(alt, az) * PLAN_FACTOR_SAFETY) / scoring.f0t0
            ha = wrap180(lst - state.ra[j])
            up = (state.hmax[j] - ha) / SIDEREAL_DEG_PER_SECOND if state.hmax[j] < 180 else 1e9
            f = state.factor[j]
            weight = state.weight[j]
            needs = state.required[j] and f < 0.5
            # Wait-for-a-better-moment discount: quality here against the best quality this
            # target can still reach from now to the end of the survey -- NOT against its
            # transit. A region that is setting at dusk never comes back to transit: each
            # night it starts lower than the night before, so measured against transit it
            # looks "too low, wait" for ever and is abandoned (official card A: the RA
            # 300-345 strip, a third of the catalogue, was captured at 38-54% and hardly
            # touched after the first month).
            later_ok = prefix[state.last_night[j] + 1] - usable_before >= DEFER_MIN_NIGHTS - 0.5
            best_ha = self._best_hour_angle_left(j, ha, night_index, seconds_left, later_ok)
            sin_alt = self._sin_sin[j] + self._cos_cos[j] * math.cos(math.radians(best_ha))
            best_alt = math.degrees(math.asin(max(-1.0, min(1.0, sin_alt))))
            if best_alt > alt:
                best_model = scoring.quality_model(min(89.0, best_alt), 1.0) or 0.0
                here = scoring.quality_model(alt, 1.0) or 0.0
                if best_model > 0.0:
                    damp *= min(1.0, here / best_model) ** defer_power
            damp *= self._pressure[self._bin_of[j]] ** PRESSURE_GAMMA
            longest = min(state.max_exposure, up)
            unit_rate = state.flux[j] * model * PLAN_FACTOR_SAFETY / scoring.f0t0
            ceiling_k = unit_rate * self._quality_scale(alt, az)
            ceiling = max(1e-6, min(1.0, ceiling_k * longest))
            row = []
            for d in durations:
                if d > up or damp <= 0.0 or k <= 0.0:
                    row.append(0.0)
                    continue
                reach = min(1.0, k * d)
                # (reach - done) x (reach / ceiling): for a target that can be finished this is
                # the usual discount on half-done work; a target too faint to ever finish in
                # one exposure is judged against its own ceiling instead of being written
                # down as permanently half done
                value = weight * max(0.0, reach - f) * (reach / ceiling)
                if self.incremental_scores:
                    band = scoring.program_band(model*sky_band_scale)
                    multiplier = scoring.program_multipliers.get(band, 1.)
                    value = max(0., weight*reach*multiplier-state.best_score[j]) * (reach/ceiling)
                if needs and reach >= 0.5:
                    value += REQUIRED_BONUS
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
                nights_left = max(0.5, prefix[state.last_night[i] + 1] - usable_before)   # usable ones, per the forecast
                seeds.append((rate * (1.0 + 2.0 / nights_left), i))
        if not seeds:
            return None
        seeds.sort(key=lambda s: -s[0])
        n_anchors = 1 if state.fast_level >= 1 else JOINT_ANCHORS

        side, pitch = grid.side, grid.pitch
        half_fov, half_glass, half_side, middle = grid.fov / 2.0, grid.glass / 2.0, grid.side / 2.0, (grid.side - 1) / 2.0
        to_deg = 180.0 / math.pi
        best = None  # (rate, c_alt, c_az, duration index, cells, margins)
        for _, anchor in seeds[:n_anchors]:
            a_alt, a_az = altaz(anchor)
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
                    if depth <= 0.0:
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

    def _band_scale(self, night_index: int) -> float:
        """The factor that turns a target's quality model into the sky quality its program
        band is decided on. The learned exposure scale is the wrong thing to use: it
        contains instrument efficiency (an earthquake or a fault lowers it for weeks), and
        the band does not. So the factor is read directly off tonight's recent band checks
        -- saturated hits whose score shows which band they really fell in: the value that
        explains most of them, nearest to the previous answer. With too few checks the
        learned scale is used, as before."""
        state = self.state
        fallback = state.scale / 0.95
        checks = [c for c in list(state._band_checks)[-BAND_CHECKS:] if c[3] == night_index]
        if len(checks) < BAND_MIN_CHECKS:
            return fallback if self._band_scale_night != night_index else self._band_scale_last
        band = state.scoring.program_band
        best, best_values = -1, []
        for step in range(66):
            value = 0.30 + 0.02 * step
            agree = sum(1 for program, matched, model, _n in checks if (band(value * model) == program) == matched)
            if agree > best:
                best, best_values = agree, [value]
            elif agree == best:
                best_values.append(value)
        anchor = self._band_scale_last if self._band_scale_night == night_index else fallback
        chosen = min(best_values, key=lambda v: abs(v - anchor))
        self._band_scale_last, self._band_scale_night = chosen, night_index
        return chosen

    def _finish_plan(self, now, lst, c_alt, c_az, chosen, seconds_left, moon, altaz, hours, night_index,
                     fixed_duration=None):
        state = self.state
        scoring = state.scoring
        c_ra, c_dec = altaz_to_radec(c_alt, c_az, lst, state.lat)
        c_hmax = max_hour_angle_deg(c_dec, state.lat, state.min_alt + 0.3)
        c_ha = wrap180(lst - c_ra)

        info: dict[int, dict] = {}
        for fiber, (_, j, _margin) in chosen.items():
            alt, az = altaz(j)
            lunar = lunar_factor(moon, state.ra[j], state.dec[j], scoring.lunar_model)
            model = scoring.quality_model(alt, lunar) or 0.0
            ha = wrap180(lst - state.ra[j])
            up = (state.hmax[j] - ha) / SIDEREAL_DEG_PER_SECOND if state.hmax[j] < 180 else 1e9
            k = (state.flux[j] * model * self._quality_scale(alt, az) * PLAN_FACTOR_SAFETY) / scoring.f0t0
            info[fiber] = {"i": j, "alt": alt, "az": az, "model": model, "up": up, "k": k}
        center_up = (c_hmax - c_ha) / SIDEREAL_DEG_PER_SECOND if c_hmax < 180 else 1e9

        best = None  # (rate, duration)
        forced = self._forced_request   # (target index, threshold, seconds to deadline, request_id) or None
        forced_best = None  # (rate, duration): best candidate that completes the pursued request target
        request_best = None  # (rate, duration): best candidate that also clears a needed
                              # request target's completion_factor_threshold
        for base in (DURATIONS if fixed_duration is None else ()):
            duration = round((base * state.duration_scale) / 30.0) * 30
            duration = int(max(state.min_exposure, min(state.max_exposure, duration)))
            if duration > seconds_left or duration > center_up:
                continue
            gain = 0.0
            completes_request = False
            for item in info.values():
                if item["up"] < duration:
                    continue
                reached = min(1.0, item["k"] * duration)
                f = state.factor[item["i"]]
                gain += state.weight[item["i"]] * max(0.0, reached * reached - f * f)
                if state.required[item["i"]] and f < 0.5 and reached >= 0.5:
                    gain += REQUIRED_BONUS
                threshold = self._request_thresholds_now.get(item["i"])
                if threshold is not None and reached >= threshold:
                    completes_request = True
            rate = gain / duration
            if best is None or rate > best[0]:
                best = (rate, duration)
            if completes_request and (request_best is None or rate > request_best[0]):
                request_best = (rate, duration)
            if forced is not None and duration <= forced[2] and any(
                    item["i"] == forced[0] and item["up"] >= duration
                    and min(1.0, item["k"] * duration) >= min(1.0, forced[1] * REQUEST_MARGIN)
                    for item in info.values()) and (forced_best is None or rate > forced_best[0]):
                forced_best = (rate, duration)
        if fixed_duration is not None:
            best = (1.0, fixed_duration)      # already chosen together with the fibres (_joint_search)
        if forced is not None:
            if forced_best is None:
                return None
            best = request_best = forced_best
        if best is None:
            return None
        # Tie-break, not a bonus: duration_seconds is one value shared by all 16 fibres,
        # so this only ever swaps to a request-completing duration that is already within
        # REQUEST_RATE_TOLERANCE of the best achievable rate -- it can trade a small,
        # bounded amount of score for a shot at a reward, never meaningfully distort the
        # exposure length chosen for everything else in this pointing.
        if request_best is not None and request_best[0] >= best[0] * REQUEST_RATE_TOLERANCE:
            best = request_best
        duration = best[1]
        if best[0] <= 0.0 and forced is None:
            if state.has_recent_sample(hours):
                return None
            fallback = next((d for d in (900, 600, 300) if d <= seconds_left and d <= center_up), None)
            if fallback is None:
                return None
            duration = fallback

        assignments: dict[str, str] = {}
        for fiber, item in info.items():
            if item["up"] >= duration:
                assignments[str(fiber)] = state.ids[item["i"]]
        if not assignments:
            return None

        band_scale = self._band_scale(night_index)
        votes = {"DARK": 0.0, "BRIGHT": 0.0, "BACKUP": 0.0}
        for fiber, item in info.items():
            if str(fiber) not in assignments:
                continue
            band = scoring.program_band(item["model"] * band_scale)
            votes[band] += state.weight[item["i"]] * min(1.0, item["k"] * duration) + \
                (REQUIRED_BONUS * 0.02 if state.required[item["i"]] else 0.0)
        program, best_score = "BACKUP", float("-inf")
        for name in ("DARK", "BRIGHT", "BACKUP"):
            matched = votes[name] * scoring.program_multipliers.get(name, 1.0)
            mismatched = (votes["DARK"] + votes["BRIGHT"] + votes["BACKUP"] - votes[name]) * scoring.mismatch_multiplier
            score = matched + mismatched
            if self.incremental_scores:
                score = 0.
                for fiber, item in info.items():
                    if str(fiber) not in assignments:
                        continue
                    i = item['i']
                    band = scoring.program_band(item['model'] * band_scale)
                    predicted = state.weight[i] * min(1., item['k'] * duration) * scoring.program_multiplier(name, band)
                    score += max(0., predicted-state.best_score[i])
            if score > best_score:
                best_score, program = score, name
        if state.force_program:
            program = state.force_program

        clean = not state.all_sky_notice()
        state.pending.clear()
        state.pending_action_index = self._current_action_index
        for fiber, item in info.items():
            if str(fiber) in assignments:
                state.pending[state.ids[item["i"]]] = PendingPrediction(
                    model=item["model"], band_model=item["model"] / 0.95, alt=item["alt"], az=item["az"],
                    clean=clean and self._direction_factor(item["alt"], item["az"]) >= 1.0,
                    lst=lst, fiber=fiber,
                )
        # Hard-mode cards: command the pointing that lands on the chosen centre
        d_alt, d_az = state.pointing.offset
        cmd_alt = min(90.0, max(0.0, c_alt - d_alt))
        cmd_az = (c_az - d_az) % 360.0
        state.pending_cmd = (cmd_alt, cmd_az)
        state.pending_program = program
        state.pending_band_scale = band_scale
        state.pending_duration = duration
        state.pending_night = night_index
        if self.conditions is not None:
            self.conditions.remember(now, self._current_action_index, state.pending,
                                     program, duration)

        return {
            "action": "observe",
            "pointing": {"alt_deg": cmd_alt, "az_deg": cmd_az},
            "assignments": assignments,
            "duration_seconds": duration,
            "program": program,
        }


if __name__ == "__main__":
    # Self-tests: python3 -m agent_core.planner
    dates = ["2027-06-01", "2027-06-02", "2027-06-03"]
    rule = {"2027-06-01": 1.0, "2027-06-02": 0.1, "2027-06-03": 1.0}
    outlook = Planner._sanitize_outlook
    # 1. unusable reply -> the rule, nothing taken from the model
    for bad in (None, "not json", [], 3, {"usable": "all good"}, {"notes": "x"}):
        assert outlook(bad, dates, rule) == (rule, 0), bad
    # 2. values are clamped to [0, 1]
    assert outlook({"usable": {"2027-06-01": 1.7, "2027-06-02": -3, "2027-06-03": 0.4}}, dates, rule) == \
        ({"2027-06-01": 1.0, "2027-06-02": 0.0, "2027-06-03": 0.4}, 3)
    # 3. a night left out, or given as something that is not a number, keeps the rule's value
    assert outlook({"usable": {"2027-06-01": 0.8, "2027-06-03": "bad", "2027-06-02": True}}, dates, rule) == \
        ({"2027-06-01": 0.8, "2027-06-02": 0.1, "2027-06-03": 1.0}, 1)
    # 4. nights that were not asked about are ignored
    assert outlook({"usable": {"2027-07-09": 0.0, "2027-06-02": 0.25}}, dates, rule) == \
        ({"2027-06-01": 1.0, "2027-06-02": 0.25, "2027-06-03": 1.0}, 1)
    # 5. NaN is not a number
    assert outlook({"usable": {"2027-06-01": float("nan")}}, dates, rule) == (rule, 0)
    print("planner self-test: _sanitize_outlook OK (5 cases)")

    def night(ratio):
        return {"ratio": ratio}
    streak, tier = Planner._fault_streak, Planner._fault_tier
    # two low nights -> strong when both are deep, mid when one is borderline
    assert tier(streak([night(1.0), night(0.65), night(0.6)], 0.8), FAULT_BASE) == "strong"
    assert tier(streak([night(1.0), night(0.78), night(0.7)], 0.8), FAULT_BASE) == "mid"
    # one deep night next to one just over the line averages below it: a fault, not borderline
    assert tier(streak([night(1.0), night(0.49), night(0.71)], 0.8), FAULT_BASE) == "strong"
    # one low night, or a recovery in between, is not enough
    assert tier(streak([night(0.6), night(1.0), night(0.6)], 0.8), FAULT_BASE) == "none"
    # unjudged nights neither extend nor break the run
    assert len(streak([night(0.6), night(None), night(0.7)], 0.8)) == 2
    # 0.8-0.9 never counts; after a false report three nights at <= 0.70 are needed
    assert tier(streak([night(0.85), night(0.85)], 0.8), FAULT_BASE) == "none"
    assert tier(streak([night(0.65), night(0.65)], 0.7), FAULT_AFTER_FALSE) == "none"
    assert tier(streak([night(0.65), night(0.65), night(0.69)], 0.7), FAULT_AFTER_FALSE) == "strong"
    assert tier(streak([night(0.65), night(0.65), night(0.69)], 0.6), FAULT_AFTER_ALLOWANCE) == "none"
    print("planner self-test: fault streak/tier OK (8 cases)")

    exact = [(v, False) for v in (0.2, 0.4, 0.6, 0.8, 1.0)]
    assert _censored_median(exact) == 0.6
    # lower bounds above the dips pull the median up to where the real observations are
    assert _censored_median([(0.1, False), (0.15, False)] + [(0.5, True)] * 6 + [(0.6, False)]) == 0.6
    # mostly lower bounds: the median of all values, which is comparable night to night
    assert _censored_median([(0.1, False)] + [(0.5, True)] * 4 + [(0.9, True)]) == 0.5
    print("planner self-test: censored median OK (3 cases)")
