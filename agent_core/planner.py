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
agent's own past hits (SurveyState) -- never hidden weather truth. Once per night, one
LLM call (see `_night_advice` below) reads the forecast, the live bulletin and the
survey's own progress, and answers with BOUNDED knobs only: an exposure bias in 0.9-1.1
and whether to lean towards observation requests (plus a sky outlook that is logged).
Direction avoidance and per-class weights were tried as model outputs and removed after
a replay ablation showed them to cost points. Required targets and instrument-fault
reports never go through the model. A failed call means neutral advice (= the rule-based path) for that
night; the next night's call still runs normally.

This mirrors the anchor-search algorithm of this project's companion TypeScript
example target-for-target, so both examples solve the problem the same way.
"""
from __future__ import annotations

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
from .memory import AdviceTape, TraceLog
from .state import PendingPrediction

REQUIRED_BONUS = 60.0
DONE_FACTOR = 0.95
PLAN_FACTOR_SAFETY = 0.9
EDGE_MARGIN_DEG = 0.08
DURATIONS = (300, 450, 600, 900, 1200, 1500, 1800, 2400, 3000, 3600)
MIN_VISIBLE_SECONDS = 600
NEIGHBOUR_RADIUS_DEG = 2.1
ANCHORS = 6
ANCHOR_POOL = 300
CLOSED_KINDS = {"rain", "storm"}
BLOCKING_KINDS = {"terrain_obstruction", "rocket_launch"}
DIRECTION_AZ = {"N": 0.0, "NE": 45.0, "E": 90.0, "SE": 135.0, "S": 180.0,
                "SW": 225.0, "W": 270.0, "NW": 315.0}

# Instrument-fault detector (see "instrument fault reporting" below). Each entry is
# (a night counts as low when its median ratio is <= this fraction of the reference,
#  consecutive low nights needed before reporting).
FAULT_BASE = (0.80, 2)             # no false report since the last correct one
FAULT_AFTER_FALSE = (0.70, 3)      # after a false report, still inside the free allowance
FAULT_AFTER_ALLOWANCE = (0.60, 3)  # the next false report would cost points
FAULT_STRONG = 0.70                # at or below this on every counted night: report without asking
FAULT_MIN_SAMPLES = 8              # a night with fewer clean samples is not judged
FAULT_REF_NIGHTS = 5
FAULT_DARK_CHECKS_WANTED = 6       # force DARK on a suspicious night until this many DARK band checks exist
FAULT_SKY_CHECKS_MIN = 6           # fewer band checks than this give no sky estimate for the night
FAULT_SKY_GRID = [0.5 + 0.025 * k for k in range(41)]   # candidate sky levels, relative to the quiet reference
FAULT_SKY_OK = 0.9                 # logged as sky_ok when the night's sky estimate is at least this
FAULT_RECOVERY_RANGE = (0.6, 1.0)  # per-night factor by which an earthquake's efficiency loss shrinks
FAULT_MAX_FORCED_PER_NIGHT = 4
FAULT_MAX_LLM_WAITS = 2

FAULT_SYSTEM_PROMPT = (
    "You arbitrate a borderline instrument-fault alarm for a robotic telescope survey. "
    "For each recent night you get: ratio = that night's measured efficiency relative to "
    "what a healthy instrument should deliver (already corrected for the measured sky "
    "level and, during an earthquake sequence, for the expected night-by-night recovery); "
    "c = number of samples; sky = measured sky level (1.0 normal); quake = an earthquake "
    "notice was active; notice_kinds = public weather/event notices that night; "
    "profile = the night in two-hour blocks as [hours after night start, ratio, samples]. "
    "The last nights have ratio between 0.70 and 0.80: too low to ignore, too high to be "
    "sure. Decide between reporting a fault now and waiting one more night.\n"
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
    counting as an observation. If the estimate never falls to one half (too many lower
    bounds at the top), the largest value is returned -- the median is at least that."""
    ordered = sorted(samples, key=lambda s: (s[0], s[1]))
    surviving, at_risk = 1.0, len(ordered)
    for value, bound in ordered:
        if not bound:
            surviving *= 1.0 - 1.0 / at_risk
            if surviving <= 0.5:
                return value
        at_risk -= 1
    return ordered[-1][0]

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


def _az_distance(a: float, b: float) -> float:
    return abs(wrap180(a - b))


CLASSES = ("QSO", "ELG", "LRG", "BGS", "Star")
OUTLOOKS = ("good", "degraded", "closed")
EXPOSURE_BIAS_RANGE = (0.9, 1.1)
# Skip the nightly call when it could not finish anyway, or when the wall clock is
# getting short enough that the deterministic search should keep all of it.
ADVICE_MIN_LLM_BUDGET_SECONDS = 30.0
ADVICE_MIN_WALLCLOCK_SECONDS = 400.0

ADVICE_SYSTEM_PROMPT = (
    "You advise a robotic multi-fibre telescope survey once at the start of each night. "
    "You are given tonight's forecast notices, the next three nights' forecast, the live "
    "bulletin, the hit rate so far, per-class progress (done_weight / total_weight of "
    "science weight), how many required targets remain, active observation requests and "
    "your remaining time budget. Reply with exactly one JSON object and nothing else:\n"
    '{"sky_outlook": "good|degraded|closed", "exposure_bias": 1.0, '
    '"pursue_requests": true, "notes": "<=120 chars"}\n'
    "Rules: exposure_bias is within 0.9-1.1 (above 1 = slightly longer exposures when the "
    "sky looks poor tonight, below 1 = slightly shorter when it looks excellent; 1.0 = no "
    "change). sky_outlook describes the whole night. pursue_requests=false only if "
    "chasing the active requests is clearly not worth it. Pointing directions, target "
    "choice, required targets and instrument-fault reports are handled by fixed rules "
    "and are not affected by anything you answer. Values outside the stated range are "
    "clamped. Answer quickly; do not deliberate at length."
)


def _clamped(value, low: float, high: float, default: float = 1.0) -> float:
    if isinstance(value, bool):
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number != number:  # NaN
        return default
    return min(high, max(low, number))


class Planner:
    def __init__(self, state, log=lambda text: None):
        self.state = state
        self.log = log
        self.grid = state.fiber_grid
        self.llm = LLMClient(log=log)
        self.trace = TraceLog(log=log)
        self.tape = AdviceTape(log=log)

        self.observe_count = 0
        self.reports = 0
        self.correct_reports = 0
        self.false_reports = 0                       # since the last correct report
        self.fault_nights: list[dict] = []           # one record per closed night
        self._fault_floor = 0                        # records before this index no longer count towards a streak
        self._fault_open_night: int | None = None
        self._quiet_fallback: float | None = None    # quiet reference carried across a correct report
        # Earthquake sequence: efficiency is modelled as quiet * (1 - loss), the loss shrinking
        # by a constant factor per night; all of it estimated from the agent's own nights.
        self._quake_quiet: float | None = None       # quiet reference frozen when the sequence began
        self._quake_anchor: tuple | None = None      # (night index, loss) of the last judged non-low night
        self._quake_first: tuple | None = None       # (night index, loss) of the first judged night
        self._quake_recovery = 1.0
        self._night_kinds: dict[str, set] = {}       # night_id -> notice kinds seen in its bulletins
        self._quake_onset: dict[str, float] = {}     # night_id -> survey hours of its first earthquake bulletin
        self._report_due: str | None = None
        self._report_streak = 0
        self._awaiting_report_result = False
        self._llm_waits = 0
        self._forced_tonight = 0
        self.night_index_seen: int | None = None
        self.consecutive_reports = 0
        self._last_forecast_notices: list = []
        self.total_assigned = 0
        self.total_hit = 0
        self._current_action_index = None
        self._request_thresholds_now: dict = {}
        self.pursue_requests = True

        log(f"planner: {len(state.ids)} targets ({sum(state.required)} required), "
            f"{len(state.nights)} nights, llm model={self.llm.model} base_url={self.llm.base_url}")

    # -- top-level decision ----------------------------------------------------

    def decide(self, payload: dict) -> dict:
        state = self.state
        now = parse_utc(payload["now_utc"])
        hours = (now - state.survey_start).total_seconds() / 3600.0

        for message in list(payload.get("new_messages", [])) + [payload.get("latest_bulletin") or {}]:
            if message.get("record_type") == "bulletin" or "slot_id" in message:
                self._note_bulletin(message)
        for message in payload.get("new_messages", []):
            if message.get("record_type") == "forecast":
                self._last_forecast_notices = message.get("notices", [])
            elif message.get("record_type") == "observation_request":
                self.log(f"planner: observation request {message.get('request_id')} issued, "
                         f"{len(message.get('target_ids', []))} targets by {message.get('deadline_utc')}")
            elif message.get("record_type") == "observation_request_result":
                self.log(f"planner: observation request {message.get('request_id')} "
                         f"{message.get('status')} (reward {message.get('score_delta', 0.0)})")
        state.on_messages(payload.get("new_messages", []), payload.get("latest_bulletin"))
        state.on_result(payload.get("last_result"), hours)
        last_result = payload.get("last_result")
        if last_result and last_result.get("action") == "observe":
            self.total_assigned += int(last_result.get("assigned_count", 0))
            self.total_hit += int(last_result.get("hit_count", 0))
        self._pace(payload, now)
        self._current_action_index = payload.get("observe_action_index")
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
            self._night_advice(night_index, night_start, now, payload)
            # pursue_requests may just have changed for tonight
            self._request_thresholds_now = self._request_thresholds(payload.get("active_requests") or [])

        if (night_end - now).total_seconds() < state.min_exposure:
            nxt = state.next_night_start(now)
            if nxt is None:
                return {"action": "finish", "reason": "survey over"}
            return {"action": "wait", "until_utc": format_utc(nxt), "reason": "night ending"}

        if state.site_closed():
            return {"action": "wait", "duration_seconds": self._to_next_slot(now, night_start),
                    "reason": "bulletin: rain/storm over the whole sky"}

        self._force_dark_if_suspicious(night_index)
        action = self.plan(now, night_end, night_index, hours)
        if action is not None and state.force_program:
            self._forced_tonight += 1
        if action is None:
            return {"action": "wait", "duration_seconds": self._to_next_slot(now, night_start),
                    "reason": "nothing useful is up"}
        self.observe_count += 1
        action["reason"] = f"{len(action['assignments'])} fibres, program {action['program']}"
        return action

    def on_finish(self, payload: dict) -> None:
        self.trace.write({"event": "finish", **payload})
        self.trace.close()
        self.log(f"planner: finished termination_reason={payload.get('termination_reason')} "
                 f"observes={self.observe_count} reports={self.reports} "
                 f"(correct {self.correct_reports}) llm_calls={self.llm.calls_made}")

    def note_action(self, action: dict) -> None:
        """Called by agent.py right after an action is validated, so the consecutive-report
        counter (enforced by validation.py) stays correct even when a fallback replaced it."""
        self.consecutive_reports = self.consecutive_reports + 1 if action.get("action") == "report" else 0

    def _to_next_slot(self, now, night_start) -> int:
        slot = self.state.slot_seconds
        into = (now - night_start).total_seconds() % slot
        return int(max(60, min(3600, slot - into if into else slot)))

    def _pace(self, payload: dict, now) -> None:
        """Do less work per decision when the wall clock is short for the nights still to come."""
        state = self.state
        remaining_wall = float((payload.get("wallclock") or {}).get("remaining_seconds", 1e9))
        night_seconds = sum(max(0.0, (end - max(start, now)).total_seconds()) for start, end in state.nights if end > now)
        decisions_left = max(1.0, night_seconds / 700.0)
        per_decision = remaining_wall / decisions_left
        level = 0 if per_decision > 0.12 else 1 if per_decision > 0.04 else 2
        if level != state.fast_level:
            self.log(f"planner: pace level {level} ({per_decision * 1000:.0f} ms per decision left)")
            state.fast_level = level

    # -- LLM: one bounded call per night ------------------------------------------

    @staticmethod
    def _sanitize_advice(answer) -> dict:
        """Turn whatever the model returned into bounded advice. Anything missing,
        malformed or out of range becomes neutral, so the worst a bad reply can do is
        nothing. Returns {"ok", "sky_outlook", "exposure_bias", "pursue_requests",
        "notes"}; ok=False means the reply was unusable as a whole. Keys the interface
        does not have (e.g. directions to avoid) are ignored."""
        advice = {"ok": False, "sky_outlook": "unknown", "exposure_bias": 1.0,
                  "pursue_requests": True, "notes": ""}
        if not isinstance(answer, dict):
            return advice
        advice["ok"] = True
        outlook = str(answer.get("sky_outlook", "")).strip().lower()
        if outlook in OUTLOOKS:
            advice["sky_outlook"] = outlook
        advice["exposure_bias"] = _clamped(answer.get("exposure_bias", 1.0), *EXPOSURE_BIAS_RANGE)
        if isinstance(answer.get("pursue_requests"), bool):
            advice["pursue_requests"] = answer["pursue_requests"]
        advice["notes"] = str(answer.get("notes") or "")[:120]
        return advice

    def _progress_by_class(self) -> dict:
        state = self.state
        progress = {name: {"done_weight": 0.0, "total_weight": 0.0} for name in CLASSES}
        for i, name in enumerate(state.target_class):
            entry = progress.get(name)
            if entry is None:
                continue
            entry["total_weight"] += state.weight[i]
            if state.factor[i] >= 0.5:
                entry["done_weight"] += state.weight[i]
        return {name: {k: round(v, 1) for k, v in entry.items()} for name, entry in progress.items()}

    def _night_advice(self, night_index: int, night_start, now, payload: dict) -> None:
        """One planning question at the start of each night. The answer is sanitized into
        bounded knobs (_sanitize_advice) and applied for the rest of the night:
        state.duration_scale and self.pursue_requests.
        sky_outlook is only logged for now."""
        state = self.state
        night_date = (night_start - timedelta(hours=12)).date().isoformat()
        left = float((payload.get("wallclock") or {}).get("remaining_seconds", 0))

        next_dates = {(night_start - timedelta(hours=12) + timedelta(days=d)).date().isoformat() for d in (1, 2, 3)}
        forecast_tonight = [n for n in self._last_forecast_notices if night_date in (n.get("nights") or [])]
        forecast_next = [n for n in self._last_forecast_notices if next_dates & set(n.get("nights") or [])]
        bulletin_notices = (payload.get("latest_bulletin") or {}).get("notices", [])
        hit_rate = (self.total_hit / self.total_assigned) if self.total_assigned > 0 else 1.0
        threshold = state.scoring.required_threshold
        requests = []
        for request in payload.get("active_requests") or []:
            try:
                hours_left = round((parse_utc(request["deadline_utc"]) - now).total_seconds() / 3600.0, 1)
            except (KeyError, TypeError, ValueError):
                hours_left = None
            requests.append({"request_id": request.get("request_id"),
                             "remaining_count": request.get("remaining_count"),
                             "hours_left": hours_left})
        user_payload = {
            "night_date": night_date,
            "nights_remaining": len(state.nights) - night_index,
            "forecast_notices_for_tonight": forecast_tonight,
            "forecast_notices_next_3_nights": forecast_next,
            "current_bulletin_notices": bulletin_notices,
            "hit_rate_so_far": round(hit_rate, 3),
            "progress_by_class": self._progress_by_class(),
            "required_remaining": sum(1 for i in range(len(state.ids))
                                      if state.required[i] and state.factor[i] < threshold),
            "active_requests": requests,
            "llm_budget_left_seconds": round(self.llm.budget_left_seconds()),
        }
        if self.tape.replaying:
            advice = self._sanitize_advice(self.tape.replay(night_date))
            neutral = self._sanitize_advice(None)
            for knob, key in (("exposure_bias", "exposure_bias"), ("pursue", "pursue_requests")):
                if knob in self.tape.masked:
                    advice[key] = neutral[key]
            status = "replay"
        elif self.llm.disabled:
            advice, status = self._sanitize_advice(None), "disabled"
        elif self.llm.budget_left_seconds() < ADVICE_MIN_LLM_BUDGET_SECONDS or left < ADVICE_MIN_WALLCLOCK_SECONDS:
            advice, status = self._sanitize_advice(None), "skipped"
        else:
            advice = self._sanitize_advice(self.llm.ask_json(ADVICE_SYSTEM_PROMPT, user_payload, left))
            status = "ok" if advice["ok"] else "fell back"
        self.tape.record(night_date, {"sky_outlook": advice["sky_outlook"],
                                      "exposure_bias": advice["exposure_bias"],
                                      "pursue_requests": advice["pursue_requests"], "notes": advice["notes"]})

        state.duration_scale = advice["exposure_bias"]
        self.pursue_requests = advice["pursue_requests"]
        self.log(f"planner: night {night_date} llm advice {status} bias={advice['exposure_bias']:.2f} "
                 f"pursue={advice['pursue_requests']} outlook={advice['sky_outlook']}")
        self.trace.write({"event": "night_advice", "night_date": night_date, "status": status,
                          "exposure_bias": advice["exposure_bias"], "pursue_requests": advice["pursue_requests"],
                          "sky_outlook": advice["sky_outlook"], "notes": advice["notes"]})

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
        if self.false_reports == 0:
            return FAULT_BASE
        if self.false_reports < self.state.false_report_free_allowance:
            return FAULT_AFTER_FALSE
        return FAULT_AFTER_ALLOWANCE

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
        if self._quake_quiet is None or self._quake_anchor is None:
            return None
        anchor_night, loss = self._quake_anchor
        loss = max(0.0, loss) * self._quake_recovery ** max(0, night_index - anchor_night)
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
        only in the base mode, and only when some counted night is above FAULT_STRONG)."""
        threshold, needed = mode
        if len(run) < needed:
            return "none"
        if mode == FAULT_BASE and any(ratio > FAULT_STRONG for ratio in run[:needed]):
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
            self._fault_open_night = None
        if night_index is not None and self._fault_open_night is None:
            self._fault_open_night = night_index
            self._forced_tonight = 0
        if self._report_due is None:
            return None
        reason, self._report_due = self._report_due, None
        self.reports += 1
        self._awaiting_report_result = True
        self.log(f"planner: reporting instrument fault at {payload.get('now_utc')} ({reason})")
        return {"action": "report", "reason": reason,
                "decision_source": "llm" if reason.startswith("model") else "rule"}

    def _settle_report(self, correct: bool, payload: dict) -> None:
        self.log(f"planner: report at {payload.get('now_utc')} was {'correct' if correct else 'FALSE'}")
        self._llm_waits = 0
        if correct:
            self.correct_reports += 1
            self.false_reports = 0
            self._quiet_fallback = self._quiet_ref() or self._quake_quiet or self._quiet_fallback
            self.state.forget_quality_history()   # the instrument is repaired: learn the level again
            self.fault_nights = []
            self._fault_floor = 0
            self._quake_anchor = None             # re-anchor the recovery curve on the next judged night
            return
        self.false_reports += 1
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
            self._quake_anchor = (last["index"], 1.0 - last["level"] / self._quake_quiet)
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
            self._quake_anchor = self._quake_first = None
            self._quake_recovery = 1.0
        elif not quake:
            self._quake_quiet = self._quake_anchor = self._quake_first = None
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
        record["accepted"] = not low
        if quake and record["level"] is not None and self._quake_quiet and not low:
            loss = 1.0 - record["level"] / self._quake_quiet
            if self._quake_first is None:
                self._quake_first = (night_index, loss)
            elif night_index > self._quake_first[0] and self._quake_first[1] > 0.05 and loss > 0.0:
                factor = (loss / self._quake_first[1]) ** (1.0 / (night_index - self._quake_first[0]))
                self._quake_recovery = min(FAULT_RECOVERY_RANGE[1], max(FAULT_RECOVERY_RANGE[0], factor))
            self._quake_anchor = (night_index, loss)
        self.fault_nights.append(record)

        mode = self._fault_mode()
        run = self._fault_streak(self.fault_nights[self._fault_floor:], mode[0])
        fmt = lambda v: "n/a" if v is None else f"{v:.2f}"  # noqa: E731
        self.log(f"planner: night {date} fault-check m={fmt(record['m'])} ref={fmt(record['ref'])} "
                 f"quake={quake} sky_ok={record['sky_ok']} streak={len(run)} "
                 f"c={record['c']} sky={fmt(record['sky'])} ratio={fmt(record['ratio'])}"
                 + (f" recovery={self._quake_recovery:.2f}" if quake else ""))
        self.trace.write({"event": "fault_check", **record, "streak": len(run)})

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
        # Borderline. Every false alarm seen so far was of this kind and was gone by the
        # third night, so the rule alone waits for one more low night; the model may report
        # sooner when the picture is clearly a fault, or hold back at most
        # FAULT_MAX_LLM_WAITS times.
        needed = mode[1]
        if len(run) >= needed + FAULT_MAX_LLM_WAITS:
            self._report_due = f"rule (borderline, {len(run)} nights): {summary}"
            return
        verdict = None
        if not self.llm.disabled and not self.tape.replaying and self._llm_waits < FAULT_MAX_LLM_WAITS:
            verdict = self._arbitrate(payload)
        if verdict == "report":
            self._report_due = f"model: borderline, {summary}"
        elif verdict == "wait_one_night":
            self._llm_waits += 1
        elif len(run) > needed:
            self._report_due = f"rule (borderline, no model verdict, {len(run)} nights): {summary}"

    def _arbitrate(self, payload: dict):
        """Borderline alarm: one model call decides between reporting now and waiting one
        more night. Returns "report", "wait_one_night", or None when the call failed (the
        rule then waits for one extra low night)."""
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

    def _direction_factor(self, alt: float, az: float) -> float:
        state = self.state
        for direction in state.terrain:
            if direction in DIRECTION_AZ and alt < 50.0 and _az_distance(az, DIRECTION_AZ[direction]) <= 60.0:
                return 0.0
        factor = 1.0
        for key in state.notices:
            kind, _, direction = key.partition("|")
            if direction not in DIRECTION_AZ:
                continue
            near = _az_distance(az, DIRECTION_AZ[direction]) <= 67.5
            if kind in BLOCKING_KINDS and near and alt < 62.0:
                return 0.0
            if near and alt < 75.0:
                factor = min(factor, 0.35)
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
        if not self.pursue_requests:
            return thresholds
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

    def _value(self, i: int) -> float:
        """Planning value of fully completing target i from here (ignores how much
        exposure is achievable tonight)."""
        state = self.state
        f = state.factor[i]
        damp = 0.6 ** state.misses[i]
        threshold = state.scoring.required_threshold
        if state.required[i]:
            if f >= threshold:
                return state.weight[i] * max(0.0, 1.0 - f * f) * damp
            return (state.weight[i] * (1.0 - f * f) + REQUIRED_BONUS * (1.0 if f < 0.5 else 0.35)) * damp
        return 0.0 if f >= DONE_FACTOR else state.weight[i] * (1.0 - f * f) * damp

    # -- main planning pass -------------------------------------------------------

    def plan(self, now, night_end, night_index: int, hours: float):
        state = self.state
        state.update_scale(hours)
        lst = local_sidereal_deg(now, state.lon)
        horizon = min(night_end, state.survey_end)
        seconds_left = (horizon - now).total_seconds()
        if seconds_left < state.min_exposure:
            return None
        min_visible = min(MIN_VISIBLE_SECONDS, seconds_left) * SIDEREAL_DEG_PER_SECOND

        still_active = []
        candidates: list[tuple[float, int]] = []
        for i in state.active:
            v = self._value(i)
            if v <= 0.0:
                continue
            still_active.append(i)
            ha = wrap180(lst - state.ra[i])
            h = state.hmax[i]
            if -h <= ha <= h - min_visible:
                nights_left = max(1, state.last_night[i] - night_index + 1)
                setting = (1.0 + 0.5 * max(0.0, ha / h)) if h < 180 else 1.0
                candidates.append((v * (1.0 + 2.0 / nights_left) * setting, i))
        state.active = still_active
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
            k = (state.flux[i] * model * state.scale * PLAN_FACTOR_SAFETY) / scoring.f0t0
            ha = wrap180(lst - state.ra[i])
            up = (state.hmax[i] - ha) / SIDEREAL_DEG_PER_SECOND if state.hmax[i] < 180 else 1e9
            reach = min(1.0, k * min(state.max_exposure, up, seconds_left))
            f = state.factor[i]
            gain = state.weight[i] * max(0.0, reach * reach - f * f)
            if state.required[i] and f < 0.5 and reach >= 0.5:
                gain += REQUIRED_BONUS
            damp = (0.6 ** state.misses[i]) * (0.7 ** state.attempts[i])
            result = gain * damp * self._direction_factor(alt, az)
            achievable_cache[i] = result
            return result

        anchors: list[tuple[float, int]] = []
        for checked, (priority, i) in enumerate(candidates):
            if checked >= ANCHOR_POOL and len(anchors) >= 3 * ANCHORS:
                break
            weighted = achievable(i) * priority / max(1e-9, self._value(i))
            if weighted > 0:
                anchors.append((weighted, i))
        if not anchors:
            return None
        anchors.sort(key=lambda t: -t[0])

        n_anchors = 1 if state.fast_level >= 1 else ANCHORS
        fibers = range(self.grid.n) if state.fast_level < 2 else (5, 6, 9, 10)
        best = None  # (total, c_alt, c_az, chosen)
        tried = 0
        for _, anchor in anchors:
            if tried >= n_anchors and best is not None:
                break
            if tried >= n_anchors + 8:
                break
            tried += 1
            a_alt, a_az = altaz(anchor)
            near = [j for j in state.neighbours(state.ra[anchor], state.dec[anchor], NEIGHBOUR_RADIUS_DEG) if j in visible]
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
                    score = v * (1.0 if margin >= EDGE_MARGIN_DEG * (1 + 1.5 * state.misses[j]) else 0.4)
                    existing = chosen.get(fib)
                    if existing is None or score > existing[0]:
                        chosen[fib] = (score, j, margin)
                if not chosen:
                    continue
                total = sum(score for score, _, _ in chosen.values())
                if best is None or total > best[0]:
                    best = (total, c_alt, c_az, chosen)
        if best is None:
            return None
        _, c_alt, c_az, chosen = best
        return self._finish_plan(now, lst, c_alt, c_az, chosen, seconds_left, moon, altaz, hours, night_index)

    def _finish_plan(self, now, lst, c_alt, c_az, chosen, seconds_left, moon, altaz, hours, night_index):
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
            k = (state.flux[j] * model * state.scale * PLAN_FACTOR_SAFETY) / scoring.f0t0
            info[fiber] = {"i": j, "alt": alt, "az": az, "model": model, "up": up, "k": k}
        center_up = (c_hmax - c_ha) / SIDEREAL_DEG_PER_SECOND if c_hmax < 180 else 1e9

        best = None  # (rate, duration)
        request_best = None  # (rate, duration): best candidate that also clears a needed
                              # request target's completion_factor_threshold
        for base in DURATIONS:
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
        if best[0] <= 0.0:
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

        band_scale = state.scale / 0.95
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
                )
        state.pending_program = program
        state.pending_duration = duration
        state.pending_night = night_index

        return {
            "action": "observe",
            "pointing": {"alt_deg": c_alt, "az_deg": c_az},
            "assignments": assignments,
            "duration_seconds": duration,
            "program": program,
        }


if __name__ == "__main__":
    # Self-test for the advice sanitizer: python3 -m agent_core.planner
    clean = Planner._sanitize_advice
    neutral = {"ok": False, "sky_outlook": "unknown", "exposure_bias": 1.0, "pursue_requests": True, "notes": ""}

    # 1. out-of-range bias is clamped, both ends
    assert clean({"exposure_bias": 1.4})["exposure_bias"] == 1.1
    assert clean({"exposure_bias": 0.2})["exposure_bias"] == 0.9

    # 2. unusable reply -> neutral
    for bad in (None, "not json", [], 3):
        assert clean(bad) == neutral, clean(bad)

    # 3. wrong types inside an object -> that knob neutral, reply still counts
    a = clean({"exposure_bias": "nan", "pursue_requests": "no", "sky_outlook": 7})
    assert a == {**neutral, "ok": True}, a
    assert clean({"exposure_bias": True})["exposure_bias"] == 1.0

    # 4. knobs the interface no longer has are ignored, never applied
    a = clean({"sky_outlook": "closed", "avoid_directions": ["N", "NE", "E", "SE", "S", "SW", "W", "NW"],
               "class_weights": {"QSO": 3.0}, "duration_scale": 0.7, "pursue_requests": False})
    assert a == {"ok": True, "sky_outlook": "closed", "exposure_bias": 1.0, "pursue_requests": False, "notes": ""}, a

    # 5. a normal reply passes through
    a = clean({"sky_outlook": "Degraded", "exposure_bias": "1.05", "pursue_requests": True, "notes": "x" * 300})
    assert a["ok"] and a["sky_outlook"] == "degraded" and a["exposure_bias"] == 1.05 and a["pursue_requests"], a
    assert len(a["notes"]) == 120, a
    print("planner self-test: _sanitize_advice OK (5 cases)")

    def night(ratio):
        return {"ratio": ratio}
    streak, tier = Planner._fault_streak, Planner._fault_tier
    # two low nights -> strong when both are deep, mid when one is borderline
    assert tier(streak([night(1.0), night(0.65), night(0.6)], 0.8), FAULT_BASE) == "strong"
    assert tier(streak([night(1.0), night(0.75), night(0.6)], 0.8), FAULT_BASE) == "mid"
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
    # nothing but lower bounds above: the median is at least the largest of them
    assert _censored_median([(0.1, False)] + [(0.5, True)] * 5) == 0.5
    print("planner self-test: censored median OK (3 cases)")
