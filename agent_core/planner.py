"""Selected Shiming science EV and raised-horizon posterior on teammate planner.

Baseline delegates decisions to the original planner, without estimator work.
Fault detection, repair handling, pacing and search depth are inherited unchanged.
"""
from __future__ import annotations

import math
import os

from .base_planner import (Planner as TeammatePlanner, REQUEST_MARGIN, PLAN_FACTOR_SAFETY,
                           DIRECTION_AZ, BLOCKING_KINDS, _az_distance)
from .geometry import parse_utc, local_sidereal_deg, tangent_offsets
from .horizon_posterior import HorizonPosterior, trajectory, WARMUP_NIGHTS
from .miss_triage import MissTriage, minimum_altitude
from .joint_ev import joint_search_ev
from .program_calibration import ProgramCalibration, PROGRAMS
from .request_policy import RequestPolicy
from .science_value import expected_increment

VERSION = "LLM_agent_v5.0.2_merge"
MODULES = frozenset({"science_ev", "program_calibration", "request_filter"})


class Planner(TeammatePlanner):
    def __init__(self, state, log=lambda text: None):
        super().__init__(state, log)
        text = os.environ.get("MERGE501_MODULES", "science_ev").strip()
        self.modules = frozenset(x.strip() for x in text.split(",") if x.strip() and x.strip() != "baseline")
        if self.modules - MODULES:
            raise ValueError("unknown MERGE501_MODULES configuration")
        self._best_scores = [0.0] * len(state.ids) if "science_ev" in self.modules else None
        self.program_calibration = ProgramCalibration(state.scoring) if "program_calibration" in self.modules else None
        self.request_filter = RequestPolicy(state) if "request_filter" in self.modules else None
        self._probabilities_cache = {}
        self._request_now = state.survey_start
        self.horizon = HorizonPosterior(state.min_alt) if os.environ.get("MERGE502_HORIZON", "1") == "1" else None
        self._horizon_night = -1
        self._horizon_pending = None
        self._miss_geometry = None
        self.miss_triage = MissTriage()
        self._horizon_planned = None
        self._horizon_cache = {}
        self._planning_lst = 0.0
        log(f"planner: {VERSION}; modules={','.join(sorted(self.modules)) or 'baseline'}; teammate fault/search retained")
        log(f"planner: horizon={'first five nights / posterior from night six' if self.horizon else 'disabled'}")

    @staticmethod
    def _unsafe_horizon_notices(messages):
        return any(n.get('event_kind') not in ('terrain_obstruction', 'earthquake')
                   for m in messages if m
                   for n in m.get('notices', []) if m.get('record_type') == 'bulletin' or 'slot_id' in m)

    def _learn_horizon(self, payload):
        messages = list(payload.get('new_messages', [])) + [payload.get('latest_bulletin') or {}]
        self.horizon.announce([n.get('direction') for m in messages for n in m.get('notices', [])
                               if n.get('event_kind') == 'terrain_obstruction'])
        pending, self._horizon_pending = self._horizon_pending, None
        geometry,self._miss_geometry=self._miss_geometry,None
        resync = [m for m in messages if m.get('record_type') == 'state_resync']
        for m in resync:
            self.horizon.invalidate(m.get('invalidated_window') or {})
            self.miss_triage.invalidate(m.get('invalidated_window') or {})
        result = payload.get('last_result') or {}
        if resync or not pending or result.get('action') != 'observe' or result.get('valid') is False:
            return
        action_index, night, paths, unsafe = pending
        hits = {h.get('target_id'): float(h.get('score', 0.0)) for h in result.get('hits', [])}
        control = any(math.isfinite(hits.get(tid, math.nan)) and hits[tid] > 0 for tid in paths)
        unsafe = unsafe or self._unsafe_horizon_notices(messages)
        self._adjudicate_misses(payload,action_index,night,paths,geometry or {},hits,unsafe,control)
        samples = []
        for tid, path in paths.items():
            score = hits.get(tid)
            if score is None or not math.isfinite(score) or score < 0:
                continue  # Absent hit is a fibre / altitude miss, NOT terrain.
            if score > 0 or (control and not unsafe):
                samples.append((path, score > 0))
        learned = self.horizon.add(action_index, night, samples)
        if learned:
            self.trace.write({'event': 'merge502_horizon_update', 'night_index': night,
                              'action_index': action_index, 'sector_updates': learned})

    def _adjudicate_misses(self,payload,action_index,night,paths,geometry,hits,unsafe,control):
        if not self.miss_triage.enabled or self.llm.disabled:return
        failed=[];controls=[]
        for tid,path in paths.items():
            score=hits.get(tid)
            g=geometry.get(tid,{})
            row=dict(target_id=tid,score=score,**g)
            i=self.state.index_of[tid]
            row['previous_missing_hits']=self.state.misses[i]
            if score is None:
                row['receipt_status']='absent_hit';failed.append(row)
            elif math.isfinite(score) and score==0:
                row['receipt_status']='returned_zero';failed.append(row)
            elif math.isfinite(score) and score>0:
                controls.append(row)
        summary=dict(action_index=action_index,night_index=night,transient_notice_overlap=unsafe,
            absent_hits=sum(r['receipt_status']=='absent_hit' for r in failed),
            returned_zeros=sum(r['receipt_status']=='returned_zero' for r in failed),
            positive_controls=controls[:4],failed_targets=failed[:4])
        if not failed or night in self.miss_triage.asked_nights:
            self.miss_triage.remember(summary)
            return
        zeros=[r for r in failed if r['receipt_status']=='returned_zero']
        missing=[r for r in failed if r['receipt_status']=='absent_hit']
        missing.sort(key=lambda r:(-r['previous_missing_hits'],-r.get('margin_deg',0)))
        selected=zeros[:4]+missing[:4]
        selected+=(zeros[4:]+missing[4:])[:8-len(selected)]
        relevant={(int(r.get('start_azimuth_deg',0))%360)//5 for r in selected}
        context=dict(protocol='jsonl-v4',terrain_can_remove_hit=False,night_index=night,
            observe_action_index=action_index,minimum_observing_altitude_deg=self.state.min_alt,
            transient_notice_overlap=unsafe,same_exposure_positive_control=control,
            terrain_directions=sorted(x for x in self.horizon.announced if isinstance(x,str)),
            notices=sorted(self.state.notices),
            horizon_sectors=[s for s in self.horizon.snapshot()['sectors'] if int(s['azimuth_range_deg'][0])//5 in relevant],
            recent_own_exposures=list(self.miss_triage.history[-6:]),
            targets=selected,positive_controls=controls[:4])
        before=self.miss_triage.calls
        remaining=float((payload.get('wallclock') or {}).get('remaining_seconds',0))
        labels=self.miss_triage.classify(self.llm,context,remaining)
        self.miss_triage.remember(summary)
        if self.miss_triage.calls==before:return
        decisions=[];soft=[]
        for row in selected:
            tid=row['target_id'];label=labels.get(tid)
            status=row['receipt_status']
            entry=dict(target_id=tid,receipt_status=status,llm=label,soft_horizon_evidence=False)
            if label is None:
                entry['effective_cause']='insufficient'
            elif status=='absent_hit':
                entry['effective_cause']='missed_fiber' if label['p_missed_fiber']>=.6 else 'insufficient'
                entry['protocol_veto']=label['p_horizon']>=.6
                if entry['protocol_veto']:entry['veto_reason']='v4 terrain preserves hits; absent hit cannot measure terrain'
            elif label['p_horizon']>=.8 and label['p_missed_fiber']<=.05 and not unsafe and not control \
                    and row.get('minimum_altitude_deg',-90)>=self.state.min_alt \
                    and len({int(z%360)//5 for a,z in paths[tid]})==1:
                soft.append((paths[tid],label['p_horizon']))
                entry['soft_horizon_evidence']=True;entry['effective_cause']='likely_horizon'
            else:
                entry['effective_cause']='likely_horizon' if label['p_horizon']>=.8 and not unsafe else 'insufficient'
                if status=='returned_zero' and label['p_missed_fiber']>=.6:
                    entry['protocol_veto']=True;entry['veto_reason']='returned hit proves fibre membership'
            decisions.append(entry)
        learned=self.horizon.add_soft(action_index,night,soft)
        self.miss_triage.soft_labels+=len(soft)
        self.trace.write(dict(event='merge502_miss_llm',night_index=night,action_index=action_index,
            requested=len(selected),valid_labels=len(labels),decisions=decisions,
            soft_sector_updates=learned,context=context))
        self.log(f"planner: miss LLM adjudication night={night+1} labels={len(labels)} soft_sectors={learned}")

    def note_action(self, action):
        super().note_action(action)
        if self.horizon is None:
            return
        self._horizon_pending = None
        self._miss_geometry = None
        planned, self._horizon_planned = self._horizon_planned, None
        fields = ('action', 'pointing', 'assignments', 'duration_seconds', 'program')
        if not planned or any(action.get(k) != planned[0].get(k) for k in fields) or not 0 <= self._horizon_night < WARMUP_NIGHTS:
            return
        _, lst = planned
        state = self.state
        paths = {}
        geometry={}
        for tid in action.get('assignments', {}).values():
            i = state.index_of[tid]
            if state.flux[i] <= 0 or state.weight[i] <= 0:
                continue
            paths[tid] = trajectory(state.ra[i], state.dec[i], lst, state.lat, action['duration_seconds'])
            alt,az=state.pending[tid].alt,state.pending[tid].az
            offsets=tangent_offsets(alt,az,action['pointing']['alt_deg'],action['pointing']['az_deg'])
            nominal,margin=state.fiber_grid.classify(*offsets) if offsets else (None,-1.)
            fibre=next(int(f) for f,t in action['assignments'].items() if t==tid)
            geometry[tid]=dict(assigned_fiber=fibre,nominal_fiber=nominal,margin_deg=round(margin,5),
                start_altitude_deg=round(alt,5),start_azimuth_deg=round(az,5),
                minimum_altitude_deg=minimum_altitude(state.ra[i],state.dec[i],lst,state.lat,action['duration_seconds']),
                maximum_sample_altitude_deg=max(a for a,z in paths[tid]))
        unsafe = any(key.partition('|')[0] not in ('terrain_obstruction', 'earthquake') for key in state.notices)
        self._horizon_pending = (state.pending_action_index, self._horizon_night, paths, unsafe)
        self._miss_geometry=geometry

    def _environment_factor(self, alt, az, posterior=True):
        if self.horizon is None:
            return super()._direction_factor(alt, az)
        state = self.state
        warmup = 0 <= self._horizon_night < WARMUP_NIGHTS
        informed = self._horizon_night >= WARMUP_NIGHTS and self.horizon.informed(az)
        factor = 1.0
        for direction in state.terrain:
            if direction in DIRECTION_AZ and alt < 50 and _az_distance(az, DIRECTION_AZ[direction]) <= 60:
                if not informed:
                    factor = min(factor, .65 if warmup else 0.0)
        for key in state.notices:
            kind, _, direction = key.partition('|')
            if direction not in DIRECTION_AZ:
                continue
            near = _az_distance(az, DIRECTION_AZ[direction]) <= 67.5
            if kind in BLOCKING_KINDS and near and alt < 62:
                return 0.0
            if near and alt < 75:
                factor = min(factor, .35)
        # Retain local zero-score avoidance, including evidence arriving AFTER
        # the five-night height calibration. Upper bounds alone do not certify
        # low-altitude visibility, and frozen height evidence must not suppress
        # the teammate's response to newly encountered obstructions.
        for blocked_az, blocked_alt in state.blocked[-40:]:
            if _az_distance(az, blocked_az) <= 12 and alt <= blocked_alt + 3:
                factor = min(factor, .2)
        if informed and posterior:
            probability = self.horizon.clear_probability(alt, az)
            factor *= 1.0 if probability >= .995 else probability
        return factor

    def _direction_factor(self, alt, az):
        return self._environment_factor(alt, az)

    def _visibility(self, i, duration):
        if self.horizon is None or self._horizon_night < WARMUP_NIGHTS:
            return [(1.0, 1.0)]
        key = (i, duration)
        if key not in self._horizon_cache:
            state = self.state
            path = trajectory(state.ra[i], state.dec[i], self._planning_lst, state.lat, duration)
            self._horizon_cache[key] = self.horizon.visibility_distribution(path)
        return self._horizon_cache[key]

    def _exposure_gain(self, i, duration, k, model, band_scale, program=None):
        distribution = self._visibility(i, duration)
        if distribution == [(1.0, 1.0)]:
            reach = min(1.0, k * duration)
            return self._science_gain(i, reach, model, band_scale, program), float(reach >= .5)
        values = dict.fromkeys((program,) if program else PROGRAMS, 0.0)
        completes = 0.0
        state, scoring = self.state, self.state.scoring
        weight, best = state.weight[i], self._best_scores[i]
        for fraction, probability in distribution:
            reach = min(1.0, k * duration * fraction)
            if self.program_calibration is None:
                # Exact deterministic-band specialization of expected_increment;
                # avoids repeated dictionary/cache/wrapper work inside quadrature.
                band = scoring.program_band(model * fraction * band_scale)
                score = weight * reach
                mismatch_gain = max(0.0, score * scoring.mismatch_multiplier - best)
                for p in values:
                    gain = max(0.0, score * scoring.program_multipliers[p] - best) if p == band else mismatch_gain
                    values[p] += probability * gain
            else:
                for p in values:
                    # Partial terrain changes quality band as well as depth.
                    values[p] += probability * self._science_gain(i, reach, model * fraction, band_scale, p)
            completes += probability * (reach >= .5)
        # Choose a declaration before resolving the latent height, not afterwards.
        return max(values.values()), completes

    def _sync_best_scores(self, message):
        if self._best_scores is None:
            return
        state = self.state
        scores = message.get("best_scores") or []
        if scores and isinstance(scores[0], dict):
            pairs = [(r.get("target_id"), r.get("best_score", 0.0)) for r in scores]
        else:
            pairs = zip(message.get("observed_target_ids", []), scores)
        self._best_scores[:] = [0.0] * len(state.ids)
        for tid, score in pairs:
            i = state.index_of.get(tid)
            value = float(score)
            if i is not None and math.isfinite(value) and value > 0:
                self._best_scores[i] = value

    def _learn_experiments(self, payload, hours):
        state = self.state
        resync = False
        for message in payload.get("new_messages", []):
            if message.get("record_type") == "state_resync":
                self._sync_best_scores(message)
                resync = True
        result = payload.get("last_result") or {}
        if resync or result.get("action") != "observe" or result.get("valid") is False or not state.pending:
            return
        rows = []
        for hit in result.get("hits", []):
            tid, score = hit.get("target_id"), float(hit.get("score", 0.0))
            i = state.index_of.get(tid)
            prediction = state.pending.get(tid)
            if i is None or prediction is None or not math.isfinite(score) or score <= 0:
                continue
            if self._best_scores is not None:
                self._best_scores[i] = max(self._best_scores[i], score)
            if prediction.clean:
                rows.append((state.weight[i], score, prediction.model))
        if self.program_calibration is not None:
            self.program_calibration.observe(state.pending_program, rows, hours)

    def decide(self, payload):
        if self.horizon is not None:
            self._learn_horizon(payload)
            self._horizon_cache.clear()
            night = self.state.current_night(parse_utc(payload['now_utc']))
            new_night = night[0] if night else -1
            if new_night != self._horizon_night and new_night >= WARMUP_NIGHTS:
                self.trace.write({'event': 'merge502_horizon_posterior', 'night_index': new_night,
                                  **self.horizon.snapshot()})
                if self._horizon_night < WARMUP_NIGHTS:
                    snap = self.horizon.snapshot()
                    self.log(f"planner: horizon posterior active; evidence_exposures={snap['evidence_exposures']} "
                             f"informed_sectors={sum(r['used'] for r in snap['sectors'])}")
            self._horizon_night = new_night
            self._horizon_planned = None
        if self.modules:
            self._probabilities_cache.clear()
            now = parse_utc(payload["now_utc"])
            hours = (now - self.state.survey_start).total_seconds() / 3600
            self._learn_experiments(payload, hours)
            if self.program_calibration is not None:
                notices = (payload.get("latest_bulletin") or {}).get("notices", [])
                context = {f"{n.get('event_kind')}|{n.get('direction')}" for n in notices}
                self.program_calibration.advance(hours, context)
        return super().decide(payload)

    def _science_gain(self, i, reach, model, band_scale, program=None):
        state, scoring = self.state, self.state.scoring
        key = (i, model, band_scale)
        probabilities = self._probabilities_cache.get(key)
        if probabilities is None:
            if self.program_calibration is not None:
                probabilities = self.program_calibration.probabilities(model)
            else:
                band = scoring.program_band(model * band_scale)
                probabilities = {p: float(p == band) for p in PROGRAMS}
            self._probabilities_cache[key] = probabilities
        names = (program,) if program else PROGRAMS
        return max(expected_increment(state.weight[i], self._best_scores[i], reach,
                   probabilities[p], scoring.program_multipliers[p], scoring.mismatch_multiplier)
                   for p in names)

    def _joint_search(self, *args, **kwargs):
        if "science_ev" in self.modules:
            return joint_search_ev(self, *args, **kwargs)
        return super()._joint_search(*args, **kwargs)

    def _finish_plan(self, now, lst, c_alt, c_az, chosen, seconds_left, moon, altaz, hours, night_index,
                     fixed_duration=None):
        action = super()._finish_plan(now, lst, c_alt, c_az, chosen, seconds_left, moon, altaz,
                                      hours, night_index, fixed_duration=fixed_duration)
        if action is not None and self.horizon is not None:
            self._horizon_planned = (action, lst)
            for tid in action['assignments'].values():
                i = self.state.index_of[tid]
                clear = sum(fraction * probability for fraction, probability in self._visibility(i, action['duration_seconds']))
                if clear < .995:
                    # Terrain attenuation must not train the instrument-fault detector.
                    self.state.pending[tid] = self.state.pending[tid]._replace(clean=False)
        if action is None or not self.modules.intersection({"science_ev", "program_calibration"}):
            return action
        state = self.state
        band_scale = self._band_scale(night_index)
        expected = dict.fromkeys(PROGRAMS, 0.0)
        for tid in action["assignments"].values():
            i = state.index_of[tid]
            model = state.pending[tid].model
            factor = min(1.0, state.flux[i] * action["duration_seconds"] * model * state.scale * PLAN_FACTOR_SAFETY / state.scoring.f0t0)
            if "science_ev" in self.modules:
                for p in PROGRAMS:
                    k = state.flux[i] * model * state.scale * PLAN_FACTOR_SAFETY / state.scoring.f0t0
                    expected[p] += self._exposure_gain(i, action['duration_seconds'], k, model, band_scale, p)[0]
            else:
                probabilities = self.program_calibration.probabilities(model)
                for p in PROGRAMS:
                    expected[p] += state.weight[i] * factor * (state.scoring.mismatch_multiplier +
                        probabilities[p] * (state.scoring.program_multipliers[p] - state.scoring.mismatch_multiplier))
        program = state.force_program or max(PROGRAMS, key=expected.get)
        action["program"] = state.pending_program = program
        self.trace.write({"event": "merge501_program", "modules": sorted(self.modules), "program": program,
                          "expected": expected, "calibration": self.program_calibration.snapshot() if self.program_calibration else None})
        return action

    def plan(self, now, *args, **kwargs):
        self._request_now = now
        self._planning_lst = local_sidereal_deg(now, self.state.lon)
        return super().plan(now, *args, **kwargs)

    def on_finish(self, payload):
        if self.horizon is not None:
            self.trace.write({'event': 'merge502_horizon_final', **self.horizon.snapshot()})
            self.trace.write(dict(event='merge502_miss_llm_final',calls=self.miss_triage.calls,
                valid_replies=self.miss_triage.valid_replies,soft_labels=self.miss_triage.soft_labels))
        super().on_finish(payload)

    def _request_reachable_later(self, i, lst, threshold, deadline, night_index, seconds_left):
        original = super()._request_reachable_later(i, lst, threshold, deadline, night_index, seconds_left)
        if not original or self.request_filter is None:
            return original
        # A screen only: keep teammate pursuit, exposure caps, tie breaks and budget.
        return self.request_filter.possible(i, self._request_now, deadline, min(1.0, threshold * REQUEST_MARGIN))
