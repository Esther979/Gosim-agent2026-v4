"""Required-only score lower bounds; retain the teammate science/sky estimators."""
import math
import os
from .state import SurveyState


def factor_lower(score, weight, multiplier):
    if (not all(math.isfinite(v) for v in (score, weight, multiplier))
            or weight <= 0 or multiplier <= 0):
        return 0.0
    return min(1.0, max(0.0, score - .0000005) / (weight * multiplier))


class GuardedSurveyState(SurveyState):
    def __init__(self, payload):
        super().__init__(payload)
        self.utc_offset_hours = float(payload['site'].get('utc_offset_hours', 0.))
        self.required_guard_enabled = os.environ.get('MERGE503_REQUIRED_GUARD', '1') != '0'
        self.required_indices = [i for i, required in enumerate(self.required) if required]
        self.required_lower = [0.0] * len(self.ids)
        self.required_receipts = {}  # action index -> {target index: (lower, score)}
        self.required_fallback = {}
        self.required_false_completions_prevented = 0
        self.duty_log_closed = False

    def site_closed(self):
        return self.duty_log_closed or super().site_closed()

    def _apply_required_lower(self):
        for i in self.required_indices:
            self.factor[i] = self.required_lower[i]

    def on_result(self, result, hours):
        action_index = self.pending_action_index
        pending = set(self.pending)
        declared = self.pending_program
        rows = {}
        if (self.required_guard_enabled and pending and result
                and result.get('action') == 'observe' and result.get('valid') is not False):
            largest = max(self.scoring.program_multipliers.get(declared, 1.),
                          self.scoring.mismatch_multiplier)
            for hit in result.get('hits', []):
                tid = hit.get('target_id')
                i = self.index_of.get(tid)
                if tid not in pending or i is None or not self.required[i]:
                    continue
                score = float(hit.get('score', 0.))
                if not math.isfinite(score) or score < 0:
                    continue
                lower = factor_lower(score, self.weight[i], largest)
                rows[i] = (lower, score)
                self.required_lower[i] = max(self.required_lower[i], lower)
            if action_index is not None:
                self.required_receipts[action_index] = rows
        super().on_result(result, hours)
        if self.required_guard_enabled:
            threshold = self.scoring.required_threshold
            self.required_false_completions_prevented += sum(
                self.factor[i] >= threshold > self.required_lower[i] for i in rows)
            self._apply_required_lower()

    def _resync(self, message):
        super()._resync(message)
        if not self.required_guard_enabled:
            return
        window = message.get('invalidated_window') or {}
        lo, hi = window.get('action_index_start'), window.get('action_index_end_exclusive')
        if lo is None or hi is None:
            self.required_receipts.clear()
        else:
            self.required_receipts = {a: rows for a, rows in self.required_receipts.items()
                                      if not lo <= a < hi}
        values = message.get('best_scores') or []
        pairs = ((r.get('target_id'), r.get('best_score', 0.)) for r in values) if values and isinstance(values[0], dict) else zip(message.get('observed_target_ids', []), values)
        best = {self.index_of[tid]: float(score) for tid, score in pairs if tid in self.index_of}
        largest = max([self.scoring.mismatch_multiplier, *self.scoring.program_multipliers.values()])
        # Unknown/restarted history uses all public multipliers. Known surviving
        # receipts keep their tighter per-declaration lower bounds.
        self.required_fallback = {i: factor_lower(best.get(i, 0.), self.weight[i], largest)
                                  for i in self.required_indices}
        self.required_lower[:] = [0.] * len(self.ids)
        for i, lower in self.required_fallback.items():
            self.required_lower[i] = lower
        for action, rows in self.required_receipts.items():
            for i, (lower, score) in list(rows.items()):
                # A complete resync list is authoritative even if the invalidation
                # description is inconsistent or this process lost part of history.
                if not math.isfinite(best.get(i, 0.)) or score > best.get(i, 0.) + .000001:
                    del rows[i]
                    continue
                self.required_lower[i] = max(self.required_lower[i], lower)
        self._apply_required_lower()

    def required_snapshot(self):
        return dict(enabled=self.required_guard_enabled,
            confirmed=sum(self.required_lower[i] >= self.scoring.required_threshold for i in self.required_indices),
            remaining=sum(self.required_lower[i] < self.scoring.required_threshold for i in self.required_indices),
            false_completion_updates_prevented=self.required_false_completions_prevented,
            receipt_actions=len(self.required_receipts))
