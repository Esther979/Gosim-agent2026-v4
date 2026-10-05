"""Causality, score censoring and network isolation contracts; stdlib unittest."""
import io
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent_core.conditions import ConditionsEstimator, event_keys, quality_sample
from agent_core.geometry import parse_utc
from agent_core.llm_client import LLMClient
from agent_core.scoring import ScoringModel
from agent_core.state import PendingPrediction

NOW = "2026-11-02T01:00:00Z"
NEXT = "2026-11-02T01:15:00Z"


def state():
    return SimpleNamespace(
        scale=0.7, weight=[1.0], flux=[0.5], index_of={"target": 0},
        scoring=ScoringModel({}, {}),
        current_night=lambda now: (0, parse_utc("2026-11-01T23:00:00Z"),
                                  parse_utc("2026-11-02T09:00:00Z")))


def payload(now=NOW, notices=(), result=None, messages=()):
    return {"now_utc": now, "latest_bulletin": {"issued_at_utc": now, "notices": list(notices)},
            "new_messages": list(messages), "last_result": result}


def notice(kind="haze", direction="E"):
    return {"event_kind": kind, "direction": direction}


class ConditionsTests(unittest.TestCase):
    def setUp(self):
        self.model = ConditionsEstimator(state(), {"blend": 1.0})
        self.model.update(payload())

    def remember(self):
        self.model.remember(parse_utc(NOW), 7,
                            {"target": PendingPrediction(1, 1 / .95, 45, 90, True)},
                            "BACKUP", 900)

    def result(self, score):
        return {"action": "observe", "observe_index": 7,
                "hits": [{"target_id": "target", "score": score}]}

    def test_direction_wrap_and_zenith(self):
        self.assertEqual(event_keys([notice(direction="N")], 45, 355), (("haze", "N"),))
        self.assertFalse(event_keys([notice(direction="N")], 80, 355))
        self.assertEqual(event_keys([notice(direction="ALL")], 80, 355), (("haze", "ALL"),))

    def test_unknown_events_and_duplicate_notices(self):
        self.assertEqual(event_keys([notice(), notice(), notice("earthquake")], 45, 90), (("haze", "E"),))

    def test_notice_penalizes_only_relevant_direction(self):
        self.model.update(payload(notices=[notice()]))
        east, west = self.model.predict(45, 90), self.model.predict(45, 270)
        self.assertLess(east.effective_scale, west.effective_scale)
        self.assertAlmostEqual(east.band_scale, west.band_scale)

    def test_empty_bulletin_clears_previous_notice(self):
        self.model.update(payload(notices=[notice()]))
        self.model.update(payload(NEXT))
        self.assertFalse(self.model.predict(45, 90).active_events)

    def test_future_bulletin_not_used(self):
        data = payload()
        data["latest_bulletin"] = {"issued_at_utc": NEXT, "notices": [notice()]}
        self.model.update(data)
        self.assertFalse(self.model.predict(45, 90).active_events)

    def test_forecast_does_not_mean_whole_night_closed(self):
        forecast = {"record_type": "forecast", "issued_at_utc": NOW,
                    "notices": [{**notice("rain", "ALL"), "nights": ["2026-11-01"]}]}
        self.model.update(payload(messages=[forecast]))
        estimate = self.model.predict(45, 90, 3600)
        self.assertTrue(estimate.forecast_events)
        self.assertGreater(estimate.effective_scale, .5)
        self.assertEqual(estimate.usable_probability, 1.0)

    def test_future_forecast_is_not_used(self):
        forecast = {"record_type": "forecast", "issued_at_utc": NEXT,
                    "notices": [{**notice(), "nights": ["2026-11-01"]}]}
        self.model.update(payload(messages=[forecast]))
        self.assertFalse(self.model.predict(45, 90).forecast_events)

    def test_different_night_forecast_is_not_used(self):
        forecast = {"record_type": "forecast", "issued_at_utc": NOW,
                    "notices": [{**notice(), "nights": ["2026-11-03"]}]}
        self.model.update(payload(messages=[forecast]))
        self.assertFalse(self.model.predict(45, 90).forecast_events)

    def test_unsaturated_proxy_and_censored_score(self):
        scoring = self.model.state.scoring
        self.assertAlmostEqual(quality_sample(.318, 1, .5, 900, 1, "BACKUP", scoring), .3)
        self.assertIsNone(quality_sample(1.2, 1, .5, 900, 1, "DARK", scoring))
        self.assertIsNone(quality_sample(1, 1, .5, 900, 1, "BACKUP", scoring))
        self.assertIsNone(quality_sample(0, 1, .5, 900, 1, "BACKUP", scoring))

    def test_result_learns_previous_context_once(self):
        self.model.update(payload(notices=[notice()]))
        self.remember()
        self.model.update(payload(NEXT, result=self.result(.318)))
        self.assertEqual(self.model.updates, 1)
        self.assertIn((("haze", "E"),), self.model.context_samples)
        self.assertNotIn((), self.model.context_samples)
        self.model.update(payload(NEXT, result=self.result(.318)))
        self.assertEqual(self.model.updates, 1)

    def test_wrong_observation_index_is_ignored(self):
        self.remember()
        result = self.result(.318)
        result["observe_index"] = 6
        self.model.update(payload(NEXT, result=result))
        self.assertEqual(self.model.updates, 0)

    def test_geometry_miss_is_not_bad_weather(self):
        self.remember()
        self.model.update(payload(NEXT, result={"action": "observe", "observe_index": 7, "hits": []}))
        self.assertEqual(self.model.updates, 0)
        self.assertEqual(self.model.zero_hits, 0)
        self.assertEqual(self.model.predict(45, 90).usable_probability, 1)

    def test_zero_hit_is_local_and_recovers(self):
        self.remember()
        self.model.update(payload(NEXT, result=self.result(0)))
        self.assertLess(self.model.predict(45, 90).usable_probability, 1)
        self.assertEqual(self.model.predict(45, 270).usable_probability, 1)
        self.assertEqual(self.model.updates, 0)
        self.model.update(payload("2026-11-02T03:00:00Z"))
        self.assertEqual(self.model.predict(45, 90).usable_probability, 1)

    def test_parameters_and_finite_estimates(self):
        for params in ({"base_scale": float("nan")}, {"blend": 1.1}, {"event_factors": {"haze": 0}}):
            with self.assertRaises(ValueError):
                ConditionsEstimator(state(), params)
        for alt in (30, 45, 80):
            for az in range(0, 360, 30):
                estimate = self.model.predict(alt, az)
                self.assertTrue(math.isfinite(estimate.effective_scale))
                self.assertTrue(0 <= estimate.usable_probability <= 1)

