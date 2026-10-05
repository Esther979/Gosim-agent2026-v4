import json
import math
import os
from pathlib import Path
import random
import unittest
from unittest.mock import patch

from test_conditions import state, payload, notice
from agent_core.conditions import ConditionsEstimator
from agent_core.llm_client import LLMClient, require_api_key
from agent_core.weather_adapter import build_conditions
from agent_core.state import SurveyState
from agent_core.planner import Planner
from agent_core.geometry import parse_utc, local_sidereal_deg
from types import SimpleNamespace


class TransferTests(unittest.TestCase):
    def test_fully_exposed_lower_band_can_still_gain_science_score(self):
        p = Planner.__new__(Planner)
        p.state = SimpleNamespace(factor=[1.], best_score=[1.], misses=[0],
                                  weight=[1.], required=[False], scoring=state().scoring)
        p.incremental_scores = False
        self.assertEqual(p._value(0), 0.)
        p.incremental_scores = True
        self.assertAlmostEqual(p._value(0), .2)
        p.state.best_score[0] = 1.2
        self.assertEqual(p._value(0), 0.)

    def test_planning_never_commits_predicted_scores_or_completion(self):
        now = parse_utc('2026-11-02T01:00:00Z')
        ra = (local_sidereal_deg(now, -17.)-25.) % 360.
        initial = {'site': {'latitude_deg': 28., 'longitude_deg': -17.},
                   'instrument': {'grid_side': 4, 'n_fibers': 16, 'glass_side_deg': .6,
                                  'pitch_deg': .63, 'fov_side_deg': 2.52},
                   'survey': {'start_utc': '2026-11-01T00:00:00Z',
                              'end_utc': '2026-11-03T10:00:00Z',
                              'nights': [{'observing_start_utc': '2026-11-02T00:00:00Z',
                                          'observing_end_utc': '2026-11-02T08:00:00Z'}]},
                   'targets': {'columns': ['target_id', 'ra_deg', 'dec_deg', 'feature_flux',
                                           'science_weight', 'required'],
                               'rows': [['A', ra, 10., .5, 1., True]]}}
        s = SurveyState(initial)
        with patch.dict(os.environ, {'LLM_DISABLED': '1', 'AGENT_CONDITIONS': 'quality',
                                    'AGENT_SCORE_GAIN': '1'}, clear=True):
            p = Planner(s)
        p.conditions.update(payload())
        p._current_action_index = 7
        result = p.plan(now, parse_utc('2026-11-02T08:00:00Z'), 0, 25.)
        self.assertEqual(result['action'], 'observe')
        self.assertEqual(s.best_score, [0.])
        self.assertEqual(s.factor, [0.])
        self.assertEqual(set(s.pending), set(result['assignments'].values()))
        self.assertEqual(p.conditions.pending['index'], 7)

    def test_optional_transfer_uses_only_frozen_local_weights(self):
        with patch.dict(os.environ, {'AGENT_CONDITIONS': 'quality'}, clear=True):
            mode, model = build_conditions(state())
        self.assertEqual(mode, 'quality')
        model.update(payload())
        before = model.predict(45, 90, 900).effective_scale
        model.update(payload(notices=[notice('overcast', 'ALL')]))
        self.assertNotEqual(before, model.predict(45, 90, 900).effective_scale)
        with patch.dict(os.environ, {'AGENT_CONDITIONS': 'off'}, clear=True):
            self.assertEqual(build_conditions(state()), ('off', None))

    def test_offline_flag_never_calls_network(self):
        with patch.dict(os.environ, {'LLM_DISABLED': '1'}, clear=True), \
                patch('urllib.request.urlopen', side_effect=AssertionError('network forbidden')) as net:
            require_api_key()
            client = LLMClient()
            self.assertIsNone(client.ask_json('test', {}, 900))
            self.assertEqual(client.calls_made, 0)
            net.assert_not_called()

    def test_resync_restores_scores_from_public_ledger(self):
        s = SurveyState.__new__(SurveyState)
        s.ids = ['A', 'B']
        s.best_score = [1.2, .8]
        s.weight = [1., 1.]
        s.factor = [1., .8]
        s.hmax = [80., 80.]
        s.scoring = state().scoring
        s.pending = {}
        s.ledger = [(1, 'A', .5), (2, 'A', 1.)]
        s._resync({'invalidated_window': {'action_index_start': 2, 'action_index_end_exclusive': 3},
                   'best_scores': [{'target_id': 'A', 'best_score': .56}]})
        self.assertEqual(s.best_score, [.56, 0.])
        self.assertEqual(s.factor, [.5, 0.])


if __name__ == '__main__':
    unittest.main()
