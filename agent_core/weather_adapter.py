"""Optional transfer of the public-input ridge estimator into the joint planner.

The existing online scale, sky-band inference, fault detector and LLM weekly
outlook remain independent. No training data or simulator imports are deployed.
"""
import json
import os
from pathlib import Path

from .conditions import ConditionsEstimator


def build_conditions(state):
    mode = os.environ.get('AGENT_CONDITIONS', 'off')
    if mode not in ('off', 'quality'):
        raise ValueError('AGENT_CONDITIONS must be off or quality')
    if mode == 'off':
        return mode, None
    path = Path(os.environ.get('AGENT_CONDITIONS_CONFIG') or
                Path(__file__).with_name('conditions_config.json'))
    return mode, ConditionsEstimator(state, json.loads(path.read_text()))
