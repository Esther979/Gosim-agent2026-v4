import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent_core.conditions import ConditionsEstimator
from agent_core.reconstruction import FEATURES, FUTURE_FEATURES, ReconstructionModel
from test_conditions import state, payload, notice, NEXT


def identity_model():
    coefficients = {name: [0.0]*(len(FEATURES)+1) for name in ("quality", "band", "usable")}
    coefficients["quality"][1] = 1.0
    coefficients["band"][1] = 1/.95
    coefficients["usable"][0] = 1.0
    return {"features": list(FEATURES), "center": [0.]*len(FEATURES),
            "scale": [1.]*len(FEATURES), "coefficients": coefficients,
            "training_quality_rmse": .1}


class ReconstructionTests(unittest.TestCase):
    def test_future_features_keep_current_information_and_encode_lead(self):
        model = ConditionsEstimator(state())
        model.update(payload(notices=[notice()]))
        present = model.feature_vector(45, 90, 900)
        future = model.future_feature_vector(45, 90, 900, 3600)
        self.assertEqual(len(future), len(FUTURE_FEATURES))
        self.assertEqual(future[:len(FEATURES)], present)
        self.assertEqual(future[FUTURE_FEATURES.index("lead_hours")], 1)
        self.assertLess(future[FUTURE_FEATURES.index("retention")], 1)

    def test_wrong_dimension_is_not_silently_truncated(self):
        model = ReconstructionModel(identity_model())
        with self.assertRaises(ValueError):
            model.predict([1.]*len(FUTURE_FEATURES))

    def test_schema_rejects_wrong_features_and_nonfinite_weights(self):
        record = identity_model()
        record["features"] = ["wrong"]
        with self.assertRaises(ValueError):
            ReconstructionModel(record)
        record = identity_model()
        record["coefficients"]["quality"][0] = float("nan")
        with self.assertRaises(ValueError):
            ReconstructionModel(record)

    def test_public_features_have_exact_training_schema(self):
        model = ConditionsEstimator(state())
        model.update(payload(notices=[notice()]))
        values = model.feature_vector(45, 90, 900)
        self.assertEqual(len(values), len(FEATURES))
        self.assertTrue(all(math.isfinite(v) for v in values))
        self.assertEqual(values[FEATURES.index("sector_haze")], 1)
        self.assertEqual(values[FEATURES.index("all_haze")], 0)
        self.assertEqual(values[FEATURES.index("duration")], .25)
        self.assertEqual(values[FEATURES.index("scale_sector_haze")], .7)

    def test_truth_fields_do_not_change_features_or_predictions(self):
        clean = ConditionsEstimator(state(), {"supervised_model": identity_model()})
        polluted = ConditionsEstimator(state(), {"supervised_model": identity_model()})
        data = payload()
        clean.update(data)
        polluted.update({**data, "weather_truth": {"quality": 0}, "instrument_efficiency": 0,
                         "future_weather": [{"quality": 10}], "card_id": "memorise-me"})
        self.assertEqual(clean.feature_vector(45, 90, 900), polluted.feature_vector(45, 90, 900))
        self.assertEqual(clean.predict(45, 90, 900), polluted.predict(45, 90, 900))

    def test_identity_model_preserves_baseline_scale(self):
        model = ConditionsEstimator(state(), {"supervised_model": identity_model()})
        model.update(payload(notices=[notice()]))
        estimate = model.predict(45, 90, 900)
        self.assertAlmostEqual(estimate.effective_scale, .7)
        self.assertAlmostEqual(estimate.band_scale, .7/.95)
        self.assertEqual(estimate.usable_probability, 1)

    def test_quality_mean_is_not_discounted_twice(self):
        record = identity_model()
        record["coefficients"]["usable"][0] = .5
        model = ConditionsEstimator(state(), {"supervised_model": record, "supervised_blend": 1})
        model.update(payload())
        estimate = model.predict(45, 90)
        self.assertAlmostEqual(estimate.effective_scale, .7)
        self.assertAlmostEqual(estimate.quality_scale*estimate.usable_probability, .7)

    def test_unannounced_fields_and_future_forecast_cannot_leak(self):
        model = ConditionsEstimator(state())
        model.update(payload(messages=[{"record_type": "forecast", "issued_at_utc": NEXT,
            "notices": [{**notice(), "nights": ["2026-11-01"]}]}]))
        values = model.feature_vector(45, 90, 900)
        self.assertEqual(values[FEATURES.index("forecast_haze")], 0)

    def test_cached_history_expires_after_new_feedback_and_scale_change(self):
        model = ConditionsEstimator(state())
        model.update(payload())
        first = model.feature_vector(45, 90, 900)
        model.state.scale = .4
        changed = model.feature_vector(45, 90, 900)
        self.assertNotEqual(first[FEATURES.index("context_proxy")], changed[FEATURES.index("context_proxy")])
        model.update(payload(NEXT))
        from agent_core.geometry import parse_utc
        model.zeros.append((parse_utc(NEXT), 90, 45))
        model.context_samples[()] = [(parse_utc(NEXT), .2)]
        second = model.feature_vector(45, 90, 900)
        self.assertEqual(second[FEATURES.index("local_zero")], 1.)
        self.assertEqual(second[FEATURES.index("context_proxy")], .2)


if __name__ == "__main__":
    unittest.main()
