# Independent v5.0.2 merge

Preserve all earlier releases and the mainline. Based on v5.0.1 science_ev.
Keep teammate base_planner, state, agent, validation, protocol, scoring, geometry
and memory byte-identical. The HTTP client has only a scoped, optional
provider-options hook for miss triage; preserve all original nightly requests. New horizon inference belongs
in horizon_posterior.py and the planner subclass / joint science search.

First five survey nights collect public, ordinary observation receipts without
pointing offsets or dedicated calibration exposures. From night six use the
height posterior in expected science gain and required completion probability.
Keep uncertainty for unobserved directions; LLM may adjudicate missing/zero outcomes using public geometry and receipts.
Preserve official v4 semantics: absent hit cannot certify terrain; uncertain
returned zero hits may add explicitly marked weak LLM evidence. Preserve fault detection, repairs, pacing and search limits.

Only public protocol and own receipts may inform decisions. Never read hidden
card truth at runtime or package credentials, tapes, tests, logs or caches.
Experiments go in ../experiments/merge_v502. MERGE502_HORIZON=0 must recover v5.0.1.
