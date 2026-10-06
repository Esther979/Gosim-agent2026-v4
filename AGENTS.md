# Independent v5.1.0 merge

Based on v5.0.3. This version adds complete neighbourhood coverage with cached spatial indexing and per-pointing cap filtering. Preserve older releases and Shiming mainline.
Use one shared field-search helper in science joint search, baseline joint search and ordinary/forced-request fallback search. Keep geometry/scoring, gains, candidate count, pacing, required guard and duty-log interpretation unchanged.
MERGE510_NEIGHBOUR_SEARCH=0 must recover v5.0.3 decisions and original spatial lookup. Default enabled. Handle RA wrap, polar caps, rounding and different public fibre grids. Validate against full-catalogue projection.
Experiments belong in ../experiments/merge_v510; local API defaults to gosim-observer-examples/python/.env.example.deepseek. Never package keys, recordings or tests.
