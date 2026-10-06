# LLM_agent_v5.1.0_merge

Independent release based on v5.0.3. Adds complete field-neighbourhood coverage to science joint search, baseline joint search and ordinary/forced-request field selection. Retains conservative required completion, public duty logs, raised-horizon learning, gain formulas and pacing.

Each anchor performs one query enclosing all currently tested fibre placements. Its radius derives from the instrument geometry, with a pointing-rounding allowance. Each actual pointing performs a cheap spherical-cap check before exact gnomonic projection and fibre/glass/margin classification. Gains are computed lazily for targets on glass. Per-search target vectors and per-band RA keys are cached. Spatial queries handle RA wrap and polar caps.

`MERGE510_NEIGHBOUR_SEARCH=1` is the default. Set it to `0` to recover v5.0.3 lookup and decisions. Existing `MERGE501_MODULES`, `MERGE502_*` and `MERGE503_*` flags retain their roles.

Public B verification compared every one of 2,495 sampled pointings with a full visible-catalogue projection: 504 omission fields before, zero after. This is geometric coverage, not a formal B/B1 score. Validation lives outside the submission in `experiments/merge_v510`; see README.zh.md for details.

From the workspace root, run all eight practice cards in parallel:

```sh
python3 run_merge510_all8.py --jobs 8
```

Local live debugging defaults to `gosim-observer-examples/python/.env.example.deepseek`; explicit `--env-file` overrides it. Replay/offline runs disable API calls. Platform API injection and the original HTTP client are retained; the ZIP contains no credentials.

```sh
python3 pack_agent.py --out ../LLM_agent_v5.1.0_merge.zip
```
