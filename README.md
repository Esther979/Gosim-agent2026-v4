# LLM_agent_v5.0.2_merge

Adds azimuth-dependent terrain-height inference to the selected v5.0.1
science_ev configuration. MERGE502_HORIZON=1 is enabled by default; 0 restores
v5.0.1 behavior. Collect ordinary scored fibre observations during survey nights
1–5; apply their height posterior from night 6. No pointing offsets, dedicated
calibration exposure or hidden scenario inputs. See README.zh.md for inference,
weather controls, posterior integration and identifiability limits.

The early coarse terrain veto below 50 degrees becomes a 0.65 value factor so
ordinary useful observations can sample it. Unobserved sectors retain fallback
rules; at least two distinct nights are needed before using a sector posterior.
Height resolution is one degree in five-degree azimuth sectors. Normal observing
limits prevent identifying the horizon height below the public altitude floor.

Final eight-card fixed-advice replay: horizon off 48278.58; on 47566.91 (-1.47%).
All eight completed with zero planner errors. Required missing 42 to 40; report
reward 800 to 700; average runner time 64.8 to 220.1 seconds. Keep this as an
independent mechanism experiment. Most directions provide upper bounds only.
That fixed-advice comparison did not make API calls or validate the submission platform.

The current miss-triage patch adds MERGE502_MISS_LLM=1. During nights 1–5,
at most one question per night (five total, eight targets each) asks the shared
client to distinguish missed fibre, terrain and insufficient evidence. Absent
hits cannot update height under official v4 semantics. Confident returned-zero
judgments without an ordinary positive control may contribute a quarter-strength
likelihood, subject to altitude, transient-notice and single-sector gates.
Malformed output, timeout or exhausted budget falls back to the existing rules.
LLM_DISABLED/replay makes no classifier calls. Set MERGE502_MISS_LLM=0 to ablate.

Classifier requests temporarily use non-thinking mode on official DeepSeek
(3072 tokens, at most 15 s) or low reasoning effort on supported Kimi Coding
models (4096 tokens, at most 25 s), respecting shorter user timeouts. Other
providers get no extra fields. Original nightly requests are restored afterwards;
the HTTP client differs from the teammate only by a two-line optional options
hook. The shared call/time caps and last-minute wall-clock reserve still apply.
A real Kimi synthetic API smoke succeeded. Live-card integration, compatibility
replay and preserved pre-patch sources/ZIP are outside this package under
../experiments/merge_v502/llm_miss_v1. Replay scores cannot measure this LLM patch.
The Kimi live L2 run completed with 6124.779099 points and zero planner errors:
40 total LLM calls, including two valid classifier replies covering 16 targets;
no soft terrain labels were accepted. This is a single-card integration check.

An independent submission based on Gosim-agent2026-v4. The entry point,
state, fault detector, repair handling and original search settings are
retained. The selected science_ev module and horizon posterior are enabled by
default. No third-party dependencies are needed.

`MERGE501_MODULES=science_ev` is the selected default in both the manifest and
planner. Use `baseline` together with MERGE502_HORIZON=0 to delegate to the teammate planner, or opt into
`program_calibration`, `request_filter`, or comma-separated combinations for
experiments. Program probability calculations are cached per decision.

Previous v5.0.1 paired nightly-advice replay across eight practice cards: baseline 47478.35;
selected science_ev 48278.58 (+1.69%). Program calibration alone added only 0.30%;
request filtering made no score difference; the cached combination scored
48036.93, below science_ev alone. These are local results, not platform claims.
The selected configuration still loses on three cards and misses five more
required targets than the replay baseline; the gain is in aggregate science.

Upload the separately generated LLM_agent_v5.0.2_merge.zip. The platform injects
the API URL/key/model; urllib transport and nightly request defaults are unchanged
from the working teammate client. For local runs configure OPENAI_API_KEY or
KIMI_API_KEY, OPENAI_BASE_URL and OPENAI_MODEL; LLM_DISABLED=1 disables the API.

See README.zh.md for details. Tests, eight-card launchers, replay inputs and
comparison evidence are outside this package in ../experiments/merge_v502.
No credentials, replay inputs, traces or test results belong in the submission.
