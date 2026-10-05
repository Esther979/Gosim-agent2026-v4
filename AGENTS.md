# AGENTS.md -- python-agent

Guide for AI coding assistants working in this project. Humans: see README.md /
README.zh.md.

## What this is

A complete, standard-library-only Python agent for the GOSIM survey26 hackathon's
telescope-survey challenge. It speaks `participant-agent-protocol-v4` (one JSON
object per line on stdin/stdout), runs a joint pointing/fibre/exposure planner. The inherited LLM branch reads weekly
forecasts into usable-night estimates and arbitrates borderline instrument-fault
alarms. Optional transferred weather and incremental-score modules are available. Read `agent.py` first -- it is the thin
stdin/stdout loop that dispatches to everything else.


## Local integration branch (2026-10-05)

Based on origin/LLM commit d1d51da58e82e9607a38d0b40b71fd45970dbc59.
The default keeps the teammate strategy. The combined experimental candidate is
selected explicitly with AGENT_CONDITIONS=quality and AGENT_SCORE_GAIN=1.
AGENT_CONDITIONS=off and AGENT_SCORE_GAIN=0 restore its numerical policy.
The spatial-index cache is always active and preserved all four baseline scores.

- conditions.py/reconstruction.py contain frozen, public-input ridge inference;
  conditions_config.json contains fitted coefficients, never practice truth.
- weather_adapter.py loads only that model. Quality is evaluated at a representative
  900-second duration. Sky-band inference and the fault detector remain inherited.
- state.best_score stores actual received scores. Resync replaces those scores from
  the public ledger; planning must never write hypothetical credit to real state.
- The two inherited LLM stages remain intact. Two call sites are not proof that both
  succeeded in every run; inspect successful replies and applied outputs separately.
- LLM_DISABLED=1 is a numerical development ablation. Never use hidden card fields
  as runtime inputs. Offline leave-one-card-out models are supplied by the external
  benchmark harness through AGENT_CONDITIONS_CONFIG.
- Experiments, labels, source snapshots and logs belong outside this deployable repo,
  in ../gosim-observer-examples/experiments/teammate_merge/. Preserve negative trials.
- Keep credentials outside Git and archives. Do not copy the existing local .env.

Run `python3 -B -m unittest discover -s tests -q` and
`python3 -B -m agent_core.planner` after relevant changes. Inspect full-run scores,
missed required targets, termination, CPU and fault timing before changing defaults.

## Module map

```
agent.py                 entry point: stdin/stdout loop, one branch per message type
agent_core/
  protocol.py             transport: read/write one JSON object per line
  state.py                 SurveyState: catalogue + everything tracked across decisions
  geometry.py              public sky maths (sidereal time, alt/az, fibre grid)
  scoring.py               factor/score estimates from PUBLIC scoring config only
  planner.py               decision logic: wait / observe / report / finish
  llm_client.py            OpenAI-compatible chat client, defaults to Kimi Coding Plan
  memory.py                optional, best-effort JSONL decision trace (off by default)
  validation.py            protocol-legal action checking + a deterministic fallback
observer.project.json      platform project manifest (image, run command, env)
requirements.txt           none needed -- standard library only
pack_agent.py              zip this folder into a project ZIP for submission
```

## Changing the strategy

Target ranking, fibre filling and exposure sizing live in `agent_core/planner.py`.
The two LLM-advised calls (forecast avoidance + bulletin check-in) are issued from
the night-advice path in `planner.py` through `agent_core/llm_client.py`; the
instrument-fault confirmation call is a third, rarer call from the same module. You
can change the ranking heuristic, add new signals to `SurveyState`, change what (if
anything) the LLM is asked, or replace the LLM calls entirely with a different
approach -- the only hard requirement enforced by `validation.py` is that whatever
`agent.py` writes to stdout is a protocol-legal response.

## Configuring the LLM (Kimi key / base URL / model)

Copy `.env.example` to `.env` and set:

- `OPENAI_API_KEY` (or `KIMI_API_KEY`) -- required to run the bundled planner as-is.
- `OPENAI_BASE_URL` -- defaults to the Kimi Coding Plan endpoint
  (`https://api.kimi.com/coding/v1`; use `https://api.kimi.ai/coding/v1` outside
  mainland China) when unset.
- `OPENAI_MODEL` -- defaults to `k3` when unset.

Any other OpenAI-compatible `/chat/completions` endpoint works too -- just point
`OPENAI_BASE_URL` / `OPENAI_MODEL` at it. On the platform, `OPENAI_BASE_URL` /
`OPENAI_API_KEY` are injected automatically for every run (the platform's own model
proxy and a temporary credential); `.env` is never uploaded and is excluded by
`pack_agent.py`.

## Packing and uploading as a complete project

```bash
python3 pack_agent.py --out ../python-agent.zip
```

This zips the project with `observer.project.json` at the root, skipping `.env`,
`__pycache__`, and other local-only files. Upload the resulting ZIP as a complete
project on the Participate page, or push this folder to a GitHub repository and
submit the repo URL instead.

## Protocol & scoring

Full protocol reference and scoring formulas are not duplicated here -- see `docs/`
(bundled in the downloadable ZIP of this example) for the complete participant
guide in Chinese and English.
