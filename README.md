# python-agent -- GOSIM survey26 hackathon, participant-agent-protocol-v4

Team entry by Esther (Jingmeng) and Xinran Wang, built on the organisers' example
project (`gosimfoundation/hackathon-survey26`, `examples/python-agent`, CC BY-NC 4.0);
the scheduler, learning models and LLM stages described below are our own work. The
fast fault probe and the pointing-offset estimate take their ideas from the organisers'
`python-pro` reference agent (same licence), reimplemented here.

The agent is a joint pointing / fibre / exposure planner that ranks candidate fields
by expected score per second, learns the things the task card does not state from
its own results, and asks a language model a few bounded questions per run where the
inputs are natural language. Everything the model says is clamped by rules before it
can change a decision; required targets, exposures and fault reports never depend on
a model answer alone.

What is learned at run time, from the agent's own hits (never from a file):

- the sky-quality scale and the program band the sky is really in;
- **terrain**: each announced obstruction is a sector of unknown height and width; a
  grid of hypotheses is weighted by every zero-scoring and scoring hit near it, and
  pointings are valued by the probability that they are clear (`state.Ridge`);
- **instrument faults, fast**: a fault is a step down of the instrument alone, so the
  clean quality ratios of the last three exposures of the night are compared with the
  six before them; a step to under 55% that is not also a drop of the sky band (which
  weather would lower and a fault does not) is reported at once with a free report --
  one per stretch between correct reports, none in the 12 hours after an earthquake
  notice appears;
- **instrument faults, slow**: a night-by-night censored median of clean quality
  ratios, corrected for the sky level the program bands reveal, against the typical
  quiet level, with an earthquake recovery curve; it catches milder faults and owns
  the paid reports;
- **pointing offset** (Hard-mode cards): every assigned target that hit or missed its
  fibre narrows a grid of candidate offsets; once one explains the outcomes clearly
  better than none, pointings are commanded so that they land where planned;
- **pace**: the measured cost of a decision and the engine's own time per decision,
  so the search depth and the optional model calls fit under both the fair clock and
  the real-time cap the platform reports (`wallclock.wall_remaining_seconds`).

Switches in `observer.project.json`: `AGENT_SCORE_GAIN=1` values a target by its
predicted score minus the best it already has; `AGENT_CONDITIONS=off` keeps the
learned sky scale (the fitted weather model it can swap in lost in leave-one-card-out
validation). `LLM_DISABLED=1` gives a deterministic, model-free run for comparisons.

[中文说明见 README.zh.md](README.zh.md)

A small, Codex-style reference implementation for the GOSIM survey26 hackathon's
telescope-survey challenge. It speaks `participant-agent-protocol-v4` (one JSON object
per line on stdin/stdout) and is organized as a real multi-module project instead of a
single script, so you can see where each concern lives and lift whichever part you need.

It runs a real anchor-search planner (rank candidates, try each fibre as the pointing
centre, fill the other 15 with the best-value neighbours, size the exposure and pick a
program by expected rate) -- the same algorithm as this repo's companion TypeScript
example, so the two are directly comparable -- plus two LLM-advised planning calls
each night. Needs an API key to run; see Configuration below. Read it, run it, and
build your own strategy from the parts that are useful to you.

## Layout

```
agent.py                 entry point: the stdin/stdout loop, one message type per branch
agent_core/
  protocol.py             transport: read one JSON object per line, write one back
  state.py                 SurveyState: catalogue + everything tracked across decisions
  geometry.py              public sky maths: sidereal time, alt/az, the 16-fibre grid
  scoring.py               factor/score estimates from PUBLIC scoring config only
  planner.py               decision logic: wait / observe / report / finish
  llm_client.py            an OpenAI-compatible chat client, defaults to Kimi Coding Plan
  memory.py                an optional, best-effort JSONL decision trace (off by default)
  validation.py            protocol-legal action checking + a deterministic fallback
observer.project.json      the platform's project manifest (image, run command, env)
requirements.txt           none needed -- standard library only
.env.example               copy to .env and set an API key before running
pack_agent.py              zip this folder into a project ZIP for submission
```

## Why these modules

- **protocol.py** -- isolates the wire format (JSON-per-line, stdout-only-for-protocol,
  stderr-only-for-logs) so nothing else in the project has to think about it.
- **state.py** -- the single source of truth for what the agent currently knows: the
  target catalogue from `initialize` (as parallel arrays, one slot per target), a
  declination-band spatial index for fast "what's near this point" queries, per-target
  `factor`/`misses`/`attempts`, a learned sky `scale` (the median of recent quality
  samples backed out of the agent's own hits -- the hidden instrument/transparency/sky/
  seeing terms are never read from a file), weather notices, and fault-diagnosis
  history. Built once, updated on every `decision_request`. Also keeps a per-observe-
  action ledger of exact factors, so a Hard-mode `state_resync` can be answered exactly
  -- by dropping the invalidated action-index window from the ledger and recomputing
  each target's best factor from what is left -- instead of only from the resync
  message's `best_scores` (see the comment on `_resync`).
- **geometry.py** -- the public sidereal-time / alt-az / fibre-grid / lunar-factor
  formulas from the participant guide's Geometry and Scoring sections. Any agent
  needs these; they involve no scenario data.
- **scoring.py** -- the *public* parts of `initialize.payload.scoring` (`q0`,
  `flux_zero_point`, `exposure_zero_point_seconds`, `airmass_exponent`, `lunar_model`,
  `program.{bands,multipliers,mismatch_multiplier}`, `required.*`, `uniformity.*`) turned
  into factor/band/multiplier estimates.
- **planner.py** -- an anchor-search strategy: rank visible not-yet-done targets (a
  required target not yet past the public factor threshold gets a large bonus; targets
  setting soon or with few nights left rank higher); for the best few candidates, try
  every fibre as the pointing centre and fill the rest with the best-value neighbours
  that land on the glass; keep the best pointing; pick the exposure length with the best
  expected score per second and the program most assignments will match. Sleeps through
  the day, closes the shutter on an all-sky rain/storm bulletin, and reports a suspected
  instrument fault only after a sustained, unexplained quality drop confirmed over
  several nights. Time-limited observation requests (`active_requests`) get a read-only
  tie-break in the exposure-length search: among durations within a small tolerance of
  the best score-per-second rate, one that also clears a still-needed target's
  `completion_factor_threshold` wins -- it never redirects the pointing itself or chases
  a target that is not already going to be exposed anyway (there is no penalty for
  letting a request expire, so this is opportunistic, not a hunt).
- **llm_client.py** -- an OpenAI-compatible client (plain `urllib`, no SDK) defaulting
  to the Kimi Coding Plan endpoint, configurable to any other OpenAI-compatible
  `/chat/completions` endpoint by environment variable.
- **memory.py** -- an optional, best-effort JSONL decision trace (night advice, the
  finish summary), off by default. The state/planner's own working memory (learned
  scale, per-target progress, night advice, report cooldowns) lives on
  `SurveyState`/`Planner` themselves, since that is genuinely part of what they track.
- **validation.py** -- the last line of defense before anything reaches stdout: checks
  every field the protocol actually validates (ranges, duplicate fibres/targets, unknown
  fields, the consecutive-report limit) and provides `fallback_action()`, a response
  that is always legal no matter what state the agent is in.

## Where the LLM is used

Every call goes through `LLMClient.ask_json` and is answered as one JSON object that
the planner sanitises before use. There are two kinds of call, neither on the
per-decision path:

1. **Reading the forecast and the duty log** (`Planner._on_forecast`). When a weekly
   forecast arrives, the model is asked for the usable fraction of every night it
   covers (and up to seven nights ahead), given the forecast notices, the fixed-table
   estimate, and `experience`: what nights under each notice kind were really worth so
   far in this run. Any long free-text field that arrived in the meantime -- an
   observation request's `reason`, which on some cards carries informal hand-over
   notes between observers, in several languages, with code words, simple ciphers and
   corrections -- is handed over as `duty_log`. From it the model may also return
   `terrain`: how high an obstruction in some compass direction reaches. Bounds: a
   night the notes alone call bad is still planned at a quarter of its worth; a
   terrain height becomes a soft prior that the ridge model's own evidence overrides;
   a night the model leaves out keeps the rule's value.
2. **Fault arbitration** (`Planner._arbitrate`). When the rule-based detector reaches
   its reporting threshold on borderline evidence, the model sees the last nights'
   ratios, sky levels and notices and may veto the report for one more night (at most
   five times per alarm). It cannot trigger a report; the strong-evidence path never
   asks.

Calls are skipped, and the rule answers, when the model budget, the fair clock or the
projected real wall clock (engine time included) would not allow them.

### Call behaviour

`LLMClient` enforces, regardless of what the model does:

- a per-call timeout (30 s) that also shrinks as the wall clock runs low (never cuts
  into the last 60 s of the survey's own budget);
- a total model time budget per run (500 s, scaled to the card's limit);
- one attempt per question; every exception (network, timeout, bad JSON, missing
  fields) is caught and `ask_json` returns `None`, which the planner treats as "no
  answer, use the rule";
- the key is read from the environment only and never logged; the HTTP `User-Agent`
  is the one the manifest sets.

## Configuration (.env)

Copy `.env.example` to `.env` and set an API key before running locally:

```
OPENAI_API_KEY=sk-...
```

That alone is enough: the client defaults to the
[Kimi Coding Plan](https://www.kimi.com/code/docs/en/) endpoint
`https://api.kimi.com/coding/v1` with model `k3`. Accounts outside mainland China
should instead use `https://api.kimi.ai/coding/v1`:

```
OPENAI_BASE_URL=https://api.kimi.ai/coding/v1
OPENAI_API_KEY=sk-...
```

`OPENAI_BASE_URL` / `OPENAI_MODEL` override the defaults, so any other
OpenAI-compatible `/chat/completions` endpoint works too (OpenAI itself, a local proxy,
etc.) -- just point them there. `KIMI_API_KEY` is accepted as an alternate name for the
key if you'd rather set that. On the platform, `OPENAI_BASE_URL` / `OPENAI_API_KEY` are
injected automatically (the platform's own model proxy and a temporary credential); you
never need to configure them for a submission, and `.env` is never packed into the ZIP.

Without an API key (`OPENAI_API_KEY` or `KIMI_API_KEY`) configured, the process checks
at startup, before reading anything from stdin, and exits with a message on stderr and
a non-zero exit code.

This agent calls the OpenAI-compatible `/chat/completions` shape. A team that instead
chose the Anthropic Messages protocol in the workspace's **Model API** section also gets
`ANTHROPIC_BASE_URL` / `ANTHROPIC_API_KEY` injected on the platform (the Anthropic SDK's
own env vars) -- use those with the official `anthropic` SDK if your own agent speaks
Claude's Messages API instead. See `docs/model-api-keys.md` for the full comparison.

## Running it locally

This project is agent-side only -- it does not ship the simulator/scorer. Test it with
any tool that speaks the protocol on stdin/stdout (a local runner, or the platform
itself), pointing it at this folder's `agent.py`. See `docs/` for the full protocol and
scoring reference.

`agent.py` reads `initialize` / `decision_request` / `finish` on stdin and writes
`decision_response` on stdout; everything else (logs) goes to stderr, matching the
protocol exactly.

## Packaging / submitting

```bash
python3 pack_agent.py --out ../python-agent.zip
```

This zips the project with `observer.project.json` at the root
(`"protocol": "jsonl-v4"`, `run: ["python3", "-u", "agent.py"]`), skipping `.env`,
`__pycache__`, and `run_output/`. Upload the ZIP as a complete project, or push this
folder to a GitHub repository. No third-party packages are required; if you add one,
list it in `requirements.txt` **and** add a matching `build` step to
`observer.project.json` -- dependencies are not installed automatically otherwise.

## Safety properties

- Never reads any file: all state comes from stdin.
- Every decision is wrapped in `try`/`except` in `agent.py`; a planner bug produces a
  safe `wait`, never a crash or an `agent_error`.
- `validate_action()` strips anything the protocol would reject before it reaches
  stdout.
- A planning call that keeps failing just leaves that question's answer out of the
  merge for the night; the rest of the decision loop (targeting, exposure sizing,
  validation) runs the same either way.

## License / Citation

Task cards, simulated data, evaluation code and this example project are licensed under
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) (attribution, non-commercial);
please cite the GOSIM 2026 Agentic Observer Hackathon (https://create.gosim.org/survey26/). Your own agent code is not restricted
by this. See `LICENSE.md` for details.
