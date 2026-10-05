# python-agent -- an example agent for participant-agent-protocol-v4

> Branch `codex/llm-weather-incremental-score`, based on `origin/LLM` at `d1d51da`.
> The actual inherited LLM stages are weekly forecast interpretation and borderline
> fault arbitration; the historical nightly-call description below is superseded.

The project manifest enables `AGENT_CONDITIONS=quality` and `AGENT_SCORE_GAIN=1`
to reproduce the evaluated candidate: frozen public-input ridge quality estimates and actual best-score
increments. It preserves the joint scheduler, sky-band estimator and fault detector.
The quality proxy uses a representative 900-second duration, not a future rollout.
Use `LLM_DISABLED=1` for API-free numerical comparisons. Set conditions to `off` and
score gain to `0` for the original numerical policy. Credentials and training labels
are not included. Full experiments are stored outside this deployment directory at
`../gosim-observer-examples/experiments/teammate_merge/`.

Initial four-card numerical means: baseline 6148.92; candidate 6179.60 (+0.50%).
L3 regressed by about 152 points. These reused practice cards are not blind validation
or platform scores; consult the complete comparison before changing deployment.


Completed paired means: numerical all-card fit 6148.92 -> 6179.60 (+0.50%);
leave-one-card-out weather 6148.92 -> 6126.80 (-0.36%); live Kimi with at most six
attempts per card 6086.10 -> 6166.91 (+1.33%). The numerical candidate reproduced
all four scores on rerun. Each live policy/card was run once with independent model
responses. OOF excludes weather training data, not scheduler development decisions.
28 automated tests and 16 inherited planner self-checks pass. Fault arbitration is
conditional; successful execution of both LLM stages on every card is not established.
Both `observer.project.json` and `.env.example` use the evaluated settings:
`AGENT_CONDITIONS=quality`, `AGENT_SCORE_GAIN=1`, `LLM_MAX_CALLS=6`,
`LLM_TOTAL_BUDGET_SECONDS=180`, and `LLM_MAX_RETRIES=0`.
Direct local runs do not read the project manifest: copy `.env.example` to `.env`
only if no `.env` exists, or add these five settings to your existing local file.
Without explicit variables the Python fallback defaults retain the original policy.

The candidate completed one online A-D evaluation, batch
`7777491a-d1e2-43b7-b594-9359855df716`, with Kimi `k3`:

| Card | LLM_agent_v4_Jingmeng | This candidate | Difference | Required missing, original / candidate | Candidate charged CPU (s) |
|---|---:|---:|---:|---:|---:|
| A | 23512.19 | 23602.56 | +90.37 | 1 / 2 | 391.254 |
| B | 38928.46 | 39174.26 | +245.80 | 0 / 0 | 483.397 |
| C | 25230.41 | 24919.78 | -310.63 | 0 / 0 | 526.576 |
| D | 31459.16 | 32226.43 | +767.28 | 2 / 3 | 252.561 |
| Mean | 29782.55 | 29980.76 | +198.20 (+0.67%) | | |

All candidate cards ended with `survey_complete`; 22 of 24 time-limited requests
completed, the same count as the comparison. The comparison batch is
`a7f11a22-8c42-4117-9d57-a7aadbdbe5a5`; scenario hashes match per card.
This is one run per version, with different LLM budgets (24 attempts here versus
87 in the comparison), so the difference cannot be attributed to weather alone.
Both LLM stages produced usable answers on B; only the forecast stage did on A/C/D.
The six-call cap can leave no quota for late fault arbitration. No formal guarantee
of per-card two-stage success or generalization is implied.

Runtime files match the evaluated archive with SHA256
`30572035a209c48b5ed4d231d33b3432ac6cb57ca624833e136050d9fcc8fc43`.
Documentation and configuration templates were subsequently updated for this branch.


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

Once at the start of every night (`Planner._night_advice`), two questions go to the
model, each through `LLMClient.ask_json`, each answered as
`{avoid_directions, duration_scale}`:

1. **Forecast call** -- reads tonight's forecast notices plus the current bulletin, and
   asks for directions to avoid and an exposure-length scale.
2. **Bulletin + hit-rate call** -- reads tonight's live bulletin as plain text, plus the
   agent's own hit rate so far this run, and asks the same question from that angle.

The two answers are merged: directions to avoid are unioned, and the duration scale is
averaged. If a call keeps failing after its retries, that question's answer is left out
of the merge for the night (the other one still counts if it succeeded); the next
night's calls are unaffected. Target selection, fibre filling, exposure sizing, and
instrument-fault reporting are otherwise fully deterministic -- calling a model on every
one of the ~1,000-5,000 decisions in a run would blow through the wall clock; the
participant guide's own advice is "do not call a model on every decision." A third,
occasional call asks the model to confirm a suspected instrument fault before reporting
one (at most twice per run; the rule-based evidence check that triggers it runs far less
often than once per night).

### Call behaviour

`LLMClient` enforces, regardless of what the model does:

- a per-call timeout (default 12 s) that also shrinks as the wall clock runs low (never
  cuts into the last 60 s of the survey's own budget);
- a total LLM time budget per run (default 300 s, well under the 900 s wall clock);
- a call cap per run (default 100);
- up to 3 attempts for one question before moving on without an answer for it this time;
  the next scheduled call still goes ahead as normal;
- every exception (network, timeout, bad JSON, missing fields) is caught -- a failed
  attempt never raises past `ask_json`, and `ask_json` returns `None` when all attempts
  for that question are exhausted;
- the HTTP `User-Agent` header is never set or overridden.

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
