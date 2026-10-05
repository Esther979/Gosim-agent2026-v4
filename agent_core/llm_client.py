"""A tiny OpenAI-compatible chat client, configured through environment variables.
Defaults to the Kimi Coding Plan endpoint, since that is what this example ships with;
any other OpenAI-compatible `/chat/completions` endpoint works too by overriding the
base URL and model:

    OPENAI_BASE_URL   default https://api.kimi.com/coding/v1 (see README for the
                      api.kimi.ai alternative for accounts outside mainland China)
    OPENAI_MODEL      default "k3"
    OPENAI_API_KEY    bearer token (KIMI_API_KEY is also accepted)

See https://www.kimi.com/code/docs/en/ for the Kimi Coding Plan API.

Request-shape knobs (all optional, read from the environment so the platform run and a
local run can differ without code changes):

    LLM_TEMPERATURE          omitted from the request when unset (provider default); the
                             Kimi k3 reasoning model rejects any value other than 1
    LLM_MAX_TOKENS           default 3000 -- reasoning models spend "thinking" tokens inside
                             max_tokens; 250 (official) always and 1200 sometimes ran out
                             before k3 had written its JSON
    LLM_USER_AGENT           when set, sent as the HTTP User-Agent (identifying this agent
                             truthfully); when unset the header is left to the standard
                             library, exactly like the official example
    LLM_CALL_TIMEOUT_SECONDS default 30; LLM_TOTAL_BUDGET_SECONDS default 500;
    LLM_MAX_CALLS default 100; LLM_MAX_RETRIES default 0 (retries AFTER the first
                             attempt; 0 = a timed-out call is not repeated)
    LLM_DISABLED             "1" switches the model off entirely: no API key is needed,
                             ask_json returns None without touching the network, and every
                             planning step takes its rule-based answer. For deterministic
                             offline regression runs only.

Every call has a timeout and the whole run has a total time budget and a call cap, so
the LLM-advised planning steps stay well inside the 900 s wall clock. A call that fails
or times out is retried up to `max_retries` more times; if all attempts fail, that one
planning step falls back to its rule-based answer for this night -- the next night's
calls still go through normally.

Standard library only (urllib) so the example has zero third-party dependencies.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from typing import Optional

DEFAULT_BASE_URL = "https://api.kimi.com/coding/v1"
DEFAULT_MODEL = "k3"
_JSON_OBJECT = re.compile(r"\{.*\}", re.S)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default


class MissingAPIKeyError(RuntimeError):
    pass


def llm_disabled() -> bool:
    return os.environ.get("LLM_DISABLED", "").strip() == "1"


def require_api_key() -> None:
    """Raise MissingAPIKeyError if neither OPENAI_API_KEY nor KIMI_API_KEY is set.
    Called once at process startup, before reading anything from stdin.
    Does not raise when LLM_DISABLED=1 (offline deterministic runs need no key)."""
    if llm_disabled():
        return
    if not (os.environ.get("OPENAI_API_KEY", "").strip() or os.environ.get("KIMI_API_KEY", "").strip()):
        raise MissingAPIKeyError("missing API key: set OPENAI_API_KEY")


class LLMClient:
    def __init__(self, log=lambda text: None, call_timeout_seconds: float = 30.0,
                 total_budget_seconds: float = 500.0, max_calls: int = 100, max_retries: int = 0):
        self.log = log
        self.disabled = llm_disabled()
        self._disabled_logged = False
        self.base_url = os.environ.get("OPENAI_BASE_URL", "").strip().rstrip("/") or DEFAULT_BASE_URL
        self.api_key = os.environ.get("OPENAI_API_KEY", "").strip() or os.environ.get("KIMI_API_KEY", "").strip()
        self.model = os.environ.get("OPENAI_MODEL", "").strip() or DEFAULT_MODEL
        self.call_timeout_seconds = _env_float("LLM_CALL_TIMEOUT_SECONDS", call_timeout_seconds)
        self.total_budget_seconds = _env_float("LLM_TOTAL_BUDGET_SECONDS", total_budget_seconds)
        self.max_calls = int(_env_float("LLM_MAX_CALLS", max_calls))
        self.max_retries = max(0, int(_env_float("LLM_MAX_RETRIES", max_retries)))
        self.max_tokens = int(_env_float("LLM_MAX_TOKENS", 3000))
        temp = os.environ.get("LLM_TEMPERATURE", "").strip()
        self.temperature = float(temp) if temp else None
        self.user_agent = os.environ.get("LLM_USER_AGENT", "").strip()
        self.spent_seconds = 0.0
        self.calls_made = 0
        self.successes = 0
        self.log(f"llm: model={self.model} base={self.base_url} timeout={self.call_timeout_seconds:.0f}s "
                 f"budget={self.total_budget_seconds:.0f}s max_tokens={self.max_tokens} "
                 f"temperature={'omitted' if self.temperature is None else self.temperature}"
                 f"{' DISABLED' if self.disabled else ''}")

    def budget_left_seconds(self) -> float:
        return max(0.0, self.total_budget_seconds - self.spent_seconds)

    def _budget_left(self, wallclock_remaining_seconds: float) -> float:
        # Never let a model call eat into the last minute of wall clock, and never
        # exceed this run's own LLM time allowance.
        return min(self.call_timeout_seconds, self.total_budget_seconds - self.spent_seconds,
                   max(0.0, wallclock_remaining_seconds - 60.0))

    def _attempt(self, system_prompt: str, user_payload: dict, timeout: float) -> dict:
        """One HTTP attempt. Raises on any problem; the caller retries or gives up."""
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload)},
            ],
            "max_tokens": self.max_tokens,
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json", "Authorization": "Bearer " + self.api_key}
        if self.user_agent:
            headers["User-Agent"] = self.user_agent
        request = urllib.request.Request(self.base_url + "/chat/completions", data=body, method="POST", headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            # keep the provider's explanation (status + first bytes of the body) in the log;
            # the exception type alone hides e.g. "invalid temperature" or a gateway block
            try:
                detail = exc.read().decode("utf-8", "replace")[:160]
            except Exception:  # noqa: BLE001
                detail = ""
            self.log(f"llm: HTTP {exc.code}: {detail!r}")
            raise
        choice = data["choices"][0]
        text = choice["message"]["content"] or ""
        match = _JSON_OBJECT.search(text)
        if not match:
            raise ValueError(f"no JSON object in model reply (finish_reason={choice.get('finish_reason')}, "
                             f"usage={data.get('usage')})")
        parsed = json.loads(match.group(0))
        if not isinstance(parsed, dict):
            raise ValueError("model reply was not a JSON object")
        return parsed

    def ask_json(self, system_prompt: str, user_payload: dict, wallclock_remaining_seconds: float) -> Optional[dict]:
        """One planning question, answered as exactly one JSON object. Retries up to
        `max_retries` more times on failure; returns None once the budget/call cap/retries
        are exhausted, so the caller's rule-based answer can take over for this step."""
        if self.disabled:
            if not self._disabled_logged:
                self._disabled_logged = True
                self.log("llm: LLM_DISABLED=1; every step uses the rule-based path")
            return None
        last_error: Optional[Exception] = None
        for _attempt_number in range(1 + self.max_retries):
            if self.calls_made >= self.max_calls:
                self.log("llm: call cap reached for this run; using the rule-based path")
                return None
            timeout = self._budget_left(wallclock_remaining_seconds)
            if timeout < 1.5:
                self.log("llm: LLM time budget exhausted; using the rule-based path")
                return None
            started = time.monotonic()
            self.calls_made += 1
            try:
                answer = self._attempt(system_prompt, user_payload, timeout)
                self.successes += 1
                if self.successes == 1:
                    self.log(f"llm: first successful reply after {time.monotonic() - started:.1f}s")
                return answer
            except (urllib.error.URLError, OSError, ValueError, KeyError, IndexError, TypeError) as exc:
                last_error = exc
            finally:
                self.spent_seconds += time.monotonic() - started
        # a ValueError is ours (reply had no JSON object / was cut off) and its text says which
        detail = f": {str(last_error)[:120]}" if isinstance(last_error, ValueError) else ""
        self.log(f"llm: call failed after {1 + self.max_retries} attempt(s) ({type(last_error).__name__}{detail}); "
                 "this step falls back to its rule-based answer")
        return None
