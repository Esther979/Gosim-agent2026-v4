"""An optional, best-effort decision trace log.

Night-to-night and run-wide memory the planner actually leans on (learned sky
scale, per-target factor/misses/attempts, night advice, report cooldowns) lives on
`SurveyState` and `Planner` themselves -- it is genuinely part of what each of them
tracks, not a separate concern. What belongs here is purely diagnostic: an opt-in
JSONL trace of planning events (night advice, weather-event interpretation, the
finish summary), written only if `AGENT_TRACE_PATH` is set. Writing to it never
raises: a read-only sandbox or a bad path must not break the agent. Protocol logs
still only ever go to stderr (see protocol.log); this is a separate artifact.
"""
from __future__ import annotations

import json
import os
import time


class TraceLog:
    def __init__(self, log=lambda text: None):
        self.log = log
        self.path = os.environ.get("AGENT_TRACE_PATH", "").strip()
        self._handle = None
        if self.path:
            try:
                self._handle = open(self.path, "a", encoding="utf-8")
            except OSError as exc:
                self.log(f"memory: could not open AGENT_TRACE_PATH ({exc}); tracing disabled")
                self.path = ""

    def write(self, record: dict) -> None:
        if not self._handle:
            return
        try:
            record = {"ts": time.time(), **record}
            self._handle.write(json.dumps(record, separators=(",", ":")) + "\n")
            self._handle.flush()
        except OSError:
            pass  # a trace write must never interrupt the decision loop

    def close(self) -> None:
        if self._handle:
            try:
                self._handle.close()
            except OSError:
                pass


class AdviceTape:
    """Local test aid for the nightly LLM advice; inert unless one of these is set
    (the platform never sets them):

        LLM_RECORD_FILE  append each night's sanitized advice as one JSON line
                         {night_date: advice}
        LLM_REPLAY_FILE  read such a file once at startup and replay the advice by
                         night_date instead of calling the model (a night that is not
                         in the file gets neutral advice)
        LLM_MASK_KNOBS   comma-separated knobs to force neutral while replaying, among
                         exposure_bias, pursue -- for ablations

    Like TraceLog, nothing here may raise into the decision loop.
    """

    def __init__(self, log=lambda text: None):
        self.log = log
        self.record_path = os.environ.get("LLM_RECORD_FILE", "").strip()
        replay_path = os.environ.get("LLM_REPLAY_FILE", "").strip()
        self.masked = {k.strip() for k in os.environ.get("LLM_MASK_KNOBS", "").split(",") if k.strip()}
        self.replaying = bool(replay_path)
        self._nights: dict = {}
        if replay_path:
            try:
                with open(replay_path, encoding="utf-8") as handle:
                    for line in handle:
                        if line.strip():
                            self._nights.update(json.loads(line))
            except (OSError, ValueError) as exc:
                self.log(f"memory: could not read LLM_REPLAY_FILE ({exc}); replaying neutral advice")
            self.log(f"memory: replaying llm advice for {len(self._nights)} nights, masked={sorted(self.masked)}")

    def replay(self, night_date: str):
        """The recorded advice for this night (None when the file has none)."""
        return self._nights.get(night_date)

    def record(self, night_date: str, advice: dict) -> None:
        if not self.record_path:
            return
        try:
            with open(self.record_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps({night_date: advice}, separators=(",", ":")) + "\n")
        except OSError:
            pass
