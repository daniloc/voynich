# Transcripts

Complete session history of the investigation (Claude Code sessions, June 20 – July 1, 2026).
Reader's guide with per-session summaries: **docs/SESSIONS.md**.

- `raw/` — the eight original session files (`<session-id>.jsonl`, one JSON object per
  line). `9966cd54…` is the 2026-07-01 audit/K14/cleanup session, snapshotted while
  still live (it is the session that performed this reorganization). Personal
  identifiers (email, local filesystem paths) were scrubbed 2026-07-02; the archives of
  full per-subagent transcripts were removed at the same time because they embedded
  verbatim copies of third-party web pages and library-licensed images — each
  subagent's task and final report survives in `subagent_reports/`.
- `readable/` — extracted plain-text versions of the seven pre-audit sessions:
  role-tagged text blocks (user / assistant / thinking / tool calls / truncated tool
  results). Names describe content: `cipher_and_motive_session` = session `ac7bc224`
  (K11, R1, M1–M14), `k13_grounding_session` = session `dcc71ad3` (L6→K13 + the
  exhaustion meta-discussion). Note these two were originally extracted under swapped
  names; docs written before the rename may be reconciled via docs/SESSIONS.md.
- `subagent_reports/` — for each multi-agent session, every subagent's task prompt and
  final report, extracted for quick reading.

Extraction code for `readable/` lives inside the audit session's own transcript
(`raw/9966cd54….jsonl`) — it kept text blocks, truncated thinking to 1,200 chars and
tool results to 2,500 chars, and dropped binary/image payloads.
