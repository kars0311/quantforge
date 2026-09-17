"""Independent verifier for the Week-8 closeout (docs / plan / handoff sync; offline, no network).

The closeout milestone changes no behaviour, so what can go wrong is the *record* drifting from
reality: a plan box ticked early, a handoff count typed from memory, a design doc advertising a
kwarg or a stop reason the module does not have, a "green" test row naming a file that does not
exist. Each test below reads the artefact the spec named and checks it against the code, git
history, or a hand-built input — never against the builder's own summary.

- docs/TEN_WEEK_PLAN.md: exactly 24 `- [x]` lines at line start (the spec's own `grep -c`), the
  three Week-8 lines ticked and named, Week 9 / Week 10 / Stretch all `- [ ]`, and no alternate
  checkbox spelling (`[X]`, `* [x]`) that a count of `- [x]` would miss.
- handoff.md: the 2026-09-11 Week-8 entry (looked up by date + content since the 2026-09-12
  open-items fix run prepended on top of it) names everything the spec listed, records the
  suite counts verbatim, and — the adversarial part — every entry below the newest one is
  byte-identical to the committed `HEAD:handoff.md` (prepending must not rewrite history). The
  per-suite growth numbers the entry quotes (`test_agent.py` 98, holdout 15, public-mode 30) are
  checked against a real `pytest --collect-only` of those files.
  Pins advanced at the 2026-09-12 fix run: HEAD (`74af8d9`, "week 8 complete") now carries the
  Week-8 entry itself, so the byte-identity check expects HEAD's first entry to be 2026-09-11
  and the working copy's first entry to be the 2026-09-12 fix-run entry.
- docs/components/13-ai-agent.md may not overclaim: the Interface block's signature is
  `inspect.signature(run_research)`, the history-record and result keys it lists are the
  module's tuples AND the keys real records carry after a mocked run, the `stopped_because`
  vocabulary is `STOP_REASONS`, the "no `public_mode`" and "no `clamps`" pins hold.
- Behavioural checks of the doc's sentences with hand-built inputs: `_select_best` prefers the
  higher VALIDATION Sharpe, maps NaN to -inf, breaks ties to the earliest iteration and skips
  errored records; `max_iters=11` is refused before any spend or data load (the README's "hard
  10-iteration cap"); a model that never converges is cut off at exactly `max_iters` SDK calls
  with the queued extra response never requested; a budget cap that admits exactly one call (cap
  = the first turn's estimate) stops the loop with `budget` after one call and still scores the
  best-so-far once; an all-rejected run scores nothing.
- docs/components/10-ai-guardrails.md says the agent-loop clause landed (past tense) and the
  clause really exists in tests/test_holdout_isolation.py; 16-tests.md rows exist for
  `test_agent.py` and the week-8 verifiers, every file it names exists, its status header names
  both live smokes and both are `skipif`-gated on `QUANTFORGE_LIVE_AI`.
- 18-runtime-config.md / .env.example: agent.py reads no environment variable, and .env.example
  is byte-identical to HEAD.
"""

from __future__ import annotations

import inspect
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from quantforge import interchange
from quantforge.ai import agent, budget, guardrails, mcp_server
from quantforge.metrics.performance import _KEYS

_ROOT = Path(__file__).resolve().parents[1]
_PLAN = (_ROOT / "docs" / "TEN_WEEK_PLAN.md").read_text()
_HANDOFF = (_ROOT / "handoff.md").read_text()
_README = (_ROOT / "README.md").read_text()
_ENV_EXAMPLE = (_ROOT / ".env.example").read_text()
_DOC_AGENT = (_ROOT / "docs" / "components" / "13-ai-agent.md").read_text()
_DOC_GUARDRAILS = (_ROOT / "docs" / "components" / "10-ai-guardrails.md").read_text()
_DOC_TESTS = (_ROOT / "docs" / "components" / "16-tests.md").read_text()
_DOC_RUNTIME = (_ROOT / "docs" / "components" / "18-runtime-config.md").read_text()
_AGENT_SRC = (_ROOT / "src" / "quantforge" / "ai" / "agent.py").read_text()

TICKERS = ["AAPL", "MSFT", "NVDA", "JPM"]
GOAL = "find a momentum variant with Sharpe > 1 on validation"


# ---------------------------------------------------------------- helpers: docs


def _plan_sections() -> dict[str, str]:
    """{heading: body} for every '## ' / '### ' section of the plan."""
    parts = re.split(r"^#{2,3} ", _PLAN, flags=re.M)[1:]
    out = {}
    for part in parts:
        heading, _, body = part.partition("\n")
        out[heading.strip()] = body
    return out


def _plan_section(prefix: str) -> str:
    return next(body for heading, body in _plan_sections().items() if heading.startswith(prefix))


def _entries(text: str) -> list[tuple[str, str]]:
    """(date, body) for each '## YYYY-MM-DD — ...' handoff entry, in file order."""
    matches = list(re.finditer(r"^## (\d{4}-\d{2}-\d{2}) — ", text, flags=re.M))
    out = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        out.append((m.group(1), text[m.start() : end]))
    return out


def _baseline_ref() -> str | None:
    """The commit this fix run was built on top of ("week 8 complete", 74af8d9), found by
    message rather than pinned as HEAD: these pins compare the working tree against the state
    BEFORE the fix run, and must keep holding after the run itself is committed (HEAD moves;
    the baseline does not)."""
    if not (_ROOT / ".git").exists() or shutil.which("git") is None:
        return None
    proc = subprocess.run(
        ["git", "log", "--format=%H", "--grep=^week 8 complete", "-1"],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    sha = proc.stdout.strip()
    return sha or None


def _git_show(rel: str) -> str | None:
    """The committed (HEAD) version of a file, or None when git is unavailable."""
    if not (_ROOT / ".git").exists():
        return None
    proc = subprocess.run(
        ["git", "show", f"{_baseline_ref()}:{rel}"],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.stdout if proc.returncode == 0 else None


def _status_line(doc: str) -> str:
    return next(line for line in doc.splitlines() if line.startswith("**Week"))


def _doc_section(doc: str, heading: str) -> str:
    assert heading in doc, f"missing section {heading!r}"
    return doc[doc.index(heading) + len(heading) :].split("\n## ")[0]


def _row(row_file: str) -> str:
    return next(line for line in _DOC_TESTS.splitlines() if line.startswith(f"| `{row_file}`"))


# ---------------------------------------------------------------- helpers: a mocked agent run


def _usage(tokens_in: int, tokens_out: int):
    return SimpleNamespace(
        input_tokens=tokens_in,
        output_tokens=tokens_out,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
    )


def _tool_turn(name: str, data: dict, block_id: str, usage=None):
    block = SimpleNamespace(type="tool_use", id=block_id, name=name, input=data)
    return SimpleNamespace(
        content=[block], usage=usage or _usage(1000, 200), stop_reason="tool_use"
    )


def _propose(strategy, params, block_id, rationale="try it", usage=None):
    data = {"strategy": strategy, "params": params, "rationale": rationale}
    return _tool_turn("propose_experiment", data, block_id, usage)


def _done(block_id="toolu_done"):
    return _tool_turn("declare_done", {"reason": "converged"}, block_id)


class FakeClient:
    """Records every ``messages.create`` request; pops canned responses in order."""

    def __init__(self, *responses):
        self.calls: list[dict] = []
        self.queue = list(responses)
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if not self.queue:
            raise AssertionError("FakeClient received more requests than canned responses")
        return self.queue.pop(0)


def _synthetic_long(seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2010-01-01", "2026-06-30", tz="UTC")
    rets = rng.normal(0.0004, 0.01, size=(len(dates), len(TICKERS)))
    wide = pd.DataFrame(100.0 * np.exp(np.cumsum(rets, axis=0)), index=dates, columns=TICKERS)
    return interchange.to_long(wide, "prices")


_SYNTHETIC = _synthetic_long()
_real_split_data = guardrails.split_data
_real_score_holdout = guardrails.score_holdout


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    """Synthetic prices, temp ledger, dev-mode env, no API key, no real client."""
    monkeypatch.setattr(mcp_server, "_price_source", lambda: _SYNTHETIC)
    mcp_server.reset_registry()
    monkeypatch.setenv("AI_LEDGER_PATH", str(tmp_path / "ledger.json"))
    for var in (
        "PUBLIC_MODE",
        "AI_DISABLED",
        "AI_BUDGET_USD_DAILY",
        "AI_BUDGET_USD_TOTAL",
        "AI_RATE_LIMIT_PER_HOUR",
        "ANTHROPIC_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(agent, "_client", None)
    monkeypatch.setattr(
        agent, "_get_client", lambda: (_ for _ in ()).throw(AssertionError("real client"))
    )
    yield
    mcp_server.reset_registry()


def _spy_holdout(monkeypatch, seq: list[str]) -> dict[str, list]:
    calls: dict[str, list] = {"split": [], "score": []}

    def split(prices=None):
        seq.append("split_data")
        out = _real_split_data(prices)
        calls["split"].append(out[2])
        return out

    def score(handle, strategy, params, engine="python"):
        seq.append("score_holdout")
        calls["score"].append((handle, strategy, params, engine))
        return _real_score_holdout(handle, strategy, params, engine)

    monkeypatch.setattr(guardrails, "split_data", split)
    monkeypatch.setattr(guardrails, "score_holdout", score)
    return calls


# ---------------------------------------------------------------------------
# docs/TEN_WEEK_PLAN.md
# ---------------------------------------------------------------------------


def test_plan_exactly_24_ticked_lines_and_no_alternate_checkbox_spelling():
    # The spec's own check is `grep -c '^- \[x\]'`: count at line start, not substrings.
    assert len(re.findall(r"^- \[x\]", _PLAN, flags=re.M)) == 24
    assert _PLAN.count("- [x]") == 24
    # A tick written as "[X]" or "* [x]" would dodge the count above; there must be none.
    assert not re.search(r"^[-*] \[X\]", _PLAN, flags=re.M)
    assert not re.search(r"^\* \[[ x]\]", _PLAN, flags=re.M)
    # And every checkbox in the file is one of the two legal spellings.
    boxes = re.findall(r"^- \[(.)\]", _PLAN, flags=re.M)
    assert set(boxes) <= {"x", " "}


def test_plan_week8_three_lines_ticked_and_named_weeks_9_10_stretch_untouched():
    week8 = _plan_section("Week 8")
    assert week8.count("- [x]") == 3 and "- [ ]" not in week8
    assert "ai/agent.py" in week8
    assert "untouched holdout" in week8 and "iteration cap" in week8 and "budget cap" in week8
    assert "tests/test_holdout_isolation.py" in week8
    later = {h: b for h, b in _plan_sections().items() if re.match(r"Week (9|10) |Stretch", h)}
    assert len(later) == 3, sorted(later)
    for heading, body in later.items():
        assert "- [x]" not in body, f"{heading!r} has a prematurely ticked box"
        assert body.count("- [ ]") >= 3, f"{heading!r} lost checklist items"
    # Weeks 1-8 are fully ticked: 4+3+3+2+3+3+3+3 = 24.
    per_week = {"Week 1": 4, "Week 2": 3, "Week 3": 3, "Week 4": 2}
    per_week.update({f"Week {n}": 3 for n in range(5, 9)})
    assert sum(per_week.values()) == 24
    for week, n in per_week.items():
        body = _plan_section(week + " ")
        assert body.count("- [x]") == n and "- [ ]" not in body, week


# ---------------------------------------------------------------------------
# handoff.md
# ---------------------------------------------------------------------------


def test_handoff_first_entry_is_week8_2026_09_11_and_names_the_spec_items():
    # Pin advanced at the 2026-09-12 fix run: that run prepended its own entry, so the Week-8
    # entry is looked up by date + content rather than taken as entries[0]. Every needle below
    # is unchanged; the "Week-7 entry is the very next one" pin now keys off the Week-8 index.
    entries = _entries(_HANDOFF)
    date, body = next((d, b) for d, b in entries if d == "2026-09-11" and "Week 8" in b)
    idx8 = entries.index((date, body))
    assert body.startswith("## 2026-09-11 — Week 8 complete: AI research agent")
    for needle in (
        "ai/agent.py",
        "run_research",
        "propose_experiment",
        "declare_done",
        "test_agent.py",
        "test_holdout_isolation.py",
        "test_public_mode_no_codegen.py",
        "1186 passed / 2 skipped",
        "919 passed / 1 skipped",
        "test_nl_interface.py::test_live_smoke",
        "test_agent.py::test_live_smoke",
        "Agent failures",
        "uncommitted",
        "ANTHROPIC_API_KEY",
        "placeholder",
        "Next up",
        "Week 9",
    ):
        assert needle in body, f"week-8 handoff entry lacks {needle!r}"
    assert re.search(r"ruff check \.` clean", body)
    assert re.search(r"pending Kent's approval", body)
    assert re.search(r"Research (tab|mode)", body)
    # Dates descend, and the Week-7 entry is the very next one after the Week-8 entry.
    assert [d for d, _ in entries] == sorted((d for d, _ in entries), reverse=True)
    date7, body7 = entries[idx8 + 1]
    assert date7 == "2026-09-10" and body7.startswith("## 2026-09-10 — Week 7")


def test_handoff_prepend_left_every_earlier_entry_byte_identical_to_head():
    # "Intact" means intact: the committed handoff's entries must reappear unchanged below the
    # new one — not re-flowed, not re-counted, not trimmed.
    # Pin advanced at the 2026-09-12 fix run: Week 8 was committed as 74af8d9, so HEAD's first
    # entry is now the 2026-09-11 one and the working copy's first entry is the fix-run entry.
    # Every entry below it must still equal HEAD's, byte for byte.
    committed = _git_show("handoff.md")
    if committed is None:
        pytest.skip("git or the committed handoff.md is unavailable")
    old = _entries(committed)
    new = _entries(_HANDOFF)
    assert old, "committed handoff has no dated entries"
    assert old[0][0] == "2026-09-11" and "Week 8" in old[0][1]
    assert new[0][0] == "2026-09-12" and "open-items" in new[0][1]
    assert [b for _, b in new[1:]] == [b for _, b in old], "an earlier handoff entry was edited"
    assert new[1][0] == "2026-09-11" and "1186 passed / 2 skipped" in new[1][1]
    assert "919 passed / 1 skipped" in new[2][1] and "879 passed / 1 skipped" in new[2][1]
    assert new[2][0] == "2026-09-10"
    assert "488 passed / 2 skipped" in new[3][1] and new[3][0] == "2026-09-01"


def _collected_counts(*rel_files: str) -> dict[str, int]:
    """{relative path: collected tests} from a real pytest collection of those files."""
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "--color=no",  # FORCE_COLOR in a caller's shell would ANSI-wrap the summary line
            "-qq",
            "-p",
            "no:cacheprovider",
            "-o",
            "addopts=",
            *rel_files,
        ],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    counts = dict(re.findall(r"^(tests/\S+\.py): (\d+)$", proc.stdout, flags=re.M))
    assert set(counts) == set(rel_files), proc.stdout
    return {k: int(v) for k, v in counts.items()}


def test_handoff_per_suite_growth_numbers_match_a_real_collection():
    # The entry quotes how many tests each week-8 suite has; a number typed from memory (the
    # builder's first draft said 27 -> 31 for the public-mode suite; it is 26 -> 30) fails here.
    # Pin advanced at the 2026-09-12 fix run: the Week-8 entry is read by date, not as [0].
    week8 = next(b for d, b in _entries(_HANDOFF) if d == "2026-09-11" and "Week 8" in b)
    body = " ".join(week8.split())  # the markdown wraps mid-parenthesis
    counts = _collected_counts(
        "tests/test_agent.py",
        "tests/test_holdout_isolation.py",
        "tests/test_public_mode_no_codegen.py",
    )
    assert re.search(rf"test_agent\.py` \({counts['tests/test_agent.py']} tests\)", body)
    m = re.search(r"test_holdout_isolation\.py` \((\d+) → (\d+)\)", body)
    assert m and int(m.group(2)) == counts["tests/test_holdout_isolation.py"]
    assert int(m.group(1)) == 9  # the committed (week-7) file collects 9 — a historical fact
    m = re.search(r"test_public_mode_no_codegen\.py` \((\d+) → (\d+)\)", body)
    assert m and int(m.group(2)) == counts["tests/test_public_mode_no_codegen.py"]
    assert int(m.group(1)) == 26  # the committed (week-7) file collects 26, not 27


# ---------------------------------------------------------------------------
# docs/components/13-ai-agent.md may not overclaim the module
# ---------------------------------------------------------------------------


def test_agent_doc_status_signature_keys_and_stop_vocabulary_are_the_modules():
    status = _status_line(_DOC_AGENT)
    assert "built / green (wk 8)" in status and "stub" not in status.lower()
    assert "## Decisions made in build" in _DOC_AGENT
    assert "clamps" not in _DOC_AGENT
    interface = _doc_section(_DOC_AGENT, "## Interface")
    sig = inspect.signature(agent.run_research)
    assert list(sig.parameters) == ["goal", "max_iters", "tickers", "engine", "cost_bps", "client"]
    assert all(
        p.kind is inspect.Parameter.KEYWORD_ONLY for n, p in sig.parameters.items() if n != "goal"
    )
    assert sig.parameters["max_iters"].default == guardrails.MAX_AGENT_ITERS == 10
    assert sig.parameters["engine"].default == "python"
    assert sig.parameters["cost_bps"].default == 10.0
    assert sig.parameters["tickers"].default is None and sig.parameters["client"].default is None
    assert "max_iters: int = guardrails.MAX_AGENT_ITERS" in interface
    assert 'engine: str = "python"' in interface and "cost_bps: float = 10.0" in interface
    assert "tickers: list[str] | None = None" in interface and "client=None" in interface
    assert "public_mode" not in interface
    # Result keys and stop reasons in the doc are exactly the module's.
    doc_result_keys = re.findall(r'^\s*#\s+(?:Returns: \{)?"(\w+)":', interface, flags=re.M)
    assert tuple(doc_result_keys) == agent._RESULT_KEYS
    doc_stops = re.findall(r'"stopped_because": ((?:"\w+"(?: \| )?)+)', interface)
    assert doc_stops and tuple(re.findall(r'"(\w+)"', doc_stops[0])) == agent.STOP_REASONS
    assert "api_error" in agent.STOP_REASONS
    # History record keys: the doc's braces list is the module's tuple.
    m = re.search(r"History record: `\{([^}]+)\}`", _DOC_AGENT)
    assert m and tuple(k.strip() for k in m.group(1).split(",")) == agent._RECORD_KEYS
    # The done-when honestly defers the live run to the key.
    done = _doc_section(_DOC_AGENT, "## Done when")
    assert "pending" in done.lower() and "ANTHROPIC_API_KEY" in done
    assert "test_live_smoke" in done and "QUANTFORGE_LIVE_AI=1" in done


def test_agent_doc_decisions_list_covers_every_spec_point():
    decisions = " ".join(_doc_section(_DOC_AGENT, "## Decisions made in build").split())
    for needle in (
        "propose_experiment",
        "declare_done",
        "run_backtest",
        "messages.create",
        "SF-7",
        "strict",
        "validate_params",
        "validation Sharpe",
        "-inf",
        "earliest",
        "_price_source",
        "tickers",
        "after the loop",
        "error",
        "system prompt",
        "last tool",
        "last message",
        "public_mode",
        "claude-sonnet-5",
    ):
        assert needle in decisions, f"decisions list lacks {needle!r}"
    assert agent.MODEL == "claude-sonnet-5" and agent.MODEL in budget.PRICES_PER_MTOK
    # The "imports no engine or portfolio code" sentence is true of the module.
    assert not re.search(r"^(from|import) quantforge\.(engine|portfolio)", _AGENT_SRC, flags=re.M)
    assert "STRATEGIES[" not in _AGENT_SRC
    assert _AGENT_SRC.count("messages.create(") == 1


def test_history_records_and_result_carry_exactly_the_documented_keys(isolated):
    # A successful and a rejected proposal: both records must have precisely the doc's keys.
    client = FakeClient(
        _propose("momentum", {"lookback": 60}, "t1"),
        _propose("momentum", {"lookback": 999}, "t2"),
        _done(),
    )
    out = agent.run_research(GOAL, max_iters=4, client=client)
    assert tuple(out) == agent._RESULT_KEYS
    assert out["stopped_because"] == "converged" and out["error"] is None
    assert [tuple(r) for r in out["history"]] == [agent._RECORD_KEYS] * 2
    ok, bad = out["history"]
    assert ok["error"] is None and list(ok["val_metrics"]) == _KEYS
    assert bad["error"] and bad["train_metrics"] is None and bad["val_metrics"] is None
    # The README's "side by side": validation and holdout metrics share the metric keys.
    assert list(out["best"]["val_metrics"]) == _KEYS == list(out["holdout_metrics"])


# ---------------------------------------------------------------------------
# The doc's ranking sentence, with hand-built records
# ---------------------------------------------------------------------------


def _rec(i: int, val_sharpe: float, error: str | None = None) -> dict:
    return {
        "iter": i,
        "strategy": "momentum",
        "params": {"lookback": 10 * i},
        "train_metrics": {"sharpe": 99.0},  # a huge TRAIN Sharpe must never win
        "val_metrics": {"sharpe": val_sharpe},
        "rationale": "x",
        "error": error,
    }


def test_select_best_validation_sharpe_nan_as_minus_inf_ties_to_earliest_errors_skipped():
    assert agent._select_best([]) is None
    # Higher validation Sharpe wins even though every train Sharpe is 99.
    assert agent._select_best([_rec(1, 0.5), _rec(2, 1.5), _rec(3, 1.0)])["params"] == {
        "lookback": 20
    }
    # NaN ranks as -inf: a negative finite Sharpe beats it.
    assert agent._select_best([_rec(1, math.nan), _rec(2, -5.0)])["params"] == {"lookback": 20}
    # Ties (including an all-NaN history) go to the earliest iteration, whatever the order.
    assert agent._select_best([_rec(2, 1.0), _rec(1, 1.0)])["params"] == {"lookback": 10}
    assert agent._select_best([_rec(3, math.nan), _rec(1, math.nan)])["params"] == {"lookback": 10}
    # Errored records are skipped even when their (stale) metrics look best.
    hist = [_rec(1, 9.0, error="rejected"), _rec(2, 0.1)]
    assert agent._select_best(hist)["params"] == {"lookback": 20}
    assert agent._select_best([_rec(1, 9.0, error="rejected")]) is None
    # Copies, not aliases: mutating the selection cannot rewrite the timeline.
    best = agent._select_best(hist)
    best["params"]["lookback"] = -1
    assert hist[1]["params"] == {"lookback": 20}


# ---------------------------------------------------------------------------
# The README's caps, exercised
# ---------------------------------------------------------------------------


def test_readme_names_the_agent_caps_and_links_the_two_proof_suites():
    section = _doc_section(_README, "## Methodology and known limitations")
    assert "research agent" in section and "10-iteration cap" in section
    assert "MAX_AGENT_ITERS" in section and guardrails.MAX_AGENT_ITERS == 10
    assert "one" in section and re.search(r"\bonce\b", section) and "side by side" in section
    for link in ("tests/test_holdout_isolation.py", "tests/test_agent.py"):
        assert f"[`{link}`]({link})" in section
        assert (_ROOT / link).is_file()


@pytest.mark.parametrize("bad", [11, 0, -1, True, 2.0, "10"])
def test_iteration_cap_above_10_refused_before_any_spend_or_data_load(isolated, bad):
    client = FakeClient(_propose("momentum", {"lookback": 60}, "t1"))
    with pytest.raises(ValueError, match="max_iters"):
        agent.run_research(GOAL, max_iters=bad, client=client)
    assert client.calls == []
    assert mcp_server._DATASETS == {}  # no load_data happened either
    assert budget.remaining() == {"daily": 2.0, "total": 10.0}  # nothing charged


def test_non_converging_model_is_cut_off_at_exactly_max_iters_calls(isolated, monkeypatch):
    seq: list[str] = []
    calls = _spy_holdout(monkeypatch, seq)
    # Four proposals queued, only three allowed: the fourth is never requested.
    client = FakeClient(
        *[_propose("momentum", {"lookback": 20 * i}, f"t{i}") for i in (1, 2, 3, 4)]
    )
    out = agent.run_research(GOAL, max_iters=3, client=client)
    assert len(client.calls) == 3 and len(client.queue) == 1
    assert out["stopped_because"] == "max_iters" and out["error"] is None
    assert [r["iter"] for r in out["history"]] == [1, 2, 3]
    assert seq[-2:] == ["split_data", "score_holdout"] and len(calls["score"]) == 1
    assert calls["score"][0][1:] == ("momentum", out["best"]["params"], "python")
    assert calls["split"][0].consumed is True
    # Every request carried the cap in the turn counter the model was shown.
    # (The string content becomes a cache-marked text block on the wire, hence ``str``.)
    assert "You have 3 turns in total" in str(client.calls[0]["messages"][0]["content"])


def test_budget_cap_admitting_exactly_one_call_stops_with_budget_after_one(isolated, monkeypatch):
    # Hand-computed: the first turn's estimate is estimate(MODEL, 4000, 1024). A daily cap equal
    # to it admits turn 1 (landing on the cap is allowed) and, once any spend is charged, the
    # larger turn-2 estimate (5200 tokens in) cannot fit — so the loop must stop with "budget"
    # after exactly one SDK call, and still score the one winner on the holdout, once.
    est1 = budget.estimate(agent.MODEL, 4000, 1024)
    assert budget.estimate(agent.MODEL, 5200, 1024) > est1
    monkeypatch.setenv("AI_BUDGET_USD_DAILY", repr(est1))
    seq: list[str] = []
    calls = _spy_holdout(monkeypatch, seq)
    usage = _usage(1200, 200)
    client = FakeClient(
        _propose("momentum", {"lookback": 60}, "t1", usage=usage),
        _propose("momentum", {"lookback": 120}, "t2"),
        _done(),
    )
    out = agent.run_research(GOAL, max_iters=5, client=client)
    assert len(client.calls) == 1 and len(client.queue) == 2
    assert out["stopped_because"] == "budget" and out["error"] is None
    assert len(out["history"]) == 1 and out["history"][0]["error"] is None
    assert out["spend_usd"] == pytest.approx(budget.estimate(agent.MODEL, 1200, 200))
    assert seq == ["split_data", "score_holdout"]
    assert len(calls["score"]) == 1 and out["holdout_metrics"] is not None
    assert out["best"]["params"]["lookback"] == 60  # validated dict: defaults merged in too


def test_all_rejected_run_scores_nothing_and_returns_no_best(isolated, monkeypatch):
    seq: list[str] = []
    calls = _spy_holdout(monkeypatch, seq)
    client = FakeClient(
        _propose("momentum", {"lookback": 999}, "t1"),
        _propose("momentum", {"code": "import os"}, "t2"),
    )
    out = agent.run_research(GOAL, max_iters=2, client=client)
    assert out["best"] is None and out["holdout_metrics"] is None
    assert out["stopped_because"] == "max_iters"
    assert all(r["error"] for r in out["history"]) and len(out["history"]) == 2
    assert seq == [] and calls == {"split": [], "score": []}
    # No holdout value or date ever reached the model.
    for request in client.calls:
        text = str(request["messages"])
        assert "holdout" not in text.lower() and "2023-" not in text and "2024-" not in text


# ---------------------------------------------------------------------------
# docs/components/10, 16, 18 and .env.example
# ---------------------------------------------------------------------------


def test_guardrails_doc_records_the_agent_loop_clause_as_landed_and_it_really_exists():
    flat = " ".join(_DOC_GUARDRAILS.split())
    status = _status_line(_DOC_GUARDRAILS)
    assert "landed in week 8" in status and "test_holdout_isolation.py" in status
    assert flat.count("landed in week 8") >= 2
    assert "agent-loop" in flat and "week 8" in flat
    for stale in ("lands in week 8", "is added in week 8", "week-8** item", "pending"):
        assert stale not in flat, f"10-ai-guardrails.md still says {stale!r}"
    suite = (_ROOT / "tests" / "test_holdout_isolation.py").read_text()
    assert "Week 8" in suite and "run_research" in suite
    assert "score_holdout" in suite and "split_data" in suite
    assert len(re.findall(r"^def test_", suite, flags=re.M)) >= 15


def test_tests_doc_rows_status_and_live_smoke_gating():
    head = "\n".join(_DOC_TESTS.splitlines()[:6])
    assert "wk 1–8" in head and "live-AI smoke tests skip" in head
    assert "test_nl_interface.py::test_live_smoke" in head
    assert "test_agent.py::test_live_smoke" in head
    agent_row = _row("test_agent.py")
    assert "green (wk 8" in agent_row and "FakeClient" in agent_row
    for req in ("FR-9", "SF-1", "SF-7", "SF-8", "RG-4"):
        assert req in agent_row
    assert "wk 8" in _row("test_holdout_isolation.py")
    assert "agent-loop clause landed" in _row("test_holdout_isolation.py")
    assert "agent clause" in _row("test_public_mode_no_codegen.py")
    verifier_row = next(line for line in _DOC_TESTS.splitlines() if "test_agent_verify.py" in line)
    for name in (
        "test_agent_run_research_verify.py",
        "test_week8_proof_suites_verify.py",
        "test_week8_closeout_verifier.py",
    ):
        assert name in verifier_row, f"week-8 verifier row lacks {name}"
    # Every suite the inventory names exists, and both live smokes are really opt-in.
    for name in sorted(set(re.findall(r"`(test_\w+\.py)`", _DOC_TESTS))):
        assert (_ROOT / "tests" / name).is_file(), f"16-tests.md names phantom suite {name}"
    for suite in ("test_nl_interface.py", "test_agent.py"):
        src = (_ROOT / "tests" / suite).read_text()
        gate = re.search(
            r"@pytest\.mark\.skipif\(\s*os\.environ\.get\(\"QUANTFORGE_LIVE_AI\"\) != \"1\","
            r".*?\n\)\ndef test_live_smoke\(",
            src,
            flags=re.S,
        )
        assert gate, f"{suite}::test_live_smoke is not gated on QUANTFORGE_LIVE_AI"


def test_agent_reads_no_env_var_and_env_example_is_unchanged():
    assert not re.search(r"os\.(getenv|environ)", _AGENT_SRC)
    assert not re.search(r"^import os\b", _AGENT_SRC, flags=re.M)
    assert not re.search(r"^from os\b", _AGENT_SRC, flags=re.M)
    status = _status_line(_DOC_RUNTIME)
    assert "unchanged by week 8" in status and "agent.py" in status
    committed = _git_show(".env.example")
    if committed is None:
        pytest.skip("git or the committed .env.example is unavailable")
    # Week 8 left .env.example byte-identical to HEAD. The 2026-09-12 open-items fix run then
    # re-pointed the two example cap values at the dev defaults (5/25 -> 2/10), so the pin is
    # retargeted: the VARIABLE SET must still equal HEAD's (nothing added or removed), and the
    # only values allowed to differ from HEAD are the two budget caps.
    pattern = re.compile(r"^([A-Z_]+)=(.*)$", flags=re.M)
    head_vars = dict(pattern.findall(committed))
    now_vars = dict(pattern.findall(_ENV_EXAMPLE))
    assert set(now_vars) == set(head_vars), "variable added/removed in .env.example"
    changed = {k for k in head_vars if head_vars[k] != now_vars[k]}
    assert changed <= {"AI_BUDGET_USD_DAILY", "AI_BUDGET_USD_TOTAL"}, (
        f".env.example values changed beyond the budget caps: {changed}"
    )
