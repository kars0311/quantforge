"""Rigor: the research agent must NOT be able to read the untouched holdout during iteration.

Guards against the core failure mode — an LLM p-hacking against the test set. The holdout is scored
exactly once, at the end.
"""

import pytest


@pytest.mark.skip(reason="TODO(week8): implement once guardrails.split_data + agent exist")
def test_agent_cannot_access_holdout_during_iteration():
    # TODO: assert the holdout handle is opaque to the agent loop, and score_holdout() is single-use.
    raise NotImplementedError
