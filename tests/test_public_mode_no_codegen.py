"""Safety: with PUBLIC_MODE=on, no LLM-generated code can execute; only vetted strategies + params.

Closes the worst public-MCP-hook risk (arbitrary code execution on a public server).
"""

import pytest


@pytest.mark.skip(reason="TODO(week9): implement once guardrails.assert_no_codegen exists")
def test_public_mode_rejects_codegen_and_unvetted_params():
    # TODO: with PUBLIC_MODE=on, assert codegen actions are rejected and only whitelisted
    # strategies/params are allowed; also assert the budget ledger blocks calls past the cap.
    raise NotImplementedError
