# Component 18 — `.env` / `.env.example` (runtime configuration)

**Weeks:** 7–9 (vars land with their features) · **Status:** `.env.example` complete for week 7 and unchanged by week 8 (every variable below present with a comment; defaults confirmed against `budget.py` / `guardrails.py`; `ai/agent.py` reads no environment variable of its own — the SDK reads `ANTHROPIC_API_KEY`, the caps/kill-switch/PUBLIC_MODE are read by `budget`/`guardrails` on its behalf); key still a placeholder in `.env`

## Function

The single documented home for runtime settings. `.env` is gitignored (holds the real key);
`.env.example` documents every variable with safe placeholders and is the onboarding reference.

## Requirements satisfied

**SF-1** (budget caps) · **SF-3/SF-4** (public mode + passcode) · **SF-7** (key server-side only).

## Variable inventory (the contract)

| variable | consumer | default | notes |
|----------|----------|---------|-------|
| `ANTHROPIC_API_KEY` | nl_interface, agent | — | **personal** account key (see setup briefing); never committed, never client-side |
| `AI_BUDGET_USD_DAILY` | budget.allow | `2` (dev) | hard daily cap, UTC day |
| `AI_BUDGET_USD_TOTAL` | budget.allow | `10` (dev) | hard lifetime cap |
| `AI_DISABLED` | budget.allow | `off` | kill-switch: `on` denies every AI call instantly |
| `AI_RATE_LIMIT_PER_HOUR` | guardrails.rate_limit | `20` | per session/IP key |
| `AI_LEDGER_PATH` | budget | `data_cache/ai_ledger.json` | persistent spend ledger |
| `PUBLIC_MODE` | guardrails (read at call time) | `off` | `on` ⇒ parameter-only, fail-closed |
| `DEMO_PASSCODE` | app gate_live_ai | — | required when PUBLIC_MODE=on |

Defaults confirmed against the code at the week-7 closeout: `budget._cap_from_env` falls back
to `2` / `10` only outside public mode — with `PUBLIC_MODE=on` an unset, blank, or unparseable
cap resolves to `0` and `allow()` denies every call (fail-closed); `AI_RATE_LIMIT_PER_HOUR`
falls back to `20` in every mode; `AI_LEDGER_PATH` also fixes where `guardrails.rate_limit`
keeps its state (`ai_rate_limits.json` beside the ledger). `QUANTFORGE_LIVE_AI` is
deliberately *not* in this table or in `.env.example`: it is a pytest-only opt-in for the
live-API smoke test (see `tests/test_nl_interface.py`), not a runtime setting.

In deploy (component 15), the secret-valued vars come from SSM Parameter Store, the rest from
the ECS task definition — `.env` is a local-dev mechanism only.

## Done when

`.env.example` lists every variable above with a comment; the infra pre-public checklist
cross-references this table; no module reads an env var not listed here.
