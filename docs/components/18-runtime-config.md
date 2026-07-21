# Component 18 — `.env` / `.env.example` (runtime configuration)

**Weeks:** 7–9 (vars land with their features) · **Status:** `.env.example` present, key placeholder

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

In deploy (component 15), the secret-valued vars come from SSM Parameter Store, the rest from
the ECS task definition — `.env` is a local-dev mechanism only.

## Done when

`.env.example` lists every variable above with a comment; the infra pre-public checklist
cross-references this table; no module reads an env var not listed here.
