# Component 15 — `infra/` (Docker + Terraform → AWS Fargate)

**Week:** 9 (mentor-assisted) · **Status:** stub · **Depends on:** the app

## Function

Containerize the app and stand up the public URL on AWS, with an infra-level cost alarm that
works even if every line of app logic fails. Streamlit Community Cloud remains the zero-infra
fallback so a live URL exists regardless.

## Requirements satisfied

- **FR-11** — Docker → ECS Fargate via Terraform + Community Cloud fallback. **SF-6** — the
  `aws_budgets_budget` alarm, independent of app logic. **DL-2** — the live URL.

## Deliverables (no functions — resources and files)

```
infra/Dockerfile          # python:3.12-slim + r-base + R packages (arrow, PerformanceAnalytics,
                          # PortfolioAnalytics — the R engine view is core, so R ships in the
                          # image; accept the ~1.5GB image, document why)
infra/main.tf             # providers, state backend (local for a solo project)
infra/ecs.tf              # ECR repo · ECS cluster · Fargate task+service (1 task, ~0.5 vCPU/1GB)
                          # · ALB + listener + SG (80/443 in, app port 8501 internal)
infra/budget.tf           # aws_budgets_budget with email notification at 50%/80%/100% (SF-6)
infra/variables.tf        # image_tag, aws_region, budget_limit_usd, alert_email
```

Secrets (`ANTHROPIC_API_KEY`, `DEMO_PASSCODE`) go into **SSM Parameter Store** (SecureString),
referenced by the task definition — never in the image, tf state, or git.

## Pre-public checklist (from the infra README — enforced before the URL is shared)

1. `PUBLIC_MODE=on`, `DEMO_PASSCODE` set, `AI_BUDGET_USD_DAILY`/`_TOTAL` set in the task env.
2. `aws_budgets_budget` exists and its notification email is confirmed received.
3. Budget-exhausted fallback verified live (set daily cap to $0.01, confirm cached scenarios).
4. `tests/test_public_mode_no_codegen.py` green in the deployed image.

## Design notes

- Prebuilt scenario data and the price cache are **baked into the image** — the public demo
  makes zero yfinance calls (determinism + rate-limit safety).
- Community Cloud fallback: same app, `PUBLIC_MODE=on`, secrets via Streamlit's secrets manager;
  documented in `infra/README.md` as the plan-B runbook.

## Done when

- `terraform apply` from scratch yields a working URL; checklist above complete; teardown
  (`terraform destroy`) verified so the demo doesn't bill idle.
