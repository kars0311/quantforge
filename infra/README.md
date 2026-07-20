# Infra (Terraform → AWS ECS Fargate)

Mentor-assisted cloud deploy (week 9). Personal AWS account only.

```bash
cd infra
terraform init
terraform plan
terraform apply      # creates ECR + Fargate service + ALB + an AWS Budgets cost cap
```

**Non-negotiable before going public:**
- `PUBLIC_MODE=on`, `DEMO_PASSCODE` set, `AI_BUDGET_USD_*` caps set.
- The `aws_budgets_budget` cost cap exists and notifies you (independent of app logic).
- Confirm the app falls back to cached scenarios when the AI budget is exhausted.

Fallback: Streamlit Community Cloud (free, GitHub-linked) gives a live URL with zero infra.
