# Terraform skeleton — deploy the container to AWS ECS Fargate, with a budget alarm.
# Mentor-assisted (week 9). Use a PERSONAL AWS account. Keep state out of git (see .gitignore).
#
# TODO(week9): flesh out. Recommended resources:
#   - aws_ecr_repository            (push the Docker image)
#   - aws_ecs_cluster / _service / _task_definition (Fargate, serverless)
#   - aws_lb (ALB) + listener + target group (HTTPS)
#   - aws_budgets_budget            (INDEPENDENT cost backstop — see below; do NOT skip)
#
# Streamlit Community Cloud is the zero-infra fallback if Fargate isn't ready.

terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.region
}

variable "region" {
  type    = string
  default = "us-east-1"
}

variable "monthly_budget_usd" {
  type    = number
  default = 50
}

# Independent cost backstop — fires regardless of app-level budget logic.
# TODO: set notification email; consider an action that scales the service to zero on breach.
resource "aws_budgets_budget" "demo_cost_cap" {
  name         = "quant-pipeline-demo"
  budget_type  = "COST"
  limit_amount = var.monthly_budget_usd
  limit_unit   = "USD"
  time_unit    = "MONTHLY"
  # TODO: notification block (threshold 80% / 100%) -> SNS/email.
}

# TODO: ECR + ECS Fargate service + ALB.
