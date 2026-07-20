# Container for the Streamlit demo. Deployed to AWS ECS Fargate via Terraform (see infra/).
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
ENV PYTHONPATH=/app/src

# Public deploys MUST set PUBLIC_MODE=on and the budget caps (see .env.example).
EXPOSE 8501
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
