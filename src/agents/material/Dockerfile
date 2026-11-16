FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/common/ ./src/common/
COPY src/agents/material/ ./src/agents/material/
COPY config.py .

CMD ["python", "src/agents/material/agent.py"]
