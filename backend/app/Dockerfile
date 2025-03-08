FROM python:3.10-slim

WORKDIR /app

COPY ./app /app
COPY requirements.txt /app/

RUN pip install --no-cache-dir -r requirements.txt

ENV PORT=8080
ENV PYTHONUNBUFFERED=1
ENV DATA_DIR=/app/data

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
