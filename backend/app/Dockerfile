# Use a newer base image
FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variables and run command
ENV PORT=8080
ENV PYTHONUNBUFFERED=1
ENV DATA_DIR=/app/data

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
