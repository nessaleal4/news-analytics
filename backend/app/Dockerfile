# Use a newer base image with the required GLIBC
FROM python:3.12-slim

WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend requirements and install them
COPY backend/requirements.txt backend/
RUN pip install --no-cache-dir -r backend/requirements.txt

# Now copy the rest of the application
COPY . .

# Set environment variables
ENV PORT=8080
ENV PYTHONUNBUFFERED=1
ENV DATA_DIR=/app/data

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
