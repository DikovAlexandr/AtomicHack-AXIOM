FROM python:3.11-slim

ENV PIP_NO_CACHE_DIR=off \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Create a non-root user and group
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Change ownership to the new user
RUN chown -R appuser:appgroup /app

# Set home directory and matplotlib cache for non-root user
ENV HOME=/app
ENV MPLCONFIGDIR=/app/.cache/matplotlib

# Switch to the non-root user
USER appuser

EXPOSE 8501

CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
