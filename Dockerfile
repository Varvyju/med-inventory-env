# MedInventoryEnv — OpenEnv Docker image
# Hugging Face Spaces requires port 7860

FROM python:3.11-slim

# Non-root user for HF Spaces compatibility
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Install dependencies first (layer caching)
COPY server/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy server code
COPY server/ .

# Ensure correct ownership
RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 7860

# Health check so HF Space knows the container is ready
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')" || exit 1

CMD ["python", "main.py"]
