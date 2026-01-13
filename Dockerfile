FROM pytorch/pytorch:2.0.0-cpu
WORKDIR /app

# Set deterministic environment variables for submission
ENV PYTHONHASHSEED=0
ENV MC_DROPOUT_SAMPLES=10
ENV SCHEMA_VERSION=1.0

# Copy source and install dependencies (vendor at build time)
COPY . .
RUN pip install --no-cache-dir -r requirements.txt

# Recommended runtime: docker run --network=none to enforce offline execution
CMD ["python3", "src/inference.py"]
