FROM pytorch/pytorch:2.0.0-cpu
WORKDIR /app

# Set deterministic environment variables for submission
ENV PYTHONHASHSEED=0
ENV MC_DROPOUT_SAMPLES=10
ENV SCHEMA_VERSION=1.0
ENV COMPETITION_MODE=1

# Copy source and install dependencies at build time (no runtime pip installs)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bundle model weights locally (assumed in data/ directory)
COPY . .

# Ensure no internet access required at runtime
# All dependencies pre-installed, weights bundled

# Recommended runtime: docker run --network=none to enforce offline execution
CMD ["python3", "src/inference.py"]
