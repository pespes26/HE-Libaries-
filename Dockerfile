# Pinned Python 3.11 (Linux): TenSEAL 0.3.16 ships a manylinux2014 cp311 wheel.
# Pinned by digest for reproducibility (tag python:3.11-slim at time of writing).
FROM python:3.11-slim@sha256:ae52c5bef62a6bdd42cd1e8dffef86b9cd284bde9427da79839de7a4b983e7ca

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    MPLBACKEND=Agg

# Install from the fully-resolved lock (direct + transitive) for reproducible builds.
COPY requirements.lock .
RUN pip install --no-cache-dir -r requirements.lock

# Copy the project (results/ and figures/ are bind-mounted at run time).
COPY . .

# NOTE: this image intentionally runs as root. The benchmark/plots services write to
# host directories bind-mounted at results/ and figures/; a non-root UID cannot write
# to those host-owned mounts (verified: PermissionError). This is a local, localhost-only
# research tool, so root in the container carries no meaningful additional risk. Add a
# matching-UID non-root user only if this is ever exposed beyond localhost.

# Streamlit dashboard port (used by the docker-compose "dashboard" service).
EXPOSE 8501

# Default action: run the full benchmark.
CMD ["python", "experiments/run_benchmark.py"]
