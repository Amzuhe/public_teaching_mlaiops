# ITCS355 Lab 1 — training image
#
# Base image pinned BY DIGEST (Lab 1, Task 2): tags move, a digest can't. The tag stays for
# humans; Docker uses the digest. Digest = the python:3.11-slim multi-arch index on 2026-10-17
# (read from Docker Hub's registry API). To update: re-read the digest, rebuild, re-run make verify.
FROM python:3.11-slim@sha256:e88e9763f943ec1834f992a4b51e0f24500486803e8bc534e5767af9ea65f6ce AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build

# Dependencies first so this layer caches independently of your source.
COPY requirements.txt ./
# --require-hashes (Lab 1, Task 2): requirements.txt is a hashed lock (uv pip compile, linux/py3.11),
# so a silently-substituted package becomes a build failure instead of a different model.
RUN pip install --prefix=/install --require-hashes -r requirements.txt


FROM python:3.11-slim@sha256:e88e9763f943ec1834f992a4b51e0f24500486803e8bc534e5767af9ea65f6ce AS runtime

# Non-root. A training container has no reason to run as root, and graders check.
RUN useradd --create-home --uid 10001 runner
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

COPY --from=builder /install /usr/local
WORKDIR /app
COPY --chown=runner:runner src/ ./src/
COPY --chown=runner:runner cloudlayer/ ./cloudlayer/
COPY --chown=runner:runner scripts/ ./scripts/

# runner must own the workdir: MLflow creates ./mlruns there (Permission denied otherwise)
RUN chown runner:runner /app

USER runner

# Credentials NEVER enter an image layer. They arrive at runtime from SECRET_STORE_PATH
# or from the platform's identity. If you find yourself adding an ARG for a key, stop.
ENTRYPOINT ["python", "-m", "src.train"]
CMD ["--n-estimators", "200", "--max-depth", "8"]
