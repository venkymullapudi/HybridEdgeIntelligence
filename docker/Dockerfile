# Gateway image: confidence gate + Claude escalation. The open-weight model runs
# in a separate vLLM container on the same task (see docker-compose.yml / Terraform).
# Pinned by digest (multi-arch index): the tag is re-pointed on every Debian/Python patch.
FROM python:3.12-slim@sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
COPY pyproject.toml README.md LICENSE docker/constraints.txt ./
COPY hybrid_edge ./hybrid_edge
# /app is root-owned; give the unprivileged user a writable data dir so the file
# record sink's default path (data/escalations.jsonl) works.
# Exact, CI-tested dependency versions: every build of a commit installs the same set.
RUN pip install -c constraints.txt ".[aws]" && useradd --uid 10001 --create-home app \
    && mkdir -p /app/data && chown app:app /app/data

USER app
EXPOSE 8080
HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/healthz', timeout=2).status==200 else 1)"
CMD ["hybrid-edge"]
