# syntax=docker/dockerfile:1.7
# =============================================================================
#  nuance-mcp — multi-stage container for the FastMCP server
#
#  What runs in here: the *agent-side* FastMCP server with an adapter selected
#  by build arg or env var.  What does NOT run in here: the Gatan/Hitachi
#  host-process bridge plugins (those run inside their vendor application
#  GUI on the microscope PC, which is typically Windows + GMS).
#
#  Typical use:
#
#    # Simulator (CI, demos, teaching) — no hardware required
#    docker build -t nuance-mcp:simulator --build-arg ADAPTER=simulator .
#    docker run --rm -p 8000:8000 nuance-mcp:simulator
#
#    # Gatan adapter — connect to the bridge on the microscope PC
#    docker build -t nuance-mcp:gatan --build-arg ADAPTER=gatan .
#    docker run --rm -p 8000:8000 \
#      -e NUANCE_MCP_ADAPTER=gatan \
#      -e GMS_MCP_ZMQ=tcp://microscope-pc.lan:5555 \
#      nuance-mcp:gatan
#
#    # JEOL adapter (offline simulator inside the container)
#    docker run --rm -p 8000:8000 \
#      -e NUANCE_MCP_ADAPTER=jeol \
#      -e NUANCE_MCP_JEOL_MODE=offline \
#      nuance-mcp:simulator   # base image is enough; jeol extra not vendor-locked
#
#  The streamable HTTP transport is exposed on :8000 for remote MCP clients
#  (e.g. a Microsoft 365 Copilot agent through a facility-managed reverse
#  proxy).  For stdio use, override the CMD: `docker run … nuance-mcp serve
#  --adapter simulator --transport stdio`.
# =============================================================================

ARG PYTHON_VERSION=3.12
ARG ADAPTER=simulator

# -----------------------------------------------------------------------------
# Stage 1 — builder: install package and dependencies into a venv
# -----------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

# Minimum system deps for building wheels (e.g. numpy, scipy)
RUN apt-get update && apt-get install --no-install-recommends -y \
        build-essential gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY pyproject.toml README.md ./
COPY src ./src

# Install into an isolated virtualenv so the runtime stage can grab it whole.
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip wheel \
    && /opt/venv/bin/pip install ".[gatan,jeol,ollama]"

# -----------------------------------------------------------------------------
# Stage 2 — runtime: minimal image with the venv + a non-root user
# -----------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS runtime

ARG ADAPTER

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    NUANCE_MCP_ADAPTER="${ADAPTER}" \
    NUANCE_MCP_TRANSPORT="http" \
    NUANCE_MCP_HOST="0.0.0.0" \
    NUANCE_MCP_PORT="8000"

# Non-root user
RUN useradd --create-home --shell /bin/bash --uid 1000 nuance \
    && mkdir -p /app /home/nuance/.cache \
    && chown -R nuance:nuance /app /home/nuance

# Pull the prebuilt venv from the builder stage
COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY --chown=nuance:nuance examples ./examples
COPY --chown=nuance:nuance docs ./docs
COPY --chown=nuance:nuance README.md LICENSE* CHANGELOG.md ./

USER nuance

EXPOSE 8000

# Container healthcheck — the streamable HTTP transport exposes /mcp;
# the simplest liveness probe is a TCP open on the server port.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import socket,os,sys; s=socket.socket(); \
        s.settimeout(2); \
        sys.exit(0 if s.connect_ex(('127.0.0.1', int(os.environ.get('NUANCE_MCP_PORT','8000'))))==0 else 1)"

# The entrypoint reads NUANCE_MCP_ADAPTER / NUANCE_MCP_TRANSPORT at run-time so
# one image serves every backend without rebuilding.
ENTRYPOINT ["sh", "-c", "exec nuance-mcp serve \
    --adapter \"$NUANCE_MCP_ADAPTER\" \
    --transport \"$NUANCE_MCP_TRANSPORT\" \
    --host \"$NUANCE_MCP_HOST\" \
    --port \"$NUANCE_MCP_PORT\""]

# -----------------------------------------------------------------------------
# Image metadata (OCI labels)
# -----------------------------------------------------------------------------
LABEL org.opencontainers.image.title="nuance-mcp" \
      org.opencontainers.image.description="Vendor-agnostic MCP server for multimodal electron microscopy" \
      org.opencontainers.image.url="https://github.com/NUANCE-IT/nuance-mcp" \
      org.opencontainers.image.source="https://github.com/NUANCE-IT/nuance-mcp" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.authors="Roberto dos Reis, Vinayak P. Dravid"
