ARG BASE_TAG=main
ARG BASE_REGISTRY=ghcr.io/berdatalakehouse/
FROM ${BASE_REGISTRY}spark_notebook_base:${BASE_TAG}

# Switch to root to install packages
USER root

RUN apt-get update && apt-get install -y \
    # required for psycopg
    libpq-dev gcc \
    # required for git dependencies (berdl-notebook-utils)
    git \
    && rm -rf /var/lib/apt/lists/*

ENV SPARK_JARS_DIR=/usr/local/spark/jars

ENV CONFIG_DIR=/opt/config
COPY ./config/ ${CONFIG_DIR}
ENV SPARK_FAIR_SCHEDULER_CONFIG=${CONFIG_DIR}/spark-fairscheduler.xml

WORKDIR /app

# Copy the application code and dependencies
COPY pyproject.toml uv.lock .python-version ./
COPY ./src /app/src

# Install dependencies into the system conda environment
RUN eval "$(conda shell.bash hook)" && uv pip install --system .

COPY ./scripts/ /opt/scripts/
RUN chmod a+x /opt/scripts/*.sh

# Create non-root user for running the service
# Use UID 1001 to avoid conflict with jovyan user (UID 1000) from base image
RUN groupadd -g 1001 mcp && \
    useradd -u 1001 -g 1001 -m -s /bin/bash mcp && \
    chown -R mcp:mcp /app /opt/scripts

# Switch to non-root user
USER mcp

ENTRYPOINT ["/usr/bin/tini", "--", "/opt/scripts/mcp-server-entrypoint.sh"]