#!/bin/bash
set -e

# No need for setup.sh - spark_notebook_base already has Spark configured
# SPARK_HOME and SPARK_CONF_DIR are already set in the base image

# Launch the MCP server
exec python -m src.main