"""
Configuration settings for the BERDL Datalake MCP Server.

This service enables AI assistants to interact with Delta Lake tables stored in MinIO through Spark,
implementing the Model Context Protocol (MCP) for natural language data operations.
"""

import logging
import os
from functools import lru_cache

from pydantic import BaseModel, Field

APP_VERSION = "0.1.0"
SERVICE_ROOT_PATH = "/apis/mcp"


class Settings(BaseModel):
    """
    Application settings for the BERDL Datalake MCP Server.
    """

    app_name: str = "BERDL Datalake MCP Server"
    app_description: str = (
        "FastAPI service for AI assistants to interact with Delta Lake tables via Spark"
    )
    api_version: str = APP_VERSION
    service_root_path: str = os.getenv("SERVICE_ROOT_PATH", SERVICE_ROOT_PATH)
    log_level: str = Field(
        default=os.getenv("LOG_LEVEL", "INFO"),
        description="Logging level for the application",
    )


@lru_cache()
def get_settings() -> Settings:
    """
    Get the application settings.

    Uses lru_cache to avoid loading the settings for every request.
    """
    return Settings()


def configure_logging():
    """Configure logging for the application."""
    settings = get_settings()
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if settings.log_level.upper() not in logging.getLevelNamesMapping():
        logging.warning(
            "Unrecognized log level '%s'. Falling back to 'INFO'.",
            settings.log_level,
        )
