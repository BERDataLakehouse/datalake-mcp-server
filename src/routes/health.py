"""
Health check routes for the API.
"""

from fastapi import APIRouter

from src.service.models import HealthResponse

# Create a router for health endpoints
router = APIRouter(tags=["health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Returns the health status of the API.",
)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(status="healthy")
