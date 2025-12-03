"""Tests for the main application."""


def test_health_check(client):
    """Test the health check endpoint returns proper structure."""
    response = client.get("/apis/mcp/health")
    assert response.status_code == 200

    data = response.json()
    # Verify response structure (DeepHealthResponse)
    assert "status" in data
    assert "components" in data
    assert "message" in data

    # Status should be one of the valid values
    assert data["status"] in ["healthy", "unhealthy", "degraded"]

    # Components should be a list
    assert isinstance(data["components"], list)

    # Each component should have required fields
    for component in data["components"]:
        assert "name" in component
        assert "status" in component
        assert component["status"] in ["healthy", "unhealthy", "degraded"]
