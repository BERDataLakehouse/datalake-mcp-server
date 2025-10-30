import pytest
from fastapi.testclient import TestClient
from src.main import create_application


@pytest.fixture
def client():
    app = create_application()
    return TestClient(app)
