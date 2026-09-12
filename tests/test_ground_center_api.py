import pytest
from fastapi.testclient import TestClient
from ground_center_server import app

client = TestClient(app)

def test_read_protocols():
    response = client.get("/api/v1/protocols")
    assert response.status_code == 200
    data = response.json()
    assert "count" in data
    assert "protocols" in data
    assert len(data["protocols"]) > 0

def test_telemetry_state():
    response = client.get("/api/v1/telemetry/state")
    assert response.status_code == 200
    data = response.json()
    assert "protocol_id" in data
    assert "astronaut_pose" in data

def test_switch_protocol():
    response = client.post("/api/v1/protocol/select/BAS-EXP-FLUID-2026")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "PROTOCOL_SWITCH_ACK"
    
    # Verify state updated
    state_response = client.get("/api/v1/telemetry/state")
    state_data = state_response.json()
    assert state_data["protocol_id"] == "BAS-EXP-FLUID-2026"
