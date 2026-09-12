import os
import pytest
import time
from unittest.mock import MagicMock
from simulate_scenarios import (
    run_scenario_1_correct,
    run_scenario_2_skipped,
    run_scenario_3_wrong_order,
    run_scenario_4_wrong_object,
    run_scenario_fluid,
    run_scenario_avionics,
    run_scenario_emergency
)
from astronaut_monitor import ProtocolComplianceEngine, MissionControlLink

# Mock URL since we don't want to rely on the server running during unit tests
MOCK_GROUND_URL = "http://localhost:8000"

@pytest.fixture
def mock_mc_link(monkeypatch):
    """Mock the Mission Control Link to avoid actual HTTP requests during tests."""
    mock = MagicMock(spec=MissionControlLink)
    monkeypatch.setattr("simulate_scenarios.MissionControlLink", lambda ground_url: mock)
    return mock

def test_scenario_1_correct(mock_mc_link):
    """Test standard 100% compliant flow."""
    result = run_scenario_1_correct("configs/protocol_bio.json", MOCK_GROUND_URL)
    assert result is True

def test_scenario_2_skipped(mock_mc_link, caplog):
    """Test deviation generation when a step is skipped."""
    run_scenario_2_skipped("configs/protocol_bio.json", MOCK_GROUND_URL)
    assert any("skipped" in record.message.lower() for record in caplog.records)

def test_scenario_3_wrong_order(mock_mc_link, caplog):
    """Test deviation generation for out-of-order steps."""
    run_scenario_3_wrong_order("configs/protocol_bio.json", MOCK_GROUND_URL)
    assert any("out of order" in record.message.lower() for record in caplog.records)

def test_scenario_4_wrong_object(mock_mc_link, caplog):
    """Test deviation for interacting with the wrong payload."""
    run_scenario_4_wrong_object("configs/protocol_bio.json", MOCK_GROUND_URL)
    assert any("expected component_a, but component_b" in record.message.lower() for record in caplog.records)

def test_scenario_fluid(mock_mc_link):
    result = run_scenario_fluid("configs/protocol_fluid.json", MOCK_GROUND_URL)
    assert result is True

def test_scenario_avionics(mock_mc_link):
    result = run_scenario_avionics("configs/protocol_avionics.json", MOCK_GROUND_URL)
    assert result is True

def test_scenario_emergency(mock_mc_link):
    result = run_scenario_emergency("configs/protocol_emergency.json", MOCK_GROUND_URL)
    assert result is True
