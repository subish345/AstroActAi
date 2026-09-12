import pytest
from unittest.mock import patch, MagicMock
from astronaut_monitor import AstronautMonitoringSystem

@patch("astronaut_monitor.cv2.VideoCapture")
def test_camera_fallback_to_synthetic(mock_video_capture, caplog):
    """
    Test that if the physical camera cannot be opened, the system falls back
    to the synthetic feed mode without crashing.
    """
    # Mock VideoCapture to return False for isOpened()
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = False
    mock_video_capture.return_value = mock_cap
    
    # We want to exit the loop quickly in the test.
    # The run() method loops infinitely for synthetic feed unless we break.
    # We'll mock time.sleep to raise an exception to break the while loop.
    with patch("time.sleep", return_value=None):
        system = AstronautMonitoringSystem(
            protocol_path="configs/protocol_bio.json"
        )
        system.run(camera_source=99, headless=True)
            
    assert any("falling back to synthetic feed mode" in record.message.lower() for record in caplog.records)
