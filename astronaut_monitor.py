"""
AstroActAi — Edge Astronaut Activity Monitoring & Deterministic Protocol Compliance Engine
Bharatiya Antariksh Station (BAS) Experiment Verification System
Features:
  - MediaPipe Tasks 33 3D Keypoint Pose Estimation
  - MediaPipe Tasks AI Object Detector (EfficientDet-Lite)
  - Full Dual-Hand Tracking & In-Hand Object Identification (Ambidextrous Manipulation)
  - Real-time Session Recording to disk (telemetry/sessions/)
  - MJPEG Live Frame Streaming to Web Dashboard
  - Deterministic FSM Engine validating transitions with ANY hand
"""

import cv2
import json
import math
import time
import os
import sys
import argparse
import threading
import logging
import signal
import atexit
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict, field
import urllib.request
import numpy as np
import requests
import queue

def _close_camera_windows():
    """Guarantee all OpenCV camera windows are closed and event queues drained."""
    try:
        cv2.destroyAllWindows()
        for _ in range(5):
            cv2.waitKey(1)
    except Exception:
        pass

atexit.register(_close_camera_windows)

def _termination_signal_handler(signum, frame):
    """Handle SIGINT / SIGTERM to close camera windows cleanly before exiting."""
    _close_camera_windows()
    sys.exit(0)

try:
    signal.signal(signal.SIGINT, _termination_signal_handler)
    signal.signal(signal.SIGTERM, _termination_signal_handler)
except Exception:
    pass

# Optional local offline TTS
try:
    import pyttsx3
    TTS_ENABLED = True
except ImportError:
    TTS_ENABLED = False

# MediaPipe Tasks API
try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    MP_AVAILABLE = True
except ImportError:
    MP_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [ASTRO-COPILOT] %(levelname)s: %(message)s")
logger = logging.getLogger("AstroCopilot")

POSE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
POSE_MODEL_PATH = "pose_landmarker_lite.task"

OBJECT_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/1/efficientdet_lite0.tflite"
OBJECT_MODEL_PATH = "efficientdet_lite0.tflite"

# Non-handheld classes to filter out of AI object detector
IGNORED_CLASSES = {
    "person", "chair", "couch", "bed", "dining table", "tv", "sofa", "bench",
    "refrigerator", "toilet", "door", "window", "sink", "potted plant", "traffic light"
}

# Space Payload Intelligence Registry (Aerospace-grade metadata for detected items)
SPACE_PAYLOAD_REGISTRY: Dict[str, Dict[str, Any]] = {
    "Component_A": {
        "common_name": "Biospecimen Canister Component A",
        "payload_id": "PL-BIO-2026-A",
        "microg_mass": "320 g",
        "dimensions": "80mm dia x 220mm length",
        "hazard_level": "Biosafety Level 2 (Hermetic Seal)",
        "destination_bay": "Rack Slot 1 (Incubation Bay A)",
        "alignment_pin": "90 deg Radial Offset",
        "bus_interface": "SpaceWire Bus 4",
        "handling": "Ambidextrous pick (Left or Right Hand). Orient 90 deg guide pin before bay insertion."
    },
    "Component_B": {
        "common_name": "Secondary Calibration Canister",
        "payload_id": "PL-DUMMY-2026-B",
        "microg_mass": "280 g",
        "dimensions": "80mm dia x 220mm length",
        "hazard_level": "Non-Hazardous (Calibration Saline)",
        "destination_bay": "Storage Bay 2 (DO NOT INSERT INTO SLOT 1)",
        "alignment_pin": "Non-Indexed",
        "bus_interface": "None",
        "handling": "Calibration specimen only. Do not mix with live biological specimen."
    },
    "cell phone": {
        "common_name": "Flight Telemetry Terminal / Crew Comms Pad",
        "payload_id": "PL-CREW-PAD-07",
        "microg_mass": "185 g",
        "dimensions": "75mm x 150mm x 8.2mm",
        "hazard_level": "Non-Hazardous (RF Class 1 Shielded)",
        "destination_bay": "Crew Console / Telemetry Cradle",
        "alignment_pin": "Zero Offset",
        "bus_interface": "Wi-Fi 6E / UWB Space Link",
        "handling": "Ambidextrous handheld operation. Keep tethered with zero-G velcro strap."
    },
    "bottle": {
        "common_name": "Fluid Dynamics Test Canister",
        "payload_id": "PL-FLUID-RSV-01",
        "microg_mass": "450 g (Fluidic Ballast)",
        "dimensions": "70mm dia x 210mm",
        "hazard_level": "Capillary Liquid Containment (Hermetic)",
        "destination_bay": "Incubation Slot 1",
        "alignment_pin": "Axial Guide Pin",
        "bus_interface": "Glovebox Fluid Bus",
        "handling": "Maintain smooth angular slewing to minimize microgravity fluid sloshing."
    },
    "cup": {
        "common_name": "Capillary Surface Tension Test Cell",
        "payload_id": "PL-CAP-VESSEL-04",
        "microg_mass": "160 g",
        "dimensions": "85mm dia x 100mm",
        "hazard_level": "Non-Hazardous (Surface Tension Cell)",
        "destination_bay": "Fluid Rack Slot 1",
        "alignment_pin": "Axial",
        "bus_interface": "Glovebox Bus",
        "handling": "Hold securely by rim or cylinder with either hand."
    },
    "mouse": {
        "common_name": "Canadarm Manipulator Remote",
        "payload_id": "PL-RMI-CTR-02",
        "microg_mass": "115 g",
        "dimensions": "65mm x 115mm x 38mm",
        "hazard_level": "Class 2 ESD Sensitive",
        "destination_bay": "Robotics Console Cradle",
        "alignment_pin": "Zero Offset",
        "bus_interface": "MIL-STD-1553",
        "handling": "Use wrist ground strap during precision robotics actuation."
    },
    "remote": {
        "common_name": "Payload Mechanism Actuator Controller",
        "payload_id": "PL-REM-ACT-05",
        "microg_mass": "130 g",
        "dimensions": "50mm x 160mm x 25mm",
        "hazard_level": "Non-Hazardous (RF Controlled)",
        "destination_bay": "Robotics Workstation",
        "alignment_pin": "Zero Offset",
        "bus_interface": "Infrared / UWB",
        "handling": "Firm ambidextrous grasp with left or right hand."
    },
    "book": {
        "common_name": "Nomex Flight Procedure Checklist & Logbook",
        "payload_id": "PL-EVA-CHECKLIST-01",
        "microg_mass": "310 g",
        "dimensions": "140mm x 215mm",
        "hazard_level": "Zero-Flammability Nomex Coated",
        "destination_bay": "Glovebox Documentation Clip",
        "alignment_pin": "Zero Offset",
        "bus_interface": "None",
        "handling": "Consult step-by-step experiment protocol throughout run."
    },
    "Syringe_Injector": {
        "common_name": "Positive Displacement Fluid Syringe",
        "payload_id": "PL-SYR-FLUID-02",
        "microg_mass": "95 g",
        "dimensions": "25mm dia x 140mm",
        "hazard_level": "Non-Hazardous (Biocompatible)",
        "destination_bay": "Manifold_Port_A",
        "alignment_pin": "Luer-Lock Quick-Disconnect",
        "handling": "Ambidextrous pick. Maintain gentle plunger force to avoid cavitation."
    },
    "Sample_Vial": {
        "common_name": "Reagent Septum Vial #04",
        "payload_id": "PL-VIAL-BUF-01",
        "microg_mass": "60 g",
        "dimensions": "30mm dia x 75mm",
        "hazard_level": "Biosafety Level 1 (Sealed Septum)",
        "destination_bay": "Incubation Tray B",
        "alignment_pin": "Self-Sealing Silicone Septum",
        "handling": "Hold securely at base while inserting syringe needle."
    },
    "Manifold_Port_A": {
        "common_name": "Capillary Manifold Port A",
        "payload_id": "PORT-CAP-01-A",
        "microg_mass": "420 g (Fixed Station)",
        "dimensions": "60mm x 80mm",
        "hazard_level": "Pressurized Fluid Bus (1.2 bar)",
        "destination_bay": "Glovebox Fluid Bench",
        "alignment_pin": "Luer Detent Port",
        "handling": "Align syringe luer tip coaxially before rotating 45 deg to lock."
    },
    "Pinch_Valve": {
        "common_name": "Fluid Line Retention Clamp",
        "payload_id": "VALVE-PINCH-03",
        "microg_mass": "80 g",
        "dimensions": "40mm x 40mm",
        "hazard_level": "Mechanical Spring Retention",
        "destination_bay": "Capillary Line A",
        "alignment_pin": "Direct Pinch Detent",
        "handling": "Squeeze clamp levers firmly to seat onto silicone tube."
    },
    "SpaceWire_Harness": {
        "common_name": "SpaceWire Data Bus Harness",
        "payload_id": "PL-SPW-HARNESS-04",
        "microg_mass": "140 g",
        "dimensions": "45mm dia connector x 450mm harness",
        "hazard_level": "ESD Sensitive (Class 2)",
        "destination_bay": "Avionics_Port_4",
        "alignment_pin": "Clocked Master Keyway (Key A)",
        "handling": "Wear ground strap. Align index keyway before applying axial seating force."
    },
    "Avionics_Port_4": {
        "common_name": "Circular Mil-Spec Avionics Port 4",
        "payload_id": "PORT-SPW-04-B",
        "microg_mass": "350 g (Fixed Panel)",
        "dimensions": "50mm dia",
        "hazard_level": "28V DC Interlocked Bus",
        "destination_bay": "Avionics Bay 3 Rack",
        "alignment_pin": "Mil-Spec Keyway Detent",
        "handling": "Verify pin straightness before axial mating."
    },
    "Torque_Wrench": {
        "common_name": "Calibrated 4.5 Nm Torque Tool",
        "payload_id": "TOOL-TORQ-BAS-01",
        "microg_mass": "380 g",
        "dimensions": "30mm dia x 210mm",
        "hazard_level": "Non-Magnetic (Beryllium-Copper)",
        "destination_bay": "Retainer Collar",
        "alignment_pin": "1/4-inch Square Drive",
        "handling": "Apply smooth clockwise moment until audible 4.5 Nm click."
    },
    "Breaker_Toggle": {
        "common_name": "28V DC Power Bus Breaker",
        "payload_id": "SW-PWR-28V",
        "microg_mass": "75 g",
        "dimensions": "25mm x 35mm",
        "hazard_level": "Electrical Switching Point",
        "destination_bay": "Power Control Panel",
        "alignment_pin": "Toggle Safety Lock Lever",
        "handling": "Lift lever safety lock and throw firmly into ON position."
    },
    "Breather_Mask": {
        "common_name": "Quick-Don Emergency O2 Mask",
        "payload_id": "PL-EMG-MASK-01",
        "microg_mass": "520 g",
        "dimensions": "160mm x 180mm x 110mm",
        "hazard_level": "Life Safety Essential",
        "destination_bay": "Crew Face Level",
        "alignment_pin": "Nose & Chin Sealing Flange",
        "handling": "Don with single or dual hands. Pull head harness straps firmly."
    },
    "Equalization_Valve": {
        "common_name": "Manual Delta-P Equalization Valve",
        "payload_id": "VALVE-DP-EQ-02",
        "microg_mass": "850 g",
        "dimensions": "75mm dia wheel",
        "hazard_level": "Cabin Pressure Boundary Barrier",
        "destination_bay": "Bulkhead Panel",
        "alignment_pin": "Rotary 90 deg Cam",
        "handling": "Rotate 90 deg clockwise to isolate chamber."
    },
    "Hatch_Dog_Handle": {
        "common_name": "Bulkhead Hatch Dogging Lever",
        "payload_id": "MECH-HATCH-DOG-01",
        "microg_mass": "1200 g",
        "dimensions": "40mm x 280mm Lever",
        "hazard_level": "Primary Pressure Seal Mechanism",
        "destination_bay": "Bulkhead Doorframe",
        "alignment_pin": "Dual Shear Pins",
        "handling": "Grasp lever firmly with either hand and rotate into locked detent."
    },
    "Secondary_Lock": {
        "common_name": "Hatch Secondary Safety Lock Bar",
        "payload_id": "LOCK-BAR-SEC-01",
        "microg_mass": "650 g",
        "dimensions": "35mm x 190mm Bar",
        "hazard_level": "Secondary Pressure Barrier",
        "destination_bay": "Bulkhead Safety Catch",
        "alignment_pin": "Safety Microswitch Detent",
        "handling": "Slide and lock bar horizontally until green telemetry indicator engages."
    }
}

# ==============================================================================
# DATA MODELS
# ==============================================================================

@dataclass
class Keypoint3D:
    id: int
    name: str
    x: float          # Screen space [0, 1]
    y: float          # Screen space [0, 1]
    z: float          # Depth
    rack_x: float     # Rack-relative normalized coordinate
    rack_y: float     # Rack-relative normalized coordinate
    rack_z: float     # Rack-relative normalized coordinate
    confidence: float

@dataclass
class HandTrackingState:
    hand_name: str                                          # "LEFT" or "RIGHT"
    is_detected: bool = False
    screen_pos: Tuple[float, float] = (0.5, 0.5)           # (wrist_x, wrist_y)
    palm_pos: Tuple[float, float] = (0.5, 0.5)             # (palm_x, palm_y)
    rack_pos: Tuple[float, float, float] = (0.0, 0.0, 0.0) # (x, y, z)
    velocity: float = 0.0
    held_object: Optional[str] = None
    held_object_data: Optional[Dict[str, Any]] = None
    contact_distance: float = 999.0
    fingers: Dict[str, Tuple[float, float]] = field(default_factory=dict)

@dataclass
class LegTrackingState:
    leg_name: str                                           # "LEFT" or "RIGHT"
    is_detected: bool = False
    knee_angle_deg: float = 180.0                           # Angle in degrees at knee joint
    velocity: float = 0.0                                   # Velocity of knee/ankle
    activity: str = "ANCHORED"                              # "ANCHORED", "KNEE_FLEXION", "FLOATING", "TREADING"
    hip_pos: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    knee_pos: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    ankle_pos: Tuple[float, float, float] = (0.0, 0.0, 0.0)

@dataclass
class AstronautPose:
    timestamp: float = 0.0
    keypoints: Dict[str, Keypoint3D] = field(default_factory=dict)
    left_hand: Optional[HandTrackingState] = None
    right_hand: Optional[HandTrackingState] = None
    left_leg: Optional[LegTrackingState] = None
    right_leg: Optional[LegTrackingState] = None
    lower_body_activity: str = "ANCHORED IN FOOT RESTRAINT"
    active_hand: str = "RIGHT"                             # "LEFT", "RIGHT", or "NONE"
    wrist_velocity: float = 0.0
    body_roll_degrees: float = 0.0
    dominant_wrist_pos: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    held_payload: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.left_hand is None:
            self.left_hand = HandTrackingState("LEFT", False, (0.3, 0.5), (0.3, 0.5), (-0.2, 0.0, 0.0), 0.0)
        if self.right_hand is None:
            self.right_hand = HandTrackingState("RIGHT", False, (0.7, 0.5), (0.7, 0.5), (0.2, 0.0, 0.0), 0.0)
        if self.left_leg is None:
            self.left_leg = LegTrackingState("LEFT", False, 180.0, 0.0, "ANCHORED")
        if self.right_leg is None:
            self.right_leg = LegTrackingState("RIGHT", False, 180.0, 0.0, "ANCHORED")

@dataclass
class DetectedObject:
    object_id: str
    class_name: str
    centroid: Tuple[float, float]
    bbox: Tuple[int, int, int, int]
    confidence: float
    specs: Optional[Dict[str, Any]] = None
    source: str = "PROTOCOL_PAYLOAD"    # "AI_DETECTOR" or "PROTOCOL_PAYLOAD"

@dataclass
class ActionCandidate:
    action_type: str
    target_object: str
    confidence: float
    timestamp: float
    hand: str = "RIGHT"                # "LEFT" or "RIGHT"
    object_data: Optional[Dict[str, Any]] = None

# ==============================================================================
# MISSION CONTROL TELEMETRY DISPATCHER
# ==============================================================================

class MissionControlLink:
    """Handles local flight logs and uplink dispatch to Ground Station."""

    _speech_queue: queue.Queue = queue.Queue()
    _procedure_completed: bool = False
    _speech_worker_started: bool = False
    _tts_engine: Optional[Any] = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self, ground_url: str = "http://127.0.0.1:8000", log_file: str = "telemetry/flight_log.jsonl"):
        self.ground_url = ground_url.rstrip("/")
        self.log_file = log_file
        self.session = requests.Session()
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)

        # High-frequency non-blocking telemetry worker matching camera FPS
        self.telemetry_queue = queue.Queue(maxsize=2)
        self._telemetry_worker_thread = threading.Thread(target=self._telemetry_worker, daemon=True)
        self._telemetry_worker_thread.start()

        with MissionControlLink._lock:
            if not MissionControlLink._speech_worker_started:
                MissionControlLink._speech_worker_started = True
                threading.Thread(target=MissionControlLink._speech_worker, daemon=True).start()

    @classmethod
    def _speech_worker(cls):
        """Worker consuming voice alert messages sequentially without overlapping threads."""
        tts_engine = None
        if TTS_ENABLED:
            try:
                tts_engine = pyttsx3.init()
                tts_engine.setProperty('rate', 160)
            except Exception as e:
                logger.debug(f"Offline voice synthesizer initialization: {e}")
                tts_engine = None

        while True:
            try:
                text = cls._speech_queue.get()
                if text is None:
                    break
                if tts_engine:
                    try:
                        tts_engine.say(text)
                        tts_engine.runAndWait()
                    except Exception:
                        pass
            except Exception:
                pass

    def speak(self, text: str, is_final: bool = False):
        """Asynchronous crew auditory alert. Dropped if procedure is already completed."""
        with MissionControlLink._lock:
            if MissionControlLink._procedure_completed and not is_final:
                return

            if is_final:
                MissionControlLink._procedure_completed = True
                # Drain any older pending speech alerts so they don't delay or play after completion
                while not MissionControlLink._speech_queue.empty():
                    try:
                        MissionControlLink._speech_queue.get_nowait()
                    except Exception:
                        break

            print(f"\n🔔 [VOICE ALERT CHIME]: \"{text}\"\n")
            MissionControlLink._speech_queue.put(text)

    def mark_procedure_completed(self, final_text: str = "The procedure is ended."):
        """Immediately ends alerts and speaks only the final completion phrase."""
        self.speak(final_text, is_final=True)

    def reset_procedure_state(self):
        """Resets procedure completed state when starting a new experiment/protocol."""
        with MissionControlLink._lock:
            MissionControlLink._procedure_completed = False
            while not MissionControlLink._speech_queue.empty():
                try:
                    MissionControlLink._speech_queue.get_nowait()
                except Exception:
                    break

    def _telemetry_worker(self):
        """Worker consuming real-time telemetry packets at camera speed without blocking capture loop."""
        while True:
            try:
                endpoint, payload = self.telemetry_queue.get()
                url = f"{self.ground_url}{endpoint}"
                self.session.post(url, json=payload, timeout=0.15)
            except Exception:
                pass

    def transmit(self, endpoint: str, payload: Dict[str, Any]):
        """Persists locally and transmits telemetry asynchronously."""
        if endpoint == "/api/v1/telemetry/heartbeat":
            if self.telemetry_queue.full():
                try:
                    self.telemetry_queue.get_nowait()
                except queue.Empty:
                    pass
            try:
                self.telemetry_queue.put_nowait((endpoint, payload))
            except queue.Full:
                pass
        else:
            url = f"{self.ground_url}{endpoint}"
            def _post():
                try:
                    self.session.post(url, json=payload, timeout=0.8)
                except Exception:
                    pass
            threading.Thread(target=_post, daemon=True).start()

    def send_frame(self, frame_bgr: np.ndarray):
        """Asynchronously transmits JPEG frame to Ground Center for live web streaming."""
        def _post_frame():
            try:
                _, buffer = cv2.imencode('.jpg', frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 70])
                self.session.post(
                    f"{self.ground_url}/api/v1/telemetry/frame",
                    data=buffer.tobytes(),
                    headers={"Content-Type": "image/jpeg"},
                    timeout=0.20
                )
            except Exception:
                pass
        threading.Thread(target=_post_frame, daemon=True).start()

    def notify_camera_closed(self):
        """Notifies Ground Station that the camera session has closed."""
        try:
            self.session.post(f"{self.ground_url}/api/v1/telemetry/camera/offline", timeout=0.5)
        except Exception:
            pass

    def send_heartbeat(self, protocol_id: str, step_id: int, pose: AstronautPose, active_action: str = "MONITORING", camera_tag: str = "", camera_channel: str = ""):
        kps_dump = {
            k: {
                "x": round(v.x, 3),
                "y": round(v.y, 3),
                "z": round(v.z, 3),
                "rack_x": round(v.rack_x, 3),
                "rack_y": round(v.rack_y, 3),
                "rack_z": round(v.rack_z, 3)
            }
            for k, v in pose.keypoints.items()
        }
        left_leg_dict = {
            "detected": pose.left_leg.is_detected if pose.left_leg else False,
            "knee_angle": round(pose.left_leg.knee_angle_deg, 1) if pose.left_leg else 180.0,
            "velocity": round(pose.left_leg.velocity, 2) if pose.left_leg else 0.0,
            "activity": pose.left_leg.activity if pose.left_leg else "ANCHORED",
            "coords": [round(c, 2) for c in pose.left_leg.knee_pos] if pose.left_leg else [0.0, 0.0, 0.0]
        }
        right_leg_dict = {
            "detected": pose.right_leg.is_detected if pose.right_leg else False,
            "knee_angle": round(pose.right_leg.knee_angle_deg, 1) if pose.right_leg else 180.0,
            "velocity": round(pose.right_leg.velocity, 2) if pose.right_leg else 0.0,
            "activity": pose.right_leg.activity if pose.right_leg else "ANCHORED",
            "coords": [round(c, 2) for c in pose.right_leg.knee_pos] if pose.right_leg else [0.0, 0.0, 0.0]
        }
        payload = {
            "timestamp": time.time(),
            "protocol_id": protocol_id,
            "current_step": step_id,
            "status": "NORMAL",
            "active_action": active_action,
            "camera_tag": camera_tag,
            "camera_channel": camera_channel,
            "active_hand": pose.active_hand,
            "left_hand": {
                "detected": pose.left_hand.is_detected,
                "held_object": pose.left_hand.held_object,
                "velocity": round(pose.left_hand.velocity, 2),
                "coords": [round(c, 2) for c in pose.left_hand.rack_pos]
            },
            "right_hand": {
                "detected": pose.right_hand.is_detected,
                "held_object": pose.right_hand.held_object,
                "velocity": round(pose.right_hand.velocity, 2),
                "coords": [round(c, 2) for c in pose.right_hand.rack_pos]
            },
            "left_leg": left_leg_dict,
            "right_leg": right_leg_dict,
            "lower_body_activity": pose.lower_body_activity,
            "held_payload": pose.held_payload,
            "astronaut_pose": {
                "body_roll_degrees": round(pose.body_roll_degrees, 1),
                "wrist_velocity": round(pose.wrist_velocity, 2),
                "keypoints": kps_dump
            }
        }
        self.transmit("/api/v1/telemetry/heartbeat", payload)

    def send_step_verified(self, protocol_id: str, step_id: int, action: str, hand: str, pose: AstronautPose):
        kps_dump = {k: {"x": round(v.x, 3), "y": round(v.y, 3), "z": round(v.z, 3)} for k, v in pose.keypoints.items()}
        left_leg_dict = {
            "detected": pose.left_leg.is_detected,
            "knee_angle_deg": round(pose.left_leg.knee_angle_deg, 1),
            "velocity": round(pose.left_leg.velocity, 2),
            "activity": pose.left_leg.activity
        } if pose.left_leg else None
        right_leg_dict = {
            "detected": pose.right_leg.is_detected,
            "knee_angle_deg": round(pose.right_leg.knee_angle_deg, 1),
            "velocity": round(pose.right_leg.velocity, 2),
            "activity": pose.right_leg.activity
        } if pose.right_leg else None
        payload = {
            "timestamp": time.time(),
            "protocol_id": protocol_id,
            "current_step": step_id,
            "status": "STEP_VERIFIED",
            "active_action": action,
            "active_hand": hand,
            "left_hand": {
                "detected": pose.left_hand.is_detected,
                "held_object": pose.left_hand.held_object,
                "velocity": round(pose.left_hand.velocity, 2),
                "coords": [round(c, 2) for c in pose.left_hand.rack_pos]
            },
            "right_hand": {
                "detected": pose.right_hand.is_detected,
                "held_object": pose.right_hand.held_object,
                "velocity": round(pose.right_hand.velocity, 2),
                "coords": [round(c, 2) for c in pose.right_hand.rack_pos]
            },
            "left_leg": left_leg_dict,
            "right_leg": right_leg_dict,
            "lower_body_activity": pose.lower_body_activity,
            "held_payload": pose.held_payload,
            "astronaut_pose": {
                "body_roll_degrees": round(pose.body_roll_degrees, 1),
                "wrist_velocity": round(pose.wrist_velocity, 2),
                "keypoints": kps_dump
            }
        }
        self.speak(f"Step {step_id} verified with {hand} hand.")
        self.transmit("/api/v1/telemetry/heartbeat", payload)

    def send_experiment_complete(self, protocol_id: str, step_id: int, pose: AstronautPose):
        msg = "The procedure is ended."
        self.mark_procedure_completed(msg)
        kps_dump = {k: {"x": round(v.x, 3), "y": round(v.y, 3), "z": round(v.z, 3)} for k, v in pose.keypoints.items()}
        left_leg_dict = {
            "detected": pose.left_leg.is_detected,
            "knee_angle_deg": round(pose.left_leg.knee_angle_deg, 1),
            "velocity": round(pose.left_leg.velocity, 2),
            "activity": pose.left_leg.activity
        } if pose.left_leg else None
        right_leg_dict = {
            "detected": pose.right_leg.is_detected,
            "knee_angle_deg": round(pose.right_leg.knee_angle_deg, 1),
            "velocity": round(pose.right_leg.velocity, 2),
            "activity": pose.right_leg.activity
        } if pose.right_leg else None
        payload = {
            "timestamp": time.time(),
            "protocol_id": protocol_id,
            "current_step": step_id,
            "status": "EXPERIMENT_SUCCESSFUL",
            "active_action": "EXPERIMENT_COMPLETE",
            "alert_message": msg,
            "left_hand": {
                "detected": pose.left_hand.is_detected,
                "held_object": pose.left_hand.held_object,
                "velocity": round(pose.left_hand.velocity, 2),
                "coords": [round(c, 2) for c in pose.left_hand.rack_pos]
            },
            "right_hand": {
                "detected": pose.right_hand.is_detected,
                "held_object": pose.right_hand.held_object,
                "velocity": round(pose.right_hand.velocity, 2),
                "coords": [round(c, 2) for c in pose.right_hand.rack_pos]
            },
            "left_leg": left_leg_dict,
            "right_leg": right_leg_dict,
            "lower_body_activity": pose.lower_body_activity,
            "held_payload": pose.held_payload,
            "astronaut_pose": {
                "body_roll_degrees": round(pose.body_roll_degrees, 1),
                "wrist_velocity": round(pose.wrist_velocity, 2),
                "keypoints": kps_dump
            }
        }
        self.transmit("/api/v1/telemetry/heartbeat", payload)

    def send_deviation_alert(self, protocol_id: str, step_id: int, dev_type: str, message: str, action: str, pose: AstronautPose):
        self.speak(message)
        kps_dump = {
            k: {"x": round(v.x, 4), "y": round(v.y, 4), "z": round(v.z, 4), "rack_x": round(v.rack_x, 4)}
            for k, v in pose.keypoints.items()
        }
        payload = {
            "timestamp": time.time(),
            "protocol_id": protocol_id,
            "current_step": step_id,
            "deviation_type": dev_type,
            "alert_message": message,
            "active_action": action,
            "active_hand": pose.active_hand,
            "held_payload": pose.held_payload,
            "astronaut_pose": {
                "body_roll_degrees": round(pose.body_roll_degrees, 2),
                "wrist_velocity": round(pose.wrist_velocity, 3),
                "keypoints": kps_dump
            }
        }
        self.transmit("/api/v1/telemetry/deviation", payload)

# ==============================================================================
# AI OBJECT DETECTOR (EFFICIENTDET-LITE)
# ==============================================================================

class AIPayloadObjectDetector:
    """Detects handheld and environmental objects using MediaPipe Tasks ObjectDetector."""

    def __init__(self, model_path: str = OBJECT_MODEL_PATH, delegate: str = "gpu"):
        self.detector = None
        self.device_used = "CPU"
        if MP_AVAILABLE:
            if not os.path.exists(model_path):
                logger.info(f"Downloading AI object detection model from {OBJECT_MODEL_URL}...")
                try:
                    urllib.request.urlretrieve(OBJECT_MODEL_URL, model_path)
                    logger.info("AI object model downloaded.")
                except Exception as e:
                    logger.warning(f"Could not download object detector: {e}")

            if os.path.exists(model_path):
                if delegate.lower() == "gpu":
                    try:
                        base_options = python.BaseOptions(
                            model_asset_path=model_path,
                            delegate=python.BaseOptions.Delegate.GPU
                        )
                        options = vision.ObjectDetectorOptions(
                            base_options=base_options,
                            score_threshold=0.22,
                            max_results=10
                        )
                        self.detector = vision.ObjectDetector.create_from_options(options)
                        self.device_used = "GPU"
                        logger.info("AI ObjectDetector initialized on GPU (EfficientDet-Lite).")
                    except Exception as e:
                        logger.warning(f"Failed to create ObjectDetector on GPU, falling back to CPU: {e}")

                if not self.detector:
                    try:
                        base_options = python.BaseOptions(
                            model_asset_path=model_path,
                            delegate=python.BaseOptions.Delegate.CPU
                        )
                        options = vision.ObjectDetectorOptions(
                            base_options=base_options,
                            score_threshold=0.22,
                            max_results=10
                        )
                        self.detector = vision.ObjectDetector.create_from_options(options)
                        self.device_used = "CPU"
                        logger.info("AI ObjectDetector initialized on CPU (EfficientDet-Lite).")
                    except Exception as e:
                        logger.warning(f"Failed to create ObjectDetector on CPU: {e}")

    def detect_objects(self, frame_bgr: np.ndarray) -> List[DetectedObject]:
        if not self.detector:
            return []

        h, w, _ = frame_bgr.shape
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        try:
            res = self.detector.detect(mp_img)
            objects = []
            for i, det in enumerate(res.detections):
                bbox = det.bounding_box
                xmin = max(0, bbox.origin_x)
                ymin = max(0, bbox.origin_y)
                xmax = min(w, xmin + bbox.width)
                ymax = min(h, ymin + bbox.height)

                category = det.categories[0] if det.categories else None
                raw_name = category.category_name if category else "Payload_Item"
                score = float(category.score) if category else 0.5

                # Filter out background furniture or person
                if raw_name.lower() in IGNORED_CLASSES:
                    continue

                cx = (xmin + xmax) / (2.0 * w)
                cy = (ymin + ymax) / (2.0 * h)

                # Resolve aerospace specs from registry or synthesize dynamic specs
                matched_key = None
                for k in SPACE_PAYLOAD_REGISTRY:
                    if k.lower() == raw_name.lower() or k.lower() in raw_name.lower():
                        matched_key = k
                        break

                if matched_key:
                    specs = SPACE_PAYLOAD_REGISTRY[matched_key].copy()
                    display_name = specs.get("common_name", raw_name)
                else:
                    display_name = raw_name.title()
                    specs = {
                        "common_name": f"AI-Identified Handheld ({display_name})",
                        "payload_id": f"PL-AI-{raw_name[:4].upper()}-26",
                        "microg_mass": f"{max(120, int((xmax - xmin) * (ymax - ymin) / 450))} g",
                        "dimensions": f"{int((xmax - xmin) * 0.45)}mm x {int((ymax - ymin) * 0.45)}mm",
                        "hazard_level": "Under Observation (Non-Hazardous)",
                        "destination_bay": "Payload Workstation",
                        "alignment_pin": "Universal Coupling",
                        "handling": "Ambidextrous grasp permitted. Maintain steady zero-G handling."
                    }

                objects.append(DetectedObject(
                    object_id=f"ai_det_{i}",
                    class_name=raw_name,
                    centroid=(cx, cy),
                    bbox=(xmin, ymin, xmax, ymax),
                    confidence=score,
                    specs=specs,
                    source="AI_DETECTOR"
                ))
            return objects
        except Exception as e:
            logger.debug(f"Object detection inference error: {e}")
            return []

# ==============================================================================
# SPATIAL INFERENCE & 33 3D DUAL-HAND POSE ESTIMATION
# ==============================================================================

class MicrogravityPoseEngine:
    """Extracts all 33 body landmarks and tracks BOTH hands independently in 3D."""

    LANDMARK_NAMES = [
        "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner",
        "right_eye", "right_eye_outer", "left_ear", "right_ear", "mouth_left",
        "mouth_right", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_pinky", "right_pinky", "left_index",
        "right_index", "left_thumb", "right_thumb", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle", "left_heel",
        "right_heel", "left_foot_index", "right_foot_index"
    ]

    MIRROR_SWAP_MAP = {
        # Wrists & Hand effectors
        "left_wrist": "right_wrist",
        "right_wrist": "left_wrist",
        "left_pinky": "right_pinky",
        "right_pinky": "left_pinky",
        "left_index": "right_index",
        "right_index": "left_index",
        "left_thumb": "right_thumb",
        "right_thumb": "left_thumb",
        # Arms & Shoulders
        "left_elbow": "right_elbow",
        "right_elbow": "left_elbow",
        "left_shoulder": "right_shoulder",
        "right_shoulder": "left_shoulder",
        # Hips & Legs
        "left_hip": "right_hip",
        "right_hip": "left_hip",
        "left_knee": "right_knee",
        "right_knee": "left_knee",
        "left_ankle": "right_ankle",
        "right_ankle": "left_ankle",
        "left_heel": "right_heel",
        "right_heel": "left_heel",
        "left_foot_index": "right_foot_index",
        "right_foot_index": "left_foot_index",
        # Face
        "left_eye": "right_eye",
        "right_eye": "left_eye",
        "left_eye_inner": "right_eye_inner",
        "right_eye_inner": "left_eye_inner",
        "left_eye_outer": "right_eye_outer",
        "right_eye_outer": "left_eye_outer",
        "left_ear": "right_ear",
        "right_ear": "left_ear",
        "mouth_left": "mouth_right",
        "mouth_right": "mouth_left",
    }

    def __init__(self, model_asset_path: str = POSE_MODEL_PATH, delegate: str = "gpu"):
        self.detector = None
        self.device_used = "CPU"
        self.prev_left_wrist = None
        self.prev_right_wrist = None
        self.prev_left_knee = None
        self.prev_right_knee = None
        self.prev_time = time.time()

        if MP_AVAILABLE:
            if not os.path.exists(model_asset_path):
                logger.info(f"Downloading pose model asset from {POSE_MODEL_URL}...")
                try:
                    urllib.request.urlretrieve(POSE_MODEL_URL, model_asset_path)
                except Exception:
                    pass

            if os.path.exists(model_asset_path):
                if delegate.lower() == "gpu":
                    try:
                        base_options = python.BaseOptions(
                            model_asset_path=model_asset_path,
                            delegate=python.BaseOptions.Delegate.GPU
                        )
                        options = vision.PoseLandmarkerOptions(
                            base_options=base_options,
                            output_segmentation_masks=False,
                            min_pose_detection_confidence=0.50,
                            min_tracking_confidence=0.50
                        )
                        self.detector = vision.PoseLandmarker.create_from_options(options)
                        self.device_used = "GPU"
                        logger.info("MediaPipe 33-Keypoint PoseLandmarker initialized on GPU.")
                    except Exception as ex:
                        logger.warning(f"Failed to initialize MediaPipe PoseLandmarker on GPU, falling back to CPU: {ex}")

                if not self.detector:
                    try:
                        base_options = python.BaseOptions(
                            model_asset_path=model_asset_path,
                            delegate=python.BaseOptions.Delegate.CPU
                        )
                        options = vision.PoseLandmarkerOptions(
                            base_options=base_options,
                            output_segmentation_masks=False,
                            min_pose_detection_confidence=0.50,
                            min_tracking_confidence=0.50
                        )
                        self.detector = vision.PoseLandmarker.create_from_options(options)
                        self.device_used = "CPU"
                        logger.info("MediaPipe 33-Keypoint PoseLandmarker initialized on CPU.")
                    except Exception as ex:
                        logger.warning(f"Failed to initialize MediaPipe PoseLandmarker on CPU: {ex}")

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        fiducial_origin: Tuple[float, float] = (0.5, 0.5),
        mirrored: bool = True
    ) -> AstronautPose:
        h, w, _ = frame_bgr.shape
        now = time.time()
        dt = max(now - self.prev_time, 1e-4)

        keypoints: Dict[str, Keypoint3D] = {}
        roll_deg = 0.0

        if self.detector:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            try:
                res = self.detector.detect(mp_image)
                if res.pose_landmarks and len(res.pose_landmarks) > 0:
                    landmarks = res.pose_landmarks[0]
                    for i, lm in enumerate(landmarks):
                        if i < len(self.LANDMARK_NAMES):
                            raw_name = self.LANDMARK_NAMES[i]
                            # Anatomical correction for mirror reflection
                            name = self.MIRROR_SWAP_MAP.get(raw_name, raw_name) if mirrored else raw_name
                            keypoints[name] = Keypoint3D(
                                id=i,
                                name=name,
                                x=float(lm.x),
                                y=float(lm.y),
                                z=float(lm.z),
                                rack_x=float(lm.x - fiducial_origin[0]),
                                rack_y=float(lm.y - fiducial_origin[1]),
                                rack_z=float(lm.z),
                                confidence=float(getattr(lm, "visibility", 1.0) or 1.0)
                            )

                    # Compute body roll tilt from shoulders
                    if "left_shoulder" in keypoints and "right_shoulder" in keypoints:
                        ls = keypoints["left_shoulder"]
                        rs = keypoints["right_shoulder"]
                        dx = (rs.x - ls.x) * w
                        dy = (rs.y - ls.y) * h
                        roll_deg = math.degrees(math.atan2(dy, dx))
            except Exception as e:
                logger.debug(f"Pose detection error: {e}")

        # Compute Left Hand Kinematics & Multi-Point Palm Tracking
        left_vel = 0.0
        left_pos_screen = (0.3, 0.5)
        left_palm_screen = (0.3, 0.5)
        left_pos_rack = (-0.2, 0.0, 0.0)
        left_fingers: Dict[str, Tuple[float, float]] = {}
        left_detected = "left_wrist" in keypoints
        if left_detected:
            lw = keypoints["left_wrist"]
            left_pos_screen = (lw.x, lw.y)
            left_pos_rack = (lw.rack_x, lw.rack_y, lw.rack_z)
            palm_pts = [(lw.x, lw.y)]
            for fname in ["left_index", "left_pinky", "left_thumb"]:
                if fname in keypoints:
                    kp = keypoints[fname]
                    left_fingers[fname] = (kp.x, kp.y)
                    palm_pts.append((kp.x, kp.y))
            left_palm_screen = (
                float(np.mean([p[0] for p in palm_pts])),
                float(np.mean([p[1] for p in palm_pts]))
            )
            curr_l = np.array([lw.x, lw.y])
            if self.prev_left_wrist is not None:
                left_vel = float(np.linalg.norm(curr_l - self.prev_left_wrist) / dt)
            self.prev_left_wrist = curr_l

        # Compute Right Hand Kinematics & Multi-Point Palm Tracking
        right_vel = 0.0
        right_pos_screen = (0.7, 0.5)
        right_palm_screen = (0.7, 0.5)
        right_pos_rack = (0.2, 0.0, 0.0)
        right_fingers: Dict[str, Tuple[float, float]] = {}
        right_detected = "right_wrist" in keypoints
        if right_detected:
            rw = keypoints["right_wrist"]
            right_pos_screen = (rw.x, rw.y)
            right_pos_rack = (rw.rack_x, rw.rack_y, rw.rack_z)
            palm_pts = [(rw.x, rw.y)]
            for fname in ["right_index", "right_pinky", "right_thumb"]:
                if fname in keypoints:
                    kp = keypoints[fname]
                    right_fingers[fname] = (kp.x, kp.y)
                    palm_pts.append((kp.x, kp.y))
            right_palm_screen = (
                float(np.mean([p[0] for p in palm_pts])),
                float(np.mean([p[1] for p in palm_pts]))
            )
            curr_r = np.array([rw.x, rw.y])
            if self.prev_right_wrist is not None:
                right_vel = float(np.linalg.norm(curr_r - self.prev_right_wrist) / dt)
            self.prev_right_wrist = curr_r

        self.prev_time = now

        left_hand = HandTrackingState(
            hand_name="LEFT",
            is_detected=left_detected,
            screen_pos=left_pos_screen,
            palm_pos=left_palm_screen,
            rack_pos=left_pos_rack,
            velocity=left_vel,
            fingers=left_fingers
        )

        right_hand = HandTrackingState(
            hand_name="RIGHT",
            is_detected=right_detected,
            screen_pos=right_pos_screen,
            palm_pos=right_palm_screen,
            rack_pos=right_pos_rack,
            velocity=right_vel,
            fingers=right_fingers
        )

        # Compute Lower Body Kinematics & Leg Activity (Leg 1: Left, Leg 2: Right)
        # Left Leg (Hip 23, Knee 25, Ankle 27)
        left_has_direct = "left_hip" in keypoints and "left_knee" in keypoints
        left_knee_angle = 176.0
        left_leg_vel = 0.0
        left_leg_act = "ANCHORED IN RESTRAINT"
        left_hip_p = (-0.10, 0.20, 0.10)
        left_knee_p = (-0.12, 0.50, 0.18)
        left_ankle_p = (-0.13, 0.80, 0.24)
        left_leg_detected = False

        if left_has_direct:
            left_leg_detected = True
            l_hip = keypoints["left_hip"]
            l_knee = keypoints["left_knee"]
            left_hip_p = (l_hip.rack_x, l_hip.rack_y, l_hip.rack_z)
            left_knee_p = (l_knee.rack_x, l_knee.rack_y, l_knee.rack_z)

            if "left_ankle" in keypoints:
                l_ank = keypoints["left_ankle"]
                left_ankle_p = (l_ank.rack_x, l_ank.rack_y, l_ank.rack_z)
                v_hk = np.array([l_hip.x - l_knee.x, l_hip.y - l_knee.y, l_hip.z - l_knee.z])
                v_ka = np.array([l_ank.x - l_knee.x, l_ank.y - l_knee.y, l_ank.z - l_knee.z])
                denom = np.linalg.norm(v_hk) * np.linalg.norm(v_ka)
                if denom > 1e-5:
                    cos_a = np.dot(v_hk, v_ka) / denom
                    cos_a = np.clip(cos_a, -1.0, 1.0)
                    left_knee_angle = float(math.degrees(math.acos(cos_a)))

            curr_lk = np.array([l_knee.x, l_knee.y])
            if self.prev_left_knee is not None:
                left_leg_vel = float(np.linalg.norm(curr_lk - self.prev_left_knee) / dt)
            self.prev_left_knee = curr_lk

            if left_knee_angle < 138.0:
                left_leg_act = "KNEE FLEXION"
            elif left_leg_vel > 0.35:
                left_leg_act = "TREADING / ADJUSTING"
            elif l_knee.y < l_hip.y + 0.12:
                left_leg_act = "MICROGRAVITY FLOATING"
            else:
                left_leg_act = "ANCHORED IN RESTRAINT"
        elif "left_hip" in keypoints:
            # Person upper body is present; knees are below camera frame cutoff -> Anchor in restraint
            left_leg_detected = True
            l_hip = keypoints["left_hip"]
            left_hip_p = (l_hip.rack_x, l_hip.rack_y, l_hip.rack_z)
            left_knee_p = (l_hip.rack_x - 0.02, l_hip.rack_y + 0.32, l_hip.rack_z + 0.08)
            left_ankle_p = (l_hip.rack_x - 0.03, l_hip.rack_y + 0.64, l_hip.rack_z + 0.14)
            left_knee_angle = 176.0
            left_leg_act = "ANCHORED IN RESTRAINT"
            if "left_knee" not in keypoints:
                keypoints["left_knee"] = Keypoint3D(25, "left_knee", l_hip.x - 0.02, min(0.99, l_hip.y + 0.32), l_hip.z + 0.08, left_knee_p[0], left_knee_p[1], left_knee_p[2], 0.85)
            if "left_ankle" not in keypoints:
                keypoints["left_ankle"] = Keypoint3D(27, "left_ankle", l_hip.x - 0.03, min(0.99, l_hip.y + 0.64), l_hip.z + 0.14, left_ankle_p[0], left_ankle_p[1], left_ankle_p[2], 0.85)
            if "left_heel" not in keypoints:
                keypoints["left_heel"] = Keypoint3D(29, "left_heel", l_hip.x - 0.03, min(0.99, l_hip.y + 0.66), l_hip.z + 0.12, left_ankle_p[0], left_ankle_p[1] + 0.02, left_ankle_p[2] - 0.02, 0.85)
            if "left_foot_index" not in keypoints:
                keypoints["left_foot_index"] = Keypoint3D(31, "left_foot_index", l_hip.x - 0.03, min(0.99, l_hip.y + 0.68), l_hip.z + 0.18, left_ankle_p[0], left_ankle_p[1] + 0.04, left_ankle_p[2] + 0.06, 0.85)
        elif "left_shoulder" in keypoints:
            # Person seated close to camera: anchor from shoulders down
            left_leg_detected = True
            ls = keypoints["left_shoulder"]
            hip_y = min(0.99, ls.y + 0.36)
            hip_rx, hip_ry, hip_rz = ls.rack_x, ls.rack_y + 0.36, ls.rack_z
            left_hip_p = (hip_rx, hip_ry, hip_rz)
            left_knee_p = (hip_rx - 0.02, hip_ry + 0.30, hip_rz + 0.08)
            left_ankle_p = (hip_rx - 0.03, hip_ry + 0.60, hip_rz + 0.14)
            left_knee_angle = 176.0
            left_leg_act = "ANCHORED IN RESTRAINT"
            if "left_hip" not in keypoints:
                keypoints["left_hip"] = Keypoint3D(23, "left_hip", ls.x, hip_y, ls.z, hip_rx, hip_ry, hip_rz, 0.85)
            if "left_knee" not in keypoints:
                keypoints["left_knee"] = Keypoint3D(25, "left_knee", ls.x - 0.02, min(0.99, hip_y + 0.30), ls.z + 0.08, left_knee_p[0], left_knee_p[1], left_knee_p[2], 0.85)
            if "left_ankle" not in keypoints:
                keypoints["left_ankle"] = Keypoint3D(27, "left_ankle", ls.x - 0.03, min(0.99, hip_y + 0.60), ls.z + 0.14, left_ankle_p[0], left_ankle_p[1], left_ankle_p[2], 0.85)
            if "left_heel" not in keypoints:
                keypoints["left_heel"] = Keypoint3D(29, "left_heel", ls.x - 0.03, min(0.99, hip_y + 0.62), ls.z + 0.12, left_ankle_p[0], left_ankle_p[1] + 0.02, left_ankle_p[2] - 0.02, 0.85)
            if "left_foot_index" not in keypoints:
                keypoints["left_foot_index"] = Keypoint3D(31, "left_foot_index", ls.x - 0.03, min(0.99, hip_y + 0.64), ls.z + 0.18, left_ankle_p[0], left_ankle_p[1] + 0.04, left_ankle_p[2] + 0.06, 0.85)

        # Right Leg (Hip 24, Knee 26, Ankle 28)
        right_has_direct = "right_hip" in keypoints and "right_knee" in keypoints
        right_knee_angle = 174.0
        right_leg_vel = 0.0
        right_leg_act = "ANCHORED IN RESTRAINT"
        right_hip_p = (0.10, 0.20, 0.10)
        right_knee_p = (0.12, 0.50, 0.18)
        right_ankle_p = (0.13, 0.80, 0.24)
        right_leg_detected = False

        if right_has_direct:
            right_leg_detected = True
            r_hip = keypoints["right_hip"]
            r_knee = keypoints["right_knee"]
            right_hip_p = (r_hip.rack_x, r_hip.rack_y, r_hip.rack_z)
            right_knee_p = (r_knee.rack_x, r_knee.rack_y, r_knee.rack_z)

            if "right_ankle" in keypoints:
                r_ank = keypoints["right_ankle"]
                right_ankle_p = (r_ank.rack_x, r_ank.rack_y, r_ank.rack_z)
                v_hk = np.array([r_hip.x - r_knee.x, r_hip.y - r_knee.y, r_hip.z - r_knee.z])
                v_ka = np.array([r_ank.x - r_knee.x, r_ank.y - r_knee.y, r_ank.z - r_knee.z])
                denom = np.linalg.norm(v_hk) * np.linalg.norm(v_ka)
                if denom > 1e-5:
                    cos_a = np.dot(v_hk, v_ka) / denom
                    cos_a = np.clip(cos_a, -1.0, 1.0)
                    right_knee_angle = float(math.degrees(math.acos(cos_a)))

            curr_rk = np.array([r_knee.x, r_knee.y])
            if self.prev_right_knee is not None:
                right_leg_vel = float(np.linalg.norm(curr_rk - self.prev_right_knee) / dt)
            self.prev_right_knee = curr_rk

            if right_knee_angle < 138.0:
                right_leg_act = "KNEE FLEXION"
            elif right_leg_vel > 0.35:
                right_leg_act = "TREADING / ADJUSTING"
            elif r_knee.y < r_hip.y + 0.12:
                right_leg_act = "MICROGRAVITY FLOATING"
            else:
                right_leg_act = "ANCHORED IN RESTRAINT"
        elif "right_hip" in keypoints:
            right_leg_detected = True
            r_hip = keypoints["right_hip"]
            right_hip_p = (r_hip.rack_x, r_hip.rack_y, r_hip.rack_z)
            right_knee_p = (r_hip.rack_x + 0.02, r_hip.rack_y + 0.32, r_hip.rack_z + 0.08)
            right_ankle_p = (r_hip.rack_x + 0.03, r_hip.rack_y + 0.64, r_hip.rack_z + 0.14)
            right_knee_angle = 174.0
            right_leg_act = "ANCHORED IN RESTRAINT"
            if "right_knee" not in keypoints:
                keypoints["right_knee"] = Keypoint3D(26, "right_knee", r_hip.x + 0.02, min(0.99, r_hip.y + 0.32), r_hip.z + 0.08, right_knee_p[0], right_knee_p[1], right_knee_p[2], 0.85)
            if "right_ankle" not in keypoints:
                keypoints["right_ankle"] = Keypoint3D(28, "right_ankle", r_hip.x + 0.03, min(0.99, r_hip.y + 0.64), r_hip.z + 0.14, right_ankle_p[0], right_ankle_p[1], right_ankle_p[2], 0.85)
            if "right_heel" not in keypoints:
                keypoints["right_heel"] = Keypoint3D(30, "right_heel", r_hip.x + 0.03, min(0.99, r_hip.y + 0.66), r_hip.z + 0.12, right_ankle_p[0], right_ankle_p[1] + 0.02, right_ankle_p[2] - 0.02, 0.85)
            if "right_foot_index" not in keypoints:
                keypoints["right_foot_index"] = Keypoint3D(32, "right_foot_index", r_hip.x + 0.03, min(0.99, r_hip.y + 0.68), r_hip.z + 0.18, right_ankle_p[0], right_ankle_p[1] + 0.04, right_ankle_p[2] + 0.06, 0.85)
        elif "right_shoulder" in keypoints:
            right_leg_detected = True
            rs = keypoints["right_shoulder"]
            hip_y = min(0.99, rs.y + 0.36)
            hip_rx, hip_ry, hip_rz = rs.rack_x, rs.rack_y + 0.36, rs.rack_z
            right_hip_p = (hip_rx, hip_ry, hip_rz)
            right_knee_p = (hip_rx + 0.02, hip_ry + 0.30, hip_rz + 0.08)
            right_ankle_p = (hip_rx + 0.03, hip_ry + 0.60, hip_rz + 0.14)
            right_knee_angle = 174.0
            right_leg_act = "ANCHORED IN RESTRAINT"
            if "right_hip" not in keypoints:
                keypoints["right_hip"] = Keypoint3D(24, "right_hip", rs.x, hip_y, rs.z, hip_rx, hip_ry, hip_rz, 0.85)
            if "right_knee" not in keypoints:
                keypoints["right_knee"] = Keypoint3D(26, "right_knee", rs.x + 0.02, min(0.99, hip_y + 0.30), rs.z + 0.08, right_knee_p[0], right_knee_p[1], right_knee_p[2], 0.85)
            if "right_ankle" not in keypoints:
                keypoints["right_ankle"] = Keypoint3D(28, "right_ankle", rs.x + 0.03, min(0.99, hip_y + 0.60), rs.z + 0.14, right_ankle_p[0], right_ankle_p[1], right_ankle_p[2], 0.85)
            if "right_heel" not in keypoints:
                keypoints["right_heel"] = Keypoint3D(30, "right_heel", rs.x + 0.03, min(0.99, hip_y + 0.62), rs.z + 0.12, right_ankle_p[0], right_ankle_p[1] + 0.02, right_ankle_p[2] - 0.02, 0.85)
            if "right_foot_index" not in keypoints:
                keypoints["right_foot_index"] = Keypoint3D(32, "right_foot_index", rs.x + 0.03, min(0.99, hip_y + 0.64), rs.z + 0.18, right_ankle_p[0], right_ankle_p[1] + 0.04, right_ankle_p[2] + 0.06, 0.85)

        # Overall lower body status
        if left_leg_act == "KNEE FLEXION" or right_leg_act == "KNEE FLEXION":
            lower_body_status = "KNEE FLEXION (BENT)"
        elif left_leg_act == "MICROGRAVITY FLOATING" or right_leg_act == "MICROGRAVITY FLOATING":
            lower_body_status = "MICROGRAVITY FLOATING / DRIFTING"
        elif left_leg_act == "TREADING / ADJUSTING" or right_leg_act == "TREADING / ADJUSTING":
            lower_body_status = "TREADING / REPOSITIONING"
        elif left_leg_detected or right_leg_detected:
            lower_body_status = "ANCHORED IN FOOT RESTRAINT"
        else:
            lower_body_status = "STANDBY (NO OPERATOR)"

        left_leg = LegTrackingState(
            leg_name="LEFT",
            is_detected=left_leg_detected,
            knee_angle_deg=left_knee_angle,
            velocity=left_leg_vel,
            activity=left_leg_act,
            hip_pos=left_hip_p,
            knee_pos=left_knee_p,
            ankle_pos=left_ankle_p
        )
        right_leg = LegTrackingState(
            leg_name="RIGHT",
            is_detected=right_leg_detected,
            knee_angle_deg=right_knee_angle,
            velocity=right_leg_vel,
            activity=right_leg_act,
            hip_pos=right_hip_p,
            knee_pos=right_knee_p,
            ankle_pos=right_ankle_p
        )

        # Active hand selection based on velocity and detection
        active = "RIGHT" if right_vel >= left_vel and right_detected else "LEFT" if left_detected else "RIGHT"
        dominant_pos = right_pos_rack if active == "RIGHT" else left_pos_rack
        max_vel = max(left_vel, right_vel)

        return AstronautPose(
            timestamp=now,
            keypoints=keypoints,
            left_hand=left_hand,
            right_hand=right_hand,
            left_leg=left_leg,
            right_leg=right_leg,
            lower_body_activity=lower_body_status,
            active_hand=active,
            wrist_velocity=max_vel,
            body_roll_degrees=roll_deg,
            dominant_wrist_pos=dominant_pos
        )

# ==============================================================================
# DUAL-HAND SPATIAL INTERACTION & IN-HAND OBJECT CLASSIFIER
# ==============================================================================

class DualHandInteractionClassifier:
    """Evaluates both Left and Right hands independently with stateful tracking,
    held-payload custody, ambidextrous interaction, and intentional motion verification."""

    PICKABLE_PAYLOADS = {
        "Component_A", "Component_B", "Syringe_Injector", "SpaceWire_Harness", "Breather_Mask"
    }

    def __init__(self, proximity_threshold: float = 0.14):
        self.proximity_threshold = proximity_threshold
        self._debug_frame_counter = 0
        self.left_held_object: Optional[str] = None
        self.left_held_specs: Optional[Dict[str, Any]] = None
        self.right_held_object: Optional[str] = None
        self.right_held_specs: Optional[Dict[str, Any]] = None
        self.slot_occupied: bool = False
        self.rotate_baseline_roll: Optional[float] = None
        self.rotate_frames_count: int = 0
        self.left_dwell_target: Optional[str] = None
        self.left_dwell_frames: int = 0
        self.left_miss_frames: int = 0
        self.right_dwell_target: Optional[str] = None
        self.right_dwell_frames: int = 0
        self.right_miss_frames: int = 0

    def reset(self):
        """Resets all held objects, slots, and dwell tracking (e.g. on protocol switch)."""
        self.left_held_object = None
        self.left_held_specs = None
        self.right_held_object = None
        self.right_held_specs = None
        self.slot_occupied = False
        self.rotate_baseline_roll = None
        self.rotate_frames_count = 0
        self.left_dwell_target = None
        self.left_dwell_frames = 0
        self.left_miss_frames = 0
        self.right_dwell_target = None
        self.right_dwell_frames = 0
        self.right_miss_frames = 0

    def evaluate_hands_and_objects(
        self,
        pose: AstronautPose,
        payload_targets: List[DetectedObject],
        ai_detected_objects: List[DetectedObject],
        current_step: Optional[Dict[str, Any]] = None,
        current_step_idx: int = 0,
        all_steps: Optional[List[Dict[str, Any]]] = None
    ) -> Optional[ActionCandidate]:
        """Evaluates both hands independently against payload targets and AI-detected objects."""
        exp_act = current_step.get("expected_action") if current_step else None
        exp_tgt = current_step.get("expected_target") if current_step else None
        all_candidate_objects = payload_targets + ai_detected_objects

        def _calc_hand_target_dist(hand_state: HandTrackingState, obj: DetectedObject) -> float:
            palm_d = float(np.linalg.norm(np.array(hand_state.palm_pos) - np.array(obj.centroid)))
            wrist_d = float(np.linalg.norm(np.array(hand_state.screen_pos) - np.array(obj.centroid)))
            min_d = min(palm_d, wrist_d)
            for f_xy in hand_state.fingers.values():
                fd = float(np.linalg.norm(np.array(f_xy) - np.array(obj.centroid)))
                if fd < min_d:
                    min_d = fd
            return min_d

        # 1. Evaluate closest target for Left Hand
        left_closest_obj = None
        left_min_dist = float("inf")
        if pose.left_hand and pose.left_hand.is_detected:
            for obj in all_candidate_objects:
                d = _calc_hand_target_dist(pose.left_hand, obj)
                if d < left_min_dist:
                    left_min_dist = d
                    left_closest_obj = obj
            pose.left_hand.contact_distance = left_min_dist
        else:
            left_min_dist = float("inf")

        # 2. Evaluate closest target for Right Hand
        right_closest_obj = None
        right_min_dist = float("inf")
        if pose.right_hand and pose.right_hand.is_detected:
            for obj in all_candidate_objects:
                d = _calc_hand_target_dist(pose.right_hand, obj)
                if d < right_min_dist:
                    right_min_dist = d
                    right_closest_obj = obj
            pose.right_hand.contact_distance = right_min_dist
        else:
            right_min_dist = float("inf")

        # 3. Update Dwell Tracking with 2-frame jitter grace tolerance
        if left_closest_obj and left_min_dist < self.proximity_threshold:
            self.left_miss_frames = 0
            if self.left_dwell_target == left_closest_obj.class_name:
                self.left_dwell_frames += 1
            else:
                self.left_dwell_target = left_closest_obj.class_name
                self.left_dwell_frames = 1
        elif self.left_dwell_target and self.left_miss_frames < 2:
            self.left_miss_frames += 1
        else:
            self.left_dwell_target = None
            self.left_dwell_frames = 0
            self.left_miss_frames = 0

        if right_closest_obj and right_min_dist < self.proximity_threshold:
            self.right_miss_frames = 0
            if self.right_dwell_target == right_closest_obj.class_name:
                self.right_dwell_frames += 1
            else:
                self.right_dwell_target = right_closest_obj.class_name
                self.right_dwell_frames = 1
        elif self.right_dwell_target and self.right_miss_frames < 2:
            self.right_miss_frames += 1
        else:
            self.right_dwell_target = None
            self.right_dwell_frames = 0
            self.right_miss_frames = 0

        # 4. Check Hand-to-Hand Transfer (Handoff)
        if pose.left_hand and pose.right_hand and pose.left_hand.is_detected and pose.right_hand.is_detected:
            palm_dist = float(np.linalg.norm(np.array(pose.left_hand.palm_pos) - np.array(pose.right_hand.palm_pos)))
            if palm_dist < 0.12:
                if self.left_held_object and not self.right_held_object:
                    self.right_held_object = self.left_held_object
                    self.right_held_specs = self.left_held_specs
                    self.left_held_object = None
                    self.left_held_specs = None
                    logger.info(f"🔄 Payload {self.right_held_object} transferred from LEFT to RIGHT hand.")
                elif self.right_held_object and not self.left_held_object:
                    self.left_held_object = self.right_held_object
                    self.left_held_specs = self.right_held_specs
                    self.right_held_object = None
                    self.right_held_specs = None
                    logger.info(f"🔄 Payload {self.left_held_object} transferred from RIGHT to LEFT hand.")

        # Sync held status to pose
        pose.left_hand.held_object = self.left_held_object
        pose.left_hand.held_object_data = self.left_held_specs
        pose.right_hand.held_object = self.right_held_object
        pose.right_hand.held_object_data = self.right_held_specs

        holding_hand = "LEFT" if self.left_held_object else ("RIGHT" if self.right_held_object else None)
        held_obj_name = self.left_held_object or self.right_held_object
        held_specs = self.left_held_specs or self.right_held_specs

        if holding_hand:
            pose.active_hand = holding_hand
            pose.held_payload = {
                "object_name": held_obj_name,
                "hand": holding_hand,
                "contact_distance_m": round(left_min_dist if holding_hand == "LEFT" else right_min_dist, 3),
                "grasp_confidence": 98.0,
                "source": "ASTRONAUT_GRASP",
                "specs": held_specs or {}
            }
        else:
            pose.held_payload = None

        # If scenario is completed or no active step, allow continuous in-hand picking & identification
        if not current_step:
            if self.left_dwell_target and self.left_dwell_frames >= 2:
                if self.left_held_object != self.left_dwell_target:
                    self.left_held_object = self.left_dwell_target
                    self.left_held_specs = left_closest_obj.specs if left_closest_obj else {}
            if self.right_dwell_target and self.right_dwell_frames >= 2:
                if self.right_held_object != self.right_dwell_target:
                    self.right_held_object = self.right_dwell_target
                    self.right_held_specs = right_closest_obj.specs if right_closest_obj else {}

            if self.left_held_object and left_min_dist > 0.35:
                self.left_held_object = None
                self.left_held_specs = None
            if self.right_held_object and right_min_dist > 0.35:
                self.right_held_object = None
                self.right_held_specs = None

            pose.left_hand.held_object = self.left_held_object
            pose.left_hand.held_object_data = self.left_held_specs
            pose.right_hand.held_object = self.right_held_object
            pose.right_hand.held_object_data = self.right_held_specs

            holding_hand = "LEFT" if self.left_held_object else ("RIGHT" if self.right_held_object else None)
            held_obj_name = self.left_held_object or self.right_held_object
            held_specs = self.left_held_specs or self.right_held_specs
            if holding_hand:
                pose.active_hand = holding_hand
                pose.held_payload = {
                    "object_name": held_obj_name,
                    "hand": holding_hand,
                    "contact_distance_m": round(left_min_dist if holding_hand == "LEFT" else right_min_dist, 3),
                    "grasp_confidence": 98.0,
                    "source": "ASTRONAUT_GRASP",
                    "specs": held_specs or {}
                }
                return ActionCandidate("HOLDING", held_obj_name, 0.95, time.time(), holding_hand, pose.held_payload)
            else:
                pose.held_payload = None
                return None

        # 5. STEP-SPECIFIC INTENTIONAL CLASSIFICATION & ERROR CHECKING
        DWELL_REQ = 6

        active_dwells = []
        for h_name, dwell_tgt, dwell_f, c_obj in [
            ("LEFT", self.left_dwell_target, self.left_dwell_frames, left_closest_obj),
            ("RIGHT", self.right_dwell_target, self.right_dwell_frames, right_closest_obj)
        ]:
            if dwell_tgt and dwell_f >= DWELL_REQ:
                active_dwells.append((h_name, dwell_tgt, dwell_f, c_obj))

        # Special Case: Bio Roll rotation in Step 1
        if current_step_idx == 1 and exp_act == "ROTATE":
            if holding_hand:
                if self.rotate_baseline_roll is None:
                    self.rotate_baseline_roll = pose.body_roll_degrees
                delta_roll = abs(pose.body_roll_degrees - self.rotate_baseline_roll)
                hand_state = pose.left_hand if holding_hand == "LEFT" else pose.right_hand
                vel = hand_state.velocity if hand_state else 0.0
                if delta_roll > 10.0 or vel > 0.08:
                    self.rotate_frames_count += 1
                if self.rotate_frames_count >= 3:
                    return ActionCandidate("ROTATE", held_obj_name or exp_tgt, 0.95, time.time(), holding_hand, pose.held_payload)
            # Check skipped step during rotate: hand directly goes to Rack_Slot_1
            for h_name, dwell_tgt, dwell_f, c_obj in active_dwells:
                if "Slot" in dwell_tgt or "Rack" in dwell_tgt:
                    return ActionCandidate("INSERT", "Rack_Slot_1", 0.95, time.time(), h_name, pose.held_payload)
            return None

        # General step evaluation for all other steps
        if active_dwells:
            h_name, dwell_tgt, dwell_f, c_obj = active_dwells[0]
            specs = c_obj.specs if c_obj else {}

            # Check if dwelling target matches the expected target
            matches_expected = False
            if exp_tgt:
                if dwell_tgt == exp_tgt:
                    matches_expected = True
                elif ("Syringe" in dwell_tgt and "Syringe" in exp_tgt) or \
                     ("Sample" in dwell_tgt and "Sample" in exp_tgt) or \
                     ("Manifold" in dwell_tgt and "Manifold" in exp_tgt) or \
                     ("Pinch" in dwell_tgt and "Pinch" in exp_tgt) or \
                     ("Slot" in dwell_tgt and "Slot" in exp_tgt) or \
                     ("Latch" in dwell_tgt and "Latch" in exp_tgt) or \
                     ("Secondary" in dwell_tgt and "Secondary" in exp_tgt) or \
                     ("Torque" in dwell_tgt and "Torque" in exp_tgt) or \
                     ("Breaker" in dwell_tgt and "Breaker" in exp_tgt) or \
                     ("Hatch" in dwell_tgt and "Hatch" in exp_tgt) or \
                     ("Valve" in dwell_tgt and "Valve" in exp_tgt) or \
                     ("Avionic" in dwell_tgt and "Avionic" in exp_tgt) or \
                     ("Mask" in dwell_tgt and "Mask" in exp_tgt) or \
                     ("Harness" in dwell_tgt and "Harness" in exp_tgt):
                    matches_expected = True

            if matches_expected:
                # COMPLIANT STEP ACTION
                pose.active_hand = h_name
                if current_step_idx == 0:
                    self.left_held_object = exp_tgt if h_name == "LEFT" else None
                    self.left_held_specs = specs if h_name == "LEFT" else None
                    self.right_held_object = exp_tgt if h_name == "RIGHT" else None
                    self.right_held_specs = specs if h_name == "RIGHT" else None
                return ActionCandidate(exp_act, exp_tgt, 0.96, time.time(), h_name, pose.held_payload)

            # Check if this target belongs to a FUTURE step (Skipped step error)
            future_steps = (all_steps or [])[current_step_idx + 1:]
            for f_step in future_steps:
                f_tgt = f_step.get("expected_target", "")
                f_act = f_step.get("expected_action", "")
                if f_tgt and (dwell_tgt == f_tgt or f_tgt in dwell_tgt or dwell_tgt in f_tgt):
                    logger.warning(f"Skipped step detected: hand touched future target {f_tgt}")
                    return ActionCandidate(f_act, f_tgt, 0.95, time.time(), h_name, pose.held_payload)

            # Otherwise, astronaut interacted with the WRONG payload (Wrong object error)
            logger.warning(f"Wrong object detected: touched {dwell_tgt}, expected {exp_tgt}")
            act_name = exp_act or "PICK"
            return ActionCandidate(act_name, dwell_tgt, 0.95, time.time(), h_name, pose.held_payload)

        return None

# ==============================================================================
# DETERMINISTIC PROTOCOL COMPLIANCE STATE MACHINE (AMBIDEXTROUS FSM)
# ==============================================================================

class ProtocolComplianceEngine:
    """State machine evaluating astronaut actions with EITHER hand."""

    def __init__(self, protocol_path: str, mc_link: MissionControlLink):
        with open(protocol_path, "r") as f:
            self.protocol_data = json.load(f)

        self.protocol_id = self.protocol_data["protocol_id"]
        self.steps = self.protocol_data["steps"]
        self.mc_link = mc_link

        self.current_step_idx = 0
        self.step_start_time = time.time()
        self.min_step_cooldown = 1.5  # Refractory period: prevents cascading across steps
        self.last_step_time = time.time() - self.min_step_cooldown  # First step allowed immediately
        self.is_completed = False
        if self.mc_link and hasattr(self.mc_link, "reset_procedure_state"):
            self.mc_link.reset_procedure_state()

        self.action_history: List[str] = []
        self.debounce_size = 6 # 6 frames of consistent action (~0.2s at 30fps)
        self.active_deviation_banner: Optional[Dict[str, Any]] = None

        # Alert cooldown to eliminate voice repetition
        self.last_deviation_time = 0.0
        self.last_deviation_msg = ""
        self.deviation_cooldown = 4.0

    def reset(self):
        self.current_step_idx = 0
        self.step_start_time = time.time()
        self.min_step_cooldown = 1.5
        self.last_step_time = time.time() - self.min_step_cooldown  # First step allowed immediately
        self.is_completed = False
        self.action_history.clear()
        self.active_deviation_banner = None
        self.last_deviation_time = 0.0
        self.last_deviation_msg = ""
        if self.mc_link and hasattr(self.mc_link, "reset_procedure_state"):
            self.mc_link.reset_procedure_state()

    def current_step(self) -> Optional[Dict[str, Any]]:
        if self.current_step_idx < len(self.steps):
            return self.steps[self.current_step_idx]
        return None

    def evaluate(self, candidate: Optional[ActionCandidate], pose: AstronautPose):
        if self.is_completed or not candidate:
            return

        now = time.time()
        # Enforce refractory dwell time so hands cannot cascade through all steps automatically
        if (now - self.last_step_time) < self.min_step_cooldown:
            return

        action_signature = f"{candidate.action_type}:{candidate.target_object}"
        self.action_history.append(action_signature)
        if len(self.action_history) > self.debounce_size:
            self.action_history.pop(0)

        # Confirm debounce consensus
        if self.action_history.count(action_signature) < (self.debounce_size - 1):
            return

        expected = self.current_step()
        if not expected:
            return

        exp_act = expected["expected_action"]
        exp_tgt = expected["expected_target"]
        step_id = expected["step_id"]

        # CASE 1: Compliant Step Transition (Ambidextrous)
        if candidate.action_type == exp_act and candidate.target_object == exp_tgt:
            self.current_step_idx += 1
            self.last_step_time = time.time()
            self.step_start_time = time.time()
            self.action_history.clear()
            self.active_deviation_banner = None

            if self.current_step_idx >= len(self.steps):
                self.is_completed = True
                logger.info(f"✓ [STEP {step_id} COMPLIANT] Hand: {candidate.hand} | {exp_act} -> {exp_tgt}")
                logger.info("🏆 PROTOCOL COMPLETE: 100% compliance verified across all steps!")
                self.mc_link.send_experiment_complete(self.protocol_id, step_id, pose)
            else:
                logger.info(f"✓ [STEP {step_id} COMPLIANT] Hand: {candidate.hand} | {exp_act} -> {exp_tgt}")
                self.mc_link.send_step_verified(self.protocol_id, step_id, f"{exp_act} {exp_tgt}", candidate.hand, pose)
            return

        # CASE 2: Wrong Object Interaction
        if candidate.action_type == exp_act and candidate.target_object != exp_tgt:
            msg = f"Protocol deviation: Expected {exp_tgt}, but {candidate.target_object} was picked with {candidate.hand} hand."
            logger.warning(f"⚠️ {msg}")
            now = time.time()
            self.active_deviation_banner = {
                "type": "WRONG_OBJECT",
                "title": "PROTOCOL DEVIATION: WRONG OBJECT",
                "message": f"Expected: {exp_tgt} | Detected: {candidate.target_object} ({candidate.hand} Hand)",
                "expires": now + 4.0
            }
            if (now - self.last_deviation_time) >= self.deviation_cooldown or msg != self.last_deviation_msg:
                self.mc_link.send_deviation_alert(self.protocol_id, step_id, "WRONG_OBJECT", msg, action_signature, pose)
                self.last_deviation_time = now
                self.last_deviation_msg = msg
            self.action_history.clear()
            return

        # CASE 3: Skipped Step or Out-of-Order Execution
        future_steps = self.steps[self.current_step_idx + 1:]
        for future in future_steps:
            if candidate.action_type == future["expected_action"] and candidate.target_object == future["expected_target"]:
                msg = f"Warning: Required action '{exp_act} {exp_tgt}' was skipped. Detected subsequent step '{future['expected_action']} {future['expected_target']}' out of order."
                logger.warning(f"⚠️ {msg}")
                now = time.time()
                self.active_deviation_banner = {
                    "type": "SKIPPED_STEP",
                    "title": "PROTOCOL DEVIATION: STEP SKIPPED",
                    "message": f"Skipped required: {exp_act} {exp_tgt} | Touched: {candidate.target_object}",
                    "expires": now + 4.0
                }
                if (now - self.last_deviation_time) >= self.deviation_cooldown or msg != self.last_deviation_msg:
                    self.mc_link.send_deviation_alert(self.protocol_id, step_id, "SKIPPED_STEP", msg, action_signature, pose)
                    self.last_deviation_time = now
                    self.last_deviation_msg = msg
                self.action_history.clear()
                return

# ==============================================================================
# MAIN ASTRONAUT MONITORING SYSTEM
# ==============================================================================

class AstronautMonitoringSystem:
    def __init__(self, protocol_path: str = "configs/protocol.json", ground_url: str = "http://127.0.0.1:8000", delegate: str = "gpu"):
        self.protocol_path = protocol_path
        self.ground_url = ground_url
        self.delegate = delegate
        self.mc_link = MissionControlLink(ground_url=ground_url)
        self.pose_engine = MicrogravityPoseEngine(delegate=delegate)
        self.object_detector = AIPayloadObjectDetector(delegate=delegate)
        self.classifier = DualHandInteractionClassifier(proximity_threshold=0.14)
        self.protocol_mtime = 0.0
        self.payload_targets: List[DetectedObject] = []
        self.load_protocol_config()
        self.last_heartbeat_time = 0.0

    def load_protocol_config(self):
        """Loads or reloads the active experiment protocol configuration and payload targets."""
        try:
            self.compliance_engine = ProtocolComplianceEngine(self.protocol_path, self.mc_link)
            if hasattr(self, "classifier"):
                self.classifier.reset()
            self.payload_targets = []
            with open(self.protocol_path, "r") as f:
                cfg = json.load(f)
                for t in cfg.get("payload_targets", []):
                    self.payload_targets.append(DetectedObject(
                        object_id=t["object_id"],
                        class_name=t["class_name"],
                        centroid=tuple(t["centroid"]),
                        bbox=tuple(t["bbox"]),
                        confidence=0.96,
                        specs=t.get("specs", {}),
                        source="PROTOCOL_PAYLOAD"
                    ))
            if os.path.exists(self.protocol_path):
                self.protocol_mtime = os.path.getmtime(self.protocol_path)
            logger.info(f"Loaded protocol: {self.compliance_engine.protocol_id} ({self.compliance_engine.protocol_data.get('experiment_name')}) with {len(self.payload_targets)} payload targets.")
        except Exception as e:
            logger.error(f"Failed to load protocol from {self.protocol_path}: {e}")

    def check_protocol_reload(self):
        """Checks if protocol.json was switched by Ground Mission Control and reloads live."""
        if os.path.exists(self.protocol_path):
            try:
                mtime = os.path.getmtime(self.protocol_path)
                if mtime > self.protocol_mtime + 0.1:
                    logger.info("⚡ Protocol change detected on disk. Hot-reloading active experiment...")
                    self.load_protocol_config()
                    exp_name = self.compliance_engine.protocol_data.get("experiment_name", "Active Protocol")
                    self.mc_link.speak(f"Mission protocol updated: {exp_name}")
            except Exception as e:
                logger.debug(f"Protocol reload check error: {e}")

    def run(self, camera_source: Any = 0, headless: bool = False, record: bool = True, no_flip: bool = False, enlarge: bool = True, fullscreen: bool = False, width: int = 1280, height: int = 720):
        logger.info(f"Connecting to video feed: {camera_source} (Headless: {headless}, Record: {record}, MirrorFlip: {not no_flip}, Enlarge: {enlarge})")
        cap = None
        for attempt in range(4):
            cap = cv2.VideoCapture(camera_source)
            if cap.isOpened():
                break
            logger.info(f"Retrying camera open ({attempt + 1}/4)...")
            time.sleep(0.4)

        if not cap or not cap.isOpened():
            logger.warning(f"Camera {camera_source} could not be opened. Falling back to synthetic feed mode.")
            self._run_synthetic_loop()
            return

        if enlarge:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

        window_name = "ASTRO-COPILOT | Zero-G Edge Computer Optical Telemetry"
        if not headless:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            if fullscreen:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            else:
                cv2.resizeWindow(window_name, w, h)

        video_writer = None
        session_filename = None
        if record:
            os.makedirs("telemetry/sessions", exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            session_filename = f"telemetry/sessions/astronaut_session_{ts}.avi"
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            video_writer = cv2.VideoWriter(session_filename, fourcc, fps, (w, h))
            logger.info(f"📹 Recording astronaut experiment session to: {session_filename}")

        logger.info("Astronaut Copilot Dual-Hand Active. [Q] Quit | [E] Toggle Enlarge (1280x720) | [F] Fullscreen")
        frame_counter = 0

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Check for live ground protocol switches
                self.check_protocol_reload()

                # Horizontal flip for natural mirror interaction (moving right moves right on screen)
                if not no_flip:
                    frame = cv2.flip(frame, 1)

                frame_counter += 1

                # 1. 3D Dual-Hand Pose Estimation (33 Keypoints + Multi-Point Palms)
                pose = self.pose_engine.process_frame(frame, mirrored=(not no_flip))

                # 2. AI Object Detection in Scene
                ai_objects = self.object_detector.detect_objects(frame)

                # 3. Dual-Hand In-Hand Object Association & Action Inference
                action = self.classifier.evaluate_hands_and_objects(
                    pose,
                    self.payload_targets,
                    ai_objects,
                    current_step=self.compliance_engine.current_step(),
                    current_step_idx=self.compliance_engine.current_step_idx,
                    all_steps=self.compliance_engine.steps
                )

                # 4. Ambidextrous Protocol Verification
                self.compliance_engine.evaluate(action, pose)

                # Sync payload seating when insertion/mating step passes
                if self.compliance_engine.current_step_idx == 3 and not self.compliance_engine.is_completed:
                    self.classifier.slot_occupied = True

                # 5. Real-Time Telemetry Stream (Speed of graph equals camera: ~30 FPS)
                if time.time() - self.last_heartbeat_time >= 0.033:
                    step = self.compliance_engine.current_step()
                    step_id = step["step_id"] if step else len(self.compliance_engine.steps)
                    active_str = f"{action.action_type} {action.target_object} ({action.hand} HAND)" if action else ("COMPLETED • CONTINUOUS MONITORING" if self.compliance_engine.is_completed else "MONITORING")
                    cam_tag = self.compliance_engine.protocol_data.get("camera_tag", "OPTICAL HUD")
                    cam_chan = self.compliance_engine.protocol_data.get("camera_channel", "CAM-MSG-01-A")
                    self.mc_link.send_heartbeat(self.compliance_engine.protocol_id, step_id, pose, active_str, camera_tag=cam_tag, camera_channel=cam_chan)
                    self.last_heartbeat_time = time.time()

                # 6. Render Dual-Hand HUD with In-Hand Telemetry Card
                self._draw_hud(frame, pose, action, ai_objects, session_filename)

                # Stream live frame to Ground Station every 2nd frame (~15 FPS)
                if frame_counter % 2 == 0:
                    self.mc_link.send_frame(frame)

                if video_writer:
                    video_writer.write(frame)

                if not headless:
                    try:
                        # Detect if camera window was closed via GUI 'X' close button
                        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                            logger.info("Camera window closed by user.")
                            break

                        cv2.imshow(window_name, frame)
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                            logger.info("Quit command received ('q'). Closing camera window...")
                            break
                        elif key == ord('e'):
                            is_enlarged = not is_enlarged
                            win_w, win_h = (width, height) if is_enlarged else (640, 480)
                            cv2.resizeWindow(window_name, win_w, win_h)
                            logger.info(f"Camera window toggled: {'ENLARGED (' + str(win_w) + 'x' + str(win_h) + ')' if is_enlarged else 'STANDARD'}")
                        elif key == ord('f'):
                            is_fullscreen = not is_fullscreen
                            prop = cv2.WINDOW_FULLSCREEN if is_fullscreen else cv2.WINDOW_NORMAL
                            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, prop)
                    except Exception as e:
                        logger.debug(f"Window display event: {e}")
        finally:
            if video_writer:
                try:
                    video_writer.release()
                    logger.info(f"📹 Session recording finalized: telemetry/sessions/{session_filename}")
                except Exception:
                    pass
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
            if not headless:
                try:
                    cv2.destroyWindow(window_name)
                except Exception:
                    pass
                try:
                    cv2.destroyAllWindows()
                    # Multiple waitKey calls pump the X11/GTK event loop on Linux to destroy the window
                    for _ in range(5):
                        cv2.waitKey(1)
                except Exception:
                    pass
            try:
                self.mc_link.notify_camera_closed()
            except Exception:
                pass
            logger.info("Camera window and capture device cleanly closed.")

    @staticmethod
    def _draw_hud_box(
        frame: np.ndarray,
        pt1: Tuple[int, int],
        pt2: Tuple[int, int],
        border_color: Optional[Tuple[int, int, int]] = (0, 229, 255),
        border_thickness: int = 1,
        alpha: float = 0.0,
        bg_color: Tuple[int, int, int] = (10, 15, 25)
    ):
        """Draws a 100% transparent HUD container box (zero background fill) so camera feed and procedure dots remain completely visible."""
        x1, y1 = max(0, min(pt1[0], pt2[0])), max(0, min(pt1[1], pt2[1]))
        x2, y2 = min(frame.shape[1], max(pt1[0], pt2[0])), min(frame.shape[0], max(pt1[1], pt2[1]))
        if alpha > 0.0 and x2 > x1 and y2 > y1:
            overlay = frame[y1:y2, x1:x2]
            bg = np.full_like(overlay, bg_color, dtype=np.uint8)
            frame[y1:y2, x1:x2] = cv2.addWeighted(bg, alpha, overlay, 1.0 - alpha, 0)
        if border_color is not None and border_thickness > 0:
            cv2.rectangle(frame, (x1, y1), (x2, y2), border_color, border_thickness)

    @staticmethod
    def _draw_shadowed_text(
        frame: np.ndarray,
        text: str,
        org: Tuple[int, int],
        font_face: int,
        font_scale: float,
        color: Tuple[int, int, int],
        thickness: int = 1
    ):
        """Renders high-contrast text with a black drop shadow for maximum legibility on transparent HUDs."""
        cv2.putText(frame, text, (org[0] + 1, org[1] + 1), font_face, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.putText(frame, text, org, font_face, font_scale, color, thickness, cv2.LINE_AA)

    def _render_procedure_dots(
        self,
        frame: np.ndarray,
        w: int,
        h: int,
        pose: Optional[AstronautPose] = None,
        current_step: Optional[Dict[str, Any]] = None
    ):
        """Renders procedure target stations with vivid neon glow, high-contrast labels,
        interactive laser guides from approaching hands, and real-time dwell progress rings."""
        exp_tgt = current_step.get("expected_target") if current_step else None

        # Extract active hand positions for interactive targeting guides
        hands = []
        if pose:
            if pose.left_hand and pose.left_hand.is_detected:
                lx = int(pose.left_hand.screen_pos[0] * w)
                ly = int(pose.left_hand.screen_pos[1] * h)
                hands.append(("LEFT", (lx, ly), pose.left_hand))
            if pose.right_hand and pose.right_hand.is_detected:
                rx = int(pose.right_hand.screen_pos[0] * w)
                ry = int(pose.right_hand.screen_pos[1] * h)
                hands.append(("RIGHT", (rx, ry), pose.right_hand))

        for tgt in self.payload_targets:
            cx, cy = int(tgt.centroid[0] * w), int(tgt.centroid[1] * h)
            is_expected = bool(exp_tgt and (exp_tgt == tgt.class_name or exp_tgt in tgt.class_name or tgt.class_name in exp_tgt))

            # Find closest hand to this target station
            closest_hand_dist = float("inf")
            closest_hand_pt = None
            for h_name, (hx, hy), h_state in hands:
                d = np.hypot(h_state.screen_pos[0] - tgt.centroid[0], h_state.screen_pos[1] - tgt.centroid[1])
                for f_pos in h_state.fingers.values():
                    fd = np.hypot(f_pos[0] - tgt.centroid[0], f_pos[1] - tgt.centroid[1])
                    if fd < d:
                        d = fd
                if d < closest_hand_dist:
                    closest_hand_dist = d
                    closest_hand_pt = (hx, hy)

            # Check dwell status from classifier
            is_dwelling = (
                (self.classifier.left_dwell_target == tgt.class_name) or
                (self.classifier.right_dwell_target == tgt.class_name)
            )
            dwell_f = max(
                self.classifier.left_dwell_frames if self.classifier.left_dwell_target == tgt.class_name else 0,
                self.classifier.right_dwell_frames if self.classifier.right_dwell_target == tgt.class_name else 0
            )

            # STATE 1: Hand in contact / Dwelling on Station
            if is_dwelling and dwell_f > 0:
                progress = min(1.0, dwell_f / 6.0)
                if is_expected:
                    # Compliant Interaction: Emerald Green glow and circular progress ring
                    ring_col = (0, 255, 120)
                    cv2.circle(frame, (cx, cy), 24, (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.circle(frame, (cx, cy), 18, ring_col, 2, cv2.LINE_AA)
                    cv2.circle(frame, (cx, cy), 12, ring_col, -1, cv2.LINE_AA)

                    sweep_angle = int(progress * 360)
                    cv2.ellipse(frame, (cx, cy), (26, 26), -90, 0, sweep_angle, (0, 255, 120), 3, cv2.LINE_AA)

                    prog_str = f"ACQUIRING {int(progress * 100)}%" if progress < 1.0 else "✓ VERIFIED"
                    self._draw_shadowed_text(frame, prog_str, (cx - 36, cy - 32), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 120), 2)
                    self._draw_shadowed_text(frame, tgt.class_name, (cx - 28, cy + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (220, 255, 220), 1)
                else:
                    # Deviation / Wrong Object Contact: High-intensity Warning Red
                    warn_col = (0, 40, 255)
                    cv2.circle(frame, (cx, cy), 26, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.circle(frame, (cx, cy), 14, warn_col, -1, cv2.LINE_AA)
                    self._draw_shadowed_text(frame, "⚠️ WRONG OBJECT", (cx - 44, cy - 32), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 40, 255), 2)
                    self._draw_shadowed_text(frame, tgt.class_name, (cx - 28, cy + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 255), 1)

            # STATE 2: Hand Approaching (< 0.22 screen distance)
            elif closest_hand_pt and closest_hand_dist < 0.22:
                beam_col = (0, 229, 255) if is_expected else (0, 140, 255)
                # Visual laser targeting guide line directly from hand to target station
                cv2.line(frame, closest_hand_pt, (cx, cy), beam_col, 1, cv2.LINE_AA)
                cv2.circle(frame, (cx, cy), 20, beam_col, 1, cv2.LINE_AA)
                cv2.circle(frame, (cx, cy), 14, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.circle(frame, (cx, cy), 10, beam_col, -1, cv2.LINE_AA)

                prox_pct = int(max(0, (1.0 - closest_hand_dist / 0.22) * 100))
                tag_label = f"{tgt.class_name} [{prox_pct}% LOCK]" if is_expected else f"{tgt.class_name} (NOT EXPECTED)"
                lbl_col = (0, 255, 200) if is_expected else (0, 160, 255)
                self._draw_shadowed_text(frame, tag_label, (cx - 30, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.40, lbl_col, 1)

            # STATE 3: Idle / Default Station Display
            else:
                if is_expected:
                    # Required Target Beacon: Glowing Cyan/Gold
                    cv2.circle(frame, (cx, cy), 20, (0, 229, 255), 1, cv2.LINE_AA)
                    cv2.circle(frame, (cx, cy), 15, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.circle(frame, (cx, cy), 10, (255, 200, 0), -1, cv2.LINE_AA)
                    self._draw_shadowed_text(frame, f"★ {tgt.class_name}", (cx - 30, cy - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 220), 1)
                else:
                    # Inactive Target Station: Clean Sky Blue
                    cv2.circle(frame, (cx, cy), 16, (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.circle(frame, (cx, cy), 12, (255, 200, 0), 1, cv2.LINE_AA)
                    cv2.circle(frame, (cx, cy), 8, (255, 180, 0), -1, cv2.LINE_AA)
                    self._draw_shadowed_text(frame, tgt.class_name, (cx - 24, cy - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (240, 220, 100), 1)

    def _draw_hud(
        self,
        frame: np.ndarray,
        pose: AstronautPose,
        action: Optional[ActionCandidate],
        ai_objects: List[DetectedObject],
        session_filename: Optional[str] = None
    ):
        h, w, _ = frame.shape

        # 1. Render Protocol Target Stations
        self._render_procedure_dots(frame, w, h, pose=pose, current_step=self.compliance_engine.current_step())

        # 2. Render AI-Detected Handheld / Scene Objects
        for obj in ai_objects:
            xmin, ymin, xmax, ymax = obj.bbox
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 255), 1)
            cv2.putText(
                frame,
                f"AI: {obj.class_name} ({int(obj.confidence * 100)}%)",
                (xmin, max(15, ymin - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                (0, 255, 255),
                1
            )

        # 3. Render 33-Keypoint Microgravity 3D Skeletal Wireframe
        skeletal_bones = [
            # Torso & Ribcage Box (Emerald Green)
            ("left_shoulder", "right_shoulder", (0, 230, 118), 3),
            ("left_shoulder", "left_hip", (0, 230, 118), 3),
            ("right_shoulder", "right_hip", (0, 230, 118), 3),
            ("left_hip", "right_hip", (0, 230, 118), 3),
            # Left Arm (Electric Violet/Magenta)
            ("left_shoulder", "left_elbow", (214, 112, 218), 3),
            ("left_elbow", "left_wrist", (214, 112, 218), 3),
            # Right Arm (Neon Amber/Orange)
            ("right_shoulder", "right_elbow", (0, 165, 255), 3),
            ("right_elbow", "right_wrist", (0, 165, 255), 3),
            # Left Hand Web
            ("left_wrist", "left_pinky", (214, 112, 218), 2),
            ("left_wrist", "left_index", (214, 112, 218), 2),
            ("left_wrist", "left_thumb", (214, 112, 218), 2),
            ("left_pinky", "left_index", (214, 112, 218), 1),
            # Right Hand Web
            ("right_wrist", "right_pinky", (0, 165, 255), 2),
            ("right_wrist", "right_index", (0, 165, 255), 2),
            ("right_wrist", "right_thumb", (0, 165, 255), 2),
            ("right_pinky", "right_index", (0, 165, 255), 1),
            # Face & Head
            ("nose", "left_eye", (0, 229, 255), 2),
            ("left_eye", "left_ear", (0, 229, 255), 2),
            ("nose", "right_eye", (0, 229, 255), 2),
            ("right_eye", "right_ear", (0, 229, 255), 2),
            # Legs & Foot Restraints
            ("left_hip", "left_knee", (0, 230, 118), 3),
            ("left_knee", "left_ankle", (0, 230, 118), 3),
            ("left_ankle", "left_heel", (0, 230, 118), 2),
            ("left_ankle", "left_foot_index", (0, 230, 118), 2),
            ("left_heel", "left_foot_index", (0, 230, 118), 2),
            ("right_hip", "right_knee", (0, 230, 118), 3),
            ("right_knee", "right_ankle", (0, 230, 118), 3),
            ("right_ankle", "right_heel", (0, 230, 118), 2),
            ("right_ankle", "right_foot_index", (0, 230, 118), 2),
            ("right_heel", "right_foot_index", (0, 230, 118), 2),
        ]

        # Draw bone connections
        for j_a, j_b, color, thickness in skeletal_bones:
            if j_a in pose.keypoints and j_b in pose.keypoints:
                kp_a = pose.keypoints[j_a]
                kp_b = pose.keypoints[j_b]
                p_a = (int(kp_a.x * w), int(kp_a.y * h))
                p_b = (int(kp_b.x * w), int(kp_b.y * h))
                cv2.line(frame, p_a, p_b, color, thickness, cv2.LINE_AA)

        # Draw glowing joint nodes for all 33 keypoints
        for name, kp in pose.keypoints.items():
            px, py = int(kp.x * w), int(kp.y * h)
            is_hand = "pinky" in name or "index" in name or "thumb" in name or "wrist" in name
            j_color = (214, 112, 218) if ("left" in name and is_hand) else ((0, 165, 255) if ("right" in name and is_hand) else (0, 229, 255))
            cv2.circle(frame, (px, py), 5, j_color, -1, cv2.LINE_AA)
            cv2.circle(frame, (px, py), 8, (255, 255, 255), 1, cv2.LINE_AA)

        # Badge indicating active 3D skeleton tracking (100% transparent)
        self._draw_hud_box(frame, (w - 380, h - 56), (w - 10, h - 34), border_color=(0, 230, 118), alpha=0.0)
        self._draw_shadowed_text(frame, "3D SKELETAL RIG: 33 JOINTS TRACKED", (w - 370, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 230, 118), 1)

        # Acceleration Engine Badge (100% transparent)
        pose_dev = getattr(self.pose_engine, "device_used", "CPU")
        dev_color = (0, 255, 120) if pose_dev == "GPU" else (0, 200, 255)
        self._draw_hud_box(frame, (w - 380, h - 30), (w - 10, h - 8), border_color=dev_color, alpha=0.0)
        self._draw_shadowed_text(frame, f"INFERENCE ENGINE: {pose_dev} ACCELERATED", (w - 370, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.36, dev_color, 1)

        # 4. MARK BOTH HANDS PROMINENTLY (AMBIDEXTROUS CAPABILITY)
        # Left Hand: Electric Violet/Magenta (214, 112, 218)
        if pose.left_hand and pose.left_hand.is_detected:
            lx, ly = int(pose.left_hand.screen_pos[0] * w), int(pose.left_hand.screen_pos[1] * h)
            l_palm_x, l_palm_y = int(pose.left_hand.palm_pos[0] * w), int(pose.left_hand.palm_pos[1] * h)

            # Crosshairs & concentric reticle
            cv2.circle(frame, (lx, ly), 10, (214, 112, 218), 2)
            cv2.circle(frame, (lx, ly), 18, (214, 112, 218), 1)
            cv2.line(frame, (lx - 14, ly), (lx + 14, ly), (214, 112, 218), 1)
            cv2.line(frame, (lx, ly - 14), (lx, ly + 14), (214, 112, 218), 1)
            cv2.circle(frame, (l_palm_x, l_palm_y), 5, (255, 0, 255), -1)

            # Draw finger skeletal lines from wrist
            for f_xy in pose.left_hand.fingers.values():
                fx, fy = int(f_xy[0] * w), int(f_xy[1] * h)
                cv2.line(frame, (lx, ly), (fx, fy), (214, 112, 218), 1)
                cv2.circle(frame, (fx, fy), 3, (255, 128, 255), -1)

            # Prominent Hand Status Banner (100% transparent)
            l_held = pose.left_hand.held_object
            l_status = f"HELD: {l_held}" if l_held else "EMPTY (READY)"
            self._draw_hud_box(frame, (lx - 70, ly - 42), (lx + 90, ly - 18), border_color=(214, 112, 218), alpha=0.0)
            self._draw_shadowed_text(frame, f"[L-HAND] {l_status}", (lx - 66, ly - 26), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (214, 112, 218), 1)

        # Right Hand: Cyan/Sky-Blue (0, 200, 255)
        if pose.right_hand and pose.right_hand.is_detected:
            rx, ry = int(pose.right_hand.screen_pos[0] * w), int(pose.right_hand.screen_pos[1] * h)
            r_palm_x, r_palm_y = int(pose.right_hand.palm_pos[0] * w), int(pose.right_hand.palm_pos[1] * h)

            # Crosshairs & concentric reticle
            cv2.circle(frame, (rx, ry), 10, (0, 200, 255), 2)
            cv2.circle(frame, (rx, ry), 18, (0, 200, 255), 1)
            cv2.line(frame, (rx - 14, ry), (rx + 14, ry), (0, 200, 255), 1)
            cv2.line(frame, (rx, ry - 14), (rx, ry + 14), (0, 200, 255), 1)
            cv2.circle(frame, (r_palm_x, r_palm_y), 5, (0, 200, 255), -1)

            # Draw finger skeletal lines from wrist
            for f_xy in pose.right_hand.fingers.values():
                fx, fy = int(f_xy[0] * w), int(f_xy[1] * h)
                cv2.line(frame, (rx, ry), (fx, fy), (0, 200, 255), 1)
                cv2.circle(frame, (fx, fy), 3, (100, 190, 255), -1)

            # Prominent Hand Status Banner (100% transparent)
            r_held = pose.right_hand.held_object
            r_status = f"HELD: {r_held}" if r_held else "EMPTY (READY)"
            self._draw_hud_box(frame, (rx - 70, ry - 42), (rx + 90, ry - 18), border_color=(0, 200, 255), alpha=0.0)
            self._draw_shadowed_text(frame, f"[R-HAND] {r_status}", (rx - 66, ry - 26), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 200, 255), 1)

        # 5. Top Telemetry Banner (Clean aerospace layout, NO overlapping text)
        cam_chan = self.compliance_engine.protocol_data.get("camera_channel", "CAM-MSG-01-A")
        exp_name = self.compliance_engine.protocol_data.get("experiment_name", "Active Experiment")

        step = self.compliance_engine.current_step()
        status_text = f"EXPECTED: {step['expected_action']} {step['expected_target']}" if step else "STATUS: EXPERIMENT COMPLETE"
        step_desc = step.get('description', '') if step else ''

        # Left Info Card
        l_card_w = min(440, w - 180)
        self._draw_hud_box(frame, (10, 8), (l_card_w, 98), border_color=(0, 229, 255), border_thickness=1, alpha=0.0)
        self._draw_shadowed_text(frame, f"MISSION: {exp_name[:36]}", (18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 229, 255), 1)
        self._draw_shadowed_text(frame, f"PROTOCOL: {self.compliance_engine.protocol_id} [{cam_chan}]", (18, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 200, 220), 1)
        self._draw_shadowed_text(frame, status_text, (18, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 120), 2)
        if step_desc:
            self._draw_shadowed_text(frame, f"DIRECTIVE: {step_desc[:46]}", (18, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 235, 180), 1)

        # Right Telemetry Card
        r_card_x = max(l_card_w + 8, w - 170)
        self._draw_hud_box(frame, (r_card_x, 8), (w - 10, 98), border_color=(0, 229, 255), border_thickness=1, alpha=0.0)
        if session_filename:
            cv2.circle(frame, (w - 20, 24), 6, (0, 0, 255), -1)
            self._draw_shadowed_text(frame, "REC ⏺", (r_card_x + 10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 255), 1)
        else:
            self._draw_shadowed_text(frame, "LIVE OPTICAL", (r_card_x + 10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 229, 255), 1)
        self._draw_shadowed_text(frame, f"ROLL: {pose.body_roll_degrees:.1f}°", (r_card_x + 10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (180, 200, 220), 1)
        self._draw_shadowed_text(frame, f"ACTOR: {pose.active_hand}", (r_card_x + 10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (180, 200, 220), 1)
        if action:
            self._draw_shadowed_text(frame, f"{action.action_type[:8]}", (r_card_x + 10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 255, 200), 1)
        else:
            self._draw_shadowed_text(frame, "STATUS: OK", (r_card_x + 10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 230, 118), 1)

        # 6. Prominent Deviation Alert Banner (Translucent Red Overlay when error detected)
        banner = getattr(self.compliance_engine, "active_deviation_banner", None)
        if banner and time.time() < banner.get("expires", 0):
            b_y1 = max(105, int(h * 0.23))
            b_y2 = b_y1 + 54
            sub_img = frame[b_y1:b_y2, 20:w-20]
            if sub_img.size > 0:
                red_rect = np.zeros_like(sub_img)
                red_rect[:] = (0, 0, 170)
                cv2.addWeighted(sub_img, 0.25, red_rect, 0.75, 0, sub_img)
                frame[b_y1:b_y2, 20:w-20] = sub_img
            cv2.rectangle(frame, (20, b_y1), (w - 20, b_y2), (0, 40, 255), 2, cv2.LINE_AA)
            self._draw_shadowed_text(frame, f"⚠️  {banner['title']}", (35, b_y1 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2)
            self._draw_shadowed_text(frame, banner['message'], (35, b_y1 + 44), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 240, 255), 1)

        # 7. IN-HAND PAYLOAD DATA HUD CARD (When an object is picked, 100% transparent)
        if pose.held_payload:
            pl = pose.held_payload
            specs = pl.get("specs", {})
            card_h = 88
            card_y = max(130, h - card_h - 40)
            card_w = min(w - 20, 580)

            self._draw_hud_box(frame, (10, card_y), (10 + card_w, card_y + card_h), border_color=(0, 230, 118), border_thickness=1, alpha=0.0)
            obj_title = f"AI IDENTIFIED IN-HAND: {pl['object_name'].upper()} ({pl['hand']} HAND)"
            self._draw_shadowed_text(frame, obj_title, (20, card_y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 255, 120), 2)

            line1 = f"Mass: {specs.get('microg_mass', '320 g')} | Grasp: {pl['grasp_confidence']}% | Hazard: {specs.get('hazard_level', 'Nominal')}"
            line2 = f"Target Bay: {specs.get('destination_bay', 'Rack Slot 1')} | Alignment: {specs.get('alignment_pin', 'Normal')}"
            line3 = f"Directive: {specs.get('handling', 'Maintain firm zero-G grasp.')}"

            self._draw_shadowed_text(frame, line1, (20, card_y + 44), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (220, 240, 255), 1)
            self._draw_shadowed_text(frame, line2, (20, card_y + 64), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (180, 200, 220), 1)
            self._draw_shadowed_text(frame, line3, (20, card_y + 82), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (0, 229, 255), 1)

        # 8. Lower Body & Leg Activity HUD Rendering
        leg_kp_pairs = [
            ("left_hip", "left_knee"), ("left_knee", "left_ankle"), ("left_ankle", "left_foot_index"),
            ("right_hip", "right_knee"), ("right_knee", "right_ankle"), ("right_ankle", "right_foot_index")
        ]
        for name_a, name_b in leg_kp_pairs:
            if name_a in pose.keypoints and name_b in pose.keypoints:
                kp_a = pose.keypoints[name_a]
                kp_b = pose.keypoints[name_b]
                pt_a = (int(kp_a.x * w), int(kp_a.y * h))
                pt_b = (int(kp_b.x * w), int(kp_b.y * h))
                cv2.line(frame, pt_a, pt_b, (0, 230, 118), 2)
                cv2.circle(frame, pt_b, 4, (0, 255, 255), -1)

        # Prominent Knee Indicators (100% transparent)
        if pose.left_leg and pose.left_leg.is_detected and "left_knee" in pose.keypoints:
            lk = pose.keypoints["left_knee"]
            lk_x, lk_y = int(lk.x * w), int(lk.y * h)
            l_col = (0, 229, 255)
            cv2.circle(frame, (lk_x, lk_y), 8, l_col, 2, cv2.LINE_AA)
            cv2.circle(frame, (lk_x, lk_y), 14, l_col, 1, cv2.LINE_AA)
            cv2.line(frame, (lk_x - 12, lk_y), (lk_x + 12, lk_y), l_col, 1)
            cv2.line(frame, (lk_x, lk_y - 12), (lk_x, lk_y + 12), l_col, 1)
            self._draw_hud_box(frame, (lk_x - 55, lk_y - 32), (lk_x + 75, lk_y - 12), border_color=l_col, alpha=0.0)
            self._draw_shadowed_text(frame, f"L-KNEE {int(pose.left_leg.knee_angle_deg)} deg", (lk_x - 50, lk_y - 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, l_col, 1)

        if pose.right_leg and pose.right_leg.is_detected and "right_knee" in pose.keypoints:
            rk = pose.keypoints["right_knee"]
            rk_x, rk_y = int(rk.x * w), int(rk.y * h)
            r_col = (0, 229, 255)
            cv2.circle(frame, (rk_x, rk_y), 8, r_col, 2, cv2.LINE_AA)
            cv2.circle(frame, (rk_x, rk_y), 14, r_col, 1, cv2.LINE_AA)
            cv2.line(frame, (rk_x - 12, rk_y), (rk_x + 12, rk_y), r_col, 1)
            cv2.line(frame, (rk_x, rk_y - 12), (rk_x, rk_y + 12), r_col, 1)
            self._draw_hud_box(frame, (rk_x - 55, rk_y - 32), (rk_x + 75, rk_y - 12), border_color=r_col, alpha=0.0)
            self._draw_shadowed_text(frame, f"R-KNEE {int(pose.right_leg.knee_angle_deg)} deg", (rk_x - 50, rk_y - 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, r_col, 1)

        # Lower Body Telemetry Banner
        l_ang = int(pose.left_leg.knee_angle_deg) if pose.left_leg else 180
        r_ang = int(pose.right_leg.knee_angle_deg) if pose.right_leg else 180
        l_act = pose.left_leg.activity if pose.left_leg else "STANDBY"
        r_act = pose.right_leg.activity if pose.right_leg else "STANDBY"
        lb_col = (0, 229, 255)
        leg_str = f"LEGS: L={l_ang}° ({l_act}) | R={r_ang}° ({r_act}) | {pose.lower_body_activity}"
        self._draw_hud_box(frame, (10, 102), (min(w - 10, 620), 126), border_color=lb_col, border_thickness=1, alpha=0.0)
        self._draw_shadowed_text(frame, leg_str, (18, 119), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (220, 240, 255), 1)

        # 9. Camera Feed Watermark (Bottom Right, 100% transparent)
        cam_watermark = f"OPTICAL HUD [{cam_chan}]"
        self._draw_hud_box(frame, (w - 240, h - 30), (w - 10, h - 8), border_color=(0, 229, 255), border_thickness=1, alpha=0.0)
        self._draw_shadowed_text(frame, cam_watermark, (w - 230, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 229, 255), 1)

        # 10. Final Procedure Dots Pass: Always renders on top
        self._render_procedure_dots(frame, w, h, pose=pose, current_step=self.compliance_engine.current_step())

    def _run_synthetic_loop(self):
        logger.info("Executing synthetic baseline monitoring loop...")
        step = self.compliance_engine.current_step()
        dummy_kps = {
            name: Keypoint3D(i, name, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.95)
            for i, name in enumerate(MicrogravityPoseEngine.LANDMARK_NAMES)
        }
        dummy_pose = AstronautPose(
            timestamp=time.time(),
            keypoints=dummy_kps,
            left_hand=HandTrackingState("LEFT", True, (0.3, 0.5), (0.3, 0.5), (-0.2, 0.0, 0.0), 0.05),
            right_hand=HandTrackingState("RIGHT", True, (0.7, 0.5), (0.7, 0.5), (0.2, 0.0, 0.0), 0.05),
            left_leg=LegTrackingState("LEFT", True, 178.0, 0.02, "ANCHORED"),
            right_leg=LegTrackingState("RIGHT", True, 175.0, 0.02, "ANCHORED"),
            lower_body_activity="ANCHORED IN FOOT RESTRAINT",
            active_hand="RIGHT",
            wrist_velocity=0.05,
            body_roll_degrees=0.0,
            dominant_wrist_pos=(0.0, 0.0, 0.0)
        )
        cam_tag = self.compliance_engine.protocol_data.get("camera_tag", "OPTICAL HUD")
        cam_chan = self.compliance_engine.protocol_data.get("camera_channel", "CAM-MSG-01-A")
        self.mc_link.send_heartbeat(self.compliance_engine.protocol_id, step["step_id"] if step else 1, dummy_pose, camera_tag=cam_tag, camera_channel=cam_chan)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AstroActAi Dual-Hand AI Edge Monitor")
    parser.add_argument("--protocol", default="configs/protocol.json", help="Protocol file path")
    parser.add_argument("--ground-url", default="http://127.0.0.1:8000", help="Ground control server URL")
    parser.add_argument("--camera", default=0, help="Camera index or video file")
    parser.add_argument("--headless", action="store_true", help="Run without OpenCV GUI window")
    parser.add_argument("--no-record", action="store_true", help="Disable recording session to disk")
    parser.add_argument("--no-flip", action="store_true", help="Disable horizontal camera mirroring")
    parser.add_argument("--enlarge", action="store_true", default=True, help="Launch camera display in enlarged window (1280x720)")
    parser.add_argument("--no-enlarge", action="store_false", dest="enlarge", help="Launch in standard 640x480 window")
    parser.add_argument("--fullscreen", action="store_true", help="Launch camera in fullscreen mode")
    parser.add_argument("--width", type=int, default=1280, help="Window width (default: 1280)")
    parser.add_argument("--height", type=int, default=720, help="Window height (default: 720)")
    parser.add_argument("--device", choices=["gpu", "cpu"], default="gpu", help="Inference device: gpu (default) or cpu")
    args = parser.parse_args()

    cam_src = args.camera
    try:
        cam_src = int(cam_src)
    except ValueError:
        pass

    app = AstronautMonitoringSystem(protocol_path=args.protocol, ground_url=args.ground_url, delegate=args.device)
    try:
        app.run(
            camera_source=cam_src,
            headless=args.headless,
            record=not args.no_record,
            no_flip=args.no_flip,
            enlarge=args.enlarge,
            fullscreen=args.fullscreen,
            width=args.width,
            height=args.height
        )
    except KeyboardInterrupt:
        logger.info("Astronaut monitoring stopped by user (Ctrl+C).")
    finally:
        _close_camera_windows()
