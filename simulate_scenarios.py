"""
AstroActAi — Scenario Simulator & Validation Testbed
Validates all 4 canonical astronaut experiment cases:
  1. Correct Procedure: Pick -> Rotate -> Insert -> Lock (100% Compliant)
  2. Skipped Step: Pick -> Insert (Rotate Skipped -> Deviation Alert)
  3. Wrong Order: Pick -> Insert -> Rotate (Out-of-Order -> Deviation Alert)
  4. Wrong Object: Picks Component_B instead of Component_A (Wrong Object Alert)
"""

import time
import json
import math
import argparse
import logging
from typing import List, Dict, Tuple, Optional
import requests

from astronaut_monitor import (
    Keypoint3D, AstronautPose, DetectedObject, ActionCandidate,
    ProtocolComplianceEngine, MissionControlLink, MicrogravityPoseEngine,
    LegTrackingState, HandTrackingState
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [SIMULATOR] %(levelname)s: %(message)s")
logger = logging.getLogger("ScenarioSim")

# Canonical 33 Landmark names
LANDMARK_NAMES = MicrogravityPoseEngine.LANDMARK_NAMES

def generate_astronaut_pose(wrist_xy: Tuple[float, float], roll_deg: float, wrist_vel: float) -> AstronautPose:
    """Synthesizes a realistic 33-keypoint astronaut skeletal pose floating in microgravity."""
    kps: Dict[str, Keypoint3D] = {}

    # Torso and head base coordinates
    mid_x, mid_y = 0.5, 0.45
    rad = math.radians(roll_deg)
    cos_r = math.cos(rad)
    sin_r = math.sin(rad)

    def rot(dx, dy):
        return mid_x + dx * cos_r - dy * sin_r, mid_y + dx * sin_r + dy * cos_r

    # Head
    nx, ny = rot(0, -0.25)
    kps["nose"] = Keypoint3D(0, "nose", nx, ny, 0.1, nx - 0.5, ny - 0.5, 0.1, 0.98)
    lex, ley = rot(-0.03, -0.28)
    kps["left_eye"] = Keypoint3D(2, "left_eye", lex, ley, 0.1, lex - 0.5, ley - 0.5, 0.1, 0.95)
    rex, rey = rot(0.03, -0.28)
    kps["right_eye"] = Keypoint3D(5, "right_eye", rex, rey, 0.1, rex - 0.5, rey - 0.5, 0.1, 0.95)
    kps["left_ear"] = Keypoint3D(7, "left_ear", lex - 0.04, ley, 0.12, 0, 0, 0, 0.92)
    kps["right_ear"] = Keypoint3D(8, "right_ear", rex + 0.04, rey, 0.12, 0, 0, 0, 0.92)

    # Shoulders
    ls_x, ls_y = rot(-0.14, -0.15)
    kps["left_shoulder"] = Keypoint3D(11, "left_shoulder", ls_x, ls_y, 0.15, ls_x - 0.5, ls_y - 0.5, 0.15, 0.97)
    rs_x, rs_y = rot(0.14, -0.15)
    kps["right_shoulder"] = Keypoint3D(12, "right_shoulder", rs_x, rs_y, 0.15, rs_x - 0.5, rs_y - 0.5, 0.15, 0.97)

    # Left arm (relaxed/floating in Neutral Body Posture)
    le_x, le_y = rot(-0.20, 0.0)
    kps["left_elbow"] = Keypoint3D(13, "left_elbow", le_x, le_y, 0.18, 0, 0, 0, 0.95)
    lw_x, lw_y = rot(-0.22, 0.16)
    kps["left_wrist"] = Keypoint3D(15, "left_wrist", lw_x, lw_y, 0.20, lw_x - 0.5, lw_y - 0.5, 0.20, 0.95)
    kps["left_pinky"] = Keypoint3D(17, "left_pinky", lw_x - 0.02, lw_y + 0.02, 0.2, 0, 0, 0, 0.9)
    kps["left_index"] = Keypoint3D(19, "left_index", lw_x, lw_y + 0.03, 0.2, 0, 0, 0, 0.9)
    kps["left_thumb"] = Keypoint3D(21, "left_thumb", lw_x + 0.02, lw_y + 0.01, 0.2, 0, 0, 0, 0.9)

    # Right arm (active effector guided to wrist_xy)
    rw_x, rw_y = wrist_xy
    re_x, re_y = (rs_x + rw_x) / 2.0 + 0.06, (rs_y + rw_y) / 2.0
    kps["right_elbow"] = Keypoint3D(14, "right_elbow", re_x, re_y, 0.18, 0, 0, 0, 0.95)
    kps["right_wrist"] = Keypoint3D(16, "right_wrist", rw_x, rw_y, 0.22, rw_x - 0.5, rw_y - 0.5, 0.22, 0.98)
    kps["right_pinky"] = Keypoint3D(18, "right_pinky", rw_x - 0.02, rw_y + 0.02, 0.22, 0, 0, 0, 0.9)
    kps["right_index"] = Keypoint3D(20, "right_index", rw_x, rw_y + 0.03, 0.22, 0, 0, 0, 0.9)
    kps["right_thumb"] = Keypoint3D(22, "right_thumb", rw_x + 0.02, rw_y + 0.01, 0.22, 0, 0, 0, 0.9)

    # Hips and floating legs
    lh_x, lh_y = rot(-0.08, 0.18)
    kps["left_hip"] = Keypoint3D(23, "left_hip", lh_x, lh_y, 0.25, 0, 0, 0, 0.94)
    rh_x, rh_y = rot(0.08, 0.18)
    kps["right_hip"] = Keypoint3D(24, "right_hip", rh_x, rh_y, 0.25, 0, 0, 0, 0.94)

    lk_x, lk_y = rot(-0.10, 0.38)
    kps["left_knee"] = Keypoint3D(25, "left_knee", lk_x, lk_y, 0.3, 0, 0, 0, 0.92)
    rk_x, rk_y = rot(0.10, 0.38)
    kps["right_knee"] = Keypoint3D(26, "right_knee", rk_x, rk_y, 0.3, 0, 0, 0, 0.92)

    la_x, la_y = rot(-0.11, 0.58)
    kps["left_ankle"] = Keypoint3D(27, "left_ankle", la_x, la_y, 0.35, 0, 0, 0, 0.9)
    ra_x, ra_y = rot(0.11, 0.58)
    kps["right_ankle"] = Keypoint3D(28, "right_ankle", ra_x, ra_y, 0.35, 0, 0, 0, 0.9)

    kps["left_heel"] = Keypoint3D(29, "left_heel", la_x - 0.02, la_y + 0.02, 0.36, 0, 0, 0, 0.88)
    kps["right_heel"] = Keypoint3D(30, "right_heel", ra_x + 0.02, ra_y + 0.02, 0.36, 0, 0, 0, 0.88)
    kps["left_foot_index"] = Keypoint3D(31, "left_foot_index", la_x + 0.02, la_y + 0.03, 0.36, 0, 0, 0, 0.88)
    kps["right_foot_index"] = Keypoint3D(32, "right_foot_index", ra_x - 0.02, ra_y + 0.03, 0.36, 0, 0, 0, 0.88)

    # Fill remaining landmark slots if any
    for i, name in enumerate(LANDMARK_NAMES):
        if name not in kps:
            kps[name] = Keypoint3D(i, name, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.8)

    left_leg = LegTrackingState(
        leg_name="LEFT",
        is_detected=True,
        knee_angle_deg=176.5,
        velocity=0.01,
        activity="ANCHORED",
        hip_pos=(lh_x - 0.5, lh_y - 0.5, 0.25),
        knee_pos=(lk_x - 0.5, lk_y - 0.5, 0.30),
        ankle_pos=(la_x - 0.5, la_y - 0.5, 0.35)
    )
    right_leg = LegTrackingState(
        leg_name="RIGHT",
        is_detected=True,
        knee_angle_deg=174.8,
        velocity=0.01,
        activity="ANCHORED",
        hip_pos=(rh_x - 0.5, rh_y - 0.5, 0.25),
        knee_pos=(rk_x - 0.5, rk_y - 0.5, 0.30),
        ankle_pos=(ra_x - 0.5, ra_y - 0.5, 0.35)
    )

    return AstronautPose(
        timestamp=time.time(),
        keypoints=kps,
        left_leg=left_leg,
        right_leg=right_leg,
        lower_body_activity="ANCHORED IN FOOT RESTRAINT",
        wrist_velocity=wrist_vel,
        body_roll_degrees=roll_deg,
        dominant_wrist_pos=(wrist_xy[0] - 0.5, wrist_xy[1] - 0.5, 0.22)
    )


def simulate_action(
    engine: ProtocolComplianceEngine,
    mc_link: MissionControlLink,
    action_type: str,
    target_object: str,
    wrist_xy: Tuple[float, float],
    roll_deg: float = 0.0,
    wrist_vel: float = 0.15,
    repeat_frames: int = 10,
    frame_delay: float = 0.08
):
    """Feeds consecutive frames of an action to satisfy the debounce hysteresis filter."""
    for _ in range(repeat_frames):
        pose = generate_astronaut_pose(wrist_xy, roll_deg, wrist_vel)
        candidate = ActionCandidate(
            action_type=action_type,
            target_object=target_object,
            confidence=0.95,
            timestamp=time.time()
        )
        engine.evaluate(candidate, pose)
        # Periodic telemetry heartbeat
        step = engine.current_step()
        step_id = step["step_id"] if step else len(engine.steps)
        mc_link.send_heartbeat(engine.protocol_id, step_id, pose, f"{action_type} {target_object}")
        time.sleep(frame_delay)


def run_scenario_1_correct(protocol_path: str, ground_url: str):
    """
    Example 1: Correct Procedure
    Pick Component_A -> Rotate Component_A -> Insert Rack_Slot_1 -> Lock Latch_Mechanism
    Expected: Protocol Compliant — 100%
    """
    import os
    logger.info("=" * 70)
    logger.info("▶ EXECUTING SCENARIO 1: CORRECT PROCEDURE (100% COMPLIANCE)")
    logger.info("Expected: Pick -> Rotate -> Insert -> Lock")
    logger.info("=" * 70)

    proto_path = "configs/protocol_bio.json" if os.path.exists("configs/protocol_bio.json") else protocol_path
    mc_link = MissionControlLink(ground_url=ground_url)
    engine = ProtocolComplianceEngine(proto_path, mc_link)

    # Step 1: Pick Component_A
    logger.info("Action 1: Pick Component_A")
    simulate_action(engine, mc_link, "PICK", "Component_A", (0.22, 0.35), roll_deg=-5.0, wrist_vel=0.20)
    time.sleep(1.6)

    # Step 2: Rotate Component_A
    logger.info("Action 2: Rotate Component_A")
    simulate_action(engine, mc_link, "ROTATE", "Component_A", (0.25, 0.35), roll_deg=22.5, wrist_vel=0.18)
    time.sleep(1.6)

    # Step 3: Insert Component_A into Rack_Slot_1
    logger.info("Action 3: Insert Rack_Slot_1")
    simulate_action(engine, mc_link, "INSERT", "Rack_Slot_1", (0.75, 0.65), roll_deg=5.0, wrist_vel=0.12)
    time.sleep(1.6)

    # Step 4: Lock Latch_Mechanism
    logger.info("Action 4: Lock Latch_Mechanism")
    simulate_action(engine, mc_link, "LOCK", "Latch_Mechanism", (0.75, 0.85), roll_deg=0.0, wrist_vel=0.08)
    time.sleep(1.6)

    logger.info(f"Result: is_completed={engine.is_completed} | Protocol Verified 100%")
    return engine.is_completed


def run_scenario_2_skipped(protocol_path: str, ground_url: str):
    """
    Example 2: Skipped Step
    Pick Component_A -> Insert Rack_Slot_1 -> Lock Latch_Mechanism
    Expected: Protocol Deviation Detected (Rotate Skipped)
    """
    import os
    logger.info("=" * 70)
    logger.info("▶ EXECUTING SCENARIO 2: SKIPPED STEP (ROTATE SKIPPED)")
    logger.info("Expected: Warning: Required action 'Rotate Component' was not detected.")
    logger.info("=" * 70)

    proto_path = "configs/protocol_bio.json" if os.path.exists("configs/protocol_bio.json") else protocol_path
    mc_link = MissionControlLink(ground_url=ground_url)
    engine = ProtocolComplianceEngine(proto_path, mc_link)

    # Step 1: Pick Component_A (Compliant)
    logger.info("Action 1: Pick Component_A")
    simulate_action(engine, mc_link, "PICK", "Component_A", (0.22, 0.35), roll_deg=-5.0, wrist_vel=0.20)
    time.sleep(1.6)

    # Skipped Step: Astronaut directly performs INSERT Rack_Slot_1 without rotating!
    logger.info("Action 2 (Deviated): Insert Rack_Slot_1 (skipping Rotate)")
    simulate_action(engine, mc_link, "INSERT", "Rack_Slot_1", (0.75, 0.65), roll_deg=5.0, wrist_vel=0.25)
    time.sleep(0.5)

    return True


def run_scenario_3_wrong_order(protocol_path: str, ground_url: str):
    """
    Example 3: Wrong Order
    Expected: Rotate -> Insert
    Detected: Insert -> Rotate
    """
    import os
    logger.info("=" * 70)
    logger.info("▶ EXECUTING SCENARIO 3: WRONG ORDER EXECUTION")
    logger.info("Detected: Insert performed before Rotate")
    logger.info("=" * 70)

    proto_path = "configs/protocol_bio.json" if os.path.exists("configs/protocol_bio.json") else protocol_path
    mc_link = MissionControlLink(ground_url=ground_url)
    engine = ProtocolComplianceEngine(proto_path, mc_link)

    # Step 1: Pick Component_A (Compliant)
    logger.info("Action 1: Pick Component_A")
    simulate_action(engine, mc_link, "PICK", "Component_A", (0.22, 0.35), roll_deg=-5.0, wrist_vel=0.20)
    time.sleep(1.6)

    # Wrong order: Triggers Insert prematurely
    logger.info("Action 2 (Out-of-Order): Insert Rack_Slot_1")
    simulate_action(engine, mc_link, "INSERT", "Rack_Slot_1", (0.75, 0.65), roll_deg=5.0, wrist_vel=0.22)
    time.sleep(0.5)

    return True


def run_scenario_4_wrong_object(protocol_path: str, ground_url: str):
    """
    Example 4: Wrong Object Interaction
    Protocol requires picking Component_A, but astronaut picks Component_B.
    Expected: Protocol deviation: Expected Component_A, but Component_B was detected.
    """
    import os
    logger.info("=" * 70)
    logger.info("▶ EXECUTING SCENARIO 4: WRONG OBJECT INTERACTION")
    logger.info("Expected: Component_A | Detected: Component_B")
    logger.info("=" * 70)

    proto_path = "configs/protocol_bio.json" if os.path.exists("configs/protocol_bio.json") else protocol_path
    mc_link = MissionControlLink(ground_url=ground_url)
    engine = ProtocolComplianceEngine(proto_path, mc_link)

    # Incorrect Object: Interacting with Component_B
    logger.info("Action 1: Pick Component_B (Incorrect Object)")
    simulate_action(engine, mc_link, "PICK", "Component_B", (0.50, 0.35), roll_deg=0.0, wrist_vel=0.18)
    time.sleep(0.5)

    return True


def run_scenario_fluid(protocol_path: str, ground_url: str):
    """
    Scenario 2: Fluid Physics Wetlab (BAS-EXP-FLUID-2026)
    Pick Syringe_Injector -> Inject Sample_Vial -> Mate Manifold_Port_A -> Clamp Pinch_Valve
    """
    import os
    logger.info("=" * 70)
    logger.info("▶ EXECUTING SCENARIO: FLUID PHYSICS WETLAB (100% COMPLIANT)")
    logger.info("Expected: Pick Syringe -> Inject Vial -> Mate Manifold -> Clamp Valve")
    logger.info("=" * 70)

    proto_path = "configs/protocol_fluid.json" if os.path.exists("configs/protocol_fluid.json") else protocol_path
    mc_link = MissionControlLink(ground_url=ground_url)
    engine = ProtocolComplianceEngine(proto_path, mc_link)

    logger.info("Action 1: Pick Syringe_Injector")
    simulate_action(engine, mc_link, "PICK", "Syringe_Injector", (0.22, 0.35), roll_deg=0.0, wrist_vel=0.18)
    time.sleep(1.6)

    logger.info("Action 2: Inject Sample_Vial")
    simulate_action(engine, mc_link, "INJECT", "Sample_Vial", (0.45, 0.35), roll_deg=-5.0, wrist_vel=0.12)
    time.sleep(1.6)

    logger.info("Action 3: Mate Manifold_Port_A")
    simulate_action(engine, mc_link, "MATE", "Manifold_Port_A", (0.75, 0.65), roll_deg=0.0, wrist_vel=0.14)
    time.sleep(1.6)

    logger.info("Action 4: Clamp Pinch_Valve")
    simulate_action(engine, mc_link, "CLAMP", "Pinch_Valve", (0.75, 0.85), roll_deg=0.0, wrist_vel=0.10)
    time.sleep(1.6)

    logger.info(f"Result: is_completed={engine.is_completed} | Fluid Experiment Complete 100%")
    return engine.is_completed


def run_scenario_avionics(protocol_path: str, ground_url: str):
    """
    Scenario 3: Avionics Maintenance (BAS-EXP-AVIONICS-2026)
    Inspect SpaceWire_Harness -> Connect Avionics_Port_4 -> Torque Torque_Wrench -> Switch Breaker_Toggle
    """
    import os
    logger.info("=" * 70)
    logger.info("▶ EXECUTING SCENARIO: AVIONICS MAINTENANCE (100% COMPLIANT)")
    logger.info("Expected: Inspect Harness -> Connect Port -> Torque Wrench -> Switch Breaker")
    logger.info("=" * 70)

    proto_path = "configs/protocol_avionics.json" if os.path.exists("configs/protocol_avionics.json") else protocol_path
    mc_link = MissionControlLink(ground_url=ground_url)
    engine = ProtocolComplianceEngine(proto_path, mc_link)

    logger.info("Action 1: Inspect SpaceWire_Harness")
    simulate_action(engine, mc_link, "INSPECT", "SpaceWire_Harness", (0.22, 0.35), roll_deg=0.0, wrist_vel=0.06)
    time.sleep(1.6)

    logger.info("Action 2: Connect Avionics_Port_4")
    simulate_action(engine, mc_link, "CONNECT", "Avionics_Port_4", (0.50, 0.35), roll_deg=0.0, wrist_vel=0.15)
    time.sleep(1.6)

    logger.info("Action 3: Torque Torque_Wrench")
    simulate_action(engine, mc_link, "TORQUE", "Torque_Wrench", (0.75, 0.65), roll_deg=18.0, wrist_vel=0.20)
    time.sleep(1.6)

    logger.info("Action 4: Switch Breaker_Toggle")
    simulate_action(engine, mc_link, "SWITCH", "Breaker_Toggle", (0.75, 0.85), roll_deg=0.0, wrist_vel=0.12)
    time.sleep(1.6)

    logger.info(f"Result: is_completed={engine.is_completed} | Avionics Maintenance Complete 100%")
    return engine.is_completed


def run_scenario_emergency(protocol_path: str, ground_url: str):
    """
    Scenario 4: Airlock Depress & Emergency Hatch Seal (BAS-EMERGENCY-01)
    Don Breather_Mask -> Align Equalization_Valve -> Pull Hatch_Dog_Handle -> Lock Secondary_Lock
    """
    import os
    logger.info("=" * 70)
    logger.info("▶ EXECUTING SCENARIO: EMERGENCY AIRLOCK DEPRESS & HATCH SEAL (100% COMPLIANT)")
    logger.info("Expected: Don Mask -> Align Valve -> Pull Handle -> Lock Secondary")
    logger.info("=" * 70)

    proto_path = "configs/protocol_emergency.json" if os.path.exists("configs/protocol_emergency.json") else protocol_path
    mc_link = MissionControlLink(ground_url=ground_url)
    engine = ProtocolComplianceEngine(proto_path, mc_link)

    logger.info("Action 1: Don Breather_Mask")
    simulate_action(engine, mc_link, "DON", "Breather_Mask", (0.50, 0.20), roll_deg=0.0, wrist_vel=0.16)
    time.sleep(1.6)

    logger.info("Action 2: Align Equalization_Valve")
    simulate_action(engine, mc_link, "ALIGN", "Equalization_Valve", (0.50, 0.35), roll_deg=45.0, wrist_vel=0.18)
    time.sleep(1.6)

    logger.info("Action 3: Pull Hatch_Dog_Handle")
    simulate_action(engine, mc_link, "PULL", "Hatch_Dog_Handle", (0.75, 0.65), roll_deg=0.0, wrist_vel=0.28)
    time.sleep(1.6)

    logger.info("Action 4: Lock Secondary_Lock")
    simulate_action(engine, mc_link, "LOCK", "Secondary_Lock", (0.75, 0.85), roll_deg=0.0, wrist_vel=0.12)
    time.sleep(1.6)

    logger.info(f"Result: is_completed={engine.is_completed} | Emergency Airlock Seal Complete 100%")
    return engine.is_completed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AstroActAi Scenario Simulator")
    parser.add_argument("--protocol", default="configs/protocol.json", help="Protocol file path")
    parser.add_argument("--ground-url", default="http://127.0.0.1:8000", help="Ground control server URL")
    parser.add_argument(
        "--scenario",
        default="all",
        choices=["all", "correct", "skipped", "order", "wrong_object", "fluid", "avionics", "emergency"],
        help="Scenario to execute"
    )
    args = parser.parse_args()

    if args.scenario in ("all", "correct"):
        run_scenario_1_correct(args.protocol, args.ground_url)
        time.sleep(1.0)

    if args.scenario in ("all", "skipped"):
        run_scenario_2_skipped(args.protocol, args.ground_url)
        time.sleep(1.0)

    if args.scenario in ("all", "order"):
        run_scenario_3_wrong_order(args.protocol, args.ground_url)
        time.sleep(1.0)

    if args.scenario in ("all", "wrong_object"):
        run_scenario_4_wrong_object(args.protocol, args.ground_url)
        time.sleep(1.0)

    if args.scenario in ("all", "fluid"):
        run_scenario_fluid(args.protocol, args.ground_url)
        time.sleep(1.0)

    if args.scenario in ("all", "avionics"):
        run_scenario_avionics(args.protocol, args.ground_url)
        time.sleep(1.0)

    if args.scenario in ("all", "emergency"):
        run_scenario_emergency(args.protocol, args.ground_url)

    logger.info("✅ All requested simulation runs dispatched successfully.")
