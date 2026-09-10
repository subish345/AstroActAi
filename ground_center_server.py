"""
Mission Control Center - Payload Operations Integration Center (POIC)
Bharatiya Antariksh Station (BAS) Experiment Verification Ground Server
"""

import os
import json
import time
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone

import asyncio
import uvicorn
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO, format="%(asctime)s [GROUND STATION] %(levelname)s: %(message)s")
logger = logging.getLogger("GroundStation")

app = FastAPI(
    title="BAS Mission Control Ground Center - Payload Ops API",
    description="Real-time astronaut skeletal telemetry, protocol compliance tracking, and deviation dispatch center",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

LOG_FILE = "telemetry/flight_log.jsonl"
PROTOCOL_FILE = "configs/protocol.json"
os.makedirs("telemetry", exist_ok=True)
os.makedirs("configs", exist_ok=True)

# In-memory telemetry cache for real-time dashboard
telemetry_state = {
    "status": "ONLINE",
    "protocol_id": "BAS-EXP-BIO-2026",
    "experiment_name": "Biological Specimen Insertion & Rack Seal",
    "camera_tag": "CAM-01 • GLOVEBOX BIO-RACK",
    "camera_channel": "CAM-MSG-01-A",
    "camera_location": "Microgravity Science Glovebox Bay",
    "current_step": 1,
    "last_heartbeat": None,
    "last_deviation": None,
    "active_action": "STANDBY",
    "active_hand": "NONE",
    "left_hand": {
        "detected": False,
        "held_object": None,
        "velocity": 0.0,
        "coords": [0.0, 0.0, 0.0]
    },
    "right_hand": {
        "detected": False,
        "held_object": None,
        "velocity": 0.0,
        "coords": [0.0, 0.0, 0.0]
    },
    "left_leg": {
        "detected": False,
        "knee_angle": 180.0,
        "velocity": 0.0,
        "activity": "ANCHORED",
        "coords": [0.0, 0.0, 0.0]
    },
    "right_leg": {
        "detected": False,
        "knee_angle": 180.0,
        "velocity": 0.0,
        "activity": "ANCHORED",
        "coords": [0.0, 0.0, 0.0]
    },
    "lower_body_activity": "ANCHORED IN FOOT RESTRAINT",
    "held_payload": None,
    "astronaut_pose": {
        "body_roll_degrees": 0.0,
        "wrist_velocity": 0.0,
        "keypoints": {}
    },
    "history": [],
    "deviations_count": 0,
    "completed": False
}

# Live Video Streaming Frame Buffer
latest_frame_bytes: Optional[bytes] = None
latest_frame_time: float = 0.0

# WebSocket Connection Manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"Dashboard client connected. Active: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info(f"Dashboard client disconnected. Remaining: {len(self.active_connections)}")

    async def broadcast(self, message: Dict[str, Any]):
        for connection in list(self.active_connections):
            try:
                await connection.send_json(message)
            except Exception:
                self.disconnect(connection)

manager = ConnectionManager()

def append_flight_log(packet: Dict[str, Any]):
    try:
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(packet) + "\n")
    except Exception as e:
        logger.error(f"Failed to append flight log: {e}")

AVAILABLE_PROTOCOLS = [
    {
        "id": "BAS-EXP-BIO-2026",
        "file": "configs/protocol_bio.json",
        "experiment_name": "Biological Specimen Insertion & Rack Seal",
        "category": "BIOLOGICAL_EXPERIMENT",
        "facility": "Bharatiya Antariksh Station (BAS) - Microgravity Science Glovebox",
        "camera_tag": "CAM-01 • GLOVEBOX BIO-RACK",
        "camera_channel": "CAM-MSG-01-A",
        "camera_location": "Microgravity Science Glovebox Bay",
        "step_count": 4,
        "actions": ["PICK", "ROTATE", "INSERT", "LOCK"]
    },
    {
        "id": "BAS-EXP-FLUID-2026",
        "file": "configs/protocol_fluid.json",
        "experiment_name": "Capillary Fluidic Injection & Valve Mating",
        "category": "FLUID_PHYSICS_WETLAB",
        "facility": "Bharatiya Antariksh Station (BAS) - Glovebox Fluid Physics Bench",
        "camera_tag": "CAM-02 • FLUID BENCH PORT-A",
        "camera_channel": "CAM-FPB-02-B",
        "camera_location": "Fluid Physics Bench Workstation",
        "step_count": 4,
        "actions": ["PICK", "INJECT", "MATE", "CLAMP"]
    },
    {
        "id": "BAS-EXP-AVIONICS-2026",
        "file": "configs/protocol_avionics.json",
        "experiment_name": "SpaceWire Bus Harness Routing & Torque",
        "category": "AVIONICS_MAINTENANCE",
        "facility": "Bharatiya Antariksh Station (BAS) - Avionics Bay 3 Rack",
        "camera_tag": "CAM-03 • AVIONICS BAY 3 RACK",
        "camera_channel": "CAM-AVN-03-C",
        "camera_location": "Avionics Bay 3 Component Rack",
        "step_count": 4,
        "actions": ["INSPECT", "CONNECT", "TORQUE", "SWITCH"]
    },
    {
        "id": "BAS-EMERGENCY-01",
        "file": "configs/protocol_emergency.json",
        "experiment_name": "Airlock Depress & Emergency Hatch Seal",
        "category": "EMERGENCY_PROCEDURE",
        "facility": "Bharatiya Antariksh Station (BAS) - Node 1 Forward Airlock Tunnel",
        "camera_tag": "CAM-04 • AIRLOCK FORWARD HATCH",
        "camera_channel": "CAM-EMG-04-D",
        "camera_location": "Node 1 Forward Airlock Bulkhead",
        "step_count": 4,
        "actions": ["DON", "ALIGN", "PULL", "LOCK"]
    }
]

@app.get("/api/v1/protocols")
async def get_protocols():
    """Returns all available experiment protocols and scenarios for crew selection."""
    return {"count": len(AVAILABLE_PROTOCOLS), "protocols": AVAILABLE_PROTOCOLS}

@app.get("/api/v1/protocol")
async def get_protocol():
    """Returns the current active experiment protocol DAG."""
    if os.path.exists(PROTOCOL_FILE):
        with open(PROTOCOL_FILE, "r") as f:
            return json.load(f)
    return {"error": "Protocol file not found"}

@app.post("/api/v1/protocol/select/{protocol_id}")
async def select_protocol(protocol_id: str):
    """Switches the active experiment protocol scenario and broadcasts to all clients."""
    matched = None
    for p in AVAILABLE_PROTOCOLS:
        if p["id"] == protocol_id:
            matched = p
            break
    
    if not matched:
        raise HTTPException(status_code=404, detail=f"Protocol {protocol_id} not found")

    target_file = matched["file"]
    if not os.path.exists(target_file):
        raise HTTPException(status_code=404, detail=f"Protocol file {target_file} missing")

    with open(target_file, "r") as f:
        proto_data = json.load(f)

    # Copy to active protocol file
    with open(PROTOCOL_FILE, "w") as f:
        json.dump(proto_data, f, indent=2)

    # Reset telemetry state for new protocol
    telemetry_state["protocol_id"] = proto_data["protocol_id"]
    telemetry_state["experiment_name"] = proto_data["experiment_name"]
    telemetry_state["camera_tag"] = proto_data.get("camera_tag", matched.get("camera_tag", "CAM-01 • GLOVEBOX BIO-RACK"))
    telemetry_state["camera_channel"] = proto_data.get("camera_channel", matched.get("camera_channel", "CAM-MSG-01-A"))
    telemetry_state["camera_location"] = proto_data.get("camera_location", matched.get("camera_location", "Payload Workstation"))
    telemetry_state["current_step"] = 1
    telemetry_state["deviations_count"] = 0
    telemetry_state["completed"] = False
    telemetry_state["active_action"] = "STANDBY"
    telemetry_state["held_payload"] = None
    telemetry_state["history"] = []

    # Broadcast protocol switch event over WebSockets
    await manager.broadcast({
        "type": "PROTOCOL_SWITCH",
        "protocol": proto_data,
        "state": telemetry_state
    })

    logger.info(f"Switched active experiment protocol to: {protocol_id} ({proto_data['experiment_name']}) | Camera: {telemetry_state['camera_tag']}")
    return {"status": "PROTOCOL_SWITCH_ACK", "protocol": proto_data}

@app.get("/api/v1/telemetry/state")
async def get_telemetry_state():
    """Returns current live telemetry snapshot."""
    return telemetry_state

@app.get("/api/v1/telemetry/logs")
async def get_flight_logs(limit: int = 100):
    """Returns the latest flight audit log entries."""
    logs = []
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            lines = f.readlines()
            for line in lines[-limit:]:
                line = line.strip()
                if line:
                    try:
                        logs.append(json.loads(line))
                    except Exception:
                        pass
    return {"count": len(logs), "logs": logs}

@app.post("/api/v1/telemetry/reset")
async def reset_telemetry():
    """Resets telemetry state for a clean mission run."""
    telemetry_state["current_step"] = 1
    telemetry_state["last_deviation"] = None
    telemetry_state["active_action"] = "STANDBY"
    telemetry_state["deviations_count"] = 0
    telemetry_state["completed"] = False
    telemetry_state["history"] = []
    await manager.broadcast({"type": "RESET", "state": telemetry_state})
    return {"status": "RESET_ACK"}

last_hb_log_time = 0.0

@app.post("/api/v1/telemetry/heartbeat")
async def receive_heartbeat(request: Request):
    """Astronaut position tracking & high-speed telemetry packet (~30 FPS matching camera)."""
    global last_hb_log_time
    data = await request.json()
    now_iso = datetime.now(timezone.utc).isoformat()
    data["received_at"] = now_iso
    data["type"] = "HEARTBEAT"

    # Update cache
    telemetry_state["last_heartbeat"] = now_iso
    telemetry_state["protocol_id"] = data.get("protocol_id", telemetry_state["protocol_id"])
    telemetry_state["experiment_name"] = data.get("experiment_name", telemetry_state["experiment_name"])
    telemetry_state["camera_tag"] = data.get("camera_tag", telemetry_state.get("camera_tag"))
    telemetry_state["camera_channel"] = data.get("camera_channel", telemetry_state.get("camera_channel"))
    telemetry_state["current_step"] = data.get("current_step", telemetry_state["current_step"])
    telemetry_state["active_action"] = data.get("active_action", telemetry_state["active_action"])
    telemetry_state["active_hand"] = data.get("active_hand", telemetry_state["active_hand"])
    telemetry_state["left_hand"] = data.get("left_hand", telemetry_state["left_hand"])
    telemetry_state["right_hand"] = data.get("right_hand", telemetry_state["right_hand"])
    telemetry_state["left_leg"] = data.get("left_leg", telemetry_state.get("left_leg"))
    telemetry_state["right_leg"] = data.get("right_leg", telemetry_state.get("right_leg"))
    telemetry_state["lower_body_activity"] = data.get("lower_body_activity", telemetry_state.get("lower_body_activity", "ANCHORED IN FOOT RESTRAINT"))
    telemetry_state["held_payload"] = data.get("held_payload", telemetry_state["held_payload"])

    pose = data.get("astronaut_pose", {})
    if pose:
        telemetry_state["astronaut_pose"] = pose

    step_status = data.get("status", "NORMAL")
    if step_status == "STEP_VERIFIED":
        telemetry_state["history"].append({
            "timestamp": now_iso,
            "step": telemetry_state["current_step"],
            "event": "STEP_VERIFIED",
            "action": telemetry_state["active_action"]
        })
    elif step_status == "EXPERIMENT_SUCCESSFUL":
        telemetry_state["completed"] = True
        telemetry_state["history"].append({
            "timestamp": now_iso,
            "step": telemetry_state["current_step"],
            "event": "EXPERIMENT_SUCCESSFUL",
            "action": "COMPLETE"
        })

    now_epoch = time.time()
    if now_epoch - last_hb_log_time >= 1.0 or step_status != "NORMAL":
        kp_count = len(pose.get("keypoints", {})) if pose else 0
        roll = pose.get("body_roll_degrees", 0.0) if pose else 0.0
        vel = pose.get("wrist_velocity", 0.0) if pose else 0.0
        lower_body = telemetry_state.get("lower_body_activity", "ANCHORED")
        logger.info(f"Telemetry 30FPS: Step={data.get('current_step')} | Roll={roll:.1f}° | WristVel={vel:.2f}m/s | LowerBody={lower_body} | 3D KPs={kp_count}")
        last_hb_log_time = now_epoch
        append_flight_log(data)

    # Broadcast to live UI at full camera frame rate
    await manager.broadcast({
        "type": "HEARTBEAT",
        "data": data,
        "state": telemetry_state
    })

    return {"status": "ACK", "received_at": now_iso}

@app.post("/api/v1/telemetry/deviation")
async def receive_deviation(request: Request):
    """High-priority incident alert dispatcher."""
    alert = await request.json()
    now_iso = datetime.now(timezone.utc).isoformat()
    alert["received_at"] = now_iso
    alert["type"] = "DEVIATION"

    telemetry_state["last_deviation"] = alert
    telemetry_state["deviations_count"] += 1
    telemetry_state["current_step"] = alert.get("current_step", telemetry_state["current_step"])
    telemetry_state["active_action"] = alert.get("active_action", telemetry_state["active_action"])
    telemetry_state["active_hand"] = alert.get("active_hand", telemetry_state["active_hand"])
    telemetry_state["held_payload"] = alert.get("held_payload", telemetry_state["held_payload"])

    pose = alert.get("astronaut_pose", {})
    if pose:
        telemetry_state["astronaut_pose"] = pose

    telemetry_state["history"].append({
        "timestamp": now_iso,
        "step": alert.get("current_step"),
        "event": "DEVIATION",
        "deviation_type": alert.get("deviation_type"),
        "message": alert.get("alert_message")
    })

    append_flight_log(alert)

    logger.error("=" * 76)
    logger.error("🚨 MISSION CONTROL DEVIATION ALERT!")
    logger.error(f"Experiment ID : {alert.get('protocol_id')}")
    logger.error(f"Step Number   : {alert.get('current_step')}")
    logger.error(f"Error Type    : {alert.get('deviation_type')}")
    logger.error(f"Message       : {alert.get('alert_message')}")
    logger.error(f"Active Action : {alert.get('active_action')}")
    logger.error("=" * 76)

    # Broadcast alert immediately to web dashboard
    await manager.broadcast({
        "type": "DEVIATION",
        "data": alert,
        "state": telemetry_state
    })

    return {"status": "ALERT_PROCESSED", "acknowledged_by": "Flight_Director", "received_at": now_iso}

@app.post("/api/v1/simulate/{scenario_id}")
async def trigger_simulation_endpoint(scenario_id: str):
    """Trigger one of the 4 canonical test scenarios via REST API."""
    import subprocess
    logger.info(f"Triggering simulation scenario: {scenario_id}")
    cmd = ["python3", "simulate_scenarios.py", "--scenario", scenario_id]
    proc = subprocess.Popen(cmd)
    return {"status": "TRIGGERED", "scenario": scenario_id, "pid": proc.pid}

@app.post("/api/v1/telemetry/frame")
async def receive_camera_frame(request: Request):
    """Receives live video frame bytes from the astronaut monitor edge engine."""
    global latest_frame_bytes, latest_frame_time
    latest_frame_bytes = await request.body()
    latest_frame_time = time.time()
    return {"status": "FRAME_RECEIVED"}

@app.post("/api/v1/telemetry/camera/offline")
async def receive_camera_offline():
    """Receives camera offline notification when astronaut monitor program terminates."""
    global latest_frame_bytes, latest_frame_time
    latest_frame_bytes = None
    latest_frame_time = 0
    await manager.broadcast({
        "type": "CAMERA_OFFLINE",
        "timestamp": time.time()
    })
    return {"status": "CAMERA_OFFLINE"}

@app.get("/api/v1/stream/video")
async def live_video_stream():
    """Streams live MJPEG camera feed to the Mission Control web dashboard."""
    async def frame_generator():
        while True:
            if latest_frame_bytes is not None and (time.time() - latest_frame_time < 5.0):
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + latest_frame_bytes + b"\r\n"
                )
            await asyncio.sleep(0.04) # ~25 FPS

    return StreamingResponse(
        frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/api/v1/recordings")
async def list_recorded_sessions():
    """Lists recorded astronaut experiment session videos in telemetry/sessions."""
    sessions_dir = "telemetry/sessions"
    recs = []
    if os.path.exists(sessions_dir):
        for f in sorted(os.listdir(sessions_dir), reverse=True):
            if f.endswith((".mp4", ".avi", ".webm")):
                fpath = os.path.join(sessions_dir, f)
                recs.append({
                    "filename": f,
                    "size_bytes": os.path.getsize(fpath),
                    "size_mb": round(os.path.getsize(fpath) / (1024 * 1024), 2),
                    "created_at": datetime.fromtimestamp(os.path.getctime(fpath), timezone.utc).isoformat()
                })
    return {"count": len(recs), "recordings": recs}

@app.get("/api/v1/recordings/{filename}")
async def get_recorded_video(filename: str):
    """Serves a recorded session video file."""
    fpath = os.path.join("telemetry/sessions", filename)
    if os.path.exists(fpath):
        return FileResponse(fpath, media_type="video/mp4")
    return JSONResponse({"error": "Video file not found"}, status_code=404)

@app.websocket("/ws/telemetry")
async def websocket_telemetry_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        # Send initial snapshot immediately
        await websocket.send_json({
            "type": "INIT",
            "state": telemetry_state
        })
        while True:
            # Keep-alive receive
            _ = await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.warning(f"WebSocket error: {e}")
        manager.disconnect(websocket)

# Mount static web dashboard
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/camera")
async def serve_camera_view():
    if os.path.exists("static/camera.html"):
        return FileResponse("static/camera.html")
    return FileResponse("static/index.html")

@app.get("/")
async def serve_dashboard():
    if os.path.exists("static/index.html"):
        return FileResponse("static/index.html")
    return JSONResponse({"status": "ONLINE", "message": "Mission Control API running. UI index.html not yet built."})

if __name__ == "__main__":
    logger.info("Starting BAS Ground Mission Control Server on http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
