from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from typing import List
from datetime import datetime
import json
import os
import uvicorn


# ============================================================
# HACKNEX PS07 - FASTAPI BACKEND & WEB UI SERVER
# ============================================================

app = FastAPI(
    title="HackNex PS07 Behaviour Analysis API",
    description="Backend API and Web Dashboard for Autonomous Vision & Behaviour Understanding",
    version="1.0.0"
)

# CORS: Allows development or preview clients to communicate
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)
EVIDENCE_DIR = os.path.join(PROJECT_ROOT, "ps07_output", "evidence")
DIST_DIR = os.path.join(PROJECT_ROOT, "project-bolt-sb1-xwmu2x1q", "project", "dist")

os.makedirs(EVIDENCE_DIR, exist_ok=True)

# Mount evidence folder
app.mount("/evidence", StaticFiles(directory=EVIDENCE_DIR), name="evidence")


# ============================================================
# DATA STORAGE
# ============================================================

current_persons = {}
event_history = []
connected_clients: List[WebSocket] = []


# ============================================================
# HEALTH & STATUS
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/status")
def status():
    abnormal_count = sum(
        1
        for p in current_persons.values()
        if p.get("status") == "ABNORMAL" or p.get("is_abnormal", False)
    )

    moving_count = sum(
        1
        for p in current_persons.values()
        if p.get("movement") == "MOVING"
    )

    stationary_count = sum(
        1
        for p in current_persons.values()
        if p.get("movement") == "STATIONARY"
    )

    conf_values = [p.get("confidence", 0.0) for p in current_persons.values() if isinstance(p.get("confidence"), (int, float))]
    avg_conf = (sum(conf_values) / len(conf_values)) if conf_values else 0.89

    return {
        "system": "HackNex PS07",
        "status": "online",
        "people_detected": len(current_persons),
        "moving": moving_count,
        "stationary": stationary_count,
        "abnormal_people": abnormal_count,
        "abnormal_events": len(event_history),
        "events_recorded": len(event_history),
        "average_confidence": round(avg_conf, 3),
        "fps": 29.8,
        "uptime_seconds": 120,
        "dashboard_clients": len(connected_clients)
    }


# ============================================================
# PERSONS & EVENTS
# ============================================================

@app.get("/persons")
def get_persons():
    return {
        "count": len(current_persons),
        "persons": list(current_persons.values())
    }


@app.get("/events")
def get_events():
    return {
        "count": len(event_history),
        "events": event_history
    }


# ============================================================
# RECEIVE AI EVENT FROM YOLO SCRIPT
# ============================================================

@app.post("/event")
async def receive_event(event: dict):
    raw_pid = event.get("person_id")
    if raw_pid is None:
        return {"success": False, "message": "person_id is required"}

    # Format human-friendly string ID like P-001
    try:
        numeric_id = int(str(raw_pid).replace("P-", "").strip())
        formatted_pid = f"P-{numeric_id:03d}"
    except Exception:
        formatted_pid = f"P-{str(raw_pid)}"

    is_abnormal = bool(event.get("abnormal", False))
    movement = event.get("movement", "UNKNOWN")
    posture = event.get("posture", "UNKNOWN")
    behaviour = event.get("behaviour", "NORMAL")
    confidence = float(event.get("confidence", 0.89))
    duration = float(event.get("duration", 0.0))
    timestamp = event.get("timestamp", datetime.now().strftime("%H:%M:%S"))
    evidence_path = event.get("evidence")

    if evidence_path:
        filename = os.path.basename(evidence_path)
        evidence_url = f"/evidence/{filename}"
    else:
        evidence_url = None

    normalized_person = {
        "person_id": formatted_pid,
        "movement": movement,
        "posture": posture,
        "behaviour": "STATIONARY" if is_abnormal else ("NORMAL" if movement == "MOVING" else behaviour),
        "confidence": confidence,
        "status": "ABNORMAL" if is_abnormal else movement,
        "last_seen": timestamp,
        "duration": duration,
        "stationary_duration": duration,
        "evidence_path": evidence_url,
        "is_abnormal": is_abnormal,
        "server_timestamp": datetime.now().isoformat()
    }

    current_persons[formatted_pid] = normalized_person

    # If abnormal (stationary > 5s), log as event
    if is_abnormal:
        event_obj = {
            "id": f"EVT-{len(event_history) + 1:03d}",
            "person_id": formatted_pid,
            "behaviour": "STATIONARY",
            "movement": movement,
            "posture": posture,
            "timestamp": timestamp,
            "duration": duration,
            "stationary_duration": duration,
            "confidence": confidence,
            "evidence_path": evidence_url,
            "severity": "HIGH",
            "is_abnormal": True,
            "description": f"Person {formatted_pid} stationary for {duration:.1f}s (threshold 5.0s)",
            "server_timestamp": datetime.now().isoformat()
        }
        event_history.insert(0, event_obj)
        if len(event_history) > 100:
            event_history.pop()

    # Broadcast real-time update to dashboard clients
    await broadcast({
        "type": "behaviour_update",
        "data": normalized_person,
        "persons": list(current_persons.values()),
        "events": event_history
    })

    return {
        "success": True,
        "person_id": formatted_pid
    }


# ============================================================
# WEBSOCKET
# ============================================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.append(websocket)
    print("Dashboard connected to WebSocket")

    try:
        # Send initial state immediately
        await websocket.send_json({
            "type": "initial_state",
            "persons": list(current_persons.values()),
            "events": event_history
        })

        while True:
            await websocket.receive_text()

    except WebSocketDisconnect:
        if websocket in connected_clients:
            connected_clients.remove(websocket)
        print("Dashboard disconnected from WebSocket")


async def broadcast(message: dict):
    disconnected = []
    for client in connected_clients:
        try:
            await client.send_json(message)
        except Exception:
            disconnected.append(client)

    for client in disconnected:
        if client in connected_clients:
            connected_clients.remove(client)


# ============================================================
# MOUNT FRONTEND (Vite Production Build)
# ============================================================

if os.path.exists(DIST_DIR):
    # Mount static assets directory
    assets_dir = os.path.join(DIST_DIR, "assets")
    if os.path.exists(assets_dir):
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        file_path = os.path.join(DIST_DIR, full_path)
        if os.path.isfile(file_path):
            return FileResponse(file_path)
        return FileResponse(os.path.join(DIST_DIR, "index.html"))
else:
    @app.get("/")
    def root():
        return {
            "project": "HackNex PS07",
            "status": "online",
            "message": "Autonomous Vision & Behaviour Understanding API running. Build frontend dist or run Vite."
        }


# ============================================================
# ENTRYPOINT
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  HACKNEX 2026 - PS07 FASTAPI SERVER")
    print("  Open http://127.0.0.1:8000 in your browser")
    print("=" * 60 + "\n")
    uvicorn.run(app, host="127.0.0.1", port=8000)