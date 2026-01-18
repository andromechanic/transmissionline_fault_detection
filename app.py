from flask import Flask, jsonify, send_from_directory, request, Response
from ultralytics import YOLO
import cv2
import time
import threading
import serial
import serial.tools.list_ports
import os
import numpy as np

app = Flask(__name__)

# ===============================
# LOAD YOLO MODEL
# ===============================
print("Loading YOLO model...")
model = YOLO("best.pt")
print("YOLO model loaded")

# ===============================
# SERIAL STATE
# ===============================
ser = None
connected_port = None
BAUD = 9600

# ===============================
# SERIAL HELPERS
# ===============================
def list_ports():
    return [
        {"device": p.device, "description": p.description}
        for p in serial.tools.list_ports.comports()
    ]

def connect_serial(port):
    global ser, connected_port
    try:
        ser = serial.Serial(port, BAUD, timeout=1)
        connected_port = port
        return True, f"Connected to {port}"
    except Exception as e:
        ser = None
        connected_port = None
        return False, str(e)

def send_serial(cmd):
    if ser and ser.is_open:
        try:
            ser.write(cmd.encode())
        except Exception as e:
            print("Serial write error:", e)
    else:
        print("SERIAL (fallback):", cmd)

# ===============================
# GLOBAL STATE (shared with UI)
# ===============================
last_result = {
    "hasFault": False,
    "detections": [],
    "inferenceTime": 0,
    "timestamp": time.time()
}

# ESP8266 electrical data - now stores data per device ID
esp_data = {
    "current": 0.0,
    "voltage": 0.0,
    "temp": 0.0,
    "timestamp": time.time()
}

# Store data for each device
device_data = {}

alert_active = False
frame_lock = threading.Lock()
current_frame = None
annotated_frame = None

# Track connected ESP8266 devices
esp_clients = {}

# Store fault status per device
device_fault_status = {}

# ===============================
# WEBCAM MONITOR THREAD
# ===============================
def webcam_monitor():
    global alert_active, last_result, current_frame, annotated_frame

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("❌ ERROR: Webcam not accessible")
        return

    print("📷 Webcam monitoring started")

    while True:
        try:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("⚠ Frame grab failed, retrying...")
                time.sleep(0.5)
                continue

            with frame_lock:
                current_frame = frame.copy()

            start = time.time()

            # Run inference
            results = model(frame, conf=0.5, verbose=False)[0]

            # Create annotated frame
            annotated = results.plot()
            with frame_lock:
                annotated_frame = annotated.copy()

            detections = []
            has_fault = False
            max_conf = 0

            for box in results.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = model.names[cls_id]
                
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                detections.append({
                    "class": label,
                    "confidence": round(conf, 3),
                    "bbox": {
                        "x1": round(x1, 1),
                        "y1": round(y1, 1),
                        "x2": round(x2, 1),
                        "y2": round(y2, 1)
                    }
                })

                if label.lower() == "fault":
                    has_fault = True
                    max_conf = max(max_conf, conf)

            last_result = {
                "hasFault": has_fault,
                "detections": detections,
                "inferenceTime": round((time.time() - start) * 1000, 2),
                "timestamp": time.time()
            }

            if has_fault and not alert_active:
                print("🚨 FAULT DETECTED — ALERT ACTIVE")
                alert_active = True

                send_serial(f"*1{int(max_conf * 100)}")
                
                # Keep alert active for 5 seconds
                threading.Timer(5.0, clear_alert).start()

            time.sleep(0.1)  # ~10 FPS inference

        except Exception as e:
            print("🔥 Webcam loop error:", e)
            time.sleep(1)

def clear_alert():
    global alert_active
    send_serial("*100")
    alert_active = False
    print("✅ Alert cleared — monitoring resumed")

# ===============================
# VIDEO STREAM GENERATOR
# ===============================
def generate_frames():
    """Generate frames for video streaming with YOLO annotations"""
    while True:
        with frame_lock:
            if annotated_frame is None:
                time.sleep(0.1)
                continue
            frame = annotated_frame.copy()
        
        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ret:
            continue
            
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        time.sleep(0.033)  # ~30 FPS stream

# ===============================
# ROUTES
# ===============================
@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/video_feed")
def video_feed():
    """Route for streaming annotated video frames"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/serial/ports")
def serial_ports():
    return jsonify({
        "ports": list_ports(),
        "connected": connected_port
    })

@app.route("/serial/connect", methods=["POST"])
def serial_connect():
    port = request.json.get("port")
    success, msg = connect_serial(port)
    return jsonify({
        "success": success,
        "message": msg,
        "connected": connected_port
    })

@app.route("/status")
def status():
    """Main status endpoint for both web UI and ESP8266"""
    return jsonify({
        **last_result,
        "alert": alert_active,
        "serial": connected_port,
        "espData": esp_data
    })

@app.route("/esp/data", methods=["POST"])
def esp_data_update():
    """Legacy endpoint - kept for backward compatibility"""
    global esp_data

    try:
        if not request.data:
            return jsonify({"success": False, "error": "Empty request body"}), 400

        data = request.get_json(force=True, silent=True)

        if not isinstance(data, dict):
            return jsonify({"success": False, "error": "Invalid JSON"}), 400

        current = float(data.get("current", 0))
        voltage = float(data.get("voltage", 0))
        temp = float(data.get("temp", 0))

        esp_data = {
            "current": current,
            "voltage": voltage,
            "temp": temp,
            "timestamp": time.time()
        }

        print(f"ESP Data | I={current}A V={voltage}V T={temp}°C")

        return jsonify({
            "success": True,
            "message": "ESP data received",
            "data": esp_data
        })

    except Exception as e:
        print("ESP data error:", e)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/register", methods=["POST"])
def esp_register():
    """ESP8266 device registration endpoint - Arduino compatible"""
    try:
        data = request.get_json(force=True, silent=True)
        if not data:
            return jsonify({"success": False, "error": "Invalid JSON"}), 400
            
        device_id = data.get("device_id", "unknown")
        device_ip = request.remote_addr
        
        esp_clients[device_id] = {
            "ip": device_ip,
            "last_seen": time.time()
        }
        
        # Initialize device data if not exists
        if device_id not in device_data:
            device_data[device_id] = {
                "irms": "0.00",
                "voltage": "0.00",
                "temp": "0.00",
                "timestamp": time.time()
            }
        
        # Initialize fault status
        if device_id not in device_fault_status:
            device_fault_status[device_id] = {
                "fault": False,
                "message": "OK"
            }
        
        print(f"ESP8266 registered: {device_id} from {device_ip}")
        
        return jsonify({
            "success": True,
            "message": "Device registered",
            "server_time": time.time()
        })
    except Exception as e:
        print(f"Registration error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/update_data", methods=["POST"])
def update_data():
    """Arduino-compatible endpoint for receiving electrical data"""
    global esp_data
    
    try:
        data = request.get_json(force=True, silent=True)
        if not data:
            return jsonify({"success": False, "error": "Invalid JSON"}), 400
        
        device_id = str(data.get("id", "1"))
        irms = data.get("irms", "0.00")
        voltage = data.get("voltage", "0.00")
        temp = data.get("temp", "0.00")
        
        # Store device-specific data
        device_data[device_id] = {
            "irms": irms,
            "voltage": voltage,
            "temp": temp,
            "timestamp": time.time()
        }
        
        # Update global esp_data for primary device (ID 1) for UI compatibility
        if device_id == "1":
            try:
                # Parse values, handling potential format issues
                current_val = float(irms.strip())
                voltage_val = float(voltage.strip())
                temp_val = float(temp.strip())
                
                esp_data = {
                    "current": current_val,
                    "voltage": voltage_val,
                    "temp": temp_val,
                    "timestamp": time.time()
                }
            except ValueError:
                print(f"Warning: Could not parse values - I:{irms} V:{voltage} T:{temp}")
        
        # Update last seen time
        if device_id in esp_clients:
            esp_clients[device_id]["last_seen"] = time.time()
        
        print(f"Device {device_id} Data | I={irms}A V={voltage}V T={temp}°C")
        
        return jsonify({
            "success": True,
            "message": "Data updated",
            "device_id": device_id
        })
        
    except Exception as e:
        print(f"Update data error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/get_status")
def get_status():
    """Arduino-compatible endpoint for polling fault status"""
    device_id = request.args.get("id", "1")
    
    # Check if there's an active fault
    if last_result["hasFault"] and alert_active:
        fault_class = ""
        fault_conf = 0
        
        if last_result["detections"]:
            # Find the fault detection
            for det in last_result["detections"]:
                if det["class"].lower() == "fault":
                    fault_class = det["class"]
                    fault_conf = det["confidence"]
                    break
        
        response_msg = f"FAULT {fault_class} {int(fault_conf * 100)}%"
        
        # Store fault status for this device
        device_fault_status[device_id] = {
            "fault": True,
            "message": response_msg
        }
        
        return response_msg, 200
    else:
        # No fault detected
        device_fault_status[device_id] = {
            "fault": False,
            "message": "OK"
        }
        return "OK", 200


@app.route("/esp/status")
def esp_status():
    """Simplified status endpoint optimized for ESP8266"""
    return jsonify({
        "fault": last_result["hasFault"],
        "alert": alert_active,
        "confidence": last_result["detections"][0]["confidence"] if last_result["detections"] else 0,
        "class": last_result["detections"][0]["class"] if last_result["detections"] else "",
        "time": int(time.time())
    })

@app.route("/esp/clients")
def esp_clients_list():
    """List all registered ESP8266 devices"""
    active_clients = {
        k: v for k, v in esp_clients.items()
        if time.time() - v["last_seen"] < 300  # Active in last 5 minutes
    }
    return jsonify({
        "clients": active_clients,
        "count": len(active_clients)
    })

@app.route("/devices")
def devices_list():
    """List all devices with their current data"""
    devices = []
    for device_id, data in device_data.items():
        device_info = {
            "id": device_id,
            "data": data,
            "fault_status": device_fault_status.get(device_id, {"fault": False, "message": "OK"}),
            "client_info": esp_clients.get(device_id, None)
        }
        devices.append(device_info)
    
    return jsonify({
        "devices": devices,
        "count": len(devices)
    })

# ===============================
# START SERVER
# ===============================
if __name__ == "__main__":
    t = threading.Thread(target=webcam_monitor, daemon=True)
    t.start()
    
    # Print server IP for easy ESP8266 configuration
    import socket
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    print(f"\n{'='*50}")
    print(f" Flask server starting on: {local_ip}:5000")
    print(f" Arduino endpoints available:")
    print(f"   - POST /register")
    print(f"   - POST /update_data")
    print(f"   - GET  /get_status?id=<device_id>")
    print(f"{'='*50}\n")
    
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)

    