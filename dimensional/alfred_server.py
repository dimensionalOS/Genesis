#!/usr/bin/env python3
import os
import time
import threading
import numpy as np
import cv2
from flask import Flask, Response, request, jsonify, render_template_string
import genesis as gs
from flask_cors import CORS, cross_origin

# Optional: Uncomment this if you want to force software rendering (often needed for headless rendering)
# os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"

app = Flask(__name__)
# Enable CORS for all routes
CORS(app, resources={
    r"/*": {
        "origins": [
            "https://simulation.d1cjzkg9uefv4p.amplifyapp.com",
            "https://dimensionalos.com",
            "https://www.dimensionalos.com",
            "https://sim.dimensionalos.com",
            "http://localhost:3000",  # For local development
            "http://localhost:5000",  # For local Flask
            "*"  # Allow all origins for testing
        ],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "Access-Control-Allow-Credentials"],
        "expose_headers": ["Content-Type", "Access-Control-Allow-Credentials"],
        "supports_credentials": True,
        "max_age": 3600
    }
})

# Add CORS headers to all responses
@app.after_request
def after_request(response):
    # CORS headers are handled by flask-cors CORS(app)
    return response

# Global simulation objects and shared frame storage
scene = None
robot = None
cam = None
dofs_idx = None  # list of local DOF indices for actuated joints
latest_frame = None  # will hold the JPEG-encoded frame

#############################################
# Simulation Initialization & Control Setup
#############################################
def init_simulation():
    global scene, robot, cam, dofs_idx
    # Initialize Genesis using the GPU backend
    gs.init(backend=gs.gpu)
    
    # Create the simulation scene.
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            res=(1280, 960),
            camera_pos=(3.5, 0.0, 2.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=40,
            max_FPS=60,
        ),
        sim_options=gs.options.SimOptions(
            dt=0.01,
        ),
        show_viewer=False,  # running headless
        renderer=gs.renderers.Rasterizer(),
    )
    
    # Add a ground plane.
    _ = scene.add_entity(gs.morphs.Plane())
    
    # Load the robot from your URDF file.
    robot = scene.add_entity(
        gs.morphs.URDF(
            file='../assets/alfred_base_descr.urdf',
            pos=(0, 0, 0),
            euler=(0, 0, 0),
        )
    )
    
    # Add a headless camera for offscreen rendering.
    cam = scene.add_camera(
        res=(640, 480),
        pos=(3.5, 0.0, 2.5),
        lookat=(0, 0, 0.5),
        fov=30,
        GUI=False,
    )
    
    # Build the scene so all entities are instantiated.
    scene.build()
    
    # Define the actuated joint names (do not include any fixed joints).
    joint_names = [
        # Base actuated joints
        "right_wheel_joint",
        "left_wheel_joint",
        "pillar_platform_joint",
        "pan_joint",
        "tilt_joint",
        # Left arm actuated joints
        "L_joint1",
        "L_joint2",
        "L_joint3",
        "L_joint4",
        "L_joint5",
        "L_joint6",
        "L_jaw_1_joint",
        "L_jaw_2_joint",
        # Right arm actuated joints
        "R_joint1",
        "R_joint2",
        "R_joint3",
        "R_joint4",
        "R_joint5",
        "R_joint6",
        "R_joint7",
        "R_jaw_1_joint",
        "R_jaw_2_joint",
    ]
    
    # Map joint names to local DOF indices.
    dofs_idx = []
    valid_joint_names = []
    for name in joint_names:
        joint = robot.get_joint(name)
        if joint is not None:
            dofs_idx.append(joint.dof_idx_local)
            valid_joint_names.append(name)
        else:
            print(f"Warning: Joint '{name}' not found; it may be fixed or not exported.")
    print("Valid joints for control:", valid_joint_names)
    
    num_dofs = len(valid_joint_names)
    # Set control gains (example values; tune these for your robot)
    robot.set_dofs_kp(
        kp=np.full(num_dofs, 4500.0),
        dofs_idx_local=dofs_idx,
    )
    robot.set_dofs_kv(
        kv=np.full(num_dofs, 450.0),
        dofs_idx_local=dofs_idx,
    )
    robot.set_dofs_force_range(
        lower=np.full(num_dofs, -100.0),
        upper=np.full(num_dofs, 100.0),
        dofs_idx_local=dofs_idx,
    )

#############################################
# Simulation Loop (runs in the main thread)
#############################################
def simulation_loop():
    global scene, cam, latest_frame
    # Run an infinite loop to step the simulation and render frames in real time.
    while True:
        scene.step()
        # Render an RGB frame using the headless camera
        frame = cam.render()
        if frame is not None:
            
            try:
                # Convert frame to correct format if needed
                if isinstance(frame, (list, tuple)):
                    frame = np.array(frame[0] if len(frame) > 0 else frame)
                if not isinstance(frame, np.ndarray):
                    frame = np.array(frame)
                
                # Ensure frame has correct shape (height, width, channels)
                if len(frame.shape) != 3:
                    print(f"Unexpected frame shape: {frame.shape}")
                    continue
                    
                # Convert to uint8 if needed
                if frame.dtype != np.uint8:
                    frame = (frame * 255).astype(np.uint8)
                
                # Convert RGB to BGR (OpenCV expects BGR for JPEG encoding)
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                ret, jpeg = cv2.imencode('.jpg', frame_bgr)
                if ret:
                    latest_frame = jpeg.tobytes()
            except Exception as e:
                print(f"Error processing frame: {e}")
                
        # Sleep for the simulation timestep.
        time.sleep(scene.sf_options.dt)

#############################################
# Flask Endpoints for Video Stream & Control
#############################################
@app.route('/')
def index():
    html = """
    <html>
      <head>
        <title>Genesis Simulation Stream</title>
        <style>
            #status {
                margin-top: 10px;
                padding: 5px;
            }
            .success {
                color: green;
            }
            .error {
                color: red;
            }
        </style>
        <script>
            function sendCommand(event) {
                event.preventDefault();
                const positions = document.getElementById('joint_positions').value;
                const statusDiv = document.getElementById('status');
                
                fetch('/control', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/x-www-form-urlencoded',
                    },
                    body: 'joint_positions=' + encodeURIComponent(positions)
                })
                .then(response => response.json())
                .then(data => {
                    statusDiv.textContent = data.message;
                    statusDiv.className = data.status === 'ok' ? 'success' : 'error';
                })
                .catch(error => {
                    statusDiv.textContent = 'Error: ' + error;
                    statusDiv.className = 'error';
                });
            }
        </script>
      </head>
      <body>
        <h1>Simulation Video Feed</h1>
        <img src="/video_feed" width="640" height="480" />
        <h2>Control Robot</h2>
        <form onsubmit="sendCommand(event)">
          <label>Joint Positions (comma-separated):</label><br/>
          <input type="text" id="joint_positions" name="joint_positions" size="50"/><br/>
          <input type="submit" value="Send Command"/>
        </form>
        <div id="status"></div>
      </body>
    </html>
    """
    return html

def generate_video_stream():
    global latest_frame
    while True:
        if latest_frame is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + latest_frame + b'\r\n')
        time.sleep(0.01)

@app.route('/video_feed')
def video_feed():
    return Response(generate_video_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/control', methods=['POST', 'OPTIONS'])
@cross_origin(supports_credentials=True)
def control():
    if request.method == 'OPTIONS':
        return jsonify({})

    global robot, dofs_idx
    data = None
    if request.is_json:
        data = request.get_json()
    else:
        joint_positions_str = request.form.get('joint_positions', '')
        try:
            data = {"joint_positions": [float(x.strip()) for x in joint_positions_str.split(',') if x.strip() != '"']}
        except Exception:
            return jsonify({"status": "error", "message": "Invalid input format"}), 400

    if data and "joint_positions" in data:
        try:
            target_positions = np.array(data["joint_positions"])
            if target_positions.shape[0] != len(dofs_idx):
                return jsonify({"status": "error", "message": f"Expected {len(dofs_idx)} values, got {target_positions.shape[0]}"}), 400
            robot.control_dofs_position(target_positions, dofs_idx)
            return jsonify({"status": "ok", "message": "Command accepted"})
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500
    else:
        return jsonify({"status": "error", "message": "No joint_positions provided"}), 400

@app.route('/health')
def health_check():
    return jsonify({"status": "healthy"}), 200

#############################################
# Flask Server in Background Thread
#############################################
def run_flask():
    # Run Flask server (use a production-ready WSGI server in production)
    app.run(host='0.0.0.0', port=5000, threaded=True, debug=False)

#############################################
# Main Entry Point
#############################################
if __name__ == '__main__':
    # Initialize simulation
    init_simulation()
    # Start Flask server in a background thread
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    # Run simulation loop in the main thread so that EGL context is used on the same thread
    simulation_loop()
