import numpy as np
import genesis as gs
from scipy.spatial.transform import Rotation as R
import lcm
import threading
import time
import cv2
from lcm_msgs.geometry_msgs import Twist, Vector3
from lcm_msgs.sensor_msgs import JointState, Image, PointCloud2, CameraInfo, RegionOfInterest
from lcm_msgs.tf2_msgs import TFMessage
from lcm_msgs.std_msgs import Header

class GenesisLcmHandler:
    def __init__(self, joint_names):
        self.joint_positions = {name: 0.0 for name in joint_names}
        self.base_twist = Twist()
        self.base_twist.linear = Vector3()
        self.base_twist.angular = Vector3()
        self.lc = lcm.LCM()
        self.lcm_thread = None
        self.running = True
        
        # Subscribe to joint states and base transform
        self.lc.subscribe("joint_states#sensor_msgs.JointState", self.joint_state_callback)
        self.lc.subscribe("alfred_base_center_tf_twist#sensor_msgs.Twist", self.base_twist_callback)
        
        # Start LCM thread
        self.start_lcm_thread()
        
    def joint_state_callback(self, channel, data):
        try:
            msg = JointState.decode(data)
            for name, position in zip(msg.name, msg.position):
                if name in self.joint_positions:
                    self.joint_positions[name] = position
        except Exception as e:
            print(f"Error decoding joint state message: {e}")
    
    def base_twist_callback(self, channel, data):
        try:
            self.base_twist = Twist.decode(data)
            # Make sure we have valid Vector3 objects
            if self.base_twist.linear is None:
                self.base_twist.linear = Vector3()
            if self.base_twist.angular is None:
                self.base_twist.angular = Vector3()
        except Exception as e:
            print(f"Error decoding base twist message: {e}")
    
    def lcm_thread_func(self):
        """Thread function to handle LCM messages in the background"""
        while self.running:
            try:
                self.lc.handle_timeout(10)  # 10ms timeout
            except Exception as e:
                print(f"LCM handling error: {e}")
                time.sleep(0.001)  # Prevent CPU overuse on errors
    
    def start_lcm_thread(self):
        """Start the LCM handling thread"""
        self.lcm_thread = threading.Thread(target=self.lcm_thread_func)
        self.lcm_thread.daemon = True
        self.lcm_thread.start()
    
    def stop(self):
        """Stop the LCM handling thread"""
        self.running = False
        if self.lcm_thread is not None:
            self.lcm_thread.join(timeout=1.0)
    
    def publish_camera_info(self, width, height, channel_name, frame_id="camera_center_link", fov=55):
        """
        Publish camera intrinsic parameters as a CameraInfo message
        :param width: image width in pixels
        :param height: image height in pixels
        :param channel_name: LCM channel name to publish to
        :param frame_id: tf frame ID for the camera
        :param fov: field of view in degrees (default: 55)
        """
        try:
            
            # Calculate intrinsics exactly as Genesis does
            # From Genesis: f = 0.5 * height / np.tan(np.deg2rad(0.5 * fov))
            f = 0.5 * height / np.tan(np.deg2rad(0.5 * fov))
            cx = 0.5 * width
            cy = 0.5 * height
            
            # Create a Header
            header = Header()
            header.stamp.sec = int(time.time())
            header.stamp.nsec = int((time.time() - int(time.time())) * 1e9)
            header.frame_id = frame_id
            
            # Create RegionOfInterest object explicitly first
            roi = RegionOfInterest()
            roi.x_offset = 0
            roi.y_offset = 0
            roi.height = height
            roi.width = width
            roi.do_rectify = False
            
            # Now create the CameraInfo with the ROI already created
            camera_info = CameraInfo()
            camera_info.header = header
            camera_info.height = height
            camera_info.width = width
            
            # Set intrinsics
            camera_info.K = [
                f, 0, cx,
                0, f, cy,
                0, 0, 1
            ]
            
            # R is 3x3 rotation matrix (identity for non-stereo cameras)
            camera_info.R = [
                1, 0, 0,
                0, 1, 0,
                0, 0, 1
            ]
            
            # P is a 3x4 projection matrix in row-major order
            camera_info.P = [
                f, 0, cx, 0,
                0, f, cy, 0,
                0, 0, 1, 0
            ]
            
            # Set distortion model (none for simulator)
            camera_info.distortion_model = "plumb_bob"
            camera_info.D = [0.0, 0.0, 0.0, 0.0, 0.0]  # No distortion
            
            # Assign our pre-created ROI
            camera_info.roi = roi
            
            # Set binning (usually 0 for no binning)
            camera_info.binning_x = 0
            camera_info.binning_y = 0
            
            # Publish the message
            self.lc.publish(channel_name, camera_info.encode())
            
            # Debug
            if channel_name == "head_cam_depth_info":
                print(f"Published camera info with f={f:.1f}, cx={cx:.1f}, cy={cy:.1f}")
            
        except Exception as e:
            print(f"Error publishing camera info to {channel_name}: {e}")
            import traceback
            traceback.print_exc()
    
    def publish_image(self, image_data, channel_name, encoding="bgr8", is_depth=False):
        """
        Publish an image as an LCM message
        :param image_data: numpy array containing image data
        :param channel_name: LCM channel name to publish to
        :param encoding: image encoding (bgr8, mono16, etc.)
        :param is_depth: whether this is a depth image
        """
        try:
            if image_data is None:
                return
            
            # Create a copy of the data to avoid modifying the original
            image = image_data.copy()
            
            # Create Image message
            img_msg = Image()
            
            # Create and set header
            img_msg.header = Header()
            img_msg.header.stamp.sec = int(time.time())
            img_msg.header.stamp.nsec = int((time.time() - img_msg.header.stamp.sec) * 1e9)
            img_msg.header.frame_id = "camera_center_link"
            
            # Get dimensions
            height, width = image.shape[:2]
            img_msg.height = height
            img_msg.width = width
            
            # Handle depth images
            if is_depth:
                # Convert depth to uint16 (mm precision)
                if image.dtype != np.uint16:
                    depth_mm = (image * 1000.0).astype(np.uint16)
                    img_data = depth_mm.tobytes()
                    encoding = "mono16"  # 16-bit depth
                    step = width * 2  # 2 bytes per pixel
                else:
                    img_data = image.tobytes()
                    step = width * 2
            else:
                # For color images
                if len(image.shape) == 3 and image.shape[2] == 3:
                    # Ensure 8-bit format (0-255)
                    if image.dtype != np.uint8:
                        if image.max() <= 1.0:
                            # Scale 0-1 to 0-255
                            image = (image * 255.0).astype(np.uint8)
                        else:
                            # Already in appropriate range
                            image = image.astype(np.uint8)
                    
                    # Always assume input is RGB from Genesis
                    if encoding == "bgr8":
                        # Convert RGB → BGR if using bgr8 encoding
                        # Simpler method: swap R and B channels directly
                        temp = image[:,:,0].copy()  # Store R channel
                        image[:,:,0] = image[:,:,2]  # R = B
                        image[:,:,2] = temp          # B = R
                    
                    # Ensure the data is contiguous in memory before converting to bytes
                    image = np.ascontiguousarray(image)
                    img_data = image.tobytes()
                    step = width * 3  # 3 bytes per pixel
                else:
                    # Grayscale image
                    if image.dtype != np.uint8:
                        if image.max() <= 1.0:
                            image = (image * 255.0).astype(np.uint8)
                        else:
                            image = image.astype(np.uint8)
                    
                    # Ensure contiguous memory
                    image = np.ascontiguousarray(image)
                    img_data = image.tobytes()
                    step = width  # 1 byte per pixel for grayscale
            
            # Set remaining image properties
            img_msg.encoding = encoding
            img_msg.is_bigendian = 0  # Almost always little endian on modern systems
            img_msg.step = step
            img_msg.data_length = len(img_data)
            img_msg.data = img_data
            
            # Publish
            self.lc.publish(channel_name, img_msg.encode())
            
        except Exception as e:
            print(f"Error publishing image to {channel_name}: {e}")
            import traceback
            traceback.print_exc()


def main():
    ########################## Initialization ##########################
    ########################## Joint Mapping ##########################
    # Define the list of controllable (actuated) joint names from the devkit_base_descr.urdf
    joint_names = [
        # Base joints
        "pillar_platform_joint",  # type: prismatic
        "pan_tilt_pan_joint",     # type: revolute
        "pan_tilt_head_joint",    # type: revolute
        
        # Arm joints
        "joint1",  # revolute
        "joint2",  # revolute
        "joint3",  # revolute
        "joint4",  # revolute
        "joint5",  # revolute
        "joint6",  # revolute
        "joint7",  # prismatic (gripper)
        "joint8",  # prismatic (gripper)
    ]
    
    # Initialize the LCM handler first
    genesis_lcm_handler = GenesisLcmHandler(joint_names)
    
    # Give some time for initial LCM connections
    print("Waiting for initial LCM messages...")
    time.sleep(1.0)

    # Initialize Genesis with the GPU backend.
    try:
        gs.init(backend=gs.gpu)
    except Exception as e:
        print(f"Error initializing Genesis GPU backend: {e}")
        print("Falling back to CPU backend")
        gs.init()

    # Create the simulation scene.
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos    = (0, -5, 2),
            camera_lookat = (0, 0, 0.5),
            camera_fov    = 60,
            res           = (1280, 960),
            max_FPS       = 60,
        ),
        sim_options=gs.options.SimOptions(
            dt = 0.01,
        ),
        rigid_options=gs.options.RigidOptions(
            enable_self_collision = True
        ),
        show_viewer=True,
    )

    ########################## Entities ##########################
    # Add a ground plane.
    plane = scene.add_entity(gs.morphs.Plane())

    # Load the robot from the URDF file.
    robot = scene.add_entity(
        gs.morphs.URDF(
            file  = '../assets/devkit_base_descr.urdf',
            pos   = (0, 0, 0),
            euler = (0, 0, 0),
            links_to_keep = ["camera_center_link"]
        )
        )

    box = scene.add_entity(gs.morphs.Box(pos=(2, 0, 2), size=(1, 1, 1)))

    ########################## Robot Camera ##########################
    # Add a camera that will be mounted on the pan-tilt mechanism
    head_cam = scene.add_camera(
        res = (1280, 800),  # Camera resolution
        pos = (0, 0, 0),    # Will be updated based on pan_tilt_head position
        lookat = (0, 0, 0), # Will be updated to look forward from the head
        fov = 55,           # Field of view
        GUI = False,         # Show this camera feed in a window
    )

    # Define camera offset from pan_tilt_head link origin (in local coordinates)
    # Moving the camera slightly up in the Z direction
    camera_offset = np.array([0.1, 0.0, 0.075])  # 10cm forward, 7.5cm up

    # Build the scene.
    scene.build()

    # Map the joint names to the robot's local degree-of-freedom (DOF) indices.
    dofs_idx = []
    valid_joint_names = []
    joint_index_map = {}  # Map joint names to their indices for easy lookup
    
    for i, name in enumerate(joint_names):
        joint = robot.get_joint(name)
        if joint is not None:
            dofs_idx.append(joint.dof_idx_local)
            valid_joint_names.append(name)
            joint_index_map[name] = len(valid_joint_names) - 1
        else:
            print(f"Warning: Joint '{name}' not found; it may be fixed or not exported.")

    print("Valid joint names for control:", valid_joint_names)

    ########################## Set Control Gains ##########################
    num_dofs = len(valid_joint_names)

    # Control gain values - adjust as needed for the devkit
    kp = np.array([
        10000,  # pillar_platform_joint
        250,    # pan_tilt_pan_joint
        250,    # pan_tilt_head_joint
        1000,   # joint1
        1000,   # joint2
        500,    # joint3
        500,    # joint4
        250,    # joint5
        250,    # joint6
        100,    # joint7 (gripper)
        100,    # joint8 (gripper)
    ])

    kv = np.array([
        1000,   # pillar_platform_joint
        10,     # pan_tilt_pan_joint
        10,     # pan_tilt_head_joint
        10,     # joint1
        10,     # joint2
        5,      # joint3
        5,      # joint4
        2.5,    # joint5
        2.5,    # joint6
        5,      # joint7 (gripper)
        5,      # joint8 (gripper)
    ])

    force_upper = np.array([
        100000, # pillar_platform_joint
        250,    # pan_tilt_pan_joint
        250,    # pan_tilt_head_joint
        500,    # joint1
        500,    # joint2
        250,    # joint3
        250,    # joint4
        250,    # joint5
        250,    # joint6
        100,    # joint7 (gripper)
        100,    # joint8 (gripper)
    ])

    force_lower = -force_upper

    # Ensure arrays match the number of valid joints
    if num_dofs > 0:
        kp = kp[:num_dofs]
        kv = kv[:num_dofs]
        force_upper = force_upper[:num_dofs]
        force_lower = force_lower[:num_dofs]

        robot.set_dofs_kp(
            kp             = kp,
            dofs_idx_local = dofs_idx,
        )
        robot.set_dofs_kv(
            kv             = kv,
            dofs_idx_local = dofs_idx,
        )
        robot.set_dofs_force_range(
            lower          = force_lower,
            upper          = force_upper,
            dofs_idx_local = dofs_idx,
        )
    else:
        print("WARNING: No valid joints found for control!")

    ########################## Control Loop ##########################
    target_positions = np.zeros(num_dofs)
    
    # Give the simulation time to initialize fully
    for step in range(10):
        scene.step()
    
    # Define fixed camera orientations for use in case of quaternion errors
    default_forward_direction = np.array([1.0, 0.0, 0.0])
    default_up_direction = np.array([0.0, 0.0, 1.0])
    
    # Set up LCM publishing rate limiters
    last_image_publish_time = 0
    image_publish_interval = 0.1  # Publish images at 10 Hz
    
    try:
        for step in range(10000):
            target_positions = np.array([genesis_lcm_handler.joint_positions.get(name, 0.0) for name in valid_joint_names])
            # target_positions = np.array([0.0 for name in valid_joint_names])
            if step < 100:
                # Initialize positions during startup
                if num_dofs > 0:
                    robot.set_dofs_position(target_positions, dofs_idx)
                robot.set_pos([0.0, 0.0, 0.0])
                quat = R.from_euler('zyx', [0.0, 0.0, 0.0], degrees=True).as_quat()
                robot.set_quat(quat)
                scene.step()
                continue
                
            # Get joint positions from LCM messages
            if num_dofs > 0:
                # robot.set_dofs_position(target_positions, dofs_idx)
                robot.control_dofs_position(target_positions, dofs_idx)

            # Set robot base pose to origin (can be updated to use LCM data)
            robot.set_pos([0.0, 0.0, 0.0])
            quat = R.from_euler('zyx', [0.0, 0.0, 0.0], degrees=True).as_quat()
            robot.set_quat(quat)
            
            # Update the camera position and orientation based on the pan_tilt_head link
            rgb_image = None
            depth_image = None
            
            try:
                # Get the current position of the pan_tilt_head link
                link = robot.get_link('camera_center_link')
                if link is None:
                    print("Warning: camera_center_link link not found")
                    continue
                    
                # Get head position
                head_pos = link.get_pos().cpu().numpy()
                
                # Calculate the camera position with offset, even if we can't use the link's quaternion
                # Apply a simple offset in world coordinates for now
                # camera_pos = head_pos + camera_offset
                camera_pos = head_pos
                
                # Get current joint angles directly from the target positions
                pan_angle = 0.0
                tilt_angle = 0.0
                
                if 'pan_tilt_pan_joint' in joint_index_map:
                    pan_idx = joint_index_map['pan_tilt_pan_joint']
                    pan_angle = target_positions[pan_idx]
                    
                if 'pan_tilt_head_joint' in joint_index_map:
                    tilt_idx = joint_index_map['pan_tilt_head_joint']
                    tilt_angle = -target_positions[tilt_idx]  # Negative for correct tilt direction
                
                # Calculate camera orientation directly from joint angles
                # This is more reliable than using the link's quaternion
                
                # Pan is rotation around Z axis, tilt is rotation around Y axis in the URDF
                # First create the pan rotation
                pan_rot = R.from_euler('z', pan_angle)
                
                # Calculate the base forward and up vectors
                initial_forward = np.array([1.0, 0.0, 0.0])  # Forward along X axis
                initial_up = np.array([0.0, 0.0, 1.0])       # Up along Z axis
                
                # Apply pan rotation
                forward_after_pan = pan_rot.apply(initial_forward)
                up_after_pan = pan_rot.apply(initial_up)
                
                # Get the local Y axis after pan to create the tilt rotation
                local_y_after_pan = pan_rot.apply([0, 1, 0])
                
                # Simple sanity check for the local Y axis
                if np.linalg.norm(local_y_after_pan) < 1e-6:
                    local_y_after_pan = np.array([0, 1, 0])  # Fallback to world Y axis
                
                # Create rotation matrix for tilt around local Y
                # Normalize the axis to ensure valid rotation
                local_y_after_pan = local_y_after_pan / np.linalg.norm(local_y_after_pan)
                tilt_rot = R.from_rotvec(tilt_angle * local_y_after_pan)
                
                # Apply tilt rotation
                final_forward = tilt_rot.apply(forward_after_pan)
                final_up = tilt_rot.apply(up_after_pan)
                
                # Normalize for safety
                final_forward = final_forward / np.linalg.norm(final_forward)
                final_up = final_up / np.linalg.norm(final_up)
                
                # Set the camera lookat point
                camera_lookat = camera_pos + 5.0 * final_forward
                
                # Debug output
                if step % 100 == 0:
                    print(f"Pan: {pan_angle:.2f}, Tilt: {tilt_angle:.2f}")
                    print(f"Camera pos: {camera_pos}")
                    print(f"Forward: {final_forward}, Up: {final_up}")
                
                # Update camera pose
                head_cam.set_pose(pos=camera_pos, lookat=camera_lookat, up=final_up)
                
                # Render the camera
                rgb_image, depth_image, _, _ = head_cam.render(depth=True)
                
                # Publish camera images via LCM at specified rate
                current_time = time.time()
                if current_time - last_image_publish_time >= image_publish_interval:
                    # Convert RGB tensor to numpy array if needed
                    if rgb_image is not None:
                        # Check if we need to convert from tensor to numpy
                        rgb_np = rgb_image
                        if hasattr(rgb_image, 'cpu'):
                            rgb_np = rgb_image.cpu().numpy()
                        
                        # Debug info for every 100th frame
                        if step % 100 == 0:
                            print(f"RGB image shape: {rgb_np.shape}")
                            if rgb_np.shape[0] > 10 and rgb_np.shape[1] > 10:
                                print(f"Sample RGB pixel values (10,10): {rgb_np[10,10]}")
                        
                        # Get image dimensions for camera info
                        height, width = rgb_np.shape[:2]
                        
                        # Publish camera info for RGB camera with the correct FOV (60 degrees from Genesis camera config)
                        genesis_lcm_handler.publish_camera_info(width, height, "head_cam_rgb_info#sensor_msgs.CameraInfo", fov=60)
                        
                        # Try publishing as RGB format instead of BGR
                        # Maybe Foxglove is misinterpreting the encoding
                        genesis_lcm_handler.publish_image(rgb_np, "head_cam_rgb#sensor_msgs.Image", encoding="rgb8")
                    
                    # Convert depth tensor to numpy array if needed
                    if depth_image is not None:
                        # Check if we need to convert from tensor to numpy
                        depth_np = depth_image
                        if hasattr(depth_image, 'cpu'):
                            depth_np = depth_image.cpu().numpy()
                        
                        # Get image dimensions for camera info
                        height, width = depth_np.shape[:2]
                        
                        # Publish camera info for depth camera with the correct FOV (60 degrees from Genesis camera config)
                        genesis_lcm_handler.publish_camera_info(width, height, "head_cam_depth_info#sensor_msgs.CameraInfo", fov=60)
                        
                        # Publish depth image (already in meters, keep as float32)
                        genesis_lcm_handler.publish_image(depth_np, "head_cam_depth#sensor_msgs.Image", encoding="32FC1", is_depth=True)
                    
                    last_image_publish_time = current_time
                
            except Exception as e:
                print(f"Error updating camera: {e}")
                import traceback
                traceback.print_exc()  # Print the full stack trace for better debugging
                
                # Fallback to default camera pose
                try:
                    head_cam.set_pose(
                        pos=head_pos + camera_offset,
                        lookat=head_pos + camera_offset + default_forward_direction,
                        up=default_up_direction
                    )
                except Exception as fallback_error:
                    print(f"Fallback camera update failed: {fallback_error}")
            
            # Step the simulation
            scene.step()
    except KeyboardInterrupt:
        print("Interrupted by user. Shutting down...")
    except Exception as e:
        print(f"Error in simulation loop: {e}")
        import traceback
        traceback.print_exc()  # Print the full stack trace for better debugging
    finally:
        # Clean up
        genesis_lcm_handler.stop()
        print("LCM handler stopped.")

if __name__ == "__main__":
    main()