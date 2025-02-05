import numpy as np
import genesis as gs
from scipy.spatial.transform import Rotation as R
import time
########################## Initialization ##########################
# Initialize Genesis with the GPU backend.
gs.init(backend=gs.gpu)

# Create the simulation scene.
scene = gs.Scene(
    viewer_options=gs.options.ViewerOptions(
        camera_pos    = (0, -5, 2),
        camera_lookat = (0, 0, 0.5),
        camera_fov    = 30,
        res           = (960, 640),
        max_FPS       = 60,
    ),
    sim_options=gs.options.SimOptions(
        dt = 0.01,
    ),
    show_viewer=True,
)

########################## Entities ##########################
# Add a ground plane.
plane = scene.add_entity(gs.morphs.Plane())

# Load the robot from the URDF file.
# (Make sure the file '/assets/alfred_base_descr.urdf' exists and is accessible.)
robot = scene.add_entity(
    gs.morphs.URDF(
        file  = '../assets/alfred_base_descr.urdf',
        pos   = (0, 0, 0),
        euler = (0, 0, 0),
        links_to_keep = ['head_cam_rgb_camera_frame', 'chest_cam_rgb_camera_frame']
    )
)

box = scene.add_entity(gs.morphs.Box(pos=(2, 0, 2), size=(1, 1, 1)))

########################## Robot Cameras ##########################
# head_cam_pos = robot.get_link('head_cam_rgb_camera_frame').get_pos()
# chest_cam_pos = robot.get_link('chest_cam_rgb_camera_frame').get_pos()
head_cam = scene.add_camera(
    res = (1280, 800),
    pos = (0, 0, 0),
    lookat = (0, 0, 0),
    fov = 60,
    GUI = False,
)
chest_cam = scene.add_camera(
    res = (1280, 800),
    pos = (0, 0, 0),
    lookat = (0, 0, 0),
    fov = 60,
    GUI = False,
)

# Build the scene.
scene.build()

########################## Joint Mapping ##########################
# Define the list of controllable (actuated) joint names.
# We exclude any joints that are fixed.
joint_names = [
    # Base actuated joints (continuous or prismatic)
    "right_wheel_joint",       # type: continuous
    "left_wheel_joint",        # type: continuous
    "pillar_platform_joint",   # type: prismatic
    "pan_joint",               # type: revolute
    "tilt_joint",              # type: revolute
    # (laser_joint is fixed and is not included)

    # Left arm actuated joints (from the ros2_control block)
    "L_joint1",  # revolute
    "L_joint2",  # revolute
    "L_joint3",  # revolute
    "L_joint4",  # revolute
    "L_joint5",  # revolute
    "L_joint6",  # revolute
    # Exclude L_joint_eef and L_gripper_base_joint (both fixed)
    "L_jaw_1_joint",  # prismatic (actuated)
    "L_jaw_2_joint",  # prismatic (actuated; mimic joint)

    # Right arm actuated joints (from the ros2_control block)
    "R_joint1",  # revolute
    "R_joint2",  # revolute
    "R_joint3",  # revolute
    "R_joint4",  # revolute
    "R_joint5",  # revolute
    "R_joint6",  # revolute
    "R_joint7",  # revolute
    # Exclude R_joint_eef and R_gripper_base_joint (both fixed)
    "R_jaw_1_joint",  # prismatic (actuated)
    "R_jaw_2_joint",  # prismatic (actuated; mimic joint)
]

# Map the joint names to the robot's local degree-of-freedom (DOF) indices.
dofs_idx = []
valid_joint_names = []
for name in joint_names:
    joint = robot.get_joint(name)
    if joint is not None:
        dofs_idx.append(joint.dof_idx_local)
        valid_joint_names.append(name)
    else:
        print(f"Warning: Joint '{name}' not found; it may be fixed or not exported.")

print("Valid joint names for control:", valid_joint_names)

########################## Set Control Gains ##########################
num_dofs = len(valid_joint_names)

# current gain values from previous Isaac Sim implementation- adjust as needed
kp = np.array([
    1000,     # right_wheel_joint
    1000,     # left_wheel_joint
    100000,  # pillar_platform_joint
    20,      # pan_joint
    20,      # tilt_joint
    250,     # L_joint1
    250,     # L_joint2
    100,     # L_joint3
    100,     # L_joint4
    75,      # L_joint5
    75,      # L_joint6
    100,     # L_jaw_1_joint
    100,     # L_jaw_2_joint
    250,     # R_joint1
    250,     # R_joint2
    100,     # R_joint3
    100,     # R_joint4
    100,      # R_joint5
    75,     # R_joint6
    75,     # R_joint7
    100,     # R_jaw_1_joint
    100,     # R_jaw_2_joint
])

kv = np.array([
    10,      # right_wheel_joint
    10,      # left_wheel_joint
    1000,    # pillar_platform_joint
    1,       # pan_joint
    1,       # tilt_joint
    10,      # L_joint1
    10,      # L_joint2
    5,       # L_joint3
    5,       # L_joint4
    2.5,     # L_joint5
    2.5,     # L_joint6
    10,      # L_jaw_1_joint
    10,      # L_jaw_2_joint
    10,      # R_joint1
    10,      # R_joint2
    5,       # R_joint3
    5,       # R_joint4
    5,       # R_joint5
    2.5,     # R_joint6
    2.5,     # R_joint7
    10,      # R_jaw_1_joint
    10,      # R_jaw_2_joint
])

force_upper = np.array([
    1000,    # right_wheel_joint
    1000,    # left_wheel_joint
    100000,  # pillar_platform_joint
    2.5,     # pan_joint
    2.5,     # tilt_joint
    250,     # L_joint1
    250,     # L_joint2
    150,     # L_joint3
    150,     # L_joint4
    100,     # L_joint5
    100,     # L_joint6
    50,      # L_jaw_1_joint
    50,      # L_jaw_2_joint
    250,     # R_joint1
    250,     # R_joint2
    150,     # R_joint3
    150,     # R_joint4
    150,     # R_joint5
    100,     # R_joint6
    100,     # R_joint7
    50,      # R_jaw_1_joint
    50,      # R_jaw_2_joint
])

force_lower = -force_upper

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

########################## Control Loop ##########################
for step in range(10000):
    # For demonstration, leave the arm joints at current positions and lower pillar
    target_positions = np.zeros(num_dofs)
    target_positions[2] = -0.0005 * step
    robot.control_dofs_position(target_positions, dofs_idx)

    # get the camera link poses
    head_cam_pos = robot.get_link('head_cam_rgb_camera_frame').get_pos().cpu().numpy()
    chest_cam_pos = robot.get_link('chest_cam_rgb_camera_frame').get_pos().cpu().numpy()
    head_cam_quat = R.from_quat(robot.get_link('head_cam_rgb_camera_frame').get_quat().cpu().numpy())
    chest_cam_quat = R.from_quat(robot.get_link('chest_cam_rgb_camera_frame').get_quat().cpu().numpy())
    head_cam_rot_matrix = head_cam_quat.as_matrix()
    chest_cam_rot_matrix = chest_cam_quat.as_matrix()

    # get the camera lookat and up vectors to dynamically update the camera poses
    head_x_axis = head_cam_rot_matrix[:, 0]
    chest_x_axis = chest_cam_rot_matrix[:, 0]
    head_lookat = head_cam_pos + head_x_axis
    chest_lookat = chest_cam_pos - chest_x_axis
    head_cam_up = -head_cam_rot_matrix[:, 2]
    chest_cam_up = -chest_cam_rot_matrix[:, 2]

    # update the camera poses
    head_cam.set_pose(
        pos=head_cam_pos,
        lookat=head_lookat,
        up=head_cam_up,
    )
    chest_cam.set_pose(
        pos=chest_cam_pos,
        lookat=chest_lookat,
        up=chest_cam_up,
    )

    # render the cameras, currently only depth is used so seg and normal will be null
    head_cam_rgb, head_cam_depth, head_cam_seg, head_cam_normal = head_cam.render(depth=True)
    chest_cam_rgb, chest_cam_depth, chest_cam_seg, chest_cam_normal = chest_cam.render(depth=True)
    
    # Step the simulation.
    scene.step()
    
    # if step % 50 == 0:
    #     ctrl_force = robot.get_dofs_control_force(dofs_idx)
    #     internal_force = robot.get_dofs_force(dofs_idx)
    #     print(f"Step {step}:")
    #     print("  Control force:", ctrl_force)
    #     print("  Internal force:", internal_force)

