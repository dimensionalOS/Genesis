import numpy as np
import genesis as gs

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
    )
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
# These gain values are placeholders?please adjust as needed.
kp = np.full(num_dofs, 500.0)       # Position gains
kv = np.full(num_dofs, 50.0)        # Velocity gains
force_lower = np.full(num_dofs, -100.0)
force_upper = np.full(num_dofs, 100.0)

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
# For demonstration, command a sinusoidal position trajectory to the actuated joints.
for step in range(500):
    # Create a sine-wave target position for each controlled joint.
    target_positions = np.sin(np.linspace(0, 2 * np.pi, num_dofs) + 0.01 * step)
    robot.control_dofs_position(target_positions, dofs_idx)
    
    # Step the simulation.
    scene.step()
    
    if step % 50 == 0:
        ctrl_force = robot.get_dofs_control_force(dofs_idx)
        internal_force = robot.get_dofs_force(dofs_idx)
        print(f"Step {step}:")
        print("  Control force:", ctrl_force)
        print("  Internal force:", internal_force)

