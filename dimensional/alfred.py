import numpy as np
import genesis as gs

########################## Initialization ##########################
gs.init(backend=gs.gpu)

# Create the simulation scene.
scene = gs.Scene(
    viewer_options = gs.options.ViewerOptions(
        camera_pos    = (0, -5, 2),
        camera_lookat = (0, 0, 0.5),
        camera_fov    = 30,
        res           = (960, 640),
        max_FPS       = 60,
    ),
    sim_options = gs.options.SimOptions(
        dt = 0.01,
    ),
    show_viewer = True,
)

########################## Entities ##########################
# Add a ground plane.
plane = scene.add_entity(gs.morphs.Plane())

# Load your robot from the URDF file.
robot = scene.add_entity(
    gs.morphs.URDF(
        file  = '../assets/alfred_base_descr.urdf',
        pos   = (0, 0, 0),
        euler = (0, 0, 0),
    )
)

# Build the scene so that all entities (including the robot) are instantiated.
scene.build()

########################## Joint Mapping ##########################
# Define only the controllable (actuated) joint names.
# (Based on your URDF, we omit fixed joints such as "base_parent_joint", "base_center_joint",
#  "laser_joint", "L_joint_eef", "L_gripper_base_joint", "R_joint_eef", and "R_gripper_base_joint".)
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

# Map joint names to their local DOF indices; if a joint isn't found, print a warning.
dofs_idx = []
valid_joint_names = []
for name in joint_names:
    joint = robot.get_joint(name)
    if joint is not None:
        dofs_idx.append(joint.dof_idx_local)
        valid_joint_names.append(name)
    else:
        print(f"Warning: Joint '{name}' not found; it is likely fixed.")

print("Valid joints for control:", valid_joint_names)
num_dofs = len(valid_joint_names)

########################## Set Control Gains ##########################
# Set control gains for the PD controller.
# (These gain values are example placeholders; adjust as needed.)
robot.set_dofs_kp(
    kp             = np.full(num_dofs, 4500.0),
    dofs_idx_local = dofs_idx,
)
robot.set_dofs_kv(
    kv             = np.full(num_dofs, 450.0),
    dofs_idx_local = dofs_idx,
)
robot.set_dofs_force_range(
    lower          = np.full(num_dofs, -100.0),
    upper          = np.full(num_dofs,  100.0),
    dofs_idx_local = dofs_idx,
)

########################## PD Control with Sample Targets ##########################
# Define sample target configurations for the actuated joints.
# (Make sure the target array length matches the number of controllable DOFs.)
target1 = np.linspace(0.1, 0.5, num_dofs)  # First target configuration
target2 = np.linspace(-0.5, 0.5, num_dofs) # Second target configuration
target3 = np.zeros(num_dofs)               # Third target configuration

# The built-in PD controller will persistently drive the robot toward the target.
# The simulation loop here runs for 1250 steps; after the loop ends, the simulation stops.
for i in range(1250):
    if i < 250:
        robot.control_dofs_position(target1, dofs_idx)
    elif i < 500:
        robot.control_dofs_position(target2, dofs_idx)
    elif i < 750:
        robot.control_dofs_position(target3, dofs_idx)
    elif i < 1000:
        # For demonstration, hold the final position.
        robot.control_dofs_position(target3, dofs_idx)
    else:
        # Optionally, zero out the control force (or you could continue holding position)
        robot.control_dofs_force(np.zeros(num_dofs), dofs_idx)
    
    scene.step()
    
    # Print diagnostic forces every 50 steps.
    if i % 50 == 0:
        ctrl_force = robot.get_dofs_control_force(dofs_idx)
        internal_force = robot.get_dofs_force(dofs_idx)
        print(f"Step {i}: Control force: {ctrl_force}, Internal force: {internal_force}")

print("Control loop finished. Simulation has ended.")

