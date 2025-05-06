import genesis as gs
import numpy as np

def main():
    joint_names = [
        "pillar_platform_joint",
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
        "joint7",
        "joint8",
    ]
    gs.init()

    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0, -5, 2),
            camera_lookat=(0, 0, 0.5),
            camera_fov=60,
            res=(1280, 960),
            max_FPS=60,
        ),
        sim_options=gs.options.SimOptions(
            dt=0.01,
        ),
        rigid_options=gs.options.RigidOptions(
            enable_self_collision=True,
        ),
        show_viewer=True,
    )

    plane = scene.add_entity(gs.morphs.Plane())

    robot = scene.add_entity(
        gs.morphs.URDF(
            file  = '../assets/devkit_base_descr.urdf',
            pos   = (0, 0, 0),
            euler = (0, 0, 0),
        )
    )

    box = scene.add_entity(gs.morphs.Box(pos=(2, 0, 2), size=(1, 1, 1)))
    
    # office = scene.add_entity(gs.morphs.Mesh(
    #     file="../assets/office-mod1.glb",
    #     fixed=True,
    #     euler=(-90, 180, 0),
    #     pos=(-2, 0, -13.21),
    #     convexify=False,
    #     decimate=True,
    #     decimate_face_num=10000,
    # ))

    scene.build()

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
        100000,  # pillar_platform_joint
        250,     # L_joint1
        250,     # L_joint2
        100,     # L_joint3
        100,     # L_joint4
        75,      # L_joint5
        75,      # L_joint6
        100,     # L_jaw_1_joint
        100,     # L_jaw_2_joint
    ])

    kv = np.array([
        1000,    # pillar_platform_joint
        10,      # L_joint1
        10,      # L_joint2
        5,       # L_joint3
        5,       # L_joint4
        2.5,     # L_joint5
        2.5,     # L_joint6
        10,      # L_jaw_1_joint
        10,      # L_jaw_2_joint
    ])

    force_upper = np.array([
        100000,  # pillar_platform_joint
        250,     # L_joint1
        250,     # L_joint2
        150,     # L_joint3
        150,     # L_joint4
        100,     # L_joint5
        100,     # L_joint6
        50,      # L_jaw_1_joint
        50,      # L_jaw_2_joint
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

    target_positions = np.zeros(num_dofs)
    for step in range(10000):
        if step < 100:
            robot.set_dofs_position(target_positions, dofs_idx)
            scene.step()
        robot.control_dofs_position(target_positions, dofs_idx)
        scene.step()

if __name__ == "__main__":
    main()