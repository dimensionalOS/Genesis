#!/usr/bin/env python3

import numpy as np
import os
import time
import lcm
from lcm_msgs.sensor_msgs import JointState
from pydrake.all import (
    MultibodyPlant, Parser, InverseDynamics, SceneGraph,
    DiagramBuilder, MathematicalProgram, Solve, RigidTransform,
    PiecewisePolynomial, TrajectorySource, 
    Simulator, RotationMatrix, RollPitchYaw, SpatialVelocity,
    AutoDiffXd, AddMultibodyPlantSceneGraph, JacobianWrtVariable,
    FindResourceOrThrow, eq, ge, le
)
import matplotlib.pyplot as plt
from tqdm import tqdm

class URDFInverseKinematicsWithCollisionChecker:
    def __init__(self, urdf_path, end_effector_link, kinematic_chain_joints):
        """
        Initialize the IK solver with collision checking.
        
        Args:
            urdf_path: Path to the URDF file
            end_effector_link: Name of end effector link
            kinematic_chain_joints: List of joint names to include in the kinematic chain
        """
        self.urdf_path = urdf_path
        self.end_effector_link = end_effector_link
        self.kinematic_chain_joints = kinematic_chain_joints
        
        # Initialize plant with collision detection
        self.builder = DiagramBuilder()
        self.plant, self.scene_graph = AddMultibodyPlantSceneGraph(self.builder, time_step=0.0)
        
        # Load URDF model
        self.parser = Parser(self.plant)
        model_instances = self.parser.AddModelsFromUrl(f"file://{self.urdf_path}")
        # We'll use the first model instance returned
        self.model_instance = model_instances[0] if model_instances else None
        
        # Finalize the plant
        self.plant.Finalize()
        
        # Get joint indices for the kinematic chain
        self.joint_indices = []
        for joint_name in self.kinematic_chain_joints:
            try:
                joint = self.plant.GetJointByName(joint_name)
                if joint.num_positions() > 0:  # Only add if it's an actuated joint
                    self.joint_indices.extend(
                        range(joint.position_start(), joint.position_start() + joint.num_positions())
                    )
            except RuntimeError:
                print(f"Warning: Joint {joint_name} not found in the URDF")
        
        # Get end effector body
        try:
            self.end_effector_body = self.plant.GetBodyByName(self.end_effector_link)
        except RuntimeError:
            print(f"Error: End effector link {self.end_effector_link} not found in the URDF")
            raise
        
        # Build the system and create a context
        self.diagram = self.builder.Build()
        self.diagram_context = self.diagram.CreateDefaultContext()
        self.plant_context = self.plant.GetMyContextFromRoot(self.diagram_context)
        
        # Store joint limits
        self.joint_lower_limits = []
        self.joint_upper_limits = []
        for idx in self.joint_indices:
            self.joint_lower_limits.append(self.plant.GetPositionLowerLimits()[idx])
            self.joint_upper_limits.append(self.plant.GetPositionUpperLimits()[idx])
            
        # Store collision pairs to ignore (will be populated by calibration)
        self.collision_pairs_to_ignore = set()
    
    def set_joint_positions(self, q):
        """Set joint positions in the plant context."""
        full_q = self.plant.GetPositions(self.plant_context)
        for i, idx in enumerate(self.joint_indices):
            full_q[idx] = q[i]
        self.plant.SetPositions(self.plant_context, full_q)
    
    def get_random_joint_positions(self):
        """Generate random joint positions within limits."""
        q = np.zeros(len(self.joint_indices))
        for i in range(len(self.joint_indices)):
            q[i] = np.random.uniform(self.joint_lower_limits[i], self.joint_upper_limits[i])
        return q
    
    def check_self_collisions(self):
        """Check for self-collisions in the current configuration."""
        query_port = self.scene_graph.GetOutputPort("query")
        query_object = query_port.Eval(self.scene_graph.GetMyContextFromRoot(self.diagram_context))
        
        # Get collision pairs
        collision_pairs = query_object.ComputePointPairPenetration()
        
        # Filter out ignored collision pairs
        filtered_pairs = []
        for pair in collision_pairs:
            # In this version of Drake, we'll use the geometry IDs directly as a unique identifier
            # This is a simplification but should work for our collision checking needs
            body_A_id = pair.id_A
            body_B_id = pair.id_B
            
            collision_key = (str(body_A_id), str(body_B_id))
            reverse_key = (str(body_B_id), str(body_A_id))
            
            if collision_key not in self.collision_pairs_to_ignore and reverse_key not in self.collision_pairs_to_ignore:
                filtered_pairs.append(pair)
        
        return filtered_pairs
        
    def get_collision_geometry_names(self):
        """Get the names of geometries that are in collision."""
        collision_pairs = self.check_self_collisions()
        geometry_pairs = []
        
        # Load and parse the URDF to extract link names
        import xml.etree.ElementTree as ET
        try:
            tree = ET.parse(self.urdf_path)
            root = tree.getroot()
            link_names = [link.attrib['name'] for link in root.findall('.//link')]
            print(f"\nAvailable links in the URDF: {', '.join(link_names[:5])}... (and {len(link_names)-5} more)")
        except Exception as e:
            print(f"Could not parse URDF file: {e}")
            link_names = []
        
        # Map GeometryId values to indices or offsets
        # This is just a guess, but often geometry IDs have some relationship to link indices
        geometry_id_mapping = {}
        for i, link_name in enumerate(link_names):
            # Create a few possible mappings for geometry IDs
            # Note: These are heuristics and may not be accurate
            base_id = (i+1) * 7  # Just a heuristic
            for offset in range(-3, 4):  # Try a few offsets around our guessed base ID
                potential_id = base_id + offset
                geometry_id_mapping[f"<GeometryId value={potential_id}>"] = link_name
        
        # Process each collision pair
        for pair in collision_pairs:
            # Get information from the collision pair
            geometry_A_id = str(pair.id_A)
            geometry_B_id = str(pair.id_B)
            penetration_depth = pair.depth
            
            # Extract numeric ID from the geometry ID string
            try:
                id_A_num = int(geometry_A_id.split('value=')[1].split('>')[0])
                id_B_num = int(geometry_B_id.split('value=')[1].split('>')[0])
            except:
                id_A_num = 0
                id_B_num = 0
            
            # Try to map to link names
            link_A = geometry_id_mapping.get(geometry_A_id, "")
            link_B = geometry_id_mapping.get(geometry_B_id, "")
            
            # If we don't have a direct mapping, try to guess based on the ID value
            if not link_A and len(link_names) > 0:
                # Try to associate with a link based on ID value
                try:
                    closest_idx = min(range(len(link_names)), key=lambda i: abs((i+1)*7 - id_A_num))
                    link_A = f"Possibly {link_names[closest_idx]}"
                except:
                    link_A = "Unknown link"
            
            if not link_B and len(link_names) > 0:
                try:
                    closest_idx = min(range(len(link_names)), key=lambda i: abs((i+1)*7 - id_B_num))
                    link_B = f"Possibly {link_names[closest_idx]}"
                except:
                    link_B = "Unknown link"
            
            # Get positions
            frame_A_pos = getattr(pair, 'p_WCa', None)
            frame_B_pos = getattr(pair, 'p_WCb', None)
            
            frame_A_info = f"Position {frame_A_pos}" if frame_A_pos is not None else "unknown position"
            frame_B_info = f"Position {frame_B_pos}" if frame_B_pos is not None else "unknown position"
            
            # Store collision information
            geometry_pairs.append({
                "geometry_A": geometry_A_id,
                "link_A": link_A,
                "frame_A": frame_A_info,
                "geometry_B": geometry_B_id,
                "link_B": link_B,
                "frame_B": frame_B_info,
                "depth": penetration_depth
            })
        
        return geometry_pairs
    
    def get_joint_name_positions(self, q):
        """Get a dictionary mapping joint names to their current positions."""
        self.set_joint_positions(q)
        joint_positions = {}
        
        for i, joint_name in enumerate(self.kinematic_chain_joints):
            # This is a simplification - in reality, a joint might control multiple positions
            # We'll assume a 1:1 mapping here
            if i < len(q):
                joint_positions[joint_name] = q[i]
        
        return joint_positions
    
    def run_calibration(self, num_samples=10000, collision_threshold=0.95):
        """
        Run calibration to identify frequent self-collisions to ignore.
        
        Args:
            num_samples: Number of random configurations to sample
            collision_threshold: Threshold for ignoring collision pairs (0.0-1.0)
        """
        print(f"Running calibration with {num_samples} random samples...")
        
        # Dictionary to count collisions for each pair
        collision_counts = {}
        
        # Sample random configurations and check for collisions
        for _ in tqdm(range(num_samples)):
            q = self.get_random_joint_positions()
            self.set_joint_positions(q)
            
            # Evaluate collision pairs
            query_port = self.scene_graph.GetOutputPort("query")
            query_object = query_port.Eval(self.scene_graph.GetMyContextFromRoot(self.diagram_context))
            collision_pairs = query_object.ComputePointPairPenetration()
            
            # Count collision occurrences
            for pair in collision_pairs:
                body_A_id = pair.id_A
                body_B_id = pair.id_B
                
                # Create a unique identifier for this collision pair
                # Ensure consistent ordering of the pair
                id_A_str = str(body_A_id)
                id_B_str = str(body_B_id)
                if id_A_str > id_B_str:
                    id_A_str, id_B_str = id_B_str, id_A_str
                
                collision_key = (id_A_str, id_B_str)
                collision_counts[collision_key] = collision_counts.get(collision_key, 0) + 1
        
        # Identify collision pairs to ignore based on threshold
        for pair, count in collision_counts.items():
            if count / num_samples >= collision_threshold:
                self.collision_pairs_to_ignore.add(pair)
                print(f"Ignoring collision pair: {pair} (collision rate: {count/num_samples:.2f})")
        
        print(f"Calibration complete. Ignoring {len(self.collision_pairs_to_ignore)} collision pairs.")
        return self.collision_pairs_to_ignore
    
    def forward_kinematics(self, q):
        """
        Compute forward kinematics for the given joint positions.
        
        Args:
            q: Joint positions for the kinematic chain
            
        Returns:
            end_effector_pose: RigidTransform representing the end effector pose
        """
        self.set_joint_positions(q)
        end_effector_pose = self.plant.EvalBodyPoseInWorld(self.plant_context, self.end_effector_body)
        return end_effector_pose
    
    def inverse_kinematics(self, target_pose, initial_guess=None):
        """
        Solve inverse kinematics to reach the target pose using a numerical approach.
        
        Args:
            target_pose: Target RigidTransform for the end effector
            initial_guess: Initial joint positions (if None, use random values)
            
        Returns:
            q_sol: Solution joint positions
            info: Information about the solution (success, etc)
        """
        # Set initial guess for optimization
        if initial_guess is None:
            q = self.get_random_joint_positions()
        else:
            q = initial_guess.copy()
            
        # Parameters for numerical optimization
        max_iterations = 1000
        tolerance = 1e-2
        damping = 0.5
        
        # Run iterative IK
        for iteration in range(max_iterations):
            # Set current joint positions
            self.set_joint_positions(q)
            
            # Get current end effector pose
            current_pose = self.plant.EvalBodyPoseInWorld(self.plant_context, self.end_effector_body)
            
            # Calculate position error
            pos_error = target_pose.translation() - current_pose.translation()
            
            # Simple orientation error based on rotation matrix
            R_current = current_pose.rotation().matrix()
            R_target = target_pose.rotation().matrix()
            R_error = R_target @ R_current.T
            orientation_error = np.zeros(3)
            orientation_error[0] = R_error[2, 1] - R_error[1, 2]
            orientation_error[1] = R_error[0, 2] - R_error[2, 0]
            orientation_error[2] = R_error[1, 0] - R_error[0, 1]
            
            # Combine errors into a 6D error vector
            error = np.hstack([pos_error, 0.5 * orientation_error])
            error_norm = np.linalg.norm(error)
            
            # Check convergence
            if error_norm < tolerance:
                break
                
            # Calculate Jacobian
            J = np.zeros((6, len(self.joint_indices)))
            for i in range(len(self.joint_indices)):
                # Finite difference approximation for Jacobian
                eps = 1e-6
                q_perturbed = q.copy()
                q_perturbed[i] += eps
                
                self.set_joint_positions(q_perturbed)
                perturbed_pose = self.plant.EvalBodyPoseInWorld(self.plant_context, self.end_effector_body)
                
                # Position part
                J[0:3, i] = (perturbed_pose.translation() - current_pose.translation()) / eps
                
                # Rotation part - simpler approximation
                R_perturbed = perturbed_pose.rotation().matrix()
                R_diff = R_perturbed @ R_current.T
                rot_diff = np.zeros(3)
                rot_diff[0] = R_diff[2, 1] - R_diff[1, 2]
                rot_diff[1] = R_diff[0, 2] - R_diff[2, 0]
                rot_diff[2] = R_diff[1, 0] - R_diff[0, 1]
                J[3:6, i] = rot_diff / eps
            
            # Reset to original pose for this iteration
            self.set_joint_positions(q)
            
            # Damped least squares for stability
            J_dag = J.T @ np.linalg.inv(J @ J.T + damping**2 * np.eye(6))
            
            # Update joint positions
            dq = J_dag @ error
            q = q + dq
            
            # Apply joint limits
            for i in range(len(self.joint_indices)):
                q[i] = max(min(q[i], self.joint_upper_limits[i]), self.joint_lower_limits[i])
        
        # Final evaluation of the solution
        self.set_joint_positions(q)
        final_pose = self.plant.EvalBodyPoseInWorld(self.plant_context, self.end_effector_body)
        final_pos_error = np.linalg.norm(target_pose.translation() - final_pose.translation())
        success = final_pos_error < tolerance * 10  # Slightly relaxed tolerance for success check
        
        # Check collisions
        collisions = self.check_self_collisions()
        
        # Return results
        info = {
            "success": success,
            "iterations": iteration + 1,
            "position_error": final_pos_error,
            "collision_free": len(collisions) == 0,
            "num_collisions": len(collisions)
        }
        
        return q if success else None, info
    
    def _eval_position_constraint(self, q, target_pose, axis):
        """Evaluate the position constraint along a specific axis."""
        self.set_joint_positions(q)
        current_pose = self.plant.EvalBodyPoseInWorld(self.plant_context, self.end_effector_body)
        
        # Calculate position error along the specified axis
        pos_error = current_pose.translation()[axis] - target_pose.translation()[axis]
        return [pos_error]
    
    def _eval_orientation_constraint(self, q, target_pose, axis):
        """Evaluate the orientation constraint along a specific axis."""
        self.set_joint_positions(q)
        current_pose = self.plant.EvalBodyPoseInWorld(self.plant_context, self.end_effector_body)
        
        # Calculate orientation error
        # We'll use a simplified approach based on the rotation matrix
        current_R = current_pose.rotation().matrix()
        target_R = target_pose.rotation().matrix()
        
        # Extract the main axes and compute their alignment
        current_axis = current_R[:, axis]
        target_axis = target_R[:, axis]
        
        # Dot product measures alignment - we want this to be 1
        alignment = np.dot(current_axis, target_axis)
        
        # We want alignment to be 1, so error is 1 - alignment
        return [1.0 - alignment]

def publish_joint_state_to_lcm(joint_names, joint_positions):
    """Publish joint positions to LCM.
    
    Args:
        joint_names: List of joint names
        joint_positions: List of joint positions matching the names
    """
    # Initialize LCM
    lc = lcm.LCM()
    
    # Create JointState message
    msg = JointState()
    msg.name = joint_names
    msg.position = joint_positions
    msg.velocity = []  # Empty list by default
    msg.effort = []    # Empty list by default
    msg.name_length = len(msg.name)
    msg.position_length = len(msg.position)
    msg.velocity_length = len(msg.velocity)
    msg.effort_length = len(msg.effort)
    
    # Publish the message
    lc.publish("joint_states#sensor_msgs.JointState", msg.encode())
    print("\nPublished joint state to LCM topic 'joint_states#sensor_msgs.JointState'")

def main():
    # Path to the URDF file
    urdf_path = os.path.abspath("/home/yashas/Documents/dimensional/Genesis/assets/devkit_base_descr.urdf")
    
    # Define the kinematic chain and end effector
    kinematic_chain_joints = ["pillar_platform_joint", "joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
    end_effector_link = "link6"
    
    # Create the IK solver
    ik_solver = URDFInverseKinematicsWithCollisionChecker(
        urdf_path, end_effector_link, kinematic_chain_joints
    )
    
    # Run calibration to identify links that are frequently in collision
    ik_solver.run_calibration(num_samples=10000, collision_threshold=0.95)
    
    # Test IK by first running forward kinematics
    print("\nTesting IK with FK validation:")
    
    # Create a test joint configuration
    test_q = np.zeros(len(ik_solver.joint_indices))
    # for i in range(len(ik_solver.joint_indices)):
    #     # Set to middle of joint range
    #     test_q[i] = (ik_solver.joint_lower_limits[i] + ik_solver.joint_upper_limits[i]) / 2.0
    
    # Print initial joint positions
    print("\nInitial Joint Positions:")
    joint_positions = ik_solver.get_joint_name_positions(test_q)
    for joint_name, position in joint_positions.items():
        print(f"  {joint_name}: {position:.6f} radians")
    
    # Run forward kinematics to get end effector pose
    ee_pose = ik_solver.forward_kinematics(test_q)
    # print(f"\nFK test pose:\n{ee_pose.GetAsMatrix4()}")
    print(f"\nFK test pose:\n{ee_pose.translation()}\n{ee_pose.rotation()}")
    
    # Check if initial configuration has collisions
    collisions = ik_solver.get_collision_geometry_names()
    if collisions:
        print("\nInitial configuration has collisions:")
        for collision in collisions:
            print(f"  Collision between {collision['geometry_A']} (Link: {collision['link_A']})")
            print(f"    and {collision['geometry_B']} (Link: {collision['link_B']})")
            print(f"    Penetration depth: {collision['depth']:.6f}")
            print(f"    Locations: {collision['frame_A']} and {collision['frame_B']}")
    else:
        print("\nInitial configuration is collision-free")
    
    # Modify the pose by subtracting 10cm in Z
    target_pose = RigidTransform(
        ee_pose.rotation(),
        ee_pose.translation() - np.array([0.3, 0.3, 0.8])  # Subtract 10cm in Z
    )
    print(f"\nTarget pose for IK:\n{target_pose.translation()}\n{target_pose.rotation()}")
    
    # Run inverse kinematics to solve for the target pose
    start_time = time.time()
    q_sol, info = ik_solver.inverse_kinematics(target_pose, initial_guess=test_q)
    end_time = time.time()
    
    # Print results
    print(f"\nIK solved in {end_time - start_time:.4f} seconds")
    print(f"Solution info: {info}")
    
    if info["success"]:
        print("\nIK solution found!")
        
        # Print solution joint positions
        print("\nSolution Joint Positions:")
        solution_joint_positions = ik_solver.get_joint_name_positions(q_sol)
        for joint_name, position in solution_joint_positions.items():
            initial_pos = joint_positions.get(joint_name, 0.0)
            delta = position - initial_pos
            print(f"  {joint_name}: {position:.6f} radians (change: {delta:+.6f})")
        
        # Validate solution with forward kinematics
        result_pose = ik_solver.forward_kinematics(q_sol)
        print(f"\nResult pose from IK solution:\n{result_pose.translation()}\n{result_pose.rotation()}")
        
        # Calculate position error
        pos_error = np.linalg.norm(result_pose.translation() - target_pose.translation())
        rot_error = RotationMatrix(result_pose.rotation().matrix().T @ target_pose.rotation().matrix()).ToAngleAxis().angle()
        
        print(f"\nPosition error: {pos_error:.6f} meters")
        print(f"Rotation error: {rot_error:.6f} radians")
        
        # Check for collisions in the solution
        collisions = ik_solver.get_collision_geometry_names()
        if collisions:
            print("\nSolution configuration has collisions:")
            for collision in collisions:
                print(f"  Collision between {collision['geometry_A']} (Link: {collision['link_A']})")
                print(f"    and {collision['geometry_B']} (Link: {collision['link_B']})")
                print(f"    Penetration depth: {collision['depth']:.6f}")
                print(f"    Locations: {collision['frame_A']} and {collision['frame_B']}")
        else:
            print("\nSolution configuration is collision-free")
        
        # Publish joint positions to LCM for visualization
        joint_names = list(solution_joint_positions.keys())
        joint_values = [solution_joint_positions[name] for name in joint_names]
        publish_joint_state_to_lcm(joint_names, joint_values)
        
        print("\nJoint state published to LCM for visualization")
    else:
        print("Failed to find IK solution")

if __name__ == "__main__":
    main()
