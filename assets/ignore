import numpy as np
import open3d as o3d  # For point cloud processing
import ompl.base as ob
import ompl.geometric as og
import ompl.util as ou
import pybullet as p  # For collision checking and forward kinematics
import pybullet_data
import time
import math
from scipy.spatial import KDTree  # For efficient point cloud distance queries

class MotionPlanner:
    def __init__(self, urdf_path, joint_names, point_cloud=None, collision_free=False):
        """
        Initialize the motion planner.
        
        Args:
            urdf_path: Path to the URDF file
            joint_names: List of joint names to use in planning
            point_cloud: Open3D point cloud object (optional)
            collision_free: If True, require completely collision-free paths
                           If False, minimize collisions with the point cloud
        """
        # Initialize PyBullet
        self.client_id = p.connect(p.DIRECT)  # Headless mode
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # Load the robot from URDF
        self.robot_id = p.loadURDF(urdf_path, useFixedBase=True)
        
        # Get joint information
        self.joint_indices = []
        self.joint_limits = []
        for i in range(p.getNumJoints(self.robot_id)):
            joint_info = p.getJointInfo(self.robot_id, i)
            joint_name = joint_info[1].decode('utf-8')
            if joint_name in joint_names:
                self.joint_indices.append(i)
                lower_limit = joint_info[8]
                upper_limit = joint_info[9]
                self.joint_limits.append((lower_limit, upper_limit))
        
        # Store parameters
        self.point_cloud = point_cloud
        self.collision_free = collision_free
        self.point_tree = None
        
        # Process point cloud if provided
        if self.point_cloud is not None:
            self.process_point_cloud(self.point_cloud)
        
        # OMPL setup
        self.setup_ompl_space()
    
    def process_point_cloud(self, point_cloud):
        """Process the point cloud and prepare for collision checking."""
        # Convert to numpy array for faster processing
        self.points = np.asarray(point_cloud.points)
        
        # Create KD-Tree for efficient nearest neighbor queries
        self.point_tree = KDTree(self.points)
    
    def setup_ompl_space(self):
        """Set up the OMPL state space and problem definition."""
        # Create state space (one dimension per joint)
        self.space = ob.RealVectorStateSpace(len(self.joint_indices))
        
        # Set joint limits
        bounds = ob.RealVectorBounds(len(self.joint_indices))
        for i, (lower, upper) in enumerate(self.joint_limits):
            bounds.setLow(i, lower)
            bounds.setHigh(i, upper)
        self.space.setBounds(bounds)
        
        # Create simple setup
        self.ss = og.SimpleSetup(self.space)
        
        # Set state validity checker based on mode
        if self.collision_free:
            # In collision-free mode, use binary validity checker
            self.ss.setStateValidityChecker(ob.StateValidityCheckerFn(self.binary_validity_checker))
            # Use a standard planner like RRT* without custom objective
            planner = og.RRTstar(self.ss.getSpaceInformation())
        else:
            # In collision-minimization mode, all states are technically "valid"
            self.ss.setStateValidityChecker(ob.StateValidityCheckerFn(self.cost_validity_checker))
            # Set optimization objective for collision minimization
            objective = self.create_objective()
            self.ss.setOptimizationObjective(objective)
            # Use an optimization-based planner
            planner = og.RRTstar(self.ss.getSpaceInformation())
        
        # Set the planner
        planner.setRange(0.1)  # Maximum step size
        self.ss.setPlanner(planner)
    
    def binary_validity_checker(self, state):
        """
        Standard validity checker for collision-free planning.
        Returns True if state is collision-free, False otherwise.
        """
        # Apply joint positions
        for i, joint_idx in enumerate(self.joint_indices):
            p.resetJointState(self.robot_id, joint_idx, state[i])
        
        # Check for collisions with PyBullet
        # This uses PyBullet's built-in collision detection with the loaded environment
        # Could be extended to check against point cloud if desired
        for link_idx in range(p.getNumJoints(self.robot_id)):
            contact_points = p.getContactPoints(bodyA=self.robot_id, linkIndexA=link_idx)
            if len(contact_points) > 0:
                return False  # Collision detected
        
        return True  # No collisions
    
    def cost_validity_checker(self, state):
        """
        Custom validity checker that allows collisions but with a cost.
        
        Returns True for all states, but assigns cost through motion cost.
        """
        # Always return valid, costs are handled in the optimization objective
        return True
    
    def create_objective(self):
        """Create an optimization objective that minimizes collisions."""
        # Create a custom optimization objective
        obj = ob.OptimizationObjective(self.ss.getSpaceInformation())
        
        # Set motion cost function that evaluates collision with point cloud
        obj.setCostToGoHeuristic(ob.CostToGoHeuristic(self.cost_to_go))
        obj.setMotionCostFn(ob.MotionCostFn(self.motion_cost))
        
        return obj
    
    def cost_to_go(self, state):
        """Estimated cost to go to goal from this state."""
        # For now, just return a simple Euclidean distance to goal
        # This could be improved with a more sophisticated heuristic
        goal = self.ss.getGoal().getState()
        return ob.Cost(self.state_distance(state, goal))
    
    def state_distance(self, state1, state2):
        """Compute Euclidean distance between states."""
        dist = 0.0
        for i in range(len(self.joint_indices)):
            dist += (state1[i] - state2[i]) ** 2
        return math.sqrt(dist)
    
    def motion_cost(self, state1, state2):
        """
        Compute the cost of motion between two states.
        Incorporates the collision cost with the point cloud if available.
        """
        # Get joint configurations
        joints1 = [state1[i] for i in range(len(self.joint_indices))]
        joints2 = [state2[i] for i in range(len(self.joint_indices))]
        
        # Compute collision cost along the path
        # Interpolate between states for better collision checking
        num_steps = 10
        total_cost = 0.0
        
        for step in range(num_steps + 1):
            t = float(step) / num_steps
            interp_joints = [joints1[i] + t * (joints2[i] - joints1[i]) for i in range(len(joints1))]
            
            # Apply joint positions
            for joint_idx, joint_val in zip(self.joint_indices, interp_joints):
                p.resetJointState(self.robot_id, joint_idx, joint_val)
            
            # Check for collisions
            if self.point_cloud is not None:
                # If point cloud is available, compute cost based on it
                collision_cost = self.compute_point_cloud_collision_cost()
            else:
                # If no point cloud, use standard collision checking
                # Assign high cost to any collisions
                collision_detected = False
                for link_idx in range(p.getNumJoints(self.robot_id)):
                    contact_points = p.getContactPoints(bodyA=self.robot_id, linkIndexA=link_idx)
                    if len(contact_points) > 0:
                        collision_detected = True
                        break
                
                collision_cost = 100.0 if collision_detected else 0.0
            
            total_cost += collision_cost
        
        # Average cost along the path
        avg_cost = total_cost / (num_steps + 1)
        
        # Add distance component to encourage shorter paths
        distance_cost = self.state_distance(state1, state2)
        
        # Weighted sum of collision cost and distance cost
        # Adjust weights based on your priorities
        collision_weight = 0.8
        distance_weight = 0.2
        final_cost = collision_weight * avg_cost + distance_weight * distance_cost
        
        return ob.Cost(final_cost)
    
    def compute_point_cloud_collision_cost(self):
        """
        Compute how much the current robot configuration collides with the point cloud.
        Returns a cost value based on collision severity.
        """
        if self.point_tree is None:
            return 0.0  # No point cloud, no collision cost
            
        total_cost = 0.0
        
        # Get all link positions and check distances to point cloud
        for link_idx in range(p.getNumJoints(self.robot_id)):
            link_state = p.getLinkState(self.robot_id, link_idx)
            link_position = link_state[0]  # (x, y, z)
            
            # Query KD-Tree for nearest points in the point cloud
            distances, _ = self.point_tree.query(link_position, k=5)
            
            # Compute cost based on distances
            # Points that are very close contribute more to the cost
            for dist in distances:
                if dist < 0.1:  # Collision threshold
                    # Inverse relationship: closer points = higher cost
                    total_cost += 0.1 / (dist + 0.01)  # Avoid division by zero
        
        return total_cost
    
    def plan(self, start_joints, goal_link_name_or_idx, goal_position, goal_orientation, planning_time=5.0):
        """
        Plan a motion that minimizes collisions with the point cloud.
        
        Args:
            start_joints: Starting joint positions
            goal_link_name_or_idx: Name or index of the link to reach the goal
            goal_position: Target position for the link
            goal_orientation: Target orientation for the link (quaternion)
            planning_time: Time budget for planning in seconds
            
        Returns:
            List of joint configurations along the path
        """
        # Set start state
        start = ob.State(self.space)
        for i, joint_val in enumerate(start_joints):
            start[i] = joint_val
        self.ss.setStartState(start)
        
        # Determine if goal_link_name_or_idx is an index or a name
        goal_link_idx = None
        if isinstance(goal_link_name_or_idx, int):
            # It's already an index
            goal_link_idx = goal_link_name_or_idx
            if goal_link_idx >= p.getNumJoints(self.robot_id):
                raise ValueError(f"Goal link index {goal_link_idx} is out of range")
        else:
            # It's a name, need to find the index
            for i in range(p.getNumJoints(self.robot_id)):
                joint_info = p.getJointInfo(self.robot_id, i)
                link_name = joint_info[12].decode('utf-8')
                joint_name = joint_info[1].decode('utf-8')
                if link_name == goal_link_name_or_idx or joint_name == goal_link_name_or_idx:
                    goal_link_idx = i
                    break
            
            if goal_link_idx is None:
                raise ValueError(f"Goal link '{goal_link_name_or_idx}' not found in URDF")
        
        # Create IK-based goal
        si = self.ss.getSpaceInformation()
        goal_region = ob.GoalRegion(si)
        goal_region.setThreshold(0.1)  # Acceptable distance to goal
        
        # Custom state sampler for goal region
        def is_goal_satisfied(state):
            # Apply joint positions
            for i, joint_idx in enumerate(self.joint_indices):
                p.resetJointState(self.robot_id, joint_idx, state[i])
            
            # Get link position
            link_state = p.getLinkState(self.robot_id, goal_link_idx)
            current_pos = link_state[0]
            current_orn = link_state[1]
            
            # Calculate distances
            pos_error = sum((current_pos[i] - goal_position[i])**2 for i in range(3))
            
            # Quaternion distance
            orn_error = 1.0 - sum(current_orn[i] * goal_orientation[i] for i in range(4))**2
            
            # Combined error
            total_error = math.sqrt(pos_error) + 2.0 * orn_error
            
            return total_error < 0.1  # Goal threshold
        
        goal_region.setStateValidityChecker(ob.StateValidityCheckerFn(is_goal_satisfied))
        self.ss.setGoal(goal_region)
        
        # Solve
        solved = self.ss.solve(planning_time)
        
        # Extract path if found
        if solved:
            path = self.ss.getSolutionPath()
            path.interpolate(100)  # Get smoother path with more waypoints
            
            # Extract joint values along the path
            joint_trajectory = []
            for i in range(path.getStateCount()):
                state = path.getState(i)
                joints = [state[j] for j in range(len(self.joint_indices))]
                joint_trajectory.append(joints)
            
            return joint_trajectory
        else:
            print("No solution found")
            return None
    
    def cleanup(self):
        """Disconnect from PyBullet."""
        p.disconnect(self.client_id)

def main():
    # Define robot and joints of interest
    urdf_path = "devkit_base_descr.urdf"
    joint_names = ["pillar_platform_joint", "joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
    
    # Option 2: Without point cloud, using PyBullet collision detection
    planner = MotionPlanner(urdf_path, joint_names, collision_free=True)
    
    # DEBUG: Print all available links in the robot
    print("Available links in the robot:")
    for i in range(p.getNumJoints(planner.robot_id)):
        joint_info = p.getJointInfo(planner.robot_id, i)
        print(f"  Joint {i}: {joint_info[1].decode('utf-8')}, Link: {joint_info[12].decode('utf-8')}")
    
    # Set start configuration
    start_joints = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    # Set goal pose for end effector
    goal_link_name = "link6"
    goal_position = (-0.5, 0, 1.7)
    goal_orientation = (0.9, 0, -0.4, 0)
    
    # Find the goal link index by name
    goal_link_idx = None
    for i in range(p.getNumJoints(planner.robot_id)):
        joint_info = p.getJointInfo(planner.robot_id, i)
        link_name = joint_info[12].decode('utf-8')
        if link_name == goal_link_name:
            goal_link_idx = i
            print(f"Found goal link/joint at index {i}: {link_name}")
            break
    
    if goal_link_idx is None:
        raise ValueError(f"Goal link '{goal_link_name}' not found in URDF")
    
    # Set robot to start configuration
    for i, joint_idx in enumerate(planner.joint_indices):
        p.resetJointState(planner.robot_id, joint_idx, start_joints[i])
    
    # Get and print goal link position
    link_state = p.getLinkState(planner.robot_id, goal_link_idx)
    start_position = link_state[0]
    start_orientation = link_state[1]
    
    print(f"Goal link '{goal_link_name}' at index {goal_link_idx} in start configuration:")
    print(f"  Position: {start_position}")
    print(f"  Orientation: {start_orientation}")
    print(f"  Target Position: {goal_position}")
    print(f"  Target Orientation: {goal_orientation}")
    print(f"  Position Error: {[goal_position[i] - start_position[i] for i in range(3)]}")
    
    # Use precise goal sampling with strict requirements
    precise_plan_with_goals(planner, start_joints, goal_link_idx, goal_position, goal_orientation)
    
    # Clean up
    planner.cleanup()

def quaternion_distance(q1, q2):
    """
    Calculate the distance between two quaternions.
    Returns a value between 0 (identical) and 1 (opposite).
    """
    # Compute the dot product
    dot_product = abs(sum(q1[i] * q2[i] for i in range(4)))
    
    # Clamp the dot product to [-1, 1]
    dot_product = max(-1.0, min(dot_product, 1.0))
    
    # Distance is 1 - |dot|
    return 1.0 - dot_product

def precise_plan_with_goals(planner, start_joints, goal_link_idx, goal_position, goal_orientation, planning_time=10.0):
    """
    Plan using high-precision position and orientation goals.
    """
    # Set start state
    start = ob.State(planner.space)
    for i, joint_val in enumerate(start_joints):
        start[i] = joint_val
    planner.ss.setStartState(start)
    
    # Create a standard goal that samples configurations
    si = planner.ss.getSpaceInformation()
    
    # Define the thresholds for reaching the goal - very strict now
    pos_threshold = 0.02     # 2cm for position - very strict requirement
    orient_threshold = 0.05  # Strict quaternion distance threshold (was 0.2 before)
    
    # Even higher weight for position accuracy to ensure we meet the 2cm requirement
    pos_weight = 0.95
    orient_weight = 0.05
    
    # Set up a multi-goal objective
    multi_goal = ob.GoalStates(si)
    
    # Sample many more configurations for better chances of finding good goals
    print("Phase 1: Finding configurations with precise positions (within 2cm)...")
    num_samples = 5000  # Sample many more configurations due to strict requirements
    
    # Keep track of configurations with good positions
    position_good_configs = []
    position_good_scores = []
    position_dists = []
    
    # First phase: Find configurations that achieve good positions
    for sample_idx in range(num_samples):
        # Progress indicator
        if sample_idx % 1000 == 0:
            print(f"  Sampling progress: {sample_idx}/{num_samples}")
            
        # We'll use a more systematic approach to explore the configuration space
        random_config = [0.0] * len(planner.joint_indices)
        
        # For the first joint (pillar_platform_joint) which controls the z-height,
        # systematically sample its range
        if len(planner.joint_indices) > 0:
            lower, upper = planner.joint_limits[0]
            # Sample more densely in the range that might achieve the z-height we want
            # Use sine wave sampling to focus more samples in the middle of the range
            t = sample_idx / num_samples
            alpha = math.sin(math.pi * t)
            random_config[0] = lower + alpha * (upper - lower)
        
        # For the other joints, sample randomly but with some correlation
        # This helps explore the configuration space more effectively
        for i in range(1, len(planner.joint_indices)):
            lower, upper = planner.joint_limits[i]
            # Add some correlation between joints to explore more realistic configurations
            base_value = np.random.random()
            random_config[i] = lower + base_value * (upper - lower)
        
        # Set the robot to this configuration
        for i, joint_idx in enumerate(planner.joint_indices):
            p.resetJointState(planner.robot_id, joint_idx, random_config[i])
        
        # Check the resulting end effector position
        link_state = p.getLinkState(planner.robot_id, goal_link_idx)
        pos = link_state[0]
        
        # Calculate distance to goal position
        pos_dist = math.sqrt(sum((pos[i] - goal_position[i])**2 for i in range(3)))
        
        # Check if position is good enough (within threshold)
        if pos_dist < pos_threshold:
            position_good_configs.append(random_config)
            position_good_scores.append(pos_dist)
            position_dists.append(pos_dist)
    
    print(f"Found {len(position_good_configs)} configurations with position error < {pos_threshold}m")
    
    # If we don't have enough good positions, relax the threshold - but still aiming for precision
    if len(position_good_configs) < 5:
        print("Not enough precise positions found, slightly relaxing position threshold...")
        
        # Get the best positions with a slightly relaxed threshold, but still quite precise
        relaxed_threshold = 0.05  # 5cm - still reasonably precise
        all_configs = []
        all_pos_dists = []
        
        for _ in range(5000):  # Additional systematic samples
            random_config = [0.0] * len(planner.joint_indices)
            
            # Use a grid-based approach for the first few joints
            # to systematically explore the space
            if len(planner.joint_indices) >= 3:
                # Systematically vary the first three joints
                grid_size = int(round(5000 ** (1/3)))  # Cube root of sample count
                i1 = _ % grid_size
                i2 = (_ // grid_size) % grid_size
                i3 = _ // (grid_size * grid_size)
                
                # First joint - vertical position control
                lower1, upper1 = planner.joint_limits[0]
                random_config[0] = lower1 + (i1 / (grid_size-1)) * (upper1 - lower1)
                
                # Second and third joints - major arm positioning
                lower2, upper2 = planner.joint_limits[1]
                random_config[1] = lower2 + (i2 / (grid_size-1)) * (upper2 - lower2)
                
                lower3, upper3 = planner.joint_limits[2]
                random_config[2] = lower3 + (i3 / (grid_size-1)) * (upper3 - lower3)
                
                # Randomize the remaining joints
                for i in range(3, len(planner.joint_indices)):
                    lower, upper = planner.joint_limits[i]
                    random_config[i] = lower + np.random.random() * (upper - lower)
            else:
                # For robots with fewer joints, just use random sampling
                for i, (lower, upper) in enumerate(planner.joint_limits):
                    random_config[i] = lower + np.random.random() * (upper - lower)
            
            # Set the robot to this configuration
            for i, joint_idx in enumerate(planner.joint_indices):
                p.resetJointState(planner.robot_id, joint_idx, random_config[i])
            
            # Check the resulting end effector position
            link_state = p.getLinkState(planner.robot_id, goal_link_idx)
            pos = link_state[0]
            
            # Calculate distance to goal position
            pos_dist = math.sqrt(sum((pos[i] - goal_position[i])**2 for i in range(3)))
            
            if pos_dist < relaxed_threshold:
                all_configs.append(random_config)
                all_pos_dists.append(pos_dist)
        
        # Sort by position distance and take the best ones
        if all_configs:
            sorted_indices = np.argsort(all_pos_dists)
            print(f"Found {len(all_configs)} configurations with position error < {relaxed_threshold}m")
            for idx in sorted_indices[:20]:
                position_good_configs.append(all_configs[idx])
                position_good_scores.append(all_pos_dists[idx])
                position_dists.append(all_pos_dists[idx])
    
    # If we still don't have any good positions, we need to report this
    if len(position_good_configs) == 0:
        print("WARNING: Could not find any configurations within acceptable position error.")
        print("The target position may be outside the robot's reachable workspace.")
        
        # Find the best position we could achieve
        best_config = None
        best_dist = float('inf')
        best_pos = None
        
        for _ in range(1000):
            random_config = [0.0] * len(planner.joint_indices)
            for i, (lower, upper) in enumerate(planner.joint_limits):
                random_config[i] = lower + np.random.random() * (upper - lower)
            
            # Set the robot to this configuration
            for i, joint_idx in enumerate(planner.joint_indices):
                p.resetJointState(planner.robot_id, joint_idx, random_config[i])
            
            # Check the resulting end effector position
            link_state = p.getLinkState(planner.robot_id, goal_link_idx)
            pos = link_state[0]
            
            # Calculate distance to goal position
            pos_dist = math.sqrt(sum((pos[i] - goal_position[i])**2 for i in range(3)))
            
            if pos_dist < best_dist:
                best_dist = pos_dist
                best_config = random_config
                best_pos = pos
        
        if best_config:
            print(f"Best achievable position has error: {best_dist:.4f}m")
            print(f"Best position found: {best_pos}")
            
            # Add this as our fallback
            position_good_configs.append(best_config)
            position_good_scores.append(best_dist)
            position_dists.append(best_dist)
    
    # Second phase: Among good positions, optimize for orientation
    print("Phase 2: Optimizing orientation among good position configurations...")
    
    # Final candidates
    final_candidates = []
    final_scores = []
    pos_errors = []
    orient_errors = []
    
    # For each good position config, check orientation and compute final score
    for config, pos_dist in zip(position_good_configs, position_dists):
        # Set the robot to this configuration
        for i, joint_idx in enumerate(planner.joint_indices):
            p.resetJointState(planner.robot_id, joint_idx, config[i])
        
        # Get current pose
        link_state = p.getLinkState(planner.robot_id, goal_link_idx)
        pos = link_state[0]
        orn = link_state[1]
        
        # Calculate orientation distance
        orient_dist = quaternion_distance(orn, goal_orientation)
        
        # Combined score (weighted sum)
        # Normalize position distance by threshold for fair comparison
        norm_pos_dist = min(pos_dist / pos_threshold, 1.0) if pos_threshold > 0 else 1.0
        
        score = pos_weight * norm_pos_dist + orient_weight * orient_dist
        
        final_candidates.append(config)
        final_scores.append(score)
        pos_errors.append(pos_dist)
        orient_errors.append(orient_dist)
    
    # Sort candidates by score
    sorted_indices = np.argsort(final_scores)
    
    # Add the top configurations as goals
    valid_goals = 0
    for idx in sorted_indices[:10]:  # Take top 10
        config = final_candidates[idx]
        score = final_scores[idx]
        pos_error = pos_errors[idx]
        orient_error = orient_errors[idx]
        
        goal_state = ob.State(planner.space)
        for i, val in enumerate(config):
            goal_state[i] = val
        
        multi_goal.addState(goal_state)
        valid_goals += 1
        
        # Print details about this goal configuration
        for i, joint_idx in enumerate(planner.joint_indices):
            p.resetJointState(planner.robot_id, joint_idx, config[i])
        
        link_state = p.getLinkState(planner.robot_id, goal_link_idx)
        pos = link_state[0]
        orn = link_state[1]
        
        print(f"Goal {valid_goals}: Position error: {pos_error:.4f}m, Orientation error: {orient_error:.4f}, Score: {score:.4f}")
        print(f"  Position: {pos}")
        print(f"  Orientation: {orn}")
        print(f"  Joint values: {config}")
    
    print(f"Found {valid_goals} valid goal configurations")
    
    if valid_goals == 0:
        raise ValueError("Could not find any valid goal configurations. The goal might be unreachable.")
    
    # Set the goal in our planner
    planner.ss.setGoal(multi_goal)
    
    # Use RRT* planner
    planner_setup = og.RRTstar(si)
    planner_setup.setRange(0.1)  # Set step size
    planner.ss.setPlanner(planner_setup)
    
    # Solve
    print("Planning motion...")
    solved = planner.ss.solve(planning_time)
    
    # Extract path if found
    if solved:
        path = planner.ss.getSolutionPath()
        path.interpolate(100)  # Get smoother path with more waypoints
        
        # Extract joint values along the path
        joint_trajectory = []
        for i in range(path.getStateCount()):
            state = path.getState(i)
            joints = [state[j] for j in range(len(planner.joint_indices))]
            joint_trajectory.append(joints)
        
        print(f"Found a path with {len(joint_trajectory)} waypoints")
        
        # Print a subset of the path
        num_to_print = min(10, len(joint_trajectory))
        step_size = max(1, len(joint_trajectory) // num_to_print)
        for i in range(0, len(joint_trajectory), step_size):
            print(f"Step {i}: {joint_trajectory[i]}")
        
        # Verify final position and orientation
        final_joints = joint_trajectory[-1]
        print(f"Final joints: {final_joints}")
        for i, joint_idx in enumerate(planner.joint_indices):
            p.resetJointState(planner.robot_id, joint_idx, final_joints[i])
        
        final_state = p.getLinkState(planner.robot_id, goal_link_idx)
        final_pos = final_state[0]
        final_orn = final_state[1]
        
        # Calculate errors
        pos_error = [goal_position[i] - final_pos[i] for i in range(3)]
        pos_dist = math.sqrt(sum(e**2 for e in pos_error))
        orient_dist = quaternion_distance(final_orn, goal_orientation)
        
        print("Final position and orientation achieved:")
        print(f"  Position: {final_pos}")
        print(f"  Orientation: {final_orn}")
        print(f"  Target Position: {goal_position}")
        print(f"  Target Orientation: {goal_orientation}")
        print(f"  Position Error: {pos_error}")
        print(f"  Position Distance: {pos_dist:.4f}m")
        print(f"  Orientation Distance: {orient_dist:.4f}")
        
        return joint_trajectory
    else:
        print("No solution found")
        return None

if __name__ == "__main__":
    main()