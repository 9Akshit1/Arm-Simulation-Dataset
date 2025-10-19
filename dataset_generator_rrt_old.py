import mujoco
import mujoco.viewer
import numpy as np
import random
import xml.etree.ElementTree as ET
import os
import json
import time
import cv2
import math
from datetime import datetime
import argparse
from scipy.spatial.distance import cdist
from collections import deque
import imageio  

class SimpleEffectiveRRT:
    """Simple RRT implementation based on proven successful approach"""
    
    class Node:
        def __init__(self, q):
            self.q = np.array(q)
            self.path_q = []
            self.parent = None
    
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.obstacles = []
        self.arm_body_ids = []
        self.safety_margin = 0.05
        self.table_height = 0.50
        
        # Find arm bodies for collision checking
        arm_link_names = ['base_link', 'shoulder_link', 'half_arm_1_link', 'half_arm_2_link', 
                         'forearm_link', 'spherical_wrist_1_link', 'spherical_wrist_2_link', 'bracelet_link']
        
        for name in arm_link_names:
            try:
                body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
                if body_id >= 0:
                    self.arm_body_ids.append(body_id)
            except:
                continue
        print(f"Found {len(self.arm_body_ids)} arm bodies for collision checking")
    
    def update_obstacles(self, scene_info):
        """Update obstacle positions from scene"""
        self.obstacles = []
        
        if 'objects' not in scene_info or not scene_info['objects']:
            return
            
        for obj in scene_info['objects']:
            if 'name' in obj:
                try:
                    obj_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, obj['name'])
                    if obj_body_id >= 0:
                        obj_pos = self.data.body(obj_body_id).xpos.copy()
                        
                        # Simple obstacle representation
                        obstacle = {
                            'position': obj_pos,
                            'size': 0.08,  # Fixed radius for simplicity
                            'name': obj['name']
                        }
                        self.obstacles.append(obstacle)
                        print(f"Added obstacle: {obj['name']} at {obj_pos}")
                except:
                    continue
    
    def get_joint_limits(self, arm_joint_indices):
        """Get joint limits"""
        limits = []
        for joint_idx in arm_joint_indices:
            if joint_idx < len(self.model.jnt_range):
                joint_range = self.model.jnt_range[joint_idx]
                if not (np.isinf(joint_range[0]) or np.isinf(joint_range[1])):
                    limits.append([joint_range[0] + 0.05, joint_range[1] - 0.05])
                else:
                    limits.append([-np.pi + 0.05, np.pi - 0.05])
            else:
                limits.append([-np.pi + 0.05, np.pi - 0.05])
        return limits
    
    def check_collision(self, joint_config, arm_joint_indices):
        """Simple and effective collision checking like the successful implementation"""
        # Store original configuration
        original_q = np.array([self.data.qpos[idx] for idx in arm_joint_indices])
        
        # Set test configuration
        for i, joint_idx in enumerate(arm_joint_indices):
            if i < len(joint_config):
                self.data.qpos[joint_idx] = joint_config[i]
        
        # Forward kinematics
        mujoco.mj_forward(self.model, self.data)
        
        collision_free = True
        
        # Check table collision
        for arm_body_id in self.arm_body_ids[-4:]:  # Only check end links
            arm_pos = self.data.body(arm_body_id).xpos
            if arm_pos[2] < self.table_height:
                collision_free = False
                break
        
        # Check object collisions
        if collision_free and len(self.obstacles) > 0:
            for arm_body_id in self.arm_body_ids[-3:]:  # Only check last 3 links
                arm_pos = self.data.body(arm_body_id).xpos
                
                for obstacle in self.obstacles:
                    distance = np.linalg.norm(arm_pos - obstacle['position'])
                    if distance < obstacle['size'] + self.safety_margin:
                        collision_free = False
                        break
                
                if not collision_free:
                    break
        
        # Restore original configuration
        for i, joint_idx in enumerate(arm_joint_indices):
            self.data.qpos[joint_idx] = original_q[i]
        mujoco.mj_forward(self.model, self.data)
        
        return collision_free
    
    def planning(self, start, goal, arm_joint_indices, expand_dis=0.1, path_resolution=0.02, 
                goal_sample_rate=20, max_iter=2000):
        """Main RRT planning function - simplified and effective"""
        
        joint_limits = self.get_joint_limits(arm_joint_indices)
        
        # Initialize with start node
        start_node = self.Node(start)
        node_list = [start_node]
        
        print(f"Starting RRT planning with {max_iter} iterations...")
        
        for iteration in range(max_iter):
            # Sample random node
            if np.random.randint(0, 100) > goal_sample_rate:
                # Random sampling
                rand_q = []
                for limit in joint_limits:
                    rand_q.append(np.random.uniform(limit[0], limit[1]))
                rnd_node = self.Node(rand_q)
            else:
                # Goal-biased sampling
                rnd_node = self.Node(goal)
            
            # Find nearest node
            nearest_ind = self.get_nearest_node_index(node_list, rnd_node)
            nearest_node = node_list[nearest_ind]
            
            # Steer towards random node
            new_node = self.steer(nearest_node, rnd_node, expand_dis, path_resolution, joint_limits)
            
            # Check if new node is collision-free
            if self.check_collision(new_node.q, arm_joint_indices):
                node_list.append(new_node)
                
                # Check if we reached the goal
                if self.calc_dist_to_goal(new_node.q, goal) <= expand_dis:
                    final_node = self.steer(new_node, self.Node(goal), expand_dis, path_resolution, joint_limits)
                    if self.check_collision(final_node.q, arm_joint_indices):
                        print(f"Goal reached at iteration {iteration}!")
                        return self.generate_final_path(len(node_list) - 1, node_list)
            
            if iteration % 200 == 0:
                print(f"RRT iteration {iteration}, nodes: {len(node_list)}")
        
        print("RRT could not find complete path to goal")
        
        # Return partial path to closest node
        if len(node_list) > 1:
            closest_node_idx = self.find_closest_to_goal(node_list, goal)
            if closest_node_idx > 0:
                print(f"Returning partial path to closest node (distance: {self.calc_dist_to_goal(node_list[closest_node_idx].q, goal):.3f})")
                return self.generate_final_path(closest_node_idx, node_list)
        
        return None
    
    def get_nearest_node_index(self, node_list, rnd_node):
        """Find nearest node to random node"""
        distances = [np.linalg.norm(node.q - rnd_node.q) for node in node_list]
        return distances.index(min(distances))
    
    def steer(self, from_node, to_node, extend_length, path_resolution, joint_limits):
        """Steer from one node towards another"""
        new_node = self.Node(from_node.q.copy())
        
        direction = to_node.q - from_node.q
        distance = np.linalg.norm(direction)
        
        if extend_length > distance:
            extend_length = distance
        
        # Calculate number of steps
        num_steps = max(1, int(extend_length / path_resolution))
        delta_q = direction / num_steps
        
        # Step towards target
        for i in range(num_steps):
            new_q = new_node.q + delta_q
            
            # Apply joint limits
            for j, limit in enumerate(joint_limits):
                if j < len(new_q):
                    new_q[j] = np.clip(new_q[j], limit[0], limit[1])
            
            new_node.q = new_q
            new_node.path_q.append(new_q.copy())
        
        new_node.parent = from_node
        return new_node
    
    def calc_dist_to_goal(self, q, goal):
        """Calculate distance to goal"""
        return np.linalg.norm(np.array(goal) - np.array(q))
    
    def find_closest_to_goal(self, node_list, goal):
        """Find node closest to goal"""
        distances = [self.calc_dist_to_goal(node.q, goal) for node in node_list]
        return distances.index(min(distances))
    
    def generate_final_path(self, goal_ind, node_list):
        """Generate final path from goal to start"""
        path = []
        node = node_list[goal_ind]
        
        while node.parent is not None:
            # Add intermediate path points
            path.extend(reversed(node.path_q))
            node = node.parent
        
        # Add start configuration
        path.append(node_list[0].q)
        
        # Reverse to get start-to-goal path
        final_path = path[::-1]
        print(f"Generated path with {len(final_path)} waypoints")
        return final_path
    
class GaussNewtonIK:
    """Gauss-Newton IK solver - EXACTLY matching the working all_kinova.py implementation"""
    
    def __init__(self, model, step_size=0.5, tol=0.01, max_iter=1000):
        self.model = model
        self.step_size = step_size
        self.tol = tol
        self.max_iter = max_iter
        self.iterations = 0
        self.converged = False
        
        # Pre-allocate jacobians for speed - EXACTLY like all_kinova.py
        self.jacp = np.zeros((3, model.nv))
        self.jacr = np.zeros((3, model.nv))
        
    def check_joint_limits(self, q, joint_indices):
        """Check and clamp joint limits with reduced logging"""
        q_limited = q.copy()
        clamp_count = 0
        
        for i, joint_idx in enumerate(joint_indices):
            if joint_idx < self.model.njnt and joint_idx < len(self.model.jnt_range):
                joint_range = self.model.jnt_range[joint_idx]
                # Check if joint has finite limits
                if not (np.isinf(joint_range[0]) or np.isinf(joint_range[1])):
                    original_val = q_limited[i]
                    q_limited[i] = np.clip(q_limited[i], joint_range[0], joint_range[1])
                    
                    # Only log if significant clamping occurs
                    if abs(q[i] - q_limited[i]) > 0.1:
                        clamp_count += 1
        
        # Only print summary if many joints were clamped
        if clamp_count > 2:
            print(f"Clamped {clamp_count} joints to limits")
            
        return q_limited
    
    def calculate_error(self, data, goal, body_id):
        """Calculate position error - EXACTLY like all_kinova.py"""
        current_pos = data.body(body_id).xpos
        return goal - current_pos
    
    def solve(self, data, goal, init_q, body_id, arm_joint_indices, arm_actuator_indices, table_height=0.50):
        """Solve using Gauss-Newton method with table collision avoidance"""
        # Ensure init_q matches arm DOF count
        if len(init_q) != len(arm_joint_indices):
            init_q = np.zeros(len(arm_joint_indices))

        # Set arm joint positions
        for i, joint_idx in enumerate(arm_joint_indices):
            data.qpos[joint_idx] = init_q[i]
        mujoco.mj_forward(self.model, data)

        self.iterations = 0
        self.converged = False

        for i in range(self.max_iter):
            error = self.calculate_error(data, goal, body_id)
            error_norm = np.linalg.norm(error)

            if error_norm < self.tol:
                self.converged = True
                break

            # Check if goal is below table - reject immediately
            if goal[2] < table_height:
                print(f"WARNING: Goal {goal} is below table height {table_height}, rejecting")
                break

            # CRITICAL FIX: Calculate jacobian at current position, not goal
            current_pos = data.body(body_id).xpos
            mujoco.mj_jac(self.model, data, self.jacp, self.jacr, current_pos, body_id)

            # Extract jacobian columns for arm joints only
            arm_jacp = self.jacp[:, arm_joint_indices]

            # Check for singularity
            if np.linalg.norm(arm_jacp) < 1e-8:
                print(f"Warning: Near-singular jacobian at iteration {i}")
                break

            # Gauss-Newton update with better regularization
            JTJ = arm_jacp.T @ arm_jacp
            
            # Adaptive regularization
            condition_num = np.linalg.cond(JTJ)
            if condition_num > 1e6:
                reg_factor = 1e-3
            else:
                reg_factor = 1e-4
            
            reg = reg_factor * np.eye(JTJ.shape[0])

            try:
                if np.linalg.det(JTJ + reg) > 1e-8:
                    j_inv = np.linalg.inv(JTJ + reg) @ arm_jacp.T
                else:
                    j_inv = np.linalg.pinv(arm_jacp, rcond=1e-4)
            except np.linalg.LinAlgError:
                j_inv = np.linalg.pinv(arm_jacp, rcond=1e-4)

            delta_q = j_inv @ error

            # CRITICAL FIX: Limit step size
            max_step = 0.3
            delta_q_norm = np.linalg.norm(delta_q)
            if delta_q_norm > max_step:
                delta_q = delta_q * (max_step / delta_q_norm)

            # Update arm joint positions
            current_q = np.array([data.qpos[idx] for idx in arm_joint_indices])
            new_q = current_q + self.step_size * delta_q

            # Apply joint limits
            new_q = self.check_joint_limits(new_q, arm_joint_indices)

            # Set new joint positions
            for j, joint_idx in enumerate(arm_joint_indices):
                data.qpos[joint_idx] = new_q[j]

            # Forward kinematics
            mujoco.mj_forward(self.model, data)
            
            # Check if end effector would go below table
            current_pos = data.body(body_id).xpos
            if current_pos[2] < table_height:
                print(f"WARNING: End effector at {current_pos} would go below table height {table_height}")
                # Revert to previous position
                for j, joint_idx in enumerate(arm_joint_indices):
                    data.qpos[joint_idx] = current_q[j]
                mujoco.mj_forward(self.model, data)
                break
                
            self.iterations = i + 1

        return np.array([data.qpos[idx] for idx in arm_joint_indices])

class CollisionAwareIK(GaussNewtonIK):
    """IK solver with simple RRT integration"""
    
    def __init__(self, model, rrt_system, step_size=0.5, tol=0.01, max_iter=1000):
        super().__init__(model, step_size, tol, max_iter)
        self.rrt_system = rrt_system
        
    def solve_with_obstacles(self, data, goal, init_q, body_id, arm_joint_indices, arm_actuator_indices, table_height=0.50):
        """Solve IK with RRT path planning when needed"""
        
        print(f"Solving IK with RRT for goal: {goal}")
        
        # Check if goal is valid
        if goal[2] < table_height:
            print(f"Goal {goal} is below table height")
            return None
        
        # Try direct IK first
        try:
            direct_result = self.solve(data, goal, init_q or np.zeros(len(arm_joint_indices)), 
                                     body_id, arm_joint_indices, arm_actuator_indices, table_height)
            
            # Check if direct result is collision-free
            if self.rrt_system.check_collision(direct_result, arm_joint_indices):
                print("Direct IK solution is collision-free")
                return direct_result
        except:
            pass
        
        print("Direct IK failed or has collisions, using RRT...")
        
        # Generate multiple IK solutions as potential goals
        ik_goals = self.generate_ik_candidates(data, goal, body_id, arm_joint_indices, arm_actuator_indices, table_height)
        
        if not ik_goals:
            print("No valid IK solutions found")
            return None
        
        # Get current configuration
        start_config = np.array([data.qpos[idx] for idx in arm_joint_indices])
        
        # Try RRT to each candidate goal
        for i, goal_config in enumerate(ik_goals):
            print(f"Trying RRT to candidate goal {i+1}/{len(ik_goals)}")
            
            path = self.rrt_system.planning(
                start=start_config,
                goal=goal_config,
                arm_joint_indices=arm_joint_indices,
                expand_dis=0.02,  # REDUCED: Much smaller steps for more precise planning
                path_resolution=0.005,  # REDUCED: Much finer resolution for smoother movement
                goal_sample_rate=30,  # Higher goal bias
                max_iter=2000  # More iterations for finer movement
            )
            
            if path and len(path) > 1:
                print(f"RRT found path with {len(path)} waypoints")
                return path[-1]  # Return final configuration
        
        print("RRT failed to find path to any goal")
        return None
    
    def generate_ik_candidates(self, data, goal, body_id, arm_joint_indices, arm_actuator_indices, table_height):
        """Generate multiple IK solutions as RRT goals"""
        
        candidates = []
        
        # Try different initialization strategies
        init_strategies = [
            np.zeros(len(arm_joint_indices)),
            np.array([0.0, 0.3, 0.0, 1.2, 0.0, 0.0, 0.0])[:len(arm_joint_indices)],
            np.array([0.5, 0.2, 0.0, 1.0, 0.0, 0.3, 0.0])[:len(arm_joint_indices)],
            np.array([-0.5, 0.2, 0.0, 1.0, 0.0, 0.3, 0.0])[:len(arm_joint_indices)],
        ]
        
        # Add some random initializations
        joint_limits = self.rrt_system.get_joint_limits(arm_joint_indices)
        for _ in range(3):
            random_init = []
            for limit in joint_limits:
                random_init.append(np.random.uniform(limit[0], limit[1]))
            init_strategies.append(np.array(random_init))
        
        for init_q in init_strategies:
            try:
                result_q = self.solve(data, goal, init_q, body_id, arm_joint_indices, arm_actuator_indices, table_height)
                
                # Verify this solution reaches the goal
                for j, joint_idx in enumerate(arm_joint_indices):
                    data.qpos[joint_idx] = result_q[j]
                mujoco.mj_forward(self.model, data)
                
                achieved_pos = data.body(body_id).xpos
                error = np.linalg.norm(goal - achieved_pos)
                
                if error < 0.08:  # Accept solutions within 8cm
                    candidates.append(result_q)
                    print(f"Generated IK candidate with error {error:.4f}")
                    
            except:
                continue
        
        print(f"Generated {len(candidates)} IK candidates")
        return candidates
    
class DatasetGenerator:
    def __init__(self, output_dir="dataset", assets_dir="dataset/scenes/assets", 
             validation=False, num_previous_actions=1):
        self.output_dir = output_dir
        self.assets_dir = assets_dir
        self.scene_counter = 0
        self.validation = validation
        self.num_previous_actions = num_previous_actions
        self.episode_counter = 0
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "scenes"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "annotations"), exist_ok=True)

        # Create geometric meshes if they don't exist
        self.create_geometric_meshes()
        
        # Available objects in assets folder
        self.available_objects = [
            "apple.stl", 
            "banana.stl", 
            "bottle.stl", 
            "bowl.stl", 
            "computer_mouse.stl",
            "cup.stl", 
            "minion.stl",
            "robot.stl", 
            "teddy_bear.stl", 
            #"sphere.stl",
            #"cylinder.stl",
            #"cube.stl"
        ]
        
        n = random.choice([2, 3])  # Randomize between 2 and 3 for object size generalizability

        # Object scaling factors
        self.object_scales = {
            "apple.stl": 0.12 / n, 
            "banana.stl": 0.001 / n, 
            "book.stl": 0.0065 / n, 
            "bottle.stl": 0.002 / n,  
            "bowl.stl": 0.0007 / n,  
            "computer_mouse.stl": 0.0013 / n, 
            "cup.stl": 0.001 / n,  
            "dinner_plate.stl": 0.007 / n,  
            "minion.stl": 0.0012 / n, 
            "robot.stl": 0.0004 / n, 
            "teddy_bear.stl": 0.003 / n, 
            "vase.stl": 0.0018 / n, 
            "sphere.stl": 0.02 / n, 
            "cylinder.stl": 0.02 / n, 
            "cube.stl": 0.02 / n, 
            "default": 0.002 / n
        }
                
        # Enhanced object colors with textures
        self.object_colors = [
            ("object_red", [0.8, 0.2, 0.2, 1]),
            ("object_blue", [0.2, 0.2, 0.8, 1]),
            ("object_green", [0.2, 0.8, 0.2, 1]),
            ("object_yellow", [0.8, 0.8, 0.2, 1]),
            ("object_purple", [0.8, 0.2, 0.8, 1]),
            ("object_orange", [0.8, 0.5, 0.2, 1]),
            ("object_cyan", [0.2, 0.8, 0.8, 1]),
            ("object_pink", [0.8, 0.4, 0.6, 1]),
            ("object_brown", [0.6, 0.4, 0.2, 1]),
            ("object_gray", [0.5, 0.5, 0.5, 1]),
            ("object_white", [0.9, 0.9, 0.9, 1]),
            ("object_black", [0.2, 0.2, 0.2, 1]),
            ("object_stripe_red", [0.8, 0.2, 0.2, 1]),
            ("object_stripe_blue", [0.2, 0.2, 0.8, 1]),
            ("object_dots_green", [0.2, 0.8, 0.2, 1])
        ]
        
        # Enhanced materials
        self.ground_materials = [
            "groundplane1", "groundplane2", "groundplane3", 
            "groundplane4", "groundplane5", "groundplane6"
        ]
        self.table_materials = [
            "wood_mat1", "wood_mat2", "wood_mat3", "wood_mat4", "wood_mat5",
            "marble_mat1", "marble_mat2", "metal_mat1"
        ]
        
        # Lighting configurations
        self.lighting_configs = [
            {"main": [0.5, 0, 2.0], "aux1": [1.0, 0.8, 1.5], "aux2": [-0.3, 0.8, 1.2]},
            {"main": [0.8, 0.3, 1.8], "aux1": [0.5, -0.5, 1.3], "aux2": [0.2, 1.0, 1.0]},
            {"main": [0.2, 0, 2.2], "aux1": [1.2, 0.2, 1.4], "aux2": [-0.5, 0.3, 1.6]},
            {"main": [0.6, -0.2, 1.9], "aux1": [0.8, 1.0, 1.2], "aux2": [-0.2, -0.5, 1.4]},
            {"main": [0.3, 0.5, 2.1], "aux1": [1.1, -0.3, 1.6], "aux2": [0.1, 0.9, 1.1]}
        ]

        # Based on arm base at [0.2, 0, 0.45] from gen3.xml
        self.arm_base_position = [0.2, 0, 0.45]
        self.obstacle_system = None  # Will be initialized per scene
        self.arm_reach = 0.9  # Gen3 arm reach is approximately 90cm

        # Workspace constraints for target generation
        self.workspace_limits = {
            'min_radius': 0.5,   # INCREASED: Minimum distance from arm base
            'max_radius': 0.85,  # Keep same
            'min_height': 0.9,   # INCREASED: Higher minimum height
            'max_height': 1.6,   # INCREASED: Much higher maximum
            'min_angle': -60,    # Keep same
            'max_angle': 120,    # Keep same
            'preferred_height_range': [1.0, 1.4]  # INCREASED: Much higher preferred range
        }
                                        
        # Table position variations - matching gen3_scene.xml
        self.table_positions = [
            [0.6, 0, 0.43],      # Default from gen3_scene.xml
            [0.65, 0.05, 0.43],  # Slightly right
            [0.55, -0.05, 0.43], # Slightly left
            [0.6, 0.08, 0.43],   # Forward
            [0.6, -0.08, 0.43]   # Backward
        ]

    def convert_to_base_relative_coords(self, world_pos, base_pos):
        """Convert world coordinates to base-link relative coordinates"""
        return np.array(world_pos) - np.array(base_pos)

    def convert_to_world_coords(self, base_relative_pos, base_pos):
        """Convert base-link relative coordinates to world coordinates"""
        return np.array(base_relative_pos) + np.array(base_pos)

    def create_geometric_meshes(self):
        """Create geometric primitive meshes programmatically"""
        if not os.path.exists(self.assets_dir):
            os.makedirs(self.assets_dir, exist_ok=True)
        
        # Create sphere STL
        sphere_path = os.path.join(self.assets_dir, "sphere.stl")
        if not os.path.exists(sphere_path):
            self.create_sphere_stl(sphere_path, radius=1.0, resolution=20)
        
        # Create cylinder STL  
        cylinder_path = os.path.join(self.assets_dir, "cylinder.stl")
        if not os.path.exists(cylinder_path):
            self.create_cylinder_stl(cylinder_path, radius=1.0, height=2.0, resolution=20)
        
        # Create cube STL
        cube_path = os.path.join(self.assets_dir, "cube.stl")
        if not os.path.exists(cube_path):
            self.create_cube_stl(cube_path, size=2.0)

    def create_sphere_stl(self, filepath, radius=1.0, resolution=20):
        """Create a complete sphere STL file"""
        import numpy as np
        
        vertices = []
        faces = []
        
        # Generate sphere vertices using spherical coordinates
        for i in range(resolution + 1):
            lat = np.pi * (-0.5 + float(i) / resolution)  # Latitude from -π/2 to π/2
            for j in range(resolution):
                lng = 2 * np.pi * float(j) / resolution  # Longitude from 0 to 2π
                x = radius * np.cos(lat) * np.cos(lng)
                y = radius * np.cos(lat) * np.sin(lng)
                z = radius * np.sin(lat)
                vertices.append([x, y, z])
        
        # Generate faces (triangles) connecting the vertices
        for i in range(resolution):
            for j in range(resolution):
                # Current vertex indices
                v1 = i * resolution + j
                v2 = i * resolution + (j + 1) % resolution
                v3 = (i + 1) * resolution + j
                v4 = (i + 1) * resolution + (j + 1) % resolution
                
                # Skip the poles to avoid degenerate triangles
                if i == 0:  # Top cap
                    faces.append([v3, v1, v4])
                elif i == resolution - 1:  # Bottom cap
                    faces.append([v1, v3, v2])
                else:  # Middle sections - create two triangles per quad
                    faces.append([v1, v3, v2])
                    faces.append([v2, v3, v4])
        
        self.write_stl_file(filepath, vertices, faces)

    def create_cylinder_stl(self, filepath, radius=1.0, height=2.0, resolution=20):
        """Create a complete cylinder STL file"""
        import numpy as np
        
        vertices = []
        faces = []
        
        # Create vertices for bottom circle (z = -height/2)
        bottom_center_idx = 0
        vertices.append([0, 0, -height/2])  # Bottom center
        
        for i in range(resolution):
            angle = 2 * np.pi * i / resolution
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            vertices.append([x, y, -height/2])
        
        # Create vertices for top circle (z = height/2)
        top_center_idx = resolution + 1
        vertices.append([0, 0, height/2])  # Top center
        
        for i in range(resolution):
            angle = 2 * np.pi * i / resolution
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            vertices.append([x, y, height/2])
        
        # Generate bottom face triangles (looking up from below)
        for i in range(resolution):
            next_i = (i + 1) % resolution
            bottom_v1 = 1 + i
            bottom_v2 = 1 + next_i
            faces.append([bottom_center_idx, bottom_v2, bottom_v1])  # Counter-clockwise when viewed from below
        
        # Generate top face triangles (looking down from above)
        for i in range(resolution):
            next_i = (i + 1) % resolution
            top_v1 = top_center_idx + 1 + i
            top_v2 = top_center_idx + 1 + next_i
            faces.append([top_center_idx, top_v1, top_v2])  # Counter-clockwise when viewed from above
        
        # Generate side face triangles
        for i in range(resolution):
            next_i = (i + 1) % resolution
            
            # Bottom vertices
            bottom_v1 = 1 + i
            bottom_v2 = 1 + next_i
            
            # Top vertices
            top_v1 = top_center_idx + 1 + i
            top_v2 = top_center_idx + 1 + next_i
            
            # Two triangles per side face
            faces.append([bottom_v1, top_v1, bottom_v2])  # First triangle
            faces.append([bottom_v2, top_v1, top_v2])     # Second triangle
        
        self.write_stl_file(filepath, vertices, faces)

    def create_cube_stl(self, filepath, size=2.0):
        """Create a complete cube STL file"""
        s = size / 2  # Half size for center-based coordinates
        
        # Define 8 vertices of the cube
        vertices = [
            # Bottom face (z = -s)
            [-s, -s, -s],  # 0: bottom-left-back
            [s, -s, -s],   # 1: bottom-right-back
            [s, s, -s],    # 2: bottom-right-front
            [-s, s, -s],   # 3: bottom-left-front
            
            # Top face (z = s)
            [-s, -s, s],   # 4: top-left-back
            [s, -s, s],    # 5: top-right-back
            [s, s, s],     # 6: top-right-front
            [-s, s, s]     # 7: top-left-front
        ]
        
        # Define 12 triangular faces (2 triangles per cube face)
        faces = [
            # Bottom face (z = -s) - normal pointing down (0, 0, -1)
            [0, 2, 1],  # Triangle 1
            [0, 3, 2],  # Triangle 2
            
            # Top face (z = s) - normal pointing up (0, 0, 1)
            [4, 5, 6],  # Triangle 1
            [4, 6, 7],  # Triangle 2
            
            # Front face (y = s) - normal pointing forward (0, 1, 0)
            [3, 6, 2],  # Triangle 1
            [3, 7, 6],  # Triangle 2
            
            # Back face (y = -s) - normal pointing backward (0, -1, 0)
            [0, 1, 5],  # Triangle 1
            [0, 5, 4],  # Triangle 2
            
            # Right face (x = s) - normal pointing right (1, 0, 0)
            [1, 2, 6],  # Triangle 1
            [1, 6, 5],  # Triangle 2
            
            # Left face (x = -s) - normal pointing left (-1, 0, 0)
            [0, 4, 7],  # Triangle 1
            [0, 7, 3],  # Triangle 2
        ]
        
        self.write_stl_file(filepath, vertices, faces)

    def write_stl_file(self, filepath, vertices, faces):
        """Write vertices and faces to STL file with proper normals"""
        import struct
        
        with open(filepath, 'wb') as f:
            # Write 80-byte header
            header = b'Binary STL file created by MuJoCo dataset generator' + b'\x00' * (80 - 47)
            f.write(header)
            
            # Write number of triangles
            f.write(struct.pack('<I', len(faces)))
            
            # Write triangles
            for face in faces:
                if len(face) != 3:
                    continue  # Skip invalid faces
                    
                try:
                    v1, v2, v3 = [vertices[i] for i in face]
                    
                    # Calculate normal vector using cross product
                    u = [v2[j] - v1[j] for j in range(3)]  # Edge 1
                    v = [v3[j] - v1[j] for j in range(3)]  # Edge 2
                    
                    # Cross product u × v
                    normal = [
                        u[1] * v[2] - u[2] * v[1],  # nx
                        u[2] * v[0] - u[0] * v[2],  # ny
                        u[0] * v[1] - u[1] * v[0]   # nz
                    ]
                    
                    # Normalize the normal vector
                    normal_length = (normal[0]**2 + normal[1]**2 + normal[2]**2)**0.5
                    if normal_length > 1e-10:  # Avoid division by zero
                        normal = [n / normal_length for n in normal]
                    else:
                        normal = [0, 0, 1]  # Default normal if calculation fails
                    
                    # Write normal vector (3 floats)
                    for coord in normal:
                        f.write(struct.pack('<f', float(coord)))
                    
                    # Write vertices (9 floats total)
                    for vertex in [v1, v2, v3]:
                        for coord in vertex:
                            f.write(struct.pack('<f', float(coord)))
                    
                    # Write attribute byte count (2 bytes, usually 0)
                    f.write(struct.pack('<H', 0))
                    
                except (IndexError, ValueError) as e:
                    print(f"Warning: Skipping invalid face {face}: {e}")
                    continue

    def find_arm_indices(self, model):
        """Find arm joint and actuator indices - EXACTLY matching all_kinova.py"""
        arm_joint_indices = []
        arm_actuator_indices = []
        
        print("Finding arm joints and actuators for gen3...")
        
        # Based on gen3.xml, the joints are named joint_1 through joint_7
        expected_joint_names = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6', 'joint_7']
        
        # Find joints by exact name match
        for expected_name in expected_joint_names:
            try:
                joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, expected_name)
                if joint_id >= 0:
                    arm_joint_indices.append(joint_id)
                    print(f"Found joint: {expected_name} (index {joint_id})")
            except:
                print(f"Warning: Could not find joint {expected_name}")
        
        # Find actuators by exact name match (same names as joints in gen3.xml)
        for expected_name in expected_joint_names:
            try:
                actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, expected_name)
                if actuator_id >= 0:
                    arm_actuator_indices.append(actuator_id)
                    print(f"Found actuator: {expected_name} (index {actuator_id})")
            except:
                print(f"Warning: Could not find actuator {expected_name}")
        
        # Fallback: if exact names don't work, use pattern matching
        if not arm_joint_indices:
            print("Fallback: Using pattern matching for joints...")
            for i in range(model.njnt):
                try:
                    joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
                    if joint_name and 'joint_' in joint_name:
                        arm_joint_indices.append(i)
                        print(f"Found joint (fallback): {joint_name} (index {i})")
                except:
                    continue
        
        if not arm_actuator_indices:
            print("Fallback: Using pattern matching for actuators...")
            for i in range(model.nu):
                try:
                    actuator_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                    if actuator_name and 'joint_' in actuator_name:
                        arm_actuator_indices.append(i)
                        print(f"Found actuator (fallback): {actuator_name} (index {i})")
                except:
                    continue
        
        print(f"Found {len(arm_joint_indices)} arm joints: {arm_joint_indices}")
        print(f"Found {len(arm_actuator_indices)} arm actuators: {arm_actuator_indices}")
        
        return arm_joint_indices, arm_actuator_indices

    def get_end_effector_body_id(self, model):
        """Find the end effector body ID - EXACTLY matching all_kinova.py"""
        # Based on gen3.xml, the end effector is 'bracelet_link'
        end_effector_names = [
            'bracelet_link',           # Exact name from gen3.xml
            'bracelet_with_vision_link', # Alternative
            'spherical_wrist_2_link',  # Second best option
            'spherical_wrist_1_link',  # Third option
            'ee_link',                 # Generic fallback
            'end_effector',
            'gripper',
            'tool0',
            'ee'
        ]
        
        print("Searching for end effector body...")
        
        for name in end_effector_names:
            try:
                body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
                if body_id >= 0:  # Valid body ID
                    print(f"Found end effector body: '{name}' (ID: {body_id})")
                    return body_id
            except:
                continue
        
        # Final fallback: find a body with "bracelet" or "wrist" in the name
        print("Searching for bodies with 'bracelet' or 'wrist' in name...")
        for i in range(model.nbody):
            try:
                body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
                if body_name and ('bracelet' in body_name.lower() or 'wrist' in body_name.lower()):
                    print(f"Found end effector body (pattern match): '{body_name}' (ID: {i})")
                    return i
            except:
                continue
        
        # Last resort: use the last body
        if model.nbody > 0:
            body_id = model.nbody - 1
            try:
                body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
                print(f"Using last body as end effector: '{body_name}' (ID: {body_id})")
            except:
                print(f"Using last body as end effector (ID: {body_id})")
            return body_id
        else:
            raise ValueError("No bodies found in model!")
        
    def get_end_effector_site_id(self, model):
        """Find the end effector site ID for pinch_site"""
        try:
            site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "pinch_site")
            if site_id >= 0:
                print(f"Found end effector site: 'pinch_site' (ID: {site_id})")
                return site_id
        except:
            pass
        
        print("Warning: pinch_site not found, using end effector body as fallback")
        return None
    
    def calculate_objects_center(self, scene_info):
        """Calculate the 3D center point of all objects on the table"""
        if 'objects' not in scene_info or not scene_info['objects']:
            # Fallback to table center if no objects
            table_pos = scene_info['config']['table_position']
            return [table_pos[0], table_pos[1], table_pos[2] + 0.05]
        
        total_x, total_y, total_z = 0, 0, 0
        count = 0
        
        for obj in scene_info['objects']:
            pos = obj['position']
            total_x += pos[0]
            total_y += pos[1] 
            total_z += pos[2]
            count += 1
        
        if count == 0:
            table_pos = scene_info['config']['table_position']
            return [table_pos[0], table_pos[1], table_pos[2] + 0.05]
        
        center = [total_x/count, total_y/count, total_z/count]
        print(f"Calculated objects center: {center}")
        return center
    
    def generate_valid_target_position(self, table_position, objects_center=None):
        """
        Generate a valid target position that ALWAYS points toward the table/objects.
        Ensures camera never goes behind the arm base.
        """
        import math
        
        arm_base = np.array(self.arm_base_position)
        table_pos = np.array(table_position)
        
        # If objects center is provided, use it; otherwise use table center
        if objects_center is not None:
            focus_point = np.array(objects_center)
        else:
            focus_point = table_pos
        
        # CRITICAL: Only allow positions that are FORWARD of the arm base
        # and point TOWARD the table/objects
        
        # Calculate direction from arm base to focus point
        to_focus = focus_point - arm_base
        to_focus_normalized = to_focus / np.linalg.norm(to_focus)
        
        # Generate position in a cone pointing toward the focus point
        # This ensures we never go "behind" the arm
        
        best_position = None
        best_score = -1
        
        for attempt in range(30):
            # CHANGE THESE VALUES FOR HIGHER CAMERA POSITIONS:
            lateral_offset = random.uniform(-0.15, 0.15)  # Small lateral movement
            vertical_offset = random.uniform(0.4, 0.8)    # INCREASED: Much higher above focus point
            distance_offset = random.uniform(-0.3, -0.1)  # INCREASED: Further back from focus for wider view
            
            # Calculate position
            # Start from focus point and move back toward arm slightly for good viewing
            base_to_focus = focus_point - arm_base
            distance_to_focus = np.linalg.norm(base_to_focus)
            
            # Position along the line from arm to focus, but pulled back for viewing
            viewing_distance = distance_to_focus + distance_offset
            
            # CHANGE THESE LIMITS FOR WIDER VIEW:
            viewing_distance = np.clip(viewing_distance, 
                                    0.5,   # INCREASED: Minimum distance for wider view
                                    self.workspace_limits['max_radius'])
            
            # Calculate target position
            direction = to_focus_normalized
            target_pos = arm_base + direction * viewing_distance
            
            # Add small lateral offset
            # Create perpendicular vector for lateral movement
            up_vector = np.array([0, 0, 1])
            right_vector = np.cross(direction, up_vector)
            right_vector = right_vector / (np.linalg.norm(right_vector) + 1e-8)
            
            target_pos += right_vector * lateral_offset
            target_pos[2] += vertical_offset  # Always lift up for better view
            
            # CHANGE HEIGHT CONSTRAINTS FOR OVERVIEW:
            target_pos[2] = np.clip(target_pos[2], 
                                0.9,   # INCREASED: Higher minimum height
                                1.6)   # INCREASED: Allow much higher positions
            
            # CRITICAL CHECK: Ensure position is FORWARD of arm base (positive X direction from arm)
            if target_pos[0] <= arm_base[0]:
                continue  # Skip positions behind or at arm base
                
            # Check if position points toward focus
            camera_to_focus = focus_point - target_pos
            if np.linalg.norm(camera_to_focus) < 0.3:  # INCREASED: Minimum distance for overview
                continue  # Too close
                
            # Score based on viewing angle and distance
            score = self.score_target_position(target_pos.tolist(), arm_base, focus_point, table_pos)
            
            if score > best_score:
                best_score = score
                best_position = target_pos.tolist()
        
        if best_position is None:
            # CHANGE FALLBACK POSITION FOR OVERVIEW:
            safe_pos = [
                arm_base[0] + 0.4,  # Always forward of arm
                arm_base[1] + 0.1,  # Slightly to the right
                arm_base[2] + 0.6   # INCREASED: Much higher above arm base
            ]
            best_position = safe_pos
            print("Using guaranteed safe fallback position")
        
        print(f"Generated target position: {best_position} (score: {best_score:.3f})")
        print(f"Position relative to arm base: {np.array(best_position) - arm_base}")
        return best_position

    def score_target_position(self, position, arm_base, focus_point, table_pos):
        """
        Score a target position based on multiple criteria:
        - Distance from arm base (prefer middle range)
        - Height above table (prefer elevated view)
        - Angle to focus point (prefer good viewing angle)
        - Workspace constraints compliance
        """
        pos = np.array(position)
        base = np.array(arm_base)
        focus = np.array(focus_point)
        table = np.array(table_pos)
        
        score = 0
        
        # 1. Distance from arm base (prefer middle range for stability)
        distance_to_base = np.linalg.norm(pos - base)
        # CHANGE OPTIMAL DISTANCE FOR WIDER VIEW:
        optimal_distance = 0.7  # INCREASED: Prefer further distances
        distance_score = 1.0 - abs(distance_to_base - optimal_distance) / optimal_distance
        score += distance_score * 0.3
        
        # 2. Height above table (prefer elevated positions for overview)
        height_above_table = pos[2] - table[2]
        # CHANGE HEIGHT PREFERENCES FOR OVERVIEW:
        if 0.5 <= height_above_table <= 1.2:  # INCREASED: Much higher range for table overview
            height_score = 1.0
        elif height_above_table > 1.2:
            height_score = max(0, 1.0 - (height_above_table - 1.2) / 0.5)
        else:
            height_score = max(0, height_above_table / 0.5)
        score += height_score * 0.3
        
        # 3. Viewing angle to focus point (prefer positions that can look down at table)
        to_focus = focus - pos
        distance_to_focus = np.linalg.norm(to_focus)
        
        # CHANGE MINIMUM DISTANCE FOR OVERVIEW:
        if distance_to_focus > 0.3:  # INCREASED: Allow further distances
            # Calculate angle from horizontal
            horizontal_dist = np.linalg.norm(to_focus[:2])
            vertical_dist = -to_focus[2]  # Negative because we want to look down
            
            if horizontal_dist > 0:
                look_down_angle = math.atan2(vertical_dist, horizontal_dist)
                # CHANGE PREFERRED ANGLES FOR WIDER VIEW:
                if 0.3 <= look_down_angle <= 1.0:  # INCREASED: 17-57 degrees looking down
                    angle_score = 1.0
                else:
                    angle_score = max(0, 1.0 - abs(look_down_angle - 0.65) / 0.65)
            else:
                angle_score = 0.5
        else:
            angle_score = 0
        
        score += angle_score * 0.25
        
        # 4. Prefer positions that are not too far to the right
        lateral_distance = abs(pos[1] - base[1])
        if lateral_distance < 0.4:
            lateral_score = 1.0
        else:
            lateral_score = max(0, 1.0 - (lateral_distance - 0.4) / 0.3)
        score += lateral_score * 0.15
        
        return score
    
    def move_arm_to_target_ik(self, model, data, target_position, arm_joint_indices, ee_body_id, ik_solver):
        """Move arm to target position using IK - EXACTLY matching all_kinova.py test_ik_methods"""
        print(f"Moving arm to target position: {target_position}")
        
        # EXACTLY like all_kinova.py - multiple initialization attempts for better convergence
        init_positions = [
            np.zeros(len(arm_joint_indices)),  # Original home position
            np.array([0.0, 0.5, 0.0, 1.5, 0.0, 0.0, 0.0])[:len(arm_joint_indices)],  # Bent position
            np.array([0.0, -0.5, 0.0, 1.0, 0.0, 0.5, 0.0])[:len(arm_joint_indices)]   # Alternative position
        ]
        
        best_result = None
        best_error = float('inf')
        
        for attempt, init_q_attempt in enumerate(init_positions):
            if len(init_q_attempt) != len(arm_joint_indices):
                init_q_attempt = np.zeros(len(arm_joint_indices))
            
            # Reset to this initialization
            for i, joint_idx in enumerate(arm_joint_indices):
                data.qpos[joint_idx] = init_q_attempt[i]
            mujoco.mj_forward(model, data)
            
            try:
                result_q = ik_solver.solve(data, np.array(target_position), init_q_attempt, ee_body_id, 
                                      arm_joint_indices, arm_joint_indices, table_height=0.50)
                
                # Get final end effector position
                final_pos = data.body(ee_body_id).xpos.copy()
                error = np.linalg.norm(np.array(target_position) - final_pos)
                
                # Keep best result
                if error < best_error:
                    best_error = error
                    best_result = {
                        'joint_angles': result_q,
                        'end_effector_pos': final_pos,
                        'error': error,
                        'iterations': ik_solver.iterations,
                        'converged': ik_solver.converged,
                        'attempt': attempt
                    }
                
                # If we get a good solution, stop trying
                if error < 0.05 and ik_solver.converged:
                    break
                    
            except Exception as e:
                print(f"  Attempt {attempt} failed: {e}")
                continue
        
        if best_result is None:
            # Fallback result
            final_pos = data.body(ee_body_id).xpos.copy()
            error = np.linalg.norm(np.array(target_position) - final_pos)
            best_result = {
                'joint_angles': np.array([data.qpos[idx] for idx in arm_joint_indices]),
                'end_effector_pos': final_pos,
                'error': error,
                'iterations': 0,
                'converged': False,
                'attempt': -1
            }
        
        # Set to best configuration
        for i, joint_idx in enumerate(arm_joint_indices):
            data.qpos[joint_idx] = best_result['joint_angles'][i]
        mujoco.mj_forward(model, data)
        
        print(f"IK Result: target={target_position}, achieved={best_result['end_effector_pos']}, error={best_result['error']:.4f}")
        print(f"Converged: {best_result['converged']}, Iterations: {best_result['iterations']}, Best attempt: {best_result['attempt']}")
        
        return best_result['joint_angles'], best_result['end_effector_pos'], best_result['error']

    def check_objects_exist(self):
        """Check which objects actually exist in the assets folder"""
        if not os.path.exists(self.assets_dir):
            print(f"Warning: {self.assets_dir} folder not found. Using placeholder objects.")
            return []
        
        existing_objects = []
        for obj in self.available_objects:
            if os.path.exists(os.path.join(self.assets_dir, obj)):
                existing_objects.append(obj)
        
        if existing_objects:
            print(f"Found {len(existing_objects)} objects: {existing_objects}")
        else:
            print("No STL files found, will use geometric primitives")
        
        return existing_objects

    def create_base_scene_xml(self):
        """Create the base scene XML structure WITHOUT keyframes to avoid DOF mismatch"""
        xml_content = '''<?xml version="1.0" ?>
    <mujoco model="gen3 scene">

    <!-- Inline robot definition to avoid keyframe conflicts -->
    <compiler angle="radian" meshdir="assets"/>

    <default>
        <joint damping="0.1" armature="0.01"/>
        <default class="visual">
        <geom type="mesh" contype="0" conaffinity="0" group="2" rgba="0.75294 0.75294 0.75294 1"/>
        </default>
        <default class="collision">
        <geom type="mesh" group="3"/>
        </default>
        <default class="large_actuator">
        <general gaintype="fixed" biastype="affine" gainprm="1000" biasprm="0 -1000 -200" forcerange="-39 39"/>
        </default>
        <default class="small_actuator">
        <general gaintype="fixed" biastype="affine" gainprm="1000" biasprm="0 -1000 -200" forcerange="-9 9"/>
        </default>
        <site size="0.001" rgba="0.5 0.5 0.5 0.3" group="4"/>
        <general gaintype="fixed" biastype="affine" gainprm="1000" biasprm="0 -1000 -200" forcerange="-50 50"/>
    </default>

    <option integrator="implicitfast" gravity="0 0 -9.81"/>

    <statistic center="0.4 0 0.5" extent="1.2" meansize="0.05"/>

    <visual>
        <headlight diffuse="0.6 0.6 0.6" ambient="0.1 0.1 0.1" specular="0 0 0"/>
        <rgba haze="0.15 0.25 0.35 1"/>
        <global azimuth="120" elevation="-20" fovy="57"/> <!-- Matches D410 vertical FOV -->
    </visual>

    <asset>
        <!-- Robot meshes -->
        <mesh name="base_link" file="base_link.stl"/>
        <mesh name="shoulder_link" file="shoulder_link.stl"/>
        <mesh name="half_arm_1_link" file="half_arm_1_link.stl"/>
        <mesh name="half_arm_2_link" file="half_arm_2_link.stl"/>
        <mesh name="forearm_link" file="forearm_link.stl"/>
        <mesh name="spherical_wrist_1_link" file="spherical_wrist_1_link.stl"/>
        <mesh name="spherical_wrist_2_link" file="spherical_wrist_2_link.stl"/>
        <mesh name="bracelet_with_vision_link" file="bracelet_with_vision_link.stl"/>
        
        <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
        
        <!-- Enhanced ground textures with more variety -->
        <texture type="2d" name="groundplane1" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" 
        markrgb="0.8 0.8 0.8" width="300" height="300"/>
        <texture type="2d" name="groundplane2" builtin="flat" rgb1="0.15 0.15 0.2" width="300" height="300"/>
        <texture type="2d" name="groundplane3" builtin="checker" mark="cross" rgb1="0.25 0.25 0.25" rgb2="0.35 0.35 0.35" 
        markrgb="0.6 0.6 0.6" width="300" height="300"/>
        <texture type="2d" name="groundplane4" builtin="gradient" rgb1="0.4 0.4 0.5" rgb2="0.2 0.2 0.3" width="300" height="300"/>
        <texture type="2d" name="groundplane5" builtin="checker" mark="none" rgb1="0.3 0.2 0.2" rgb2="0.4 0.3 0.3" width="400" height="400"/>
        <texture type="2d" name="groundplane6" builtin="flat" rgb1="0.1 0.3 0.1" width="300" height="300"/>
        
        <material name="groundplane1" texture="groundplane1" texuniform="true" texrepeat="8 8" reflectance="0.2"/>
        <material name="groundplane2" texture="groundplane2" texuniform="true" reflectance="0.1"/>
        <material name="groundplane3" texture="groundplane3" texuniform="true" texrepeat="6 6" reflectance="0.3"/>
        <material name="groundplane4" texture="groundplane4" texuniform="true" texrepeat="4 4" reflectance="0.15"/>
        <material name="groundplane5" texture="groundplane5" texuniform="true" texrepeat="10 10" reflectance="0.25"/>
        <material name="groundplane6" texture="groundplane6" texuniform="true" reflectance="0.05"/>

        <!-- Enhanced table textures with more wood varieties -->
        <texture type="2d" name="wood_tex1" builtin="checker" mark="edge" rgb1="0.6 0.4 0.2" rgb2="0.5 0.3 0.1" 
        markrgb="0.7 0.5 0.3" width="100" height="100"/>
        <texture type="2d" name="wood_tex2" builtin="flat" rgb1="0.4 0.3 0.2" width="100" height="100"/>
        <texture type="2d" name="wood_tex3" builtin="checker" mark="none" rgb1="0.7 0.5 0.3" rgb2="0.6 0.4 0.2" 
        width="100" height="100"/>
        <texture type="2d" name="wood_tex4" builtin="gradient" rgb1="0.8 0.6 0.4" rgb2="0.6 0.4 0.2" width="150" height="150"/>
        <texture type="2d" name="wood_tex5" builtin="checker" mark="cross" rgb1="0.5 0.35 0.2" rgb2="0.45 0.3 0.15" 
        markrgb="0.6 0.45 0.3" width="120" height="120"/>
        <texture type="2d" name="marble_tex1" builtin="flat" rgb1="0.9 0.9 0.85" width="100" height="100"/>
        <texture type="2d" name="marble_tex2" builtin="gradient" rgb1="0.95 0.95 0.9" rgb2="0.85 0.85 0.8" width="100" height="100"/>
        <texture type="2d" name="metal_tex1" builtin="flat" rgb1="0.7 0.7 0.8" width="100" height="100"/>
        
        <material name="wood_mat1" texture="wood_tex1" texuniform="true" texrepeat="4 4" reflectance="0.1"/>
        <material name="wood_mat2" texture="wood_tex2" texuniform="true" reflectance="0.05"/>
        <material name="wood_mat3" texture="wood_tex3" texuniform="true" texrepeat="5 5" reflectance="0.15"/>
        <material name="wood_mat4" texture="wood_tex4" texuniform="true" texrepeat="3 3" reflectance="0.12"/>
        <material name="wood_mat5" texture="wood_tex5" texuniform="true" texrepeat="6 6" reflectance="0.18"/>
        <material name="marble_mat1" texture="marble_tex1" texuniform="true" texrepeat="2 2" reflectance="0.4"/>
        <material name="marble_mat2" texture="marble_tex2" texuniform="true" texrepeat="2 2" reflectance="0.35"/>
        <material name="metal_mat1" texture="metal_tex1" texuniform="true" texrepeat="8 8" reflectance="0.6"/>
        
        <!-- Enhanced object materials with textures and more colors -->
        <texture type="2d" name="red_tex" builtin="flat" rgb1="0.8 0.2 0.2" width="50" height="50"/>
        <texture type="2d" name="blue_tex" builtin="flat" rgb1="0.2 0.2 0.8" width="50" height="50"/>
        <texture type="2d" name="green_tex" builtin="flat" rgb1="0.2 0.8 0.2" width="50" height="50"/>
        <texture type="2d" name="yellow_tex" builtin="flat" rgb1="0.8 0.8 0.2" width="50" height="50"/>
        <texture type="2d" name="purple_tex" builtin="flat" rgb1="0.8 0.2 0.8" width="50" height="50"/>
        <texture type="2d" name="orange_tex" builtin="flat" rgb1="0.8 0.5 0.2" width="50" height="50"/>
        <texture type="2d" name="cyan_tex" builtin="flat" rgb1="0.2 0.8 0.8" width="50" height="50"/>
        <texture type="2d" name="pink_tex" builtin="flat" rgb1="0.8 0.4 0.6" width="50" height="50"/>
        <texture type="2d" name="brown_tex" builtin="flat" rgb1="0.6 0.4 0.2" width="50" height="50"/>
        <texture type="2d" name="gray_tex" builtin="flat" rgb1="0.5 0.5 0.5" width="50" height="50"/>
        <texture type="2d" name="white_tex" builtin="flat" rgb1="0.9 0.9 0.9" width="50" height="50"/>
        <texture type="2d" name="black_tex" builtin="flat" rgb1="0.2 0.2 0.2" width="50" height="50"/>
        
        <!-- Textured object materials -->
        <texture type="2d" name="stripe_tex1" builtin="checker" mark="edge" rgb1="0.8 0.2 0.2" rgb2="0.9 0.3 0.3" 
        markrgb="0.7 0.1 0.1" width="50" height="50"/>
        <texture type="2d" name="stripe_tex2" builtin="checker" mark="edge" rgb1="0.2 0.2 0.8" rgb2="0.3 0.3 0.9" 
        markrgb="0.1 0.1 0.7" width="50" height="50"/>
        <texture type="2d" name="dots_tex1" builtin="checker" mark="cross" rgb1="0.2 0.8 0.2" rgb2="0.3 0.9 0.3" 
        markrgb="0.1 0.7 0.1" width="50" height="50"/>
        
        <material name="object_red" texture="red_tex" texuniform="true" reflectance="0.1"/>
        <material name="object_blue" texture="blue_tex" texuniform="true" reflectance="0.1"/>
        <material name="object_green" texture="green_tex" texuniform="true" reflectance="0.1"/>
        <material name="object_yellow" texture="yellow_tex" texuniform="true" reflectance="0.15"/>
        <material name="object_purple" texture="purple_tex" texuniform="true" reflectance="0.1"/>
        <material name="object_orange" texture="orange_tex" texuniform="true" reflectance="0.12"/>
        <material name="object_cyan" texture="cyan_tex" texuniform="true" reflectance="0.2"/>
        <material name="object_pink" texture="pink_tex" texuniform="true" reflectance="0.15"/>
        <material name="object_brown" texture="brown_tex" texuniform="true" reflectance="0.05"/>
        <material name="object_gray" texture="gray_tex" texuniform="true" reflectance="0.3"/>
        <material name="object_white" texture="white_tex" texuniform="true" reflectance="0.4"/>
        <material name="object_black" texture="black_tex" texuniform="true" reflectance="0.02"/>
        <material name="object_stripe_red" texture="stripe_tex1" texuniform="true" texrepeat="4 4" reflectance="0.1"/>
        <material name="object_stripe_blue" texture="stripe_tex2" texuniform="true" texrepeat="4 4" reflectance="0.1"/>
        <material name="object_dots_green" texture="dots_tex1" texuniform="true" texrepeat="6 6" reflectance="0.1"/>
    </asset>

    <worldbody>
        <!-- Randomizable lighting setup -->
        <light pos="0.5 0 2.0" directional="true" diffuse="0.8 0.8 0.8"/>
        <light pos="1.0 0.8 1.5" diffuse="0.4 0.4 0.4"/>
        <light pos="-0.3 0.8 1.2" diffuse="0.3 0.3 0.3"/>

        <!-- Ground plane -->
        <geom name="floor" size="0 0 0.05" type="plane" material="groundplane1"/>

        <!-- Larger table - positioned near arm's base, accessible from edge -->
        <geom name="table" type="box" size="0.5 0.4 0.02" pos="0.6 0 0.43" material="wood_mat1"/>

        <!-- Table legs for the larger table -->
        <geom name="table_leg1" type="cylinder" size="0.02 0.21" pos="0.2 0.3 0.21" rgba="0.4 0.2 0.1 1"/>
        <geom name="table_leg2" type="cylinder" size="0.02 0.21" pos="1.0 0.3 0.21" rgba="0.4 0.2 0.1 1"/>
        <geom name="table_leg3" type="cylinder" size="0.02 0.21" pos="0.2 -0.3 0.21" rgba="0.4 0.2 0.1 1"/>
        <geom name="table_leg4" type="cylinder" size="0.02 0.21" pos="1.0 -0.3 0.21" rgba="0.4 0.2 0.1 1"/>
        
        <!-- Robot arm positioned on table edge -->
        <body name="base_link" pos="0.2 0 0.45">
        <inertial pos="-0.000648 -0.000166 0.084487" quat="0.999294 0.00139618 -0.0118387 0.035636" mass="1.697"
            diaginertia="0.00462407 0.00449437 0.00207755"/>
        <geom class="visual" mesh="base_link"/>
        <geom class="collision" mesh="base_link"/>
        <body name="shoulder_link" pos="0 0 0.15643" quat="0 1 0 0">
            <inertial pos="-2.3e-05 -0.010364 -0.07336" quat="0.707051 0.0451246 -0.0453544 0.704263" mass="1.3773"
            diaginertia="0.00488868 0.00457 0.00135132"/>
            <joint name="joint_1" range="-6.2832 6.2832" limited="true" armature="0.1"/>
            <geom class="visual" mesh="shoulder_link"/>
            <geom class="collision" mesh="shoulder_link"/>
            <body name="half_arm_1_link" pos="0 0.005375 -0.12838" quat="1 1 0 0">
            <inertial pos="-4.4e-05 -0.09958 -0.013278" quat="0.482348 0.516286 -0.516862 0.483366" mass="1.1636"
                diaginertia="0.0113017 0.011088 0.00102532"/>
            <joint name="joint_2" range="-2.2 2.2" limited="true" armature="0.1"/>
            <geom class="visual" mesh="half_arm_1_link"/>
            <geom class="collision" mesh="half_arm_1_link"/>
            <body name="half_arm_2_link" pos="0 -0.21038 -0.006375" quat="1 -1 0 0">
                <inertial pos="-4.4e-05 -0.006641 -0.117892" quat="0.706144 0.0213722 -0.0209128 0.707437" mass="1.1636"
                diaginertia="0.0111633 0.010932 0.00100671"/>
                <joint name="joint_3" range="-6.2832 6.2832" limited="true" armature="0.1"/>
                <geom class="visual" mesh="half_arm_2_link"/>
                <geom class="collision" mesh="half_arm_2_link"/>
                <body name="forearm_link" pos="0 0.006375 -0.21038" quat="1 1 0 0">
                <inertial pos="-1.8e-05 -0.075478 -0.015006" quat="0.483678 0.515961 -0.515859 0.483455" mass="0.9302"
                    diaginertia="0.00834839 0.008147 0.000598606"/>
                <joint name="joint_4" range="-2.5656 2.5656" limited="true" armature="0.1"/>
                <geom class="visual" mesh="forearm_link"/>
                <geom class="collision" mesh="forearm_link"/>
                <body name="spherical_wrist_1_link" pos="0 -0.20843 -0.006375" quat="1 -1 0 0">
                    <inertial pos="1e-06 -0.009432 -0.063883" quat="0.703558 0.0707492 -0.0707492 0.703558" mass="0.6781"
                    diaginertia="0.00165901 0.001596 0.000346988"/>
                    <joint name="joint_5" range="-6.2832 6.2832" limited="true" armature="0.1"/>
                    <geom class="visual" mesh="spherical_wrist_1_link"/>
                    <geom class="collision" mesh="spherical_wrist_1_link"/>
                    <body name="spherical_wrist_2_link" pos="0 0.00017505 -0.10593" quat="1 1 0 0">
                    <inertial pos="1e-06 -0.045483 -0.00965" quat="0.44426 0.550121 -0.550121 0.44426" mass="0.6781"
                        diaginertia="0.00170087 0.001641 0.00035013"/>
                    <joint name="joint_6" range="-2.05 2.05" limited="true" armature="0.1"/>
                    <geom class="visual" mesh="spherical_wrist_2_link"/>
                    <geom class="collision" mesh="spherical_wrist_2_link"/>
                    <body name="bracelet_link" pos="0 -0.10593 -0.00017505" quat="1 -1 0 0">
                        <inertial pos="0.000281 0.011402 -0.029798" quat="0.394358 0.596779 -0.577293 0.393789" mass="0.5"
                        diaginertia="0.000657336 0.000587019 0.000320645"/>
                        <joint name="joint_7" range="-6.2832 6.2832" limited="true" armature="0.1"/>
                        <geom class="visual" mesh="bracelet_with_vision_link"/>
                        <geom class="collision" mesh="bracelet_with_vision_link"/>
                        
                        <!-- Camera positioned on the bracelet, facing upward/outward from the bracelet -->
                        <!--Camera FOC is usually 47 to 60 -->
                        <camera name="wrist" pos="0 -0.05639 -0.058475" quat="0 0 0 1" fovy="60" resolution="1280 720"/>  
                        
                        <!-- End effector site pointing in the same direction as camera -->
                        <site name="pinch_site" pos="0 0 -0.121525" quat="0 1 0 0"/>
                    </body>
                    </body>
                </body>
                </body>
            </body>
            </body>
        </body>
        </body>
    </worldbody>

    <actuator>
        <general class="large_actuator" name="joint_1" joint="joint_1" ctrlrange="-6.2832 6.2832"/>
        <general class="large_actuator" name="joint_2" joint="joint_2" ctrlrange="-2.2 2.2"/>
        <general class="large_actuator" name="joint_3" joint="joint_3" ctrlrange="-6.2832 6.2832"/>
        <general class="large_actuator" name="joint_4" joint="joint_4" ctrlrange="-2.5656 2.5656"/>
        <general class="small_actuator" name="joint_5" joint="joint_5" ctrlrange="-6.2832 6.2832"/>
        <general class="small_actuator" name="joint_6" joint="joint_6" ctrlrange="-2.05 2.05"/>
        <general class="small_actuator" name="joint_7" joint="joint_7" ctrlrange="-6.2832 6.2832"/>
    </actuator>

    <!-- REMOVED KEYFRAMES SECTION TO PREVENT DOF MISMATCH ERRORS -->
    <!-- The original gen3.xml keyframes become invalid when freejoint objects are added -->

    </mujoco>'''
        
        return ET.fromstring(xml_content)

    def generate_scene(self, scene_id, config=None):
        """Generate a single scene with specified or random configuration"""
        print(f"Generating scene {scene_id}...")
        
        # Create base scene
        root = self.create_base_scene_xml()

        # Apply configuration or randomize
        if config is None:
            config = self.generate_random_config()
        
        # Apply configuration to scene
        scene_info = self.apply_config_to_scene(root, config)
        scene_info['scene_id'] = scene_id
        scene_info['config'] = config
        
        # Save scene XML
        scene_filename = f"scene_{scene_id}.xml"
        scene_path = os.path.join(self.output_dir, "scenes", scene_filename)
        
        tree = ET.ElementTree(root)
        tree.write(scene_path, encoding="utf-8", xml_declaration=True)
        
        return scene_path, scene_info

    def generate_random_config(self):
        """Generate a random scene configuration with dynamic target positioning"""
        existing_objects = self.check_objects_exist()
        
        # Choose 3-4 objects only
        if existing_objects:
            num_objects = random.randint(3, 4)
            selected_objects = random.sample(existing_objects, min(num_objects, len(existing_objects)))
        else:
            num_objects = random.randint(3, 4)
            selected_objects = ["geometric"] * num_objects
        
        # Choose table position first
        table_position = random.choice(self.table_positions)
        
        # Generate dynamic target position based on table position
        table_center = [table_position[0], table_position[1], table_position[2] + 0.05]
        target_position = self.generate_valid_target_position(table_position, table_center)
        
        config = {
            'objects': selected_objects,
            'ground_material': random.choice(self.ground_materials),
            'table_material': random.choice(self.table_materials),
            'lighting_config': random.choice(self.lighting_configs),
            'target_position': target_position,  # Now dynamically generated
            'table_position': table_position,
            'lighting_intensity': {
                'main': [random.uniform(0.6, 0.9), random.uniform(0.6, 0.9), random.uniform(0.6, 0.9)],
                'aux1': [random.uniform(0.3, 0.5), random.uniform(0.3, 0.5), random.uniform(0.3, 0.5)],
                'aux2': [random.uniform(0.2, 0.4), random.uniform(0.2, 0.4), random.uniform(0.2, 0.4)]
            }
        }
        
        return config

    def apply_config_to_scene(self, root, config):
        """Apply configuration to scene XML"""
        worldbody = root.find("worldbody")
        assets = root.find("asset")
        
        # Update ground and table materials
        for geom in root.findall(".//geom[@name='floor']"):
            geom.set("material", config['ground_material'])

        for geom in root.findall(".//geom[@name='table']"):
            geom.set("material", config['table_material'])
            # Update table position
            table_pos = config['table_position']
            geom.set("pos", f"{table_pos[0]} {table_pos[1]} {table_pos[2]}")

        # Update table legs positions based on table position
        table_pos = config['table_position']
        leg_positions = [
            [table_pos[0] - 0.4, table_pos[1] + 0.3, 0.21],
            [table_pos[0] + 0.4, table_pos[1] + 0.3, 0.21],
            [table_pos[0] - 0.4, table_pos[1] - 0.3, 0.21],
            [table_pos[0] + 0.4, table_pos[1] - 0.3, 0.21]
        ]
        table_legs = []
        table_legs.extend(root.findall(".//geom[@name='table_leg1']"))
        table_legs.extend(root.findall(".//geom[@name='table_leg2']"))
        table_legs.extend(root.findall(".//geom[@name='table_leg3']"))
        table_legs.extend(root.findall(".//geom[@name='table_leg4']"))
        for i, geom in enumerate(table_legs):
            if i < len(leg_positions):
                pos = leg_positions[i]
                geom.set("pos", f"{pos[0]} {pos[1]} {pos[2]}")
        
        # Clear existing lights and add new ones
        for light in worldbody.findall("light"):
            worldbody.remove(light)
        
        # Add configured lights
        lighting_config = config['lighting_config']
        lighting_intensity = config['lighting_intensity']
        
        main_light = ET.SubElement(worldbody, "light")
        main_light.set("pos", f"{lighting_config['main'][0]} {lighting_config['main'][1]} {lighting_config['main'][2]}")
        main_light.set("directional", "true")
        main_light.set("diffuse", f"{lighting_intensity['main'][0]} {lighting_intensity['main'][1]} {lighting_intensity['main'][2]}")
        
        aux1_light = ET.SubElement(worldbody, "light")
        aux1_light.set("pos", f"{lighting_config['aux1'][0]} {lighting_config['aux1'][1]} {lighting_config['aux1'][2]}")
        aux1_light.set("diffuse", f"{lighting_intensity['aux1'][0]} {lighting_intensity['aux1'][1]} {lighting_intensity['aux1'][2]}")
        
        aux2_light = ET.SubElement(worldbody, "light")
        aux2_light.set("pos", f"{lighting_config['aux2'][0]} {lighting_config['aux2'][1]} {lighting_config['aux2'][2]}")
        aux2_light.set("diffuse", f"{lighting_intensity['aux2'][0]} {lighting_intensity['aux2'][1]} {lighting_intensity['aux2'][2]}")
        
        # Add objects
        scene_info = self.add_objects_to_scene(root, worldbody, assets, config)
        
        return scene_info
    
    def freeze_objects_after_settling(self, model, data):
        """Freeze objects in place after they have settled on the table - IMPROVED VERSION"""
        print("Freezing objects in their settled positions...")
        
        # Find all freejoint objects (our table objects)
        freejoint_indices = []
        freejoint_body_indices = []
        
        for i in range(model.njnt):
            try:
                joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
                if joint_name and 'object_' in joint_name and '_joint' in joint_name:
                    freejoint_indices.append(i)
                    # Find the body this joint belongs to
                    body_name = joint_name.replace('_joint', '')
                    try:
                        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
                        if body_id >= 0:
                            freejoint_body_indices.append(body_id)
                    except:
                        pass
            except:
                continue
        
        print(f"Found {len(freejoint_indices)} object joints to freeze")
        
        # Method 1: Set EXTREMELY high damping for all freejoint DOFs
        for joint_idx in freejoint_indices:
            if model.jnt_type[joint_idx] == mujoco.mjtJoint.mjJNT_FREE:
                # Get DOF address for this joint
                dof_adr = model.jnt_dofadr[joint_idx]
                # Set very high damping for all 6 DOFs of freejoint (3 linear + 3 angular)
                for dof_offset in range(6):
                    if dof_adr + dof_offset < model.nv:
                        model.dof_damping[dof_adr + dof_offset] = 10000.0  # MUCH higher damping
                        data.qvel[dof_adr + dof_offset] = 0.0  # Zero velocity
        
        # Method 2: Set very high friction for better grip on table
        for body_idx in freejoint_body_indices:
            for geom_idx in range(model.ngeom):
                if model.geom_bodyid[geom_idx] == body_idx:
                    # Set extremely high friction
                    model.geom_friction[geom_idx][0] = 5.0  # Much higher sliding friction
                    model.geom_friction[geom_idx][1] = 0.5   # Higher torsional friction  
                    model.geom_friction[geom_idx][2] = 0.5   # Higher rolling friction
        
        # Method 3: Zero out all object velocities completely
        for joint_idx in freejoint_indices:
            if model.jnt_type[joint_idx] == mujoco.mjtJoint.mjJNT_FREE:
                dof_adr = model.jnt_dofadr[joint_idx]
                for dof_offset in range(6):
                    if dof_adr + dof_offset < model.nv:
                        data.qvel[dof_adr + dof_offset] = 0.0
                        # Also zero acceleration
                        data.qacc[dof_adr + dof_offset] = 0.0
        
        print("Objects frozen with extreme damping, high friction, and zero velocities")
        
    def settle_objects(self, model, data, num_steps=1000):
        """Force objects to settle on the table with gravity"""
        print("Settling objects with gravity...")
        
        # Find arm joint indices to keep arm stationary
        arm_joint_indices, _ = self.find_arm_indices(model)
        
        # Store current arm position
        arm_positions = []
        for joint_idx in arm_joint_indices:
            arm_positions.append(data.qpos[joint_idx])
        
        # Run physics simulation to let objects settle
        for step in range(num_steps):
            # Keep arm stationary
            for i, joint_idx in enumerate(arm_joint_indices):
                data.qpos[joint_idx] = arm_positions[i]
                data.qvel[joint_idx] = 0  # No arm movement
            
            # Step physics
            mujoco.mj_step(model, data)
            
            # Check if objects have settled (low velocities)
            if step > 500 and step % 100 == 0:
                max_vel = 0
                for i in range(model.nbody):
                    if i > 0:  # Skip world body
                        body_vel = np.linalg.norm(data.cvel[i][:3])  # Linear velocity
                        max_vel = max(max_vel, body_vel)
                
                if max_vel < 0.01:  # Objects have settled
                    print(f"Objects settled after {step} steps")
                    

        # Freeze objects after settling
        self.freeze_objects_after_settling(model, data)
        
        print("Object settling and freezing complete")

    def add_objects_to_scene(self, root, worldbody, assets, config):
        """Add objects to the scene and return object information"""
        objects_info = []
        table_pos = config['table_position']
        
        # Define table bounds with scaling and shifting
        scale_factor = 0.6  # Shrink spawn area to 60% of original
        shift_options = [
            [-0.08, 0],    # Left (closer to arm)
            [0.05, 0],     # Right (slightly farther)
            [0, -0.08],    # Back
            [0, 0.08],     # Front
            [0, 0]         # Center (no shift)
        ]
        shift_x, shift_y = random.choice(shift_options)

        # Calculate scaled and shifted bounds
        base_width = 0.25 * scale_factor  # 0.15
        base_height = 0.15 * scale_factor  # 0.09
        table_x_min = table_pos[0] - base_width + shift_x
        table_x_max = table_pos[0] + base_width + shift_x
        table_y_min = table_pos[1] - base_height + shift_y
        table_y_max = table_pos[1] + base_height + shift_y

        # Ensure bounds stay within table limits
        table_x_min = max(table_x_min, table_pos[0] - 0.4)
        table_x_max = min(table_x_max, table_pos[0] + 0.4)
        table_y_min = max(table_y_min, table_pos[1] - 0.3)
        table_y_max = min(table_y_max, table_pos[1] + 0.3)
        table_z = table_pos[2] + 0.04  # Slightly above table surface
        
        # Keep track of placed positions to avoid overlap
        placed_positions = []
        min_distance = 0.12  # Increased minimum distance between objects
        
        for i, obj_name in enumerate(config['objects']):
            # Try to find a non-overlapping position
            attempts = 0
            while attempts < 20:  # Max attempts to find valid position
                x = random.uniform(table_x_min, table_x_max)
                y = random.uniform(table_y_min, table_y_max)
                
                # Check if position is too close to existing objects
                valid_position = True
                for prev_pos in placed_positions:
                    distance = np.sqrt((x - prev_pos[0])**2 + (y - prev_pos[1])**2)
                    if distance < min_distance:
                        valid_position = False
                        break
                
                if valid_position:
                    break
                attempts += 1
            
            # Calculate proper height based on object type and scale
            scale = self.object_scales.get(obj_name, self.object_scales["default"])
            if obj_name in ["sphere.stl", "cylinder.stl", "cube.stl"]:
                # For geometric primitives, we know exact dimensions
                if obj_name == "sphere.stl":
                    object_height = scale * 2.0  # diameter
                elif obj_name == "cylinder.stl":
                    object_height = scale * 2.0  # height
                elif obj_name == "cube.stl":
                    object_height = scale * 2.0  # edge length
            else:
                # For STL meshes, estimate height
                object_height = scale * 10  # Conservative estimate
                
            z = table_z + object_height / 2  # Bottom of object sits on table
            placed_positions.append([x, y])

            # Place objects in their normal orientation with just random Z-axis rotation
            #rotation_z = random.uniform(0, 2 * np.pi)
            rotation_z = 0
            quat = f"0 0 {np.sin(rotation_z/2)} {np.cos(rotation_z/2)}"

            '''# More realistic orientations - mostly upright with some variation
            orientation_type = random.choice(['upright', 'lying', 'tilted'])

            if orientation_type == 'upright':
                # Mostly upright with small random rotation around Z-axis
                rotation = random.uniform(0, 2 * np.pi)
                quat = f"0 0 {np.sin(rotation/2)} {np.cos(rotation/2)}"
            elif orientation_type == 'lying':
                # Lying on side - rotate around X or Y axis
                if random.choice([True, False]):
                    # Lying on side (rotate around X-axis)
                    angle = np.pi/2 + random.uniform(-0.2, 0.2)
                    quat = f"{np.sin(angle/2)} 0 0 {np.cos(angle/2)}"
                else:
                    # Lying on other side (rotate around Y-axis)
                    angle = np.pi/2 + random.uniform(-0.2, 0.2)
                    quat = f"0 {np.sin(angle/2)} 0 {np.cos(angle/2)}"
            else:  # tilted
                # Slightly tilted but mostly upright
                tilt_x = random.uniform(-0.3, 0.3)
                tilt_y = random.uniform(-0.3, 0.3)
                rotation_z = random.uniform(0, 2 * np.pi)
                # Combine rotations (simplified)
                quat = f"{np.sin(tilt_x/2)} {np.sin(tilt_y/2)} {np.sin(rotation_z/2)} {np.cos(rotation_z/2)}"'''
            
            
            # Choose random color
            color_name, color_rgba = random.choice(self.object_colors)
            
            if obj_name == "geometric":
                # Use geometric primitives
                obj_info = self.add_geometric_object(worldbody, i, x, y, z, quat, color_name)
            else:
                # Use STL mesh with proper scaling
                obj_info = self.add_mesh_object(assets, worldbody, i, obj_name, x, y, z, quat, color_name)
            
            obj_info.update({
                'position': [x, y, z],
                'rotation': quat,
                'color': color_rgba
            })
            objects_info.append(obj_info)
        
        return {'objects': objects_info}

    def add_mesh_object(self, assets, worldbody, i, obj_file, x, y, z, quat, color_name):
        """Add a mesh object to the scene with proper scaling and physics"""
        mesh_name = f"object_{i}"
        
        # Get scaling factor for this object
        scale = self.object_scales.get(obj_file, self.object_scales["default"])
        
        # Add mesh asset with scaling
        mesh = ET.SubElement(assets, "mesh")
        mesh.set("name", mesh_name)
        mesh.set("file", obj_file)
        mesh.set("scale", f"{scale} {scale} {scale}")
        
        # Calculate proper object height based on scale
        if obj_file in ["sphere.stl", "cylinder.stl", "cube.stl"]:
            if obj_file == "sphere.stl":
                object_height = scale * 2.0  # Full diameter, then half for center
            elif obj_file == "cylinder.stl":
                object_height = scale * 2.0  # Full height, then half for center
            elif obj_file == "cube.stl":
                object_height = scale * 2.0  # Full edge length, then half for center
        else:
            # For STL meshes, use scale factor properly
            object_height = scale * 8  # Better estimate for mesh objects

        # Position object so bottom sits exactly on table surface
        table_surface_z = z  # z parameter is already table surface + small offset
        proper_z = table_surface_z + (object_height / 2)  # Center at half-height above surface
        
        # Create object body with freejoint for physics
        obj_body = ET.SubElement(worldbody, "body")
        obj_body.set("name", f"object_{i}")
        obj_body.set("pos", f"{x} {y} {proper_z}")
        obj_body.set("quat", quat)

        # Add freejoint to allow the object to settle with gravity
        obj_joint = ET.SubElement(obj_body, "freejoint")
        obj_joint.set("name", f"object_{i}_joint")
        
        # Add visual geom
        obj_geom_visual = ET.SubElement(obj_body, "geom")
        obj_geom_visual.set("name", f"object_{i}_visual")
        obj_geom_visual.set("type", "mesh")
        obj_geom_visual.set("mesh", mesh_name)
        obj_geom_visual.set("material", color_name)
        obj_geom_visual.set("group", "2")
        obj_geom_visual.set("contype", "0")
        obj_geom_visual.set("conaffinity", "0")
        
        # Add collision geom with physics properties
        obj_geom_collision = ET.SubElement(obj_body, "geom")
        obj_geom_collision.set("name", f"object_{i}_collision")
        obj_geom_collision.set("type", "mesh")
        obj_geom_collision.set("mesh", mesh_name)
        obj_geom_collision.set("group", "3")
        obj_geom_collision.set("friction", "0.8 0.02 0.001")  # Better friction for stability
        obj_geom_collision.set("density", "1000")  # kg/m³
        obj_geom_collision.set("solref", "0.01 1")  # Softer contact
        obj_geom_collision.set("solimp", "0.9 0.95 0.001")  # Better contact solver
                
        return {
            'type': 'mesh',
            'file': obj_file,
            'name': f"object_{i}",
            'material': color_name,
            'scale': scale
        }

    def add_geometric_object(self, worldbody, i, x, y, z, quat, color_name):
        """Add a geometric primitive object to the scene with proper physics"""
        geometric_types = [
            {"type": "box", "size": "0.025 0.025 0.025", "height": 0.025},
            {"type": "sphere", "size": "0.02", "height": 0.02},
            {"type": "cylinder", "size": "0.02 0.03", "height": 0.03},
            {"type": "ellipsoid", "size": "0.02 0.025 0.022", "height": 0.022}
        ]

        geom_type = random.choice(geometric_types)
        object_height = geom_type["height"]
        
        # Position object so its bottom sits exactly on table surface
        table_surface_z = z  # z parameter is already table surface height
        proper_z = table_surface_z + object_height  # Center at proper height above surface

        # Create object body with freejoint for physics
        obj_body = ET.SubElement(worldbody, "body")
        obj_body.set("name", f"object_{i}")
        obj_body.set("pos", f"{x} {y} {proper_z}")
        obj_body.set("quat", quat)

        # Add freejoint to allow the object to settle with gravity
        obj_joint = ET.SubElement(obj_body, "freejoint")
        obj_joint.set("name", f"object_{i}_joint")
        
        # Add geom with physics properties
        obj_geom = ET.SubElement(obj_body, "geom")
        obj_geom.set("name", f"object_{i}")
        obj_geom.set("type", geom_type["type"])
        obj_geom.set("size", geom_type["size"])
        obj_geom.set("material", color_name)
        obj_geom.set("friction", "0.8 0.02 0.001")
        obj_geom.set("density", "800")
        obj_geom.set("solref", "0.01 1")  # Softer contact
        obj_geom.set("solimp", "0.9 0.95 0.001")  # Better contact solver
        
        return {
            'type': geom_type["type"],
            'size': geom_type["size"],
            'name': f"object_{i}",
            'material': color_name
        }
    
    def select_query_object(self, scene_info):
        """Select one target object from scene and get its ACTUAL position"""
        if 'objects' in scene_info and scene_info['objects']:
            selected_obj = random.choice(scene_info['objects'])
            print(f"DEBUG SELECTED QUERY OBJECT: {selected_obj}")
            return selected_obj
        else:
            # Fallback to table center
            table_pos = scene_info['config']['table_position']
            fallback_obj = {'position': [table_pos[0], table_pos[1], table_pos[2] + 0.05]}
            print(f"DEBUG FALLBACK QUERY OBJECT: {fallback_obj}")
            return fallback_obj
        
    def get_actual_object_position_in_sim(self, model, data, query_object):
        """Get the actual object position from the simulation (not stored position)"""
        try:
            if 'name' in query_object:
                obj_name = query_object['name']
                obj_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, obj_name)
                if obj_body_id >= 0:
                    actual_pos = data.body(obj_body_id).xpos.copy()
                    print(f"Found actual object '{obj_name}' at simulation position: {actual_pos}")
                    return actual_pos
                else:
                    print(f"Could not find object '{obj_name}' in simulation")
            return None
        except Exception as e:
            print(f"Error getting actual object position: {e}")
            return None
        
    def get_actual_object_height(self, model, data, query_object):
        """Get actual object height from simulation instead of estimating"""
        
        print(f"DEBUG OBJECT HEIGHT CALCULATION:")
        print(f"  Query object: {query_object}")
        
        if 'name' in query_object:
            try:
                obj_name = query_object['name']
                print(f"  Looking for object: {obj_name}")
                obj_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, obj_name)
                print(f"  Object body ID: {obj_body_id}")
                
                if obj_body_id >= 0:
                    obj_pos = data.body(obj_body_id).xpos.copy()
                    print(f"  Actual object position in sim: {obj_pos}")

                    # Get all geoms for this body
                    object_geoms = []
                    for geom_id in range(model.ngeom):
                        if model.geom_bodyid[geom_id] == obj_body_id:
                            object_geoms.append(geom_id)
                    
                    if object_geoms:
                        # Get object's bounding box
                        min_z = float('inf')
                        max_z = float('-inf')
                        
                        obj_pos = data.body(obj_body_id).xpos.copy()
                        
                        for geom_id in object_geoms:
                            geom_type = model.geom_type[geom_id]
                            geom_size = model.geom_size[geom_id]
                            geom_pos = data.geom_xpos[geom_id]
                            
                            # Calculate approximate height based on geom type
                            if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
                                height = geom_size[2] * 2  # Box size is half-extent
                            elif geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
                                height = geom_size[0] * 2  # Sphere radius * 2
                            elif geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
                                height = geom_size[1] * 2  # Cylinder half-height * 2
                            elif geom_type == mujoco.mjtGeom.mjGEOM_MESH:
                                # For mesh, estimate from bounding box
                                height = geom_size[2] * 2 if len(geom_size) > 2 else 0.08
                            else:
                                height = 0.05  # Default fallback
                            
                            geom_min_z = geom_pos[2] - height/2
                            geom_max_z = geom_pos[2] + height/2
                            
                            min_z = min(min_z, geom_min_z)
                            max_z = max(max_z, geom_max_z)
                        
                        actual_height = max_z - min_z
                        print(f"Calculated actual object height: {actual_height*100:.1f}cm")
                        return max(actual_height, 0.02)  # Minimum 2cm height
            except Exception as e:
                print(f"Could not calculate object height: {e}")
        
        # Fallback to stored object info or default
        if 'type' in query_object:
            if query_object['type'] == 'mesh' and 'scale' in query_object:
                scale = query_object['scale']
                return scale * 8  # Reasonable estimate for mesh objects
            elif 'size' in query_object:
                try:
                    size_values = [float(x) for x in query_object['size'].split()]
                    if query_object['type'] == 'box':
                        return size_values[2] * 2
                    elif query_object['type'] == 'sphere':
                        return size_values[0] * 2
                    elif query_object['type'] == 'cylinder':
                        return size_values[1] * 2
                except:
                    pass
        
        print("Using default object height estimate: 5cm")
        return 0.05  # Default fallback
        
    def add_target_object_marker(self, scene_root, target_position, query_object, marker_name="target_marker"):
        """Add a red sphere marker to identify the target object, positioned above the object"""
        try:
            # Find worldbody
            worldbody = scene_root.find('.//worldbody')
            if worldbody is None:
                print("Warning: Could not find worldbody to add target marker")
                return
            
            # Estimate object height to place marker above it
            object_height = 0.05  # Default 5cm height
            
            # Try to get actual object dimensions
            if 'type' in query_object:
                if query_object['type'] == 'mesh':
                    # For mesh objects, use scale factor
                    scale = query_object.get('scale', 1.0)
                    if 'file' in query_object:
                        obj_file = query_object['file']
                        if obj_file in ["sphere.stl", "cylinder.stl", "cube.stl"]:
                            if obj_file == "sphere.stl":
                                object_height = scale * 2.0  # Full diameter
                            elif obj_file == "cylinder.stl":
                                object_height = scale * 2.0  # Full height
                            elif obj_file == "cube.stl":
                                object_height = scale * 2.0  # Full edge length
                        else:
                            object_height = scale * 8  # Estimate for mesh objects
                elif query_object['type'] in ['box', 'sphere', 'cylinder', 'ellipsoid']:
                    # For geometric primitives, extract size from size string
                    if 'size' in query_object:
                        size_str = query_object['size']
                        size_values = [float(x) for x in size_str.split()]
                        if query_object['type'] == 'box':
                            object_height = size_values[2] * 2  # Z-dimension * 2 (size is half-extent)
                        elif query_object['type'] == 'sphere':
                            object_height = size_values[0] * 2  # Radius * 2 = diameter
                        elif query_object['type'] == 'cylinder':
                            object_height = size_values[1] * 2  # Height * 2 (size is half-height)
                        elif query_object['type'] == 'ellipsoid':
                            object_height = size_values[2] * 2  # Z-radius * 2
            
            # Position marker well above the top of the object
            marker_height_offset = max(0.06, object_height * 0.6)  # At least 6cm above or 60% of object height
            marker_z = target_position[2] + (object_height / 2) + marker_height_offset
            
            # Create marker body
            marker_body = ET.SubElement(worldbody, "body")
            marker_body.set("name", marker_name)
            marker_body.set("pos", f"{target_position[0]} {target_position[1]} {marker_z}")
            
            # Add red sphere geom - BIGGER and MORE OPAQUE
            marker_geom = ET.SubElement(marker_body, "geom")
            marker_geom.set("name", f"{marker_name}_geom")
            marker_geom.set("type", "sphere")
            marker_geom.set("size", "0.015")  # 1.5cm sphere
            marker_geom.set("rgba", "1 0 0 1")  # Fully opaque bright red
            marker_geom.set("group", "0")  # Main visual group (more visible)
            marker_geom.set("contype", "0")  # No collision
            marker_geom.set("conaffinity", "0")  # No collision
            
            print(f"Added red target marker above object (height: {object_height:.3f}m, marker at: {marker_z:.3f}m)")
            
        except Exception as e:
            print(f"Warning: Could not add target marker: {e}")

    def calculate_object_boundary_radius(self, model, data, query_object):
        """Calculate effective radius/boundary of the query object for collision avoidance"""
        try:
            if 'name' in query_object:
                obj_name = query_object['name']
                obj_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, obj_name)
                
                if obj_body_id >= 0:
                    # Get all geoms for this body and find maximum extent
                    max_radius = 0.0
                    
                    for geom_id in range(model.ngeom):
                        if model.geom_bodyid[geom_id] == obj_body_id:
                            geom_type = model.geom_type[geom_id]
                            geom_size = model.geom_size[geom_id]
                            
                            if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
                                # For box, use diagonal of largest face
                                max_radius = max(max_radius, np.sqrt(geom_size[0]**2 + geom_size[1]**2))
                            elif geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
                                max_radius = max(max_radius, geom_size[0])
                            elif geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
                                max_radius = max(max_radius, geom_size[0])  # Use radius
                            else:
                                max_radius = max(max_radius, max(geom_size))
                    
                    print(f"Calculated object boundary radius: {max_radius*100:.1f}cm")
                    return max(max_radius, 0.02)  # Minimum 2cm radius
            
            # Fallback based on object type
            if 'type' in query_object and 'size' in query_object:
                size_str = query_object['size']
                size_values = [float(x) for x in size_str.split()]
                return max(size_values) + 0.01  # Add 1cm safety
            
        except Exception as e:
            print(f"Error calculating object boundary: {e}")
        
        return 0.05  # Default 5cm radius

    def calculate_safe_target_position(self, query_pos_array, object_radius, safety_margin):
        """Calculate safe target position accounting for object boundaries"""
        # Position the target above the object with proper clearance
        safe_target = query_pos_array.copy()
        safe_target[2] = query_pos_array[2] + object_radius + safety_margin
        
        print(f"Safe target position: {safe_target}")
        print(f"Object radius: {object_radius*100:.1f}cm, Safety margin: {safety_margin*100:.1f}cm")
        
        return safe_target

    def setup_random_arm_pose(self, model, data, arm_joint_indices):
        """Set arm to varied poses that position pinch site within table boundaries"""
        
        # Get table and arm positions for reference
        table_center = [0.6, 0, 0.45]  # Default table position from XML
        arm_base = [0.2, 0, 0.45]      # Arm base position from XML
        
        # Get pinch site ID for positioning
        ee_site_id = self.get_end_effector_site_id(model)
        ee_body_id = self.get_end_effector_body_id(model)
        
        print("Generating random arm pose with pinch site positioned on table...")
        
        # Define table boundaries (more generous than object spawn area)
        table_x_min, table_x_max = 0.15, 1.05  # Full table width
        table_y_min, table_y_max = -0.35, 0.35  # Full table depth
        table_z_min, table_z_max = 0.5, 1.2    # Above table surface to reasonable reach height
        
        best_pose = None
        best_score = -1
        
        # Try multiple random poses and pick the best one
        for attempt in range(50):
            # Generate random joint angles within limits
            random_q = []
            for i, joint_idx in enumerate(arm_joint_indices):
                if joint_idx < len(model.jnt_range):
                    joint_range = model.jnt_range[joint_idx]
                    if not (np.isinf(joint_range[0]) or np.isinf(joint_range[1])):
                        # Use wider range for more variety, but avoid extremes
                        range_center = (joint_range[0] + joint_range[1]) / 2
                        range_span = (joint_range[1] - joint_range[0]) * 0.7  # Use 70% of full range
                        min_val = range_center - range_span / 2
                        max_val = range_center + range_span / 2
                        random_q.append(np.random.uniform(min_val, max_val))
                    else:
                        random_q.append(np.random.uniform(-np.pi * 0.6, np.pi * 0.6))
                else:
                    random_q.append(np.random.uniform(-np.pi * 0.6, np.pi * 0.6))
            
            # Apply random pose
            for j, joint_idx in enumerate(arm_joint_indices):
                data.qpos[joint_idx] = random_q[j]
            mujoco.mj_forward(model, data)
            
            # Get pinch site position (or end effector if no site)
            if ee_site_id is not None:
                pinch_pos = data.site_xpos[ee_site_id].copy()
            else:
                pinch_pos = data.body(ee_body_id).xpos.copy()
            
            # Check if pinch site is within table boundaries
            within_table = (table_x_min <= pinch_pos[0] <= table_x_max and 
                        table_y_min <= pinch_pos[1] <= table_y_max and 
                        table_z_min <= pinch_pos[2] <= table_z_max)
            
            if not within_table:
                continue
            
            # Check for self-collisions
            self_collisions = self.check_arm_collisions(model, data, arm_joint_indices, ee_site_id)
            has_self_collision = any(col['type'] in ['arm_self_collision', 'pinch_site_collision'] for col in self_collisions)
            
            if has_self_collision:
                continue
            
            # Score this pose based on multiple criteria
            score = 0
            
            # 1. Prefer central table positions (not too close to edges)
            table_center_2d = np.array([table_center[0], table_center[1]])
            pinch_center_2d = np.array([pinch_pos[0], pinch_pos[1]])
            distance_from_center = np.linalg.norm(pinch_center_2d - table_center_2d)
            center_score = max(0, 1.0 - distance_from_center / 0.4)  # Prefer within 40cm of center
            score += center_score * 0.3
            
            # 2. Prefer varied heights (not always same height)
            height_variety_score = 1.0 - abs(pinch_pos[2] - 0.85) / 0.35  # Prefer variety around 85cm
            score += height_variety_score * 0.2
            
            # 3. Prefer poses that aren't too extended or too contracted
            joint_variety_score = 0
            for q_val in random_q:
                # Prefer joints not at extreme positions
                if abs(q_val) < np.pi * 0.8:  # Not too extreme
                    joint_variety_score += 0.1
            score += min(joint_variety_score, 0.3)
            
            # 4. Bonus for forward-facing poses (positive X from base)
            if pinch_pos[0] > arm_base[0]:
                score += 0.2
            
            if score > best_score:
                best_score = score
                best_pose = random_q.copy()
            
            if score > 0.8:  # Good enough pose found
                break
        
        if best_pose is not None:
            # Apply best pose
            for j, joint_idx in enumerate(arm_joint_indices):
                data.qpos[joint_idx] = best_pose[j]
            mujoco.mj_forward(model, data)
            
            # Get final pinch position
            if ee_site_id is not None:
                final_pinch_pos = data.site_xpos[ee_site_id].copy()
            else:
                final_pinch_pos = data.body(ee_body_id).xpos.copy()
            
            print(f"Random arm pose set: pinch site at [{final_pinch_pos[0]:.3f}, {final_pinch_pos[1]:.3f}, {final_pinch_pos[2]:.3f}] (score: {best_score:.3f})")
            print(f"Pinch site is {'ON' if table_x_min <= final_pinch_pos[0] <= table_x_max and table_y_min <= final_pinch_pos[1] <= table_y_max else 'OFF'} table")
        else:
            print("WARNING: Could not find good random arm pose, using default")
            # Fallback to a known good pose
            default_pose = [0.3, 0.2, 0.0, 1.2, 0.0, 0.4, 0.0][:len(arm_joint_indices)]
            for j, joint_idx in enumerate(arm_joint_indices):
                if j < len(default_pose):
                    data.qpos[joint_idx] = default_pose[j]
            mujoco.mj_forward(model, data)

    def point_camera_with_advanced_alignment(self, model, data, arm_joint_indices, ee_body_id, target_pos):
        """Use the same camera alignment method as timestep alignment for obs_0 and query_image"""
        
        current_q = np.array([data.qpos[idx] for idx in arm_joint_indices])
        
        # Use the same wrist-only optimization as timestep camera alignment
        optimized_q = self.optimize_camera_alignment_wrist_only(
            model, data, arm_joint_indices, ee_body_id, target_pos, current_q
        )
        
        return optimized_q

    def point_camera_with_wrist_joints_only(self, model, data, arm_joint_indices, target_pos, max_iterations=50):
        """Point camera by ONLY adjusting joints 6&7 with more precision"""
        
        if len(arm_joint_indices) < 7:
            print("Warning: Need at least 7 joints for wrist pointing")
            return
        
        joint_6_idx = arm_joint_indices[5]  # joint_6 (spherical_wrist_2)
        joint_7_idx = arm_joint_indices[6]  # joint_7 (bracelet)
        
        original_j6 = data.qpos[joint_6_idx]
        original_j7 = data.qpos[joint_7_idx]
        
        best_alignment = -1.0
        best_j6, best_j7 = original_j6, original_j7
        
        # Multi-stage optimization: coarse then fine
        search_ranges = [
            (np.linspace(-0.3, 0.3, 13), np.linspace(-0.3, 0.3, 13)),  # Coarse search
            (np.linspace(-0.1, 0.1, 21), np.linspace(-0.1, 0.1, 21)),  # Medium search
            (np.linspace(-0.05, 0.05, 11), np.linspace(-0.05, 0.05, 11))  # Fine search
        ]
        
        for stage, (j6_range, j7_range) in enumerate(search_ranges):
            stage_best_alignment = best_alignment
            
            for j6_delta in j6_range:
                for j7_delta in j7_range:
                    test_j6 = best_j6 + j6_delta if stage > 0 else original_j6 + j6_delta
                    test_j7 = best_j7 + j7_delta if stage > 0 else original_j7 + j7_delta
                    
                    # Apply joint limits
                    if joint_6_idx < len(model.jnt_range):
                        joint_range = model.jnt_range[joint_6_idx]
                        if not (np.isinf(joint_range[0]) or np.isinf(joint_range[1])):
                            test_j6 = np.clip(test_j6, joint_range[0], joint_range[1])
                    
                    if joint_7_idx < len(model.jnt_range):
                        joint_range = model.jnt_range[joint_7_idx]
                        if not (np.isinf(joint_range[0]) or np.isinf(joint_range[1])):
                            test_j7 = np.clip(test_j7, joint_range[0], joint_range[1])
                    
                    # Apply and test
                    data.qpos[joint_6_idx] = test_j6
                    data.qpos[joint_7_idx] = test_j7
                    mujoco.mj_forward(model, data)
                    
                    # Calculate camera alignment
                    try:
                        camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist")
                        if camera_id >= 0:
                            camera_pos = data.cam_xpos[camera_id].copy()
                            camera_mat = data.cam_xmat[camera_id].reshape(3, 3)
                            camera_direction = -camera_mat[:, 2]
                        else:
                            raise ValueError("No camera")
                    except:
                        ee_body_id = self.get_end_effector_body_id(model)
                        camera_pos = data.body(ee_body_id).xpos.copy()
                        ee_rotation_matrix = data.body(ee_body_id).xmat.reshape(3, 3)
                        camera_direction = -ee_rotation_matrix[:, 2]
                    
                    to_target = np.array(target_pos) - camera_pos
                    distance = np.linalg.norm(to_target)
                    
                    if distance > 0.01:
                        to_target_normalized = to_target / distance
                        alignment = np.dot(camera_direction, to_target_normalized)
                        
                        # Bonus for good viewing distance
                        distance_score = 1.0
                        if distance < 0.2 or distance > 0.8:  # Penalize too close/far
                            distance_score = 0.8
                        
                        final_score = alignment * distance_score
                        
                        if final_score > best_alignment:
                            best_alignment = final_score
                            best_j6, best_j7 = test_j6, test_j7
            
            print(f"Stage {stage + 1} alignment: {best_alignment:.4f}")
            
            # If no improvement in this stage, stop early
            if best_alignment <= stage_best_alignment + 0.001:
                break
        
        # Apply final best alignment
        data.qpos[joint_6_idx] = best_j6
        data.qpos[joint_7_idx] = best_j7
        mujoco.mj_forward(model, data)
        
        print(f"Final camera pointing alignment: {best_alignment:.4f}")
        print(f"Joint 6 change: {best_j6 - original_j6:.4f}")
        print(f"Joint 7 change: {best_j7 - original_j7:.4f}")

    def optimize_camera_alignment_wrist_only(self, model, data, arm_joint_indices, ee_body_id, objects_center, initial_q):
        """Use wrist-only IK to optimize camera alignment using ONLY wrist joints (last 3 joints)
        This only moves the wrist/end effector to point the camera, not the full arm"""
        
        class WristOnlyAlignmentIK:
            def __init__(self, model, objects_center, step_size=0.3, tol=0.05, max_iter=50):
                self.model = model
                self.objects_center = np.array(objects_center)
                self.step_size = step_size
                self.tol = tol
                self.max_iter = max_iter
                
            def get_camera_info(self, data, ee_body_id):
                """Get camera position and direction"""
                try:
                    camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist")
                    if camera_id >= 0:
                        camera_pos = data.cam_xpos[camera_id].copy()
                        camera_mat = data.cam_xmat[camera_id].reshape(3, 3)
                        camera_direction = -camera_mat[:, 2]
                    else:
                        raise ValueError("No camera found")
                except:
                    camera_pos = data.body(ee_body_id).xpos.copy()
                    ee_rotation_matrix = data.body(ee_body_id).xmat.reshape(3, 3)
                    camera_direction = -ee_rotation_matrix[:, 2]
                
                return camera_pos, camera_direction
                
            def calculate_alignment_error(self, data, ee_body_id):
                """Calculate alignment error"""
                camera_pos, camera_direction = self.get_camera_info(data, ee_body_id)
                
                to_objects = self.objects_center - camera_pos
                distance = np.linalg.norm(to_objects)
                
                if distance < 0.1:
                    return np.array([5.0, 5.0, 5.0])
                
                to_objects_normalized = to_objects / distance
                alignment_error_magnitude = 1.0 - np.dot(camera_direction, to_objects_normalized)
                
                cross_product = np.cross(camera_direction, to_objects_normalized)
                cross_magnitude = np.linalg.norm(cross_product)
                
                if cross_magnitude < 1e-6:
                    if np.dot(camera_direction, to_objects_normalized) > 0:
                        error_vector = np.array([0.0, 0.0, 0.0])
                    else:
                        error_vector = np.array([1.0, 0.0, 0.0])
                else:
                    error_vector = (cross_product / cross_magnitude) * alignment_error_magnitude
                
                return error_vector
                
            def solve_wrist_only(self, data, arm_joint_indices, ee_body_id, init_q):
                # Set initial position
                for i, joint_idx in enumerate(arm_joint_indices):
                    data.qpos[joint_idx] = init_q[i]
                mujoco.mj_forward(self.model, data)
                
                best_q = init_q.copy()
                best_error_norm = float('inf')
                
                # Only adjust last 3 joints (wrist joints)
                wrist_joint_count = min(3, len(arm_joint_indices))
                
                for iteration in range(self.max_iter):
                    error = self.calculate_alignment_error(data, ee_body_id)
                    error_norm = np.linalg.norm(error)
                    
                    if error_norm < best_error_norm:
                        best_error_norm = error_norm
                        best_q = np.array([data.qpos[idx] for idx in arm_joint_indices])
                    
                    if error_norm < self.tol:
                        print(f"Wrist-only camera alignment converged in {iteration} iterations")
                        break
                    
                    current_q = np.array([data.qpos[idx] for idx in arm_joint_indices])
                    
                    # Calculate jacobian for ONLY wrist joints
                    jacobian = np.zeros((3, wrist_joint_count))
                    delta = 0.001
                    
                    for j in range(wrist_joint_count):
                        wrist_joint_idx = len(arm_joint_indices) - wrist_joint_count + j
                        
                        # Positive perturbation
                        test_q = current_q.copy()
                        test_q[wrist_joint_idx] += delta
                        for k, joint_idx in enumerate(arm_joint_indices):
                            data.qpos[joint_idx] = test_q[k]
                        mujoco.mj_forward(self.model, data)
                        error_pos = self.calculate_alignment_error(data, ee_body_id)
                        
                        # Negative perturbation  
                        test_q[wrist_joint_idx] = current_q[wrist_joint_idx] - delta
                        for k, joint_idx in enumerate(arm_joint_indices):
                            data.qpos[joint_idx] = test_q[k]
                        mujoco.mj_forward(self.model, data)
                        error_neg = self.calculate_alignment_error(data, ee_body_id)
                        
                        # Finite difference
                        jacobian[:, j] = (error_pos - error_neg) / (2 * delta)
                        
                        # Restore position
                        for k, joint_idx in enumerate(arm_joint_indices):
                            data.qpos[joint_idx] = current_q[k]
                        mujoco.mj_forward(self.model, data)
                    
                    # Solve for wrist joint updates only
                    try:
                        JTJ = jacobian.T @ jacobian
                        reg = 1e-6 * np.eye(JTJ.shape[0])
                        delta_q_wrist = -np.linalg.solve(JTJ + reg, jacobian.T @ error)
                    except:
                        delta_q_wrist = -np.linalg.pinv(jacobian, rcond=1e-4) @ error
                    
                    # Limit step size
                    delta_q_norm = np.linalg.norm(delta_q_wrist)
                    if delta_q_norm > 0.1:
                        delta_q_wrist = delta_q_wrist * (0.1 / delta_q_norm)
                    
                    # Update ONLY wrist joints
                    new_q = current_q.copy()
                    for j in range(wrist_joint_count):
                        wrist_joint_idx = len(arm_joint_indices) - wrist_joint_count + j
                        new_q[wrist_joint_idx] += self.step_size * delta_q_wrist[j]
                    
                    # Apply joint limits to wrist joints only
                    for j in range(wrist_joint_count):
                        wrist_joint_idx = len(arm_joint_indices) - wrist_joint_count + j
                        joint_idx = arm_joint_indices[wrist_joint_idx]
                        
                        if joint_idx < len(self.model.jnt_range):
                            joint_range = self.model.jnt_range[joint_idx]
                            if not (np.isinf(joint_range[0]) or np.isinf(joint_range[1])):
                                new_q[wrist_joint_idx] = np.clip(new_q[wrist_joint_idx], joint_range[0], joint_range[1])
                    
                    # Set new position
                    for i, joint_idx in enumerate(arm_joint_indices):
                        data.qpos[joint_idx] = new_q[i]
                    mujoco.mj_forward(self.model, data)
                
                # Set to best found position
                for i, joint_idx in enumerate(arm_joint_indices):
                    data.qpos[joint_idx] = best_q[i]
                mujoco.mj_forward(self.model, data)
                
                return best_q, best_error_norm
        
        # Use the wrist-only camera alignment IK
        wrist_ik = WristOnlyAlignmentIK(model, objects_center)
        optimized_q, final_error = wrist_ik.solve_wrist_only(data, arm_joint_indices, ee_body_id, initial_q)
        
        print(f"Wrist-only camera alignment completed. Final error: {final_error:.4f}")
        return optimized_q
    
    def calculate_pinch_site_orientation_alignment(self, model, data, ee_body_id, ee_site_id, use_site, query_pos):
        """Calculate how well the pinch site/end effector is oriented toward the query object"""
        try:
            # Get pinch site position and orientation
            if use_site and ee_site_id is not None:
                pinch_pos = data.site_xpos[ee_site_id].copy()
                # Get end effector orientation (the direction the pinch site is "facing")
                ee_rotation_matrix = data.body(ee_body_id).xmat.reshape(3, 3)
                pinch_direction = -ee_rotation_matrix[:, 2]  # Z-axis direction (typically pointing direction)
            else:
                pinch_pos = data.body(ee_body_id).xpos.copy()
                ee_rotation_matrix = data.body(ee_body_id).xmat.reshape(3, 3)
                pinch_direction = -ee_rotation_matrix[:, 2]
            
            # Vector from pinch site to query object
            to_query = np.array(query_pos) - pinch_pos
            distance = np.linalg.norm(to_query)
            
            if distance < 0.01:  # Too close for meaningful orientation
                return 1.0
            
            to_query_normalized = to_query / distance
            
            # Calculate alignment (dot product gives cosine of angle between vectors)
            alignment = np.dot(pinch_direction, to_query_normalized)
            
            # DEBUG: Print alignment details
            print(f"DEBUG ALIGNMENT: pinch_pos={pinch_pos}, query_pos={query_pos}")
            print(f"  to_query vector: {to_query}, distance: {distance:.3f}")
            print(f"  pinch_direction: {pinch_direction}")
            print(f"  to_query_normalized: {to_query_normalized}")
            print(f"  dot product (raw alignment): {alignment:.3f}")
            
            # Convert to 0-1 score (1 = perfectly aligned, 0 = opposite direction)
            alignment_score = (alignment + 1.0) / 2.0
            
            return max(0.0, min(1.0, alignment_score))
            
        except Exception as e:
            print(f"Error calculating pinch orientation alignment: {e}")
            return 0.0

    def orient_pinch_site_toward_query(self, model, data, arm_joint_indices, ee_body_id, ee_site_id, use_site, query_pos, query_object, max_iterations=30):
        """Slightly adjust wrist joints to orient pinch site toward query object (NO major arm movement)"""
        
        # Use actual object position from simulation instead of stored position
        actual_query_pos = self.get_actual_object_position_in_sim(model, data, query_object)
        if actual_query_pos is not None:
            query_pos = actual_query_pos
            print(f"Using ACTUAL object position from simulation: {query_pos}")
        else:
            print(f"Fallback to stored position: {query_pos}")
            
        print(f"Orienting pinch site toward query object at: {query_pos}")
        
        # Get current pinch site position
        if use_site and ee_site_id is not None:
            current_pinch_pos = data.site_xpos[ee_site_id].copy()
            print(f"Current pinch site position: {current_pinch_pos}")
        else:
            current_pinch_pos = data.body(ee_body_id).xpos.copy()
            print(f"Current end effector position: {current_pinch_pos}")
        
        # Calculate initial alignment
        initial_alignment = self.calculate_pinch_site_orientation_alignment(model, data, ee_body_id, ee_site_id, use_site, query_pos)
        print(f"Initial orientation alignment: {initial_alignment:.3f}")
        
        if initial_alignment > 0.85:  # Already well aligned (85% threshold)
            print("Already well oriented toward query object")
            return np.array([data.qpos[idx] for idx in arm_joint_indices])
        
        best_q = np.array([data.qpos[idx] for idx in arm_joint_indices])
        best_alignment = initial_alignment
        
        # Only adjust last 3 joints (wrist joints) for orientation - NO major arm movement
        wrist_joint_count = min(3, len(arm_joint_indices))
        print(f"Adjusting only last {wrist_joint_count} wrist joints for orientation")
        
        # Small orientation adjustments - these should be tiny movements
        orientation_adjustments = [
            [0, 0, 0],                          # No change
            [0.015, 0, 0], [-0.015, 0, 0],     # Tiny wrist pitch (1.5 degrees)
            [0, 0.015, 0], [0, -0.015, 0],     # Tiny wrist yaw
            [0, 0, 0.015], [0, 0, -0.015],     # Tiny wrist roll
            [0.03, 0, 0], [-0.03, 0, 0],       # Small wrist pitch (3 degrees)
            [0, 0.03, 0], [0, -0.03, 0],       # Small wrist yaw
            [0.015, 0.015, 0], [-0.015, -0.015, 0], # Combined tiny adjustments
            [0.045, 0, 0], [-0.045, 0, 0],     # Medium wrist pitch (4.5 degrees)
            [0, 0.045, 0], [0, -0.045, 0],     # Medium wrist yaw
            [0.03, 0.03, 0], [-0.03, -0.03, 0], # Combined small adjustments
        ]
        
        for adjustment in orientation_adjustments:
            try:
                test_q = best_q.copy()
                
                # Apply ONLY to wrist joints (last few joints)
                for j in range(min(len(adjustment), wrist_joint_count)):
                    wrist_joint_idx = len(test_q) - wrist_joint_count + j
                    if wrist_joint_idx >= 0:
                        test_q[wrist_joint_idx] += adjustment[j]
                
                # Apply joint limits to wrist joints
                for j in range(wrist_joint_count):
                    wrist_joint_idx = len(test_q) - wrist_joint_count + j
                    if wrist_joint_idx >= 0:
                        joint_idx = arm_joint_indices[wrist_joint_idx]
                        if joint_idx < len(model.jnt_range):
                            joint_range = model.jnt_range[joint_idx]
                            if not (np.isinf(joint_range[0]) or np.isinf(joint_range[1])):
                                test_q[wrist_joint_idx] = np.clip(test_q[wrist_joint_idx], joint_range[0], joint_range[1])
                
                # Apply test configuration
                for j, joint_idx in enumerate(arm_joint_indices):
                    data.qpos[joint_idx] = test_q[j]
                mujoco.mj_forward(model, data)
                
                # Check new alignment
                new_alignment = self.calculate_pinch_site_orientation_alignment(model, data, ee_body_id, ee_site_id, use_site, query_pos)
                
                # Also check that we didn't move the pinch site too much
                if use_site and ee_site_id is not None:
                    new_pinch_pos = data.site_xpos[ee_site_id].copy()
                else:
                    new_pinch_pos = data.body(ee_body_id).xpos.copy()
                
                movement_distance = np.linalg.norm(new_pinch_pos - current_pinch_pos)
                
                # Accept if alignment improved AND movement is minimal (less than 2cm)
                if new_alignment > best_alignment and movement_distance < 0.02:
                    best_alignment = new_alignment
                    best_q = test_q.copy()
                    print(f"  Improved orientation alignment to {new_alignment:.3f} with minimal movement ({movement_distance*100:.1f}cm)")
                
            except Exception as e:
                continue
        
        # Apply best configuration
        for j, joint_idx in enumerate(arm_joint_indices):
            data.qpos[joint_idx] = best_q[j]
        mujoco.mj_forward(model, data)
        
        # Final verification
        final_alignment = self.calculate_pinch_site_orientation_alignment(model, data, ee_body_id, ee_site_id, use_site, query_pos)
        if use_site and ee_site_id is not None:
            final_pinch_pos = data.site_xpos[ee_site_id].copy()
        else:
            final_pinch_pos = data.body(ee_body_id).xpos.copy()
        
        total_movement = np.linalg.norm(final_pinch_pos - current_pinch_pos)
        
        print(f"Final orientation alignment: {final_alignment:.3f}")
        print(f"Total pinch site movement: {total_movement*100:.1f}cm (should be minimal)")
        print("Orientation adjustment complete - arm position minimally changed")
        
        return best_q

    def align_end_effector_to_object_top(self, model, data, arm_joint_indices, ee_body_id, ee_site_id, use_site, query_pos, query_object, safety_margin):
        """UPDATED: Just return a target position for orientation reference - don't move arm there"""
        
        # Calculate object height for reference
        object_height = self.get_actual_object_height(model, data, query_object)
        
        # Create a reference target position (object center with small safety margin)
        # This is just for orientation reference, NOT for moving the arm to this position
        target_pos = np.array(query_pos).copy()
        target_pos[2] = query_pos[2] + safety_margin * 0.5  # Very small height adjustment for reference
        
        print(f"Reference target for orientation: {target_pos}")
        print(f"NOTE: This is just for orientation reference, not arm movement target")
        
        return target_pos

    def align_end_effector_wrist_only(self, model, data, arm_joint_indices, ee_body_id, ee_site_id, use_site, target_pos, query_pos_array, query_object):
        """UPDATED: Use new orientation-based alignment instead of position-based movement"""
        
        print("Using orientation-based end effector alignment (wrist joints only)")
        
        # Use actual query position instead of target_ee_pos
        actual_query_pos = self.get_actual_object_position_in_sim(model, data, query_object)
        orient_target = actual_query_pos if actual_query_pos is not None else query_pos_array

        result_q = self.orient_pinch_site_toward_query(
            model, data, arm_joint_indices, ee_body_id, ee_site_id, use_site, orient_target, query_object
        )
                
        return result_q
    
    def align_camera_for_observation(self, model, data, arm_joint_indices, ee_body_id, target_pos):
        """Improved camera alignment that actually points at the target"""
        print(f"Aligning camera to view target at: {target_pos}")
        
        # Get current arm position
        current_q = np.array([data.qpos[idx] for idx in arm_joint_indices])
        
        # Calculate optimal camera position (positioned to view target from good angle)
        target_array = np.array(target_pos)
        
        # Position camera ~30cm away from target at good viewing angle
        camera_offset = np.array([0.25, 0.1, 0.15])  # 25cm back, 10cm right, 15cm up
        desired_camera_pos = target_array + camera_offset
        
        # Use IK to move end effector (and attached camera) to desired position
        success, achieved_pos = self.move_to_waypoint_with_rrt(
            model, data, desired_camera_pos, arm_joint_indices, ee_body_id, None, 
            CollisionAwareIK(model, self.rrt_system, step_size=0.5, tol=0.01, max_iter=500)
        )
        
        if not success:
            print("Failed to move camera to optimal position, trying alternative positions...")
            # Try different camera positions
            alternative_offsets = [
                np.array([0.2, -0.1, 0.2]),   # Left side, higher
                np.array([0.3, 0, 0.1]),      # Straight back
                np.array([0.15, 0.15, 0.25]), # Right side, much higher
                np.array([0.1, -0.2, 0.15])   # Left back
            ]
            
            for offset in alternative_offsets:
                alt_camera_pos = target_array + offset
                success, achieved_pos = self.move_to_waypoint_with_rrt(
                    model, data, alt_camera_pos, arm_joint_indices, ee_body_id, None,
                    CollisionAwareIK(model, self.rrt_system, step_size=0.5, tol=0.01, max_iter=300)
                )
                if success:
                    print(f"Alternative camera position successful: {achieved_pos}")
                    break
        
        # Fine-tune camera pointing using ONLY wrist joints
        print("Fine-tuning camera pointing with wrist joints...")
        self.fine_tune_camera_pointing_wrist_only(model, data, arm_joint_indices, ee_body_id, target_pos)
        
        final_q = np.array([data.qpos[idx] for idx in arm_joint_indices])
        final_alignment = self.calculate_camera_alignment_score(model, data, ee_body_id, target_pos)
        
        print(f"Final camera alignment score: {final_alignment:.3f}")
        return final_q

    def fine_tune_camera_pointing_wrist_only(self, model, data, arm_joint_indices, ee_body_id, target_pos, max_iterations=30):
        """Fine-tune camera pointing using ONLY wrist joints (last 3 joints)"""
        
        best_q = np.array([data.qpos[idx] for idx in arm_joint_indices])
        best_alignment = self.calculate_camera_alignment_score(model, data, ee_body_id, target_pos)
        
        print(f"Starting camera fine-tuning, initial alignment: {best_alignment:.3f}")
        
        # Only adjust last 3 joints (wrist joints)
        wrist_joint_count = min(3, len(arm_joint_indices))
        
        # Systematic search for better camera pointing
        search_ranges = [
            np.linspace(-0.1, 0.1, 11),   # ±0.1 radians (~6 degrees)
            np.linspace(-0.15, 0.15, 13), # ±0.15 radians (~9 degrees)  
            np.linspace(-0.2, 0.2, 15)    # ±0.2 radians (~12 degrees)
        ]
        
        for range_idx, search_range in enumerate(search_ranges):
            improved = False
            
            # Try all combinations of adjustments for wrist joints
            for delta1 in search_range[::2]:  # Skip some values for speed
                for delta2 in search_range[::2]:
                    for delta3 in search_range[::3]:  # Even more sparse for 3rd joint
                        
                        test_q = best_q.copy()
                        
                        # Apply adjustments to last 3 joints only
                        if len(test_q) >= 3:
                            test_q[-3] += delta1  # Wrist joint 1
                            test_q[-2] += delta2  # Wrist joint 2  
                            test_q[-1] += delta3  # Wrist joint 3
                        
                        # Apply joint limits
                        for j in range(wrist_joint_count):
                            joint_idx_in_list = len(test_q) - wrist_joint_count + j
                            if joint_idx_in_list >= 0:
                                arm_joint_idx = arm_joint_indices[joint_idx_in_list]
                                if arm_joint_idx < len(model.jnt_range):
                                    joint_range = model.jnt_range[arm_joint_idx]
                                    if not (np.isinf(joint_range[0]) or np.isinf(joint_range[1])):
                                        test_q[joint_idx_in_list] = np.clip(test_q[joint_idx_in_list], joint_range[0], joint_range[1])
                        
                        # Test this configuration
                        for j, joint_idx in enumerate(arm_joint_indices):
                            data.qpos[joint_idx] = test_q[j]
                        mujoco.mj_forward(model, data)
                        
                        # Calculate alignment
                        alignment = self.calculate_camera_alignment_score(model, data, ee_body_id, target_pos)
                        
                        if alignment > best_alignment + 0.01:  # Require meaningful improvement
                            best_alignment = alignment
                            best_q = test_q.copy()
                            improved = True
                            print(f"Range {range_idx+1}: Improved alignment to {alignment:.3f}")
            
            # If we found improvement in this range, continue to next range
            if not improved:
                print(f"No improvement in range {range_idx+1}, stopping search")
                break
        
        # Apply best configuration
        for j, joint_idx in enumerate(arm_joint_indices):
            data.qpos[joint_idx] = best_q[j]
        mujoco.mj_forward(model, data)
        
        print(f"Camera fine-tuning complete. Final alignment: {best_alignment:.3f}")
        return best_q

    def align_end_effector_orientation_adaptive(self, model, data, arm_joint_indices, ee_body_id, ee_site_id, use_site, query_pos_array, query_object):
        """IMPROVED: Only apply alignment if it actually helps and doesn't break proximity"""
        print("Checking if end effector alignment is beneficial...")
        
        if len(arm_joint_indices) < 6:
            print("Warning: Not enough joints for proper alignment")
            return
        
        # Get current position and distance to target
        if use_site and ee_site_id is not None:
            current_pinch_pos = data.site_xpos[ee_site_id].copy()
        else:
            current_pinch_pos = data.body(ee_body_id).xpos.copy()
        
        initial_distance = np.linalg.norm(query_pos_array - current_pinch_pos)
        initial_q = np.array([data.qpos[idx] for idx in arm_joint_indices])
        
        # Calculate initial alignment
        initial_alignment = self.calculate_pinch_site_orientation_alignment(model, data, ee_body_id, ee_site_id, use_site, query_pos_array)
        
        print(f"Initial: distance={initial_distance*100:.1f}cm, alignment={initial_alignment:.3f}")
        
        # Only proceed if alignment is really poor AND we're not very close to target
        if initial_alignment > 0.4 or initial_distance < 0.04:
            print("Skipping alignment - either already decent alignment or too close to target")
            return
        
        # Store original state
        original_q = initial_q.copy()
        
        # CONSERVATIVE alignment - only joints 5-6, very small adjustments
        joint_5_idx = arm_joint_indices[4] if len(arm_joint_indices) > 4 else None
        joint_6_idx = arm_joint_indices[5] if len(arm_joint_indices) > 5 else None
        
        if joint_5_idx is None or joint_6_idx is None:
            print("Not enough joints for alignment")
            return
        
        best_q = original_q.copy()
        best_score = initial_alignment - (initial_distance * 2.0)  # Heavily weight proximity
        
        # Very small adjustments only
        small_adjustments = [
            [0.0, 0.0],           # No change
            [0.02, 0.0], [-0.02, 0.0],   # Tiny joint 5 adjustments
            [0.0, 0.02], [0.0, -0.02],   # Tiny joint 6 adjustments  
            [0.01, 0.01], [-0.01, -0.01], # Combined tiny
            [0.03, 0.0], [-0.03, 0.0],   # Small joint 5
            [0.0, 0.03], [0.0, -0.03],   # Small joint 6
        ]
        
        for j5_delta, j6_delta in small_adjustments:
            try:
                test_q = original_q.copy()
                test_q[4] = original_q[4] + j5_delta  # Joint 5
                test_q[5] = original_q[5] + j6_delta  # Joint 6
                
                # Apply joint limits
                if joint_5_idx < len(model.jnt_range):
                    joint_range = model.jnt_range[joint_5_idx]
                    if not (np.isinf(joint_range[0]) or np.isinf(joint_range[1])):
                        test_q[4] = np.clip(test_q[4], joint_range[0], joint_range[1])
                
                if joint_6_idx < len(model.jnt_range):
                    joint_range = model.jnt_range[joint_6_idx]
                    if not (np.isinf(joint_range[0]) or np.isinf(joint_range[1])):
                        test_q[5] = np.clip(test_q[5], joint_range[0], joint_range[1])
                
                # Apply test configuration
                for j, joint_idx in enumerate(arm_joint_indices):
                    data.qpos[joint_idx] = test_q[j]
                mujoco.mj_forward(model, data)
                
                # Check new position and alignment
                if use_site and ee_site_id is not None:
                    new_pinch_pos = data.site_xpos[ee_site_id].copy()
                else:
                    new_pinch_pos = data.body(ee_body_id).xpos.copy()
                
                new_distance = np.linalg.norm(query_pos_array - new_pinch_pos)
                new_alignment = self.calculate_pinch_site_orientation_alignment(model, data, ee_body_id, ee_site_id, use_site, query_pos_array)
                
                # CRITICAL: Only accept if position doesn't get significantly worse
                if new_distance > initial_distance + 0.01:  # Allow max 1cm position degradation
                    continue
                
                # Combined score favoring proximity over alignment
                combined_score = new_alignment - (new_distance * 2.0)
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_q = test_q.copy()
                    print(f"  Better config: distance={new_distance*100:.1f}cm, alignment={new_alignment:.3f}")
                    
            except Exception as e:
                continue
        
        # Apply best configuration
        for j, joint_idx in enumerate(arm_joint_indices):
            data.qpos[joint_idx] = best_q[j]
        mujoco.mj_forward(model, data)
        
        # Verify final result
        if use_site and ee_site_id is not None:
            final_pinch_pos = data.site_xpos[ee_site_id].copy()
        else:
            final_pinch_pos = data.body(ee_body_id).xpos.copy()
        
        final_distance = np.linalg.norm(query_pos_array - final_pinch_pos)
        final_alignment = self.calculate_pinch_site_orientation_alignment(model, data, ee_body_id, ee_site_id, use_site, query_pos_array)
        
        movement = np.linalg.norm(final_pinch_pos - current_pinch_pos)
        
        print(f"Final: distance={final_distance*100:.1f}cm, alignment={final_alignment:.3f}, moved={movement*100:.1f}cm")
        
        # If result is significantly worse, revert to original
        if final_distance > initial_distance + 0.015:  # If position degraded by >1.5cm
            print("Alignment made position worse, reverting...")
            for j, joint_idx in enumerate(arm_joint_indices):
                data.qpos[joint_idx] = original_q[j]
            mujoco.mj_forward(model, data)
        else:
            print("Alignment complete with acceptable position preservation")

    def attempt_final_distance_correction(self, model, data, arm_joint_indices, ee_body_id, ee_site_id, query_pos_array, distance_threshold, ik_solver):
        """Final attempt to bring pinch site within distance threshold"""
        print("Attempting final distance correction...")
        
        # Calculate a target position that's exactly at the threshold distance
        current_pos = data.site_xpos[ee_site_id] if ee_site_id is not None else data.body(ee_body_id).xpos
        
        direction = (query_pos_array - current_pos)
        current_distance = np.linalg.norm(direction)
        
        if current_distance > distance_threshold:
            # Move closer to target
            move_distance = current_distance - distance_threshold
            direction_normalized = direction / current_distance
            correction_target = current_pos + (direction_normalized * move_distance)
            
            print(f"Correction target: {correction_target}")
            
            # Use IK to reach correction target
            corrected_q, corrected_pos, error = self.move_arm_to_target_with_obstacles(
                model, data, correction_target, arm_joint_indices, ee_body_id, ik_solver
            )
            
            if corrected_q is not None:
                corrected_distance = np.linalg.norm(query_pos_array - corrected_pos)
                print(f"Distance correction: {current_distance*100:.1f}cm -> {corrected_distance*100:.1f}cm")
                return True
            else:
                print("Distance correction failed")
                return False
        
        return True  # Already within threshold

    def calculate_end_effector_alignment_score(self, model, data, ee_body_id, ee_site_id, use_site, target_pos):
        """Calculate how well the end effector is aligned to point at the target"""
        try:
            # Get end effector position and orientation
            if use_site and ee_site_id is not None:
                ee_pos = data.site_xpos[ee_site_id].copy()
                # Get end effector orientation (pointing direction)
                ee_rotation_matrix = data.body(ee_body_id).xmat.reshape(3, 3)
                ee_direction = -ee_rotation_matrix[:, 2]  # End effector pointing direction
            else:
                ee_pos = data.body(ee_body_id).xpos.copy()
                ee_rotation_matrix = data.body(ee_body_id).xmat.reshape(3, 3)
                ee_direction = -ee_rotation_matrix[:, 2]
            
            # Vector from end effector to target
            to_target = np.array(target_pos) - ee_pos
            distance = np.linalg.norm(to_target)
            
            if distance < 0.02:  # Too close (2cm)
                return 1.0  # Perfect if very close
            
            to_target_normalized = to_target / distance
            alignment = np.dot(ee_direction, to_target_normalized)
            
            # Bonus for good distance (not too far)
            distance_score = 1.0
            if distance > 0.15:  # Penalty for being too far (>15cm)
                distance_score = max(0.3, 1.0 - (distance - 0.15) / 0.1)
            
            return max(0.0, alignment * distance_score)
            
        except:
            return 0.0
        
    def calculate_camera_alignment_score(self, model, data, ee_body_id, target_pos):
        """Calculate how well the camera is aligned to the target position"""
        try:
            # Try to get wrist camera first
            camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist")
            if camera_id >= 0:
                camera_pos = data.cam_xpos[camera_id].copy()
                camera_mat = data.cam_xmat[camera_id].reshape(3, 3)
                camera_direction = -camera_mat[:, 2]  # Camera forward direction
                print(f"Using wrist camera at: {camera_pos}")
            else:
                # Fallback to end effector orientation
                camera_pos = data.body(ee_body_id).xpos.copy()
                ee_rotation_matrix = data.body(ee_body_id).xmat.reshape(3, 3)
                camera_direction = -ee_rotation_matrix[:, 2]
                print(f"Using end effector as camera at: {camera_pos}")
            
            # Vector from camera to target
            to_target = np.array(target_pos) - camera_pos
            distance = np.linalg.norm(to_target)
            
            print(f"Camera to target: distance={distance*100:.1f}cm, vector={to_target}")
            
            if distance < 0.02:  # Too close (2cm)
                print("Camera too close to target")
                return 0.0
            
            to_target_normalized = to_target / distance
            alignment = np.dot(camera_direction, to_target_normalized)
            
            print(f"Camera direction: {camera_direction}")
            print(f"To target normalized: {to_target_normalized}")  
            print(f"Dot product alignment: {alignment:.3f}")
            
            # Good viewing distance range: 15cm to 40cm
            distance_score = 1.0
            if distance < 0.15:  # Too close
                distance_score = distance / 0.15  # Linear penalty
            elif distance > 0.40:  # Too far
                distance_score = 0.40 / distance  # Inverse penalty
            
            # Convert alignment from [-1,1] to [0,1] range
            alignment_score = max(0.0, (alignment + 1.0) / 2.0)
            
            final_score = alignment_score * distance_score
            
            print(f"Distance score: {distance_score:.3f}, Alignment score: {alignment_score:.3f}")
            print(f"Final camera score: {final_score:.3f}")
            
            return final_score
            
        except Exception as e:
            print(f"Camera alignment calculation failed: {e}")
            return 0.0

    def fine_tune_query_alignment(self, model, data, arm_joint_indices, ee_body_id, query_pos, base_q, max_iterations=50):
        """Fine-tune arm pose for PERFECT query object centering"""
        best_q = base_q.copy()
        best_alignment = self.calculate_camera_alignment_score(model, data, ee_body_id, query_pos)
        
        print(f"Starting alignment: {best_alignment:.4f}")
        
        # More comprehensive adjustment patterns for better centering
        adjustment_rounds = [
            # Round 1: Wrist-only fine adjustments
            {
                'joints': [-3, -2, -1],  # Last 3 joints (wrist)
                'adjustments': [
                    [0, 0, 0],                          # No change
                    [0.01, 0, 0], [-0.01, 0, 0],       # Tiny pitch
                    [0, 0.01, 0], [0, -0.01, 0],       # Tiny yaw
                    [0, 0, 0.01], [0, 0, -0.01],       # Tiny roll
                    [0.02, 0, 0], [-0.02, 0, 0],       # Small pitch
                    [0, 0.02, 0], [0, -0.02, 0],       # Small yaw
                    [0.01, 0.01, 0], [-0.01, -0.01, 0], # Combined
                    [0.03, 0, 0], [-0.03, 0, 0],       # Medium pitch
                    [0, 0.03, 0], [0, -0.03, 0],       # Medium yaw
                ]
            },
            # Round 2: Include elbow for larger corrections if needed
            {
                'joints': [-4, -3, -2, -1],  # Last 4 joints (elbow + wrist)
                'adjustments': [
                    [0, 0, 0, 0],                       # No change
                    [0.01, 0, 0, 0], [-0.01, 0, 0, 0], # Elbow
                    [0, 0.02, 0, 0], [0, -0.02, 0, 0], # Wrist pitch
                    [0, 0, 0.02, 0], [0, 0, -0.02, 0], # Wrist yaw
                    [0.01, 0.01, 0, 0], [-0.01, -0.01, 0, 0], # Combined
                ]
            }
        ]
        
        improvement_threshold = 0.001  # Minimum improvement to continue
        
        for round_num, round_config in enumerate(adjustment_rounds):
            print(f"Alignment round {round_num + 1}: targeting joints {round_config['joints']}")
            round_improved = False
            
            for adjustment in round_config['adjustments']:
                try:
                    test_q = best_q.copy()
                    
                    # Apply adjustments to specified joints
                    for i, joint_offset in enumerate(round_config['joints']):
                        if len(test_q) >= abs(joint_offset) and i < len(adjustment):
                            joint_idx = len(test_q) + joint_offset if joint_offset < 0 else joint_offset
                            test_q[joint_idx] += adjustment[i]
                    
                    # Apply joint limits
                    for i, joint_idx in enumerate(arm_joint_indices):
                        if joint_idx < len(model.jnt_range):
                            joint_range = model.jnt_range[joint_idx]
                            if not (np.isinf(joint_range[0]) or np.isinf(joint_range[1])):
                                test_q[i] = np.clip(test_q[i], joint_range[0], joint_range[1])

                    # Test this configuration
                    for j, joint_idx in enumerate(arm_joint_indices):
                        data.qpos[joint_idx] = test_q[j]
                    mujoco.mj_forward(model, data)

                    alignment = self.calculate_camera_alignment_score(model, data, ee_body_id, query_pos)
                    
                    if alignment > best_alignment + improvement_threshold:
                        best_alignment = alignment
                        best_q = test_q.copy()
                        round_improved = True
                        print(f"  Improved alignment to: {alignment:.4f}")

                except Exception as e:
                    continue
            
            # If this round didn't improve much, stop early
            if not round_improved:
                print(f"  No significant improvement in round {round_num + 1}, stopping")
                break
        
        # Set best position
        for j, joint_idx in enumerate(arm_joint_indices):
            data.qpos[joint_idx] = best_q[j]
        mujoco.mj_forward(model, data)
        
        print(f"Final alignment score: {best_alignment:.4f}")
        return best_q
    
    def fine_tune_wrist_only_alignment(self, model, data, arm_joint_indices, ee_body_id, query_pos, base_q, max_iterations=50):
        """Fine-tune ONLY wrist joints for query image centering without moving whole arm"""
        best_q = base_q.copy()
        best_alignment = self.calculate_camera_alignment_score(model, data, ee_body_id, query_pos)
        
        print(f"Starting wrist-only alignment: {best_alignment:.4f}")
        
        # WRIST-ONLY adjustments (last 3 joints)
        wrist_adjustments = [
            [0, 0, 0],                          # No change
            [0.02, 0, 0], [-0.02, 0, 0],       # Small pitch
            [0, 0.02, 0], [0, -0.02, 0],       # Small yaw
            [0, 0, 0.02], [0, 0, -0.02],       # Small roll
            [0.04, 0, 0], [-0.04, 0, 0],       # Medium pitch
            [0, 0.04, 0], [0, -0.04, 0],       # Medium yaw
            [0.02, 0.02, 0], [-0.02, -0.02, 0], # Combined small
            [0.06, 0, 0], [-0.06, 0, 0],       # Larger pitch
            [0, 0.06, 0], [0, -0.06, 0],       # Larger yaw
            [0.04, 0.04, 0], [-0.04, -0.04, 0], # Combined medium
        ]
        
        for adjustment in wrist_adjustments:
            try:
                test_q = best_q.copy()
                
                # Only adjust last 3 joints (wrist)
                if len(test_q) >= 3:
                    test_q[-3] += adjustment[0]  # Wrist pitch
                    test_q[-2] += adjustment[1]  # Wrist yaw  
                    test_q[-1] += adjustment[2]  # Wrist roll

                # Apply joint limits
                for i, joint_idx in enumerate(arm_joint_indices):
                    if joint_idx < len(model.jnt_range):
                        joint_range = model.jnt_range[joint_idx]
                        if not (np.isinf(joint_range[0]) or np.isinf(joint_range[1])):
                            test_q[i] = np.clip(test_q[i], joint_range[0], joint_range[1])

                # Test this configuration
                for j, joint_idx in enumerate(arm_joint_indices):
                    data.qpos[joint_idx] = test_q[j]
                mujoco.mj_forward(model, data)

                alignment = self.calculate_camera_alignment_score(model, data, ee_body_id, query_pos)
                
                if alignment > best_alignment:
                    best_alignment = alignment
                    best_q = test_q.copy()
                    print(f"  Improved wrist alignment to: {alignment:.4f}")

            except Exception as e:
                continue
        
        # Set best position
        for j, joint_idx in enumerate(arm_joint_indices):
            data.qpos[joint_idx] = best_q[j]
        mujoco.mj_forward(model, data)
        
        print(f"Final wrist-only alignment score: {best_alignment:.4f}")
        return best_q
    
    def optimize_camera_alignment_full_arm(self, model, data, arm_joint_indices, ee_body_id, target_pos, base_q):
        """Optimize camera alignment using full arm movement for larger adjustments"""
        best_q = base_q.copy()
        best_alignment = self.calculate_camera_alignment_score(model, data, ee_body_id, target_pos)
        
        print(f"Full arm alignment starting score: {best_alignment:.4f}")
        
        # Full arm adjustment patterns (shoulder, elbow, wrist)
        full_arm_adjustments = [
            [0, 0, 0, 0, 0, 0, 0],  # No change
            # Shoulder adjustments
            [0.05, 0, 0, 0, 0, 0, 0], [-0.05, 0, 0, 0, 0, 0, 0],
            [0, 0.05, 0, 0, 0, 0, 0], [0, -0.05, 0, 0, 0, 0, 0],
            # Elbow adjustments  
            [0, 0, 0.05, 0, 0, 0, 0], [0, 0, -0.05, 0, 0, 0, 0],
            [0, 0, 0, 0.05, 0, 0, 0], [0, 0, 0, -0.05, 0, 0, 0],
            # Combined movements
            [0.03, 0.03, 0, 0, 0, 0, 0], [-0.03, -0.03, 0, 0, 0, 0, 0],
            [0.03, 0, 0.03, 0, 0, 0, 0], [-0.03, 0, -0.03, 0, 0, 0, 0],
            # Wrist fine-tuning
            [0, 0, 0, 0, 0.02, 0, 0], [0, 0, 0, 0, -0.02, 0, 0],
            [0, 0, 0, 0, 0, 0.02, 0], [0, 0, 0, 0, 0, -0.02, 0],
        ]
        
        for adjustment in full_arm_adjustments:
            try:
                test_q = base_q.copy()
                
                # Apply adjustments to all joints
                for i in range(min(len(test_q), len(adjustment))):
                    test_q[i] += adjustment[i]
                
                # Apply joint limits
                for i, joint_idx in enumerate(arm_joint_indices):
                    if i < len(test_q) and joint_idx < len(model.jnt_range):
                        joint_range = model.jnt_range[joint_idx]
                        if not (np.isinf(joint_range[0]) or np.isinf(joint_range[1])):
                            test_q[i] = np.clip(test_q[i], joint_range[0], joint_range[1])
                
                # Test configuration
                for j, joint_idx in enumerate(arm_joint_indices):
                    if j < len(test_q):
                        data.qpos[joint_idx] = test_q[j]
                mujoco.mj_forward(model, data)
                
                alignment = self.calculate_camera_alignment_score(model, data, ee_body_id, target_pos)
                
                if alignment > best_alignment:
                    best_alignment = alignment
                    best_q = test_q.copy()
                    print(f"  Improved full arm alignment to: {alignment:.4f}")
            
            except Exception as e:
                continue
        
        # Set best position
        for j, joint_idx in enumerate(arm_joint_indices):
            if j < len(best_q):
                data.qpos[joint_idx] = best_q[j]
        mujoco.mj_forward(model, data)
        
        print(f"Final full arm alignment score: {best_alignment:.4f}")
        return best_q
    
    def adjust_wrist_only_for_camera_pointing(self, model, data, arm_joint_indices, ee_body_id, target_pos, initial_q, target_name):
        """ONLY adjust wrist joints (last 3) to point camera at target - NO full arm movement"""
        
        print(f"Adjusting ONLY wrist joints to point camera at {target_name}")
        
        # Set initial position
        for i, joint_idx in enumerate(arm_joint_indices):
            data.qpos[joint_idx] = initial_q[i]
        mujoco.mj_forward(model, data)
        
        best_q = initial_q.copy()
        best_alignment = -1.0
        
        # Only adjust last 3 joints (wrist joints) - NO shoulder, elbow, or other joints
        wrist_joint_count = min(3, len(arm_joint_indices))
        print(f"Will only adjust last {wrist_joint_count} joints (wrist joints)")
        
        # Simple wrist adjustments to point camera
        wrist_adjustments = [
            [0, 0, 0],                          # No change
            [0.02, 0, 0], [-0.02, 0, 0],       # Small wrist pitch
            [0, 0.02, 0], [0, -0.02, 0],       # Small wrist yaw
            [0, 0, 0.02], [0, 0, -0.02],       # Small wrist roll
            [0.04, 0, 0], [-0.04, 0, 0],       # Medium wrist pitch
            [0, 0.04, 0], [0, -0.04, 0],       # Medium wrist yaw
            [0.02, 0.02, 0], [-0.02, -0.02, 0], # Combined adjustments
            [0.06, 0, 0], [-0.06, 0, 0],       # Larger wrist pitch
            [0, 0.06, 0], [0, -0.06, 0],       # Larger wrist yaw
        ]
        
        for adjustment in wrist_adjustments:
            try:
                test_q = initial_q.copy()
                
                # ONLY modify the last few joints (wrist joints)
                for j in range(min(len(adjustment), wrist_joint_count)):
                    wrist_joint_idx = len(test_q) - wrist_joint_count + j
                    if wrist_joint_idx >= 0:
                        test_q[wrist_joint_idx] += adjustment[j]
                
                # Apply joint limits to wrist joints only
                for j in range(wrist_joint_count):
                    wrist_joint_idx = len(test_q) - wrist_joint_count + j
                    if wrist_joint_idx >= 0:
                        joint_idx = arm_joint_indices[wrist_joint_idx]
                        if joint_idx < len(model.jnt_range):
                            joint_range = model.jnt_range[joint_idx]
                            if not (np.isinf(joint_range[0]) or np.isinf(joint_range[1])):
                                test_q[wrist_joint_idx] = np.clip(test_q[wrist_joint_idx], joint_range[0], joint_range[1])
                
                # Set test configuration
                for j, joint_idx in enumerate(arm_joint_indices):
                    data.qpos[joint_idx] = test_q[j]
                mujoco.mj_forward(model, data)
                
                # Get camera direction
                try:
                    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist")
                    if camera_id >= 0:
                        camera_pos = data.cam_xpos[camera_id].copy()
                        camera_mat = data.cam_xmat[camera_id].reshape(3, 3)
                        camera_direction = -camera_mat[:, 2]
                    else:
                        camera_pos = data.body(ee_body_id).xpos.copy()
                        ee_rotation_matrix = data.body(ee_body_id).xmat.reshape(3, 3)
                        camera_direction = -ee_rotation_matrix[:, 2]
                except:
                    camera_pos = data.body(ee_body_id).xpos.copy()
                    ee_rotation_matrix = data.body(ee_body_id).xmat.reshape(3, 3)
                    camera_direction = -ee_rotation_matrix[:, 2]
                
                # Calculate alignment to target
                to_target = np.array(target_pos) - camera_pos
                distance = np.linalg.norm(to_target)
                
                if distance > 0.05:  # Valid distance
                    to_target_normalized = to_target / distance
                    alignment = np.dot(camera_direction, to_target_normalized)
                    
                    if alignment > best_alignment:
                        best_alignment = alignment
                        best_q = test_q.copy()
                        print(f"  Improved camera alignment to {target_name}: {alignment:.3f} with wrist adjustment {adjustment}")
                
            except Exception as e:
                continue
        
        # Set best wrist configuration
        for j, joint_idx in enumerate(arm_joint_indices):
            data.qpos[joint_idx] = best_q[j]
        mujoco.mj_forward(model, data)
        
        print(f"Final camera alignment to {target_name}: {best_alignment:.3f}")
        print("ONLY wrist joints were adjusted - full arm position unchanged")
        return best_q

    def old_capture_query_image(self, model, data, episode_dir, query_object, best_q, 
                    enable_zoom=False, zoom_fov=25, center_precision='high'):
        """
        Capture perfectly centered image of query object with optional zoom
        
        Args:
            enable_zoom (bool): Whether to apply zoom (reduce FOV) for the image
            zoom_fov (float): FOV value to use when zoomed (lower = more zoom)
            center_precision (str): 'low', 'medium', 'high' - how precisely to center
        """
        try:
            # Find arm components
            arm_joint_indices, _ = self.find_arm_indices(model)
            ee_body_id = self.get_end_effector_body_id(model)
            
            # Store the original arm position
            original_q = np.array([data.qpos[idx] for idx in arm_joint_indices])
            
            # Get query object position
            if 'name' in query_object:
                obj_name = query_object['name']
                try:
                    obj_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, obj_name)
                    if obj_body_id >= 0:
                        query_pos = data.body(obj_body_id).xpos.copy()
                        print(f"Got actual object position from simulation: {query_pos}")
                    else:
                        query_pos = query_object['position']
                        print(f"Using stored object position: {query_pos}")
                except:
                    query_pos = query_object['position']
                    print(f"Fallback to stored position: {query_pos}")
            else:
                query_pos = query_object['position']
            
            print(f"Centering camera on query object at: {query_pos}")
            print(f"Zoom enabled: {enable_zoom}, FOV: {zoom_fov if enable_zoom else 'default'}")
            print(f"Centering precision: {center_precision}")
            
            # Apply centering precision settings
            max_iterations = {
                'low': 20,
                'medium': 35,
                'high': 50
            }.get(center_precision, 35)
            
            # Perform WRIST-ONLY precise alignment for query image
            best_centered_q = self.fine_tune_wrist_only_alignment(
                model, data, arm_joint_indices, ee_body_id, 
                query_pos, original_q, max_iterations
            )
            
            # Get camera and store original FOV
            camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist")
            original_fov = None
            
            if camera_id >= 0:
                # Store original FOV
                original_fov = model.cam_fovy[camera_id]
                print(f"Original camera FOV: {original_fov:.1f}°")
                
                # Apply zoom if enabled
                if enable_zoom:
                    model.cam_fovy[camera_id] = zoom_fov
                    print(f"Applied zoom FOV: {zoom_fov:.1f}°")
                
                # Capture the image
                renderer = mujoco.Renderer(model, height=480, width=640)
                renderer.update_scene(data, camera=camera_id)
                image = renderer.render()
                
                query_image_path = os.path.join(episode_dir, "query_obj.png")
                cv2.imwrite(query_image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                
                renderer.close()
                print(f"Captured perfectly centered query image: {query_image_path}")
                
                # Restore original FOV
                if original_fov is not None:
                    model.cam_fovy[camera_id] = original_fov
                    print(f"Restored original FOV: {original_fov:.1f}°")
            else:
                print("Warning: Could not find wrist camera")
            
            # Restore original arm position
            for j, joint_idx in enumerate(arm_joint_indices):
                data.qpos[joint_idx] = original_q[j]
            mujoco.mj_forward(model, data)
            print("Restored original arm position")
                        
        except Exception as e:
            print(f"Failed to capture query image: {e}")
            # Ensure we restore everything even if error occurs
            try:
                # Restore original arm position
                for j, joint_idx in enumerate(arm_joint_indices):
                    data.qpos[joint_idx] = original_q[j]
                mujoco.mj_forward(model, data)
                
                # Restore original FOV if we changed it
                if original_fov is not None and camera_id >= 0:
                    model.cam_fovy[camera_id] = original_fov
                    print("Restored original FOV after error")
            except:
                pass
    
    def capture_query_image(self, model, data, episode_dir, query_object, best_q):
        """Capture query object image (camera should already be aligned)"""
        try:
            camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist")
            if camera_id >= 0:
                renderer = mujoco.Renderer(model, height=480, width=640)
                renderer.update_scene(data, camera=camera_id)
                image = renderer.render()
                
                query_image_path = os.path.join(episode_dir, "query_obj.png")
                cv2.imwrite(query_image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                renderer.close()
                
                print(f"Captured query image: {query_image_path}")
            else:
                print("Warning: Could not find wrist camera")
        except Exception as e:
            print(f"Failed to capture query image: {e}")

    def capture_observation_image(self, model, data, obs_dir, timestep):
        """Capture observation image at current timestep"""
        try:
            camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist")
            if camera_id >= 0:
                renderer = mujoco.Renderer(model, height=480, width=640)
                renderer.update_scene(data, camera=camera_id)
                image = renderer.render()
                
                obs_image_path = os.path.join(obs_dir, f"obs_{timestep}.png")
                cv2.imwrite(obs_image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                renderer.close()
                
        except Exception as e:
            print(f"Failed to capture observation image at timestep {timestep}: {e}")

    def setup_external_camera(self, scene_root, scene_info):
        """Add external camera positioned to capture the whole simulation"""
        # Calculate good camera position based on table and objects
        table_pos = scene_info['config']['table_position']
        objects_center = self.calculate_objects_center(scene_info)
        
        # Position camera on the right side of table, elevated, looking at the scene
        camera_x = table_pos[0] + 0.6  # 60cm to the right of table center
        camera_y = table_pos[1] - 0.3  # Slightly back to get better angle
        camera_z = table_pos[2] + 0.5  # 50cm above table surface
        
        # Calculate target point (center of action)
        target_x = table_pos[0]
        target_y = table_pos[1]
        target_z = table_pos[2] + 0.2  # Slightly above table surface
        
        # Calculate orientation to look at target
        camera_pos = np.array([camera_x, camera_y, camera_z])
        target_pos = np.array([target_x, target_y, target_z])
        forward = target_pos - camera_pos
        forward = forward / np.linalg.norm(forward)
        
        # Create right and up vectors
        up = np.array([0, 0, 1])  # World up
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)
        up = up / np.linalg.norm(up)
        
        # Create camera element
        worldbody = scene_root.find('.//worldbody')
        if worldbody is not None:
            camera_elem = ET.SubElement(worldbody, 'camera')
            camera_elem.set('name', 'external_view')
            camera_elem.set('pos', f'{camera_x} {camera_y} {camera_z}')
            # Set orientation using xyaxes (right vector, up vector)
            camera_elem.set('xyaxes', f'{right[0]} {right[1]} {right[2]} {up[0]} {up[1]} {up[2]}')
            camera_elem.set('fovy', '45')  # Good field of view
            
            print(f"Added external camera at position: [{camera_x}, {camera_y}, {camera_z}]")
            print(f"Camera targeting: [{target_x}, {target_y}, {target_z}]")


    def capture_external_video_frame(self, model, data):
        """Capture a frame from the external camera for video recording"""
        try:
            camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "external_view")
            if camera_id >= 0:
                renderer = mujoco.Renderer(model, height=480, width=640)  # Match framebuffer size
                renderer.update_scene(data, camera=camera_id)
                frame = renderer.render()
                renderer.close()
                return frame
            else:
                print("External camera not found")
                return None
        except Exception as e:
            print(f"Failed to capture external video frame: {e}")
            return None

    def check_arm_collisions(self, model, data, arm_joint_indices, ee_site_id=None):
        """Check if any part of the arm is colliding with objects OR itself, including pinch site"""
        collisions = []
        
        # Get all arm body IDs
        arm_body_ids = set()
        for joint_idx in arm_joint_indices:
            try:
                # Get body that owns this joint
                for body_id in range(model.nbody):
                    for joint_id in range(model.njnt):
                        if model.jnt_bodyid[joint_id] == body_id and joint_id == joint_idx:
                            arm_body_ids.add(body_id)
                            break
            except:
                continue
        
        # Also add common arm body names
        arm_keywords = ['base_link', 'shoulder', 'arm', 'link', 'wrist', 'bracelet', 'hand', 'finger', 'gripper']
        for body_id in range(model.nbody):
            try:
                body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
                if body_name:
                    body_name_lower = body_name.lower()
                    if any(keyword in body_name_lower for keyword in arm_keywords):
                        arm_body_ids.add(body_id)
            except:
                continue
        
        print(f"Monitoring {len(arm_body_ids)} arm bodies for collisions (including self-collision)")
        
        # Check for contacts
        for i in range(data.ncon):
            contact = data.contact[i]
            geom1_id = contact.geom1
            geom2_id = contact.geom2
            
            # Get body IDs for these geoms
            body1_id = model.geom_bodyid[geom1_id]
            body2_id = model.geom_bodyid[geom2_id]
            
            # Check different collision types
            collision_type = None
            collision_info = None
            
            # Case 1: Arm vs Object collision
            if body1_id in arm_body_ids and body2_id not in arm_body_ids:
                try:
                    body2_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body2_id)
                    if body2_name and ('object_' in body2_name or 'table' in body2_name):
                        collision_type = "arm_vs_object"
                        collision_info = {'object_name': body2_name}
                except:
                    pass
            elif body2_id in arm_body_ids and body1_id not in arm_body_ids:
                try:
                    body1_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body1_id)
                    if body1_name and ('object_' in body1_name or 'table' in body1_name):
                        collision_type = "arm_vs_object"
                        collision_info = {'object_name': body1_name}
                except:
                    pass
            
            # Case 2: Arm self-collision (different arm parts touching each other)
            elif body1_id in arm_body_ids and body2_id in arm_body_ids and body1_id != body2_id:
                # Ignore adjacent link collisions (they're supposed to be connected)
                try:
                    body1_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body1_id)
                    body2_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body2_id)
                    
                    # Skip adjacent links that are supposed to be connected
                    adjacent_pairs = [
                        ('base_link', 'shoulder_link'),
                        ('shoulder_link', 'half_arm_1_link'),
                        ('half_arm_1_link', 'half_arm_2_link'),
                        ('half_arm_2_link', 'forearm_link'),
                        ('forearm_link', 'spherical_wrist_1_link'),
                        ('spherical_wrist_1_link', 'spherical_wrist_2_link'),
                        ('spherical_wrist_2_link', 'bracelet_link')
                    ]
                    
                    is_adjacent = False
                    for pair in adjacent_pairs:
                        if (pair[0] in body1_name and pair[1] in body2_name) or (pair[1] in body1_name and pair[0] in body2_name):
                            is_adjacent = True
                            break
                    
                    if not is_adjacent:
                        collision_type = "arm_self_collision"
                        collision_info = {'body1_name': body1_name, 'body2_name': body2_name}
                except:
                    collision_type = "arm_self_collision"
                    collision_info = {'body1_id': body1_id, 'body2_id': body2_id}
            
            if collision_type and collision_info:
                full_collision_info = {
                    'type': collision_type,
                    'contact_id': i,
                    'contact_pos': contact.pos.copy(),
                    'contact_force': np.linalg.norm(contact.frame[:3]),
                    **collision_info
                }
                collisions.append(full_collision_info)
        
        # Additional check: Pinch site collision with objects/environment
        if ee_site_id is not None:
            pinch_pos = data.site_xpos[ee_site_id].copy()
            
            # Check if pinch site is too close to any objects
            for body_id in range(model.nbody):
                if body_id not in arm_body_ids:  # Not part of arm
                    try:
                        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
                        if body_name and ('object_' in body_name or 'table' in body_name):
                            body_pos = data.body(body_id).xpos.copy()
                            distance = np.linalg.norm(pinch_pos - body_pos)
                            
                            # Conservative pinch site clearance
                            min_clearance = 0.02  # 2cm minimum clearance for pinch site
                            if distance < min_clearance:
                                collision_info = {
                                    'type': 'pinch_site_collision',
                                    'contact_id': -1,  # Virtual collision
                                    'object_name': body_name,
                                    'contact_pos': pinch_pos.copy(),
                                    'distance': distance,
                                    'min_clearance': min_clearance
                                }
                                collisions.append(collision_info)
                    except:
                        continue
        
        if collisions:
            for collision in collisions:
                if collision['type'] == 'arm_vs_object':
                    print(f"  🔴 Arm-Object collision: {collision.get('object_name', 'unknown')}")
                elif collision['type'] == 'arm_self_collision':
                    print(f"  🟡 Arm self-collision: {collision.get('body1_name', 'unknown')} vs {collision.get('body2_name', 'unknown')}")
                elif collision['type'] == 'pinch_site_collision':
                    print(f"  🟠 Pinch site too close: {collision.get('object_name', 'unknown')} ({collision.get('distance', 0)*100:.1f}cm)")
        
        return collisions

    def resolve_arm_collisions(self, model, data, arm_joint_indices, ee_body_id, collisions, 
                            target_pos, max_attempts=5, ee_site_id=None):
        """Attempt to resolve arm collisions by slightly adjusting arm position"""
        
        if not collisions:
            return True, np.array([data.qpos[idx] for idx in arm_joint_indices])
        
        print(f"Found {len(collisions)} arm-object collisions, attempting to resolve...")
        
        # Store original position
        original_q = np.array([data.qpos[idx] for idx in arm_joint_indices])
        best_q = original_q.copy()
        
        # Get current end effector position
        current_ee_pos = data.body(ee_body_id).xpos.copy()
        
        for attempt in range(max_attempts):
            print(f"Collision resolution attempt {attempt + 1}/{max_attempts}")
            
            # Generate slight variations to move away from collisions
            # Focus on wrist joints for fine adjustments that don't move the arm too much
            test_q = best_q.copy()
            
            # Strategy: Small random adjustments to last few joints
            adjustment_magnitude = 0.05 + (attempt * 0.02)  # Increase with attempts
            num_joints_to_adjust = min(4, len(arm_joint_indices))  # Adjust last 4 joints
            
            for i in range(num_joints_to_adjust):
                joint_idx = -(i + 1)  # Last few joints
                if abs(joint_idx) <= len(test_q):
                    # Random small adjustment
                    delta = np.random.uniform(-adjustment_magnitude, adjustment_magnitude)
                    test_q[joint_idx] += delta
            
            # Apply joint limits
            for i, joint_idx in enumerate(arm_joint_indices):
                if joint_idx < len(model.jnt_range):
                    joint_range = model.jnt_range[joint_idx]
                    if not (np.isinf(joint_range[0]) or np.isinf(joint_range[1])):
                        test_q[i] = np.clip(test_q[i], joint_range[0], joint_range[1])
            
            # Apply test configuration
            for i, joint_idx in enumerate(arm_joint_indices):
                data.qpos[joint_idx] = test_q[i]
            mujoco.mj_forward(model, data)
            
            # Check for collisions
            new_collisions = self.check_arm_collisions(model, data, arm_joint_indices, ee_site_id)
            
            if not new_collisions:
                print(f"✓ Collision resolved in attempt {attempt + 1}")
                return True, test_q
            else:
                print(f"✗ Still {len(new_collisions)} collisions after attempt {attempt + 1}")
                
                # If fewer collisions, keep this as best attempt
                if len(new_collisions) < len(collisions):
                    best_q = test_q.copy()
                    collisions = new_collisions
        
        # If we couldn't resolve all collisions, use the best attempt
        print(f"Could not fully resolve collisions after {max_attempts} attempts")
        print(f"Using best attempt with {len(collisions)} remaining collisions")
        
        # Apply best configuration
        for i, joint_idx in enumerate(arm_joint_indices):
            data.qpos[joint_idx] = best_q[i]
        mujoco.mj_forward(model, data)
        
        return len(collisions) == 0, best_q

    def check_and_resolve_collisions(self, model, data, arm_joint_indices, ee_body_id, 
                                target_pos, context="", ee_site_id=None):
        """Check for arm collisions and resolve them if found"""
        
        # Check for collisions
        collisions = self.check_arm_collisions(model, data, arm_joint_indices, ee_site_id=ee_site_id)
        
        if collisions:
            print(f"⚠️  {context}: Found collisions with objects:")
            for collision in collisions:
                print(f"   - {collision['object_name']} at {collision['contact_pos']}")
            
            # Attempt to resolve
            resolved, final_q = self.resolve_arm_collisions(
                model, data, arm_joint_indices, ee_body_id, collisions, target_pos, ee_site_id=ee_site_id
            )
            
            if resolved:
                print(f"✅ {context}: All collisions resolved")
                return True, final_q
            else:
                print(f"❌ {context}: Could not resolve all collisions")
                return False, final_q
        else:
            print(f"✅ {context}: No collisions detected")
            current_q = np.array([data.qpos[idx] for idx in arm_joint_indices])
            return True, current_q

    def save_action_space_data(self, action_data, episode_dir):
        """Save action space data to Excel file"""
        import pandas as pd
        
        # Save Excel file
        df = pd.DataFrame(action_data)
        excel_path = os.path.join(episode_dir, "action_space.xlsx")
        df.to_excel(excel_path, index=False)

        # Save CSV file
        csv_path = os.path.join(episode_dir, "action_space.csv")
        df.to_csv(csv_path, index=False)
        
        print(f"Saved action data: {excel_path} and {csv_path}")

    def plan_rrt_joint_path(self, model, data, start_q, goal_q, arm_joint_indices, ik_solver, ee_site_id=None, max_iterations=1000):
        """Plan collision-free path in joint space using RRT"""
        
        # Simple RRT implementation for joint space planning
        nodes = [start_q.copy()]
        parents = [-1]  # Parent indices
        
        step_size = 0.1  # Joint space step size
        goal_threshold = 0.05  # How close to goal is acceptable
        
        for iteration in range(max_iterations):
            # Sample random configuration or bias toward goal
            if np.random.random() < 0.1:  # 10% bias toward goal
                rand_q = goal_q.copy()
            else:
                # Sample random joint configuration within limits
                rand_q = np.array([
                    np.random.uniform(-np.pi, np.pi) for _ in range(len(start_q))
                ])
            
            # Find nearest node
            distances = [np.linalg.norm(node - rand_q) for node in nodes]
            nearest_idx = np.argmin(distances)
            nearest_q = nodes[nearest_idx]
            
            # Step toward random config
            direction = rand_q - nearest_q
            direction_norm = np.linalg.norm(direction)
            
            if direction_norm > step_size:
                direction = (direction / direction_norm) * step_size
            
            new_q = nearest_q + direction
            
            # Check if this configuration is collision-free
            if self.is_joint_config_collision_free(model, data, new_q, arm_joint_indices, ee_site_id):
                nodes.append(new_q.copy())
                parents.append(nearest_idx)
                
                # Check if we reached the goal
                if np.linalg.norm(new_q - goal_q) < goal_threshold:
                    print(f"RRT reached goal in {iteration + 1} iterations")
                    
                    # Reconstruct path
                    path = []
                    current_idx = len(nodes) - 1
                    
                    while current_idx != -1:
                        path.append(nodes[current_idx])
                        current_idx = parents[current_idx]
                    
                    path.reverse()
                    # Smooth the path to reduce unnecessary waypoints
                    smoothed_path = self.smooth_rrt_path(model, data, path, arm_joint_indices, ee_site_id)
                    return smoothed_path
            
            if (iteration + 1) % 100 == 0:
                print(f"RRT iteration {iteration + 1}/{max_iterations}, nodes: {len(nodes)}")
        
        print(f"RRT failed to find path after {max_iterations} iterations")
        return None

    def smooth_rrt_path(self, model, data, path, arm_joint_indices, ee_site_id, max_iterations=50):
        """Smooth RRT path by removing unnecessary waypoints"""
        if len(path) <= 2:
            return path
        
        smoothed = [path[0]]  # Keep start
        i = 0
        
        while i < len(path) - 1:
            # Try to connect current point to points further ahead
            max_skip = min(10, len(path) - i - 1)  # Don't skip more than 10 waypoints
            
            connected = False
            for skip in range(max_skip, 0, -1):
                if i + skip >= len(path):
                    continue
                    
                # Check if we can directly connect path[i] to path[i + skip]
                if self.can_connect_directly(model, data, path[i], path[i + skip], arm_joint_indices, ee_site_id):
                    smoothed.append(path[i + skip])
                    i = i + skip
                    connected = True
                    break
            
            if not connected:
                # Can't skip, add next waypoint
                smoothed.append(path[i + 1])
                i += 1
        
        print(f"Path smoothing: {len(path)} -> {len(smoothed)} waypoints")
        return smoothed

    def can_connect_directly(self, model, data, config1, config2, arm_joint_indices, ee_site_id, num_checks=5):
        """Check if two joint configurations can be connected directly without collision"""
        
        # Sample points along the line between config1 and config2
        for t in np.linspace(0, 1, num_checks):
            test_config = config1 + t * (config2 - config1)
            
            if not self.is_joint_config_collision_free(model, data, test_config, arm_joint_indices, ee_site_id):
                return False
        
        return True

    def is_joint_config_collision_free(self, model, data, joint_config, arm_joint_indices, ee_site_id=None):
        """Enhanced collision checking that considers pinch site as true end effector"""
        
        # Save current state
        original_q = np.array([data.qpos[idx] for idx in arm_joint_indices])
        
        try:
            # Set test configuration
            for i, joint_idx in enumerate(arm_joint_indices):
                if i < len(joint_config):
                    data.qpos[joint_idx] = joint_config[i]
            mujoco.mj_forward(model, data)
            
            # 1. Check arm collisions with objects and self-collision
            arm_collisions = self.check_arm_collisions(model, data, arm_joint_indices, ee_site_id)
            if arm_collisions:
                return False
            
            # 2. Enhanced pinch site collision checking
            if ee_site_id is not None:
                pinch_pos = data.site_xpos[ee_site_id].copy()
                
                # Check minimum clearance from all objects
                for body_id in range(model.nbody):
                    try:
                        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
                        if body_name and ('object_' in body_name or 'table' in body_name):
                            body_pos = data.body(body_id).xpos.copy()
                            distance = np.linalg.norm(pinch_pos - body_pos)
                            
                            # Get object size for proper clearance calculation
                            min_clearance = self.calculate_required_clearance(model, body_id)
                            
                            if distance < min_clearance:
                                return False
                                
                    except:
                        continue
            
            # 3. Check for joint limits violations
            for i, joint_idx in enumerate(arm_joint_indices):
                if i < len(joint_config) and joint_idx < len(model.jnt_range):
                    joint_range = model.jnt_range[joint_idx]
                    if not (np.isinf(joint_range[0]) or np.isinf(joint_range[1])):
                        if joint_config[i] < joint_range[0] or joint_config[i] > joint_range[1]:
                            return False
            
            return True
            
        except Exception as e:
            print(f"Collision check failed: {e}")
            return False
            
        finally:
            # Always restore original configuration
            for i, joint_idx in enumerate(arm_joint_indices):
                data.qpos[joint_idx] = original_q[i]
            mujoco.mj_forward(model, data)

    def calculate_required_clearance(self, model, body_id):
        """Calculate required clearance distance for a body"""
        try:
            # Get all geoms for this body
            max_size = 0.02  # Default 2cm clearance
            
            for geom_id in range(model.ngeom):
                if model.geom_bodyid[geom_id] == body_id:
                    geom_size = model.geom_size[geom_id]
                    geom_type = model.geom_type[geom_id]
                    
                    if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
                        # Use largest dimension + safety margin
                        max_size = max(max_size, max(geom_size) + 0.015)
                    elif geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
                        max_size = max(max_size, geom_size[0] + 0.015)
                    elif geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
                        max_size = max(max_size, max(geom_size[0], geom_size[1]) + 0.015)
                    else:
                        max_size = max(max_size, max(geom_size) + 0.015)
            
            return max_size
            
        except:
            return 0.03  # Default 3cm clearance
        
    def is_position_above_table_safe(self, position, scene_info, safety_margin=0.08):
        """Check if position is safely above table surface"""
        table_pos = scene_info['config']['table_position']
        table_height = table_pos[2]
        min_safe_height = table_height + safety_margin
        
        if position[2] < min_safe_height:
            print(f"Position {position} is below safe height {min_safe_height:.3f} (table at {table_height:.3f})")
            return False
        return True

    def move_to_waypoint_with_rrt(self, model, data, waypoint_pos, arm_joint_indices, ee_body_id, ee_site_id, ik_solver):
        """Move to waypoint using improved RRT with proper multi-step path generation"""
        
        # Get current joint configuration and position
        current_q = np.array([data.qpos[idx] for idx in arm_joint_indices])
        if ee_site_id is not None:
            current_pos = data.site_xpos[ee_site_id].copy()
        else:
            current_pos = data.body(ee_body_id).xpos.copy()
        
        print(f"Moving from {current_pos} to {waypoint_pos}")
        
        # Try IK to find goal configuration for this waypoint
        goal_q, achieved_pos, error = self.move_arm_to_target_with_obstacles(
            model, data, waypoint_pos, arm_joint_indices, ee_body_id, ik_solver
        )
        
        if goal_q is None:
            print(f"IK failed for waypoint: {waypoint_pos}")
            return False, np.array([0, 0, 0])
        
        # Check if direct solution is collision-free
        direct_collisions = self.check_arm_collisions(model, data, arm_joint_indices, ee_site_id)
        
        if not direct_collisions:
            print("Direct IK solution is collision-free")
            return True, achieved_pos
        
        # If direct path has collisions, use RRT to find collision-free path
        print("Direct path has collisions, using RRT...")
        path = self.plan_rrt_joint_path(model, data, current_q, goal_q, arm_joint_indices, ik_solver, ee_site_id, max_iterations=800)
        
        if path is None or len(path) < 2:
            print("RRT failed to find collision-free path")
            
            # Try recovery with lifted intermediate position
            recovery_pos = waypoint_pos.copy()
            recovery_pos[2] += 0.05  # Lift 5cm
            print(f"Attempting recovery with lifted position: {recovery_pos}")
            
            recovery_q, recovery_achieved, recovery_error = self.move_arm_to_target_with_obstacles(
                model, data, recovery_pos, arm_joint_indices, ee_body_id, ik_solver
            )
            
            if recovery_q is not None:
                recovery_collisions = self.check_arm_collisions(model, data, arm_joint_indices, ee_site_id)
                if not recovery_collisions:
                    print("Recovery solution successful")
                    return True, recovery_achieved
            
            return False, np.array([0, 0, 0])
        
        # Execute the planned path (use final waypoint)
        final_config = path[-1]
        for i, joint_idx in enumerate(arm_joint_indices):
            data.qpos[joint_idx] = final_config[i]
        mujoco.mj_forward(model, data)
        
        # Get achieved position
        if ee_site_id is not None:
            final_pos = data.site_xpos[ee_site_id].copy()
        else:
            final_pos = data.body(ee_body_id).xpos.copy()
        
        # Verify no collisions in final configuration
        final_collisions = self.check_arm_collisions(model, data, arm_joint_indices, ee_site_id)
        if final_collisions:
            print("Final configuration has collisions!")
            return False, final_pos
        
        print(f"Successfully moved via RRT path with {len(path)} waypoints")
        return True, final_pos

    def visualize_trajectory_with_spheres(self, model, data, scene_info, action_data, scene_root, scene_path, arm_joint_indices):
        """Add red spheres at all trajectory positions and reload simulation for visualization"""
        try:
            print(f"Adding trajectory visualization spheres for {len(action_data)} timesteps...")
            
            # Find worldbody to add spheres
            worldbody = scene_root.find('.//worldbody')
            if worldbody is None:
                print("Warning: Could not find worldbody to add trajectory spheres")
                return
            
            # Get pinch site ID for trajectory visualization
            ee_site_id = self.get_end_effector_site_id(model)

            # Add sphere for timestep 0 (initial position from action data)
            if len(action_data) > 0:
                # Use the x_t-1, y_t-1, z_t-1 from first action entry (this is timestep 0 position)
                initial_pos = [action_data[0]['x_t-1'], action_data[0]['y_t-1'], action_data[0]['z_t-1']]
                
                # IMPORTANT: Convert from base-relative back to world coordinates if needed
                base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
                if base_body_id >= 0:
                    base_pos = data.body(base_body_id).xpos.copy()
                    # If action data is in base-relative coordinates, convert back to world
                    world_pos = [initial_pos[0] + base_pos[0], initial_pos[1] + base_pos[1], initial_pos[2] + base_pos[2]]
                else:
                    world_pos = initial_pos
                
                sphere_body = ET.SubElement(worldbody, "body")
                sphere_body.set("name", f"pinch_trajectory_sphere_0")
                sphere_body.set("pos", f"{world_pos[0]} {world_pos[1]} {world_pos[2]}")
                
                sphere_geom = ET.SubElement(sphere_body, "geom")
                sphere_geom.set("name", f"pinch_trajectory_sphere_0_geom")
                sphere_geom.set("type", "sphere")
                sphere_geom.set("size", "0.008")
                sphere_geom.set("rgba", "1 0 0 0.8")  # Red for pinch site
                sphere_geom.set("group", "0")
                sphere_geom.set("contype", "0")
                sphere_geom.set("conaffinity", "0")
                
                print(f"Added pinch site trajectory sphere 0 at world position: {world_pos}")

            # Add spheres for timesteps 1-5 using actual pinch site positions
            final_pinch_world_pos = None
            for i, row in enumerate(action_data):
                timestep = i + 1
                # Use the x_t, y_t, z_t values (these should be pinch site positions)
                base_relative_pos = [row['x_t'], row['y_t'], row['z_t']]
                
                # Convert from base-relative back to world coordinates for visualization
                base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
                if base_body_id >= 0:
                    base_pos = data.body(base_body_id).xpos.copy()
                    world_pos = [base_relative_pos[0] + base_pos[0], base_relative_pos[1] + base_pos[1], base_relative_pos[2] + base_pos[2]]
                else:
                    world_pos = base_relative_pos
                
                # Store final position for cube placement
                if timestep == len(action_data):
                    final_pinch_world_pos = world_pos.copy()
                
                sphere_body = ET.SubElement(worldbody, "body")
                sphere_body.set("name", f"pinch_trajectory_sphere_{timestep}")
                sphere_body.set("pos", f"{world_pos[0]} {world_pos[1]} {world_pos[2]}")
                
                sphere_geom = ET.SubElement(sphere_body, "geom")
                sphere_geom.set("name", f"pinch_trajectory_sphere_{timestep}_geom")
                sphere_geom.set("type", "sphere")
                sphere_geom.set("size", "0.008")
                sphere_geom.set("rgba", "1 0 0 0.8")  # Red for pinch site trajectory
                sphere_geom.set("group", "0")
                sphere_geom.set("contype", "0")
                sphere_geom.set("conaffinity", "0")
                
                print(f"Added pinch site trajectory sphere {timestep} at world position: {world_pos}")

            # Add 1cm x 1cm x 1cm reference cube beside arm's final position
            if final_pinch_world_pos is not None:
                # Get table position for reference
                table_pos = scene_info['config']['table_position']
                table_height = table_pos[2]  # Table surface height
                
                # Position cube beside the final arm position, on the table surface
                cube_offset = 0.03  # 3cm away from final pinch position
                cube_pos = [
                    final_pinch_world_pos[0] + cube_offset,  # Slightly to the side
                    final_pinch_world_pos[1],                # Same Y as final position
                    table_height + 0.005                     # On table surface (0.5cm cube center height)
                ]
                
                cube_body = ET.SubElement(worldbody, "body")
                cube_body.set("name", "reference_cube_1cm")
                cube_body.set("pos", f"{cube_pos[0]} {cube_pos[1]} {cube_pos[2]}")
                
                cube_geom = ET.SubElement(cube_body, "geom")
                cube_geom.set("name", "reference_cube_1cm_geom")
                cube_geom.set("type", "box")
                cube_geom.set("size", "0.005 0.005 0.005")  # 1cm x 1cm x 1cm (half-sizes)
                cube_geom.set("rgba", "0 1 0 0.9")  # Bright green for visibility
                cube_geom.set("group", "0")
                cube_geom.set("contype", "0")
                cube_geom.set("conaffinity", "0")
                
                print(f"Added 1cm reference cube beside final arm position at: {cube_pos}")
                print(f"Cube is {cube_offset*100:.1f}cm away from final pinch position")

            # Save the modified scene with trajectory spheres and reference cube
            trajectory_scene_path = scene_path.replace('.xml', '_trajectory.xml')
            trajectory_tree = ET.ElementTree(scene_root)
            trajectory_tree.write(trajectory_scene_path, encoding="utf-8", xml_declaration=True)

            # Reload model with trajectory spheres and cube
            trajectory_model = mujoco.MjModel.from_xml_path(trajectory_scene_path)
            trajectory_data = mujoco.MjData(trajectory_model)

            # Re-find arm components (indices might change)
            traj_arm_joint_indices, _ = self.find_arm_indices(trajectory_model)

            # Set arm to final position from original simulation
            final_q = np.array([data.qpos[idx] for idx in arm_joint_indices])
            for j, joint_idx in enumerate(traj_arm_joint_indices):
                if j < len(final_q):
                    trajectory_data.qpos[joint_idx] = final_q[j]

            # Let physics settle
            mujoco.mj_forward(trajectory_model, trajectory_data)
            self.settle_objects(trajectory_model, trajectory_data, num_steps=100)
            
            print(f"✅ Trajectory visualization ready with {len(action_data) + 1} spheres")
            print("Red spheres show the complete trajectory path from timestep 0 to final timestep")
            
            # Launch viewer for trajectory visualization if in validation mode
            if self.validation:
                try:
                    print("Launching trajectory visualization viewer...")
                    trajectory_viewer = mujoco.viewer.launch_passive(trajectory_model, trajectory_data)
                    print("🔴 RED SPHERES show the complete arm trajectory")
                    print("Press any key to continue...")
                    input()  # Wait for user input
                    trajectory_viewer.close()
                except Exception as e:
                    print(f"Could not launch trajectory viewer: {e}")
                    print("Continuing without trajectory visualization...")
            
            return True
            
        except Exception as e:
            print(f"Failed to create trajectory visualization: {e}")
            return False

    def generate_episode(self, episode_id, distance_threshold, capture_images, use_timestep_alignment, timestep_alignment_type, visualize_trajectory):
        """Generate one episode with multiple simulations"""
        episode_dir = os.path.join(self.output_dir, f"episode_{episode_id}")
        os.makedirs(episode_dir, exist_ok=True)
        
        episode_successful = True
        
        try:
            # Generate scene for this simulation
            scene_path, scene_info = self.generate_scene(f"{episode_id}")
            
            # Process simulation with IK trajectory
            sim_success = self.process_simulation(
                scene_path, scene_info, episode_id, 
                distance_threshold, capture_images, use_timestep_alignment, timestep_alignment_type, visualize_trajectory
            )
            
            if not sim_success:
                episode_successful = False
                
        except Exception as e:
            print(f"Failed simulation episode {episode_id}: {e}")
            episode_successful = False
        
        return episode_successful
    
    def process_simulation(self, scene_path, scene_info, episode_id, 
                  distance_threshold, capture_images, use_timestep_alignment, timestep_alignment_type, visualize_trajectory):
        """Process one simulation with IK trajectory generation"""
        try:
            # Load model
            model = mujoco.MjModel.from_xml_path(scene_path)
            data = mujoco.MjData(model)
            
            # Handle keyframes
            if model.nkey > 0:
                model.key_time = np.array([])
                model.key_qpos = np.array([]).reshape(0, model.nq)
                model.key_qvel = np.array([]).reshape(0, model.nv) 
                model.key_act = np.array([]).reshape(0, model.na)
                model.nkey = 0
            
            # Find arm components
            arm_joint_indices, arm_actuator_indices = self.find_arm_indices(model)
            ee_site_id = self.get_end_effector_site_id(model)
            ee_body_id = self.get_end_effector_body_id(model)
            use_site = ee_site_id is not None
            
            # Initialize simple RRT system
            self.rrt_system = SimpleEffectiveRRT(model, data)
            self.rrt_system.update_obstacles(scene_info)
            
            # Initialize collision-aware IK solver
            basic_ik_solver = GaussNewtonIK(model, step_size=0.5, tol=0.01, max_iter=1000)
            ik_solver = CollisionAwareIK(model, self.rrt_system, step_size=0.5, tol=0.01, max_iter=1000)

            viewer_instance = None
            if self.validation:
                try:
                    viewer_instance = mujoco.viewer.launch_passive(model, data)
                    print("Live viewer launched for validation")
                    # Give viewer time to initialize
                    time.sleep(1.0)
                except:
                    print("Could not launch viewer, continuing without live display")
                    
            # Setup initial random arm pose
            self.setup_random_arm_pose(model, data, arm_joint_indices)
            self.settle_objects(model, data, num_steps=500)
            
            # Select query object
            query_object = self.select_query_object(scene_info)
            query_pos = query_object['position']

            '''# Estimate object height to move end effector to above it
            object_height = 0.05  # Default 5cm height
            # Try to get actual object dimensions
            if 'type' in query_object:
                if query_object['type'] == 'mesh':
                    # For mesh objects, use scale factor
                    scale = query_object.get('scale', 1.0)
                    if 'file' in query_object:
                        obj_file = query_object['file']
                        object_height = scale * 8  # Estimate for mesh objects
            # Position end effector well above the top of the object
            ee_height_offset = max(0.06, object_height * 0.6)  # At least 6cm above or 60% of object height
            query_pos[2] = query_pos[2] + (object_height / 2) + ee_height_offset'''
            
            # Create episode directory structure
            episode_dir = os.path.join(self.output_dir, f"episode_{episode_id}")
            obs_dir = os.path.join(episode_dir, "observation_space")
            os.makedirs(obs_dir, exist_ok=True)
            
            # Convert query_pos to array here to avoid variable scope issues
            query_pos_array = np.array(query_pos)
            print(f"DEBUG QUERY POSITION:")
            print(f"  Original query_pos: {query_pos}")
            print(f"  Query_pos_array: {query_pos_array}")
            print(f"  Query object info: {query_object}")

            # Generate IK trajectory
            trajectory_success = self.generate_ik_trajectory(
                episode_id, scene_path, scene_info, model, data, arm_joint_indices, ee_body_id, ee_site_id, use_site,
                ik_solver, basic_ik_solver, query_pos_array, episode_dir, obs_dir, distance_threshold, capture_images, query_object, use_timestep_alignment, timestep_alignment_type, visualize_trajectory
            )

            if viewer_instance is not None:
                viewer_instance.close()
            
            return trajectory_success
            
        except Exception as e:
            print(f"Error processing simulation {episode_id}: {e}")
            return False
        
    def move_arm_to_target_with_obstacles(self, model, data, target_position, arm_joint_indices, ee_body_id, ik_solver):
        """Move arm using improved IK with proper target reaching"""
        print(f"Moving arm to target: {target_position}")
        
        # Get current position
        current_pos = data.body(ee_body_id).xpos.copy()
        initial_distance = np.linalg.norm(np.array(target_position) - current_pos)
        print(f"Initial distance to target: {initial_distance*100:.1f}cm")
        
        # First try: Direct IK without collision avoidance for speed
        try:
            # Use Jacobian-based IK directly
            target_pos = np.array(target_position)
            max_iterations = 50
            step_size = 0.1
            tolerance = 0.005  # 5mm tolerance
            
            best_q = np.array([data.qpos[idx] for idx in arm_joint_indices])
            best_distance = initial_distance
            
            for iteration in range(max_iterations):
                # Forward kinematics to get current end effector position
                current_ee_pos = data.body(ee_body_id).xpos.copy()
                
                # Calculate error
                position_error = target_pos - current_ee_pos
                error_magnitude = np.linalg.norm(position_error)
                
                if error_magnitude < tolerance:
                    print(f"IK converged in {iteration} iterations, error: {error_magnitude*1000:.1f}mm")
                    break
                
                # Calculate Jacobian for end effector position
                jacobian = np.zeros((3, len(arm_joint_indices)))
                delta = 0.001
                
                current_q = np.array([data.qpos[idx] for idx in arm_joint_indices])
                
                for j, joint_idx in enumerate(arm_joint_indices):
                    # Positive perturbation
                    data.qpos[joint_idx] += delta
                    mujoco.mj_forward(model, data)
                    pos_plus = data.body(ee_body_id).xpos.copy()
                    
                    # Negative perturbation
                    data.qpos[joint_idx] = current_q[j] - delta
                    mujoco.mj_forward(model, data)
                    pos_minus = data.body(ee_body_id).xpos.copy()
                    
                    # Finite difference
                    jacobian[:, j] = (pos_plus - pos_minus) / (2 * delta)
                    
                    # Restore original position
                    data.qpos[joint_idx] = current_q[j]
                
                # Restore state for next iteration
                mujoco.mj_forward(model, data)
                
                # Solve for joint updates using damped least squares
                try:
                    JTJ = jacobian.T @ jacobian
                    damping = 0.01 * np.eye(JTJ.shape[0])
                    delta_q = jacobian.T @ np.linalg.solve(JTJ + damping, position_error)
                except:
                    delta_q = np.linalg.pinv(jacobian, rcond=1e-4) @ position_error
                
                # Limit step size
                delta_q_norm = np.linalg.norm(delta_q)
                if delta_q_norm > step_size:
                    delta_q = delta_q * (step_size / delta_q_norm)
                
                # Update joints
                new_q = current_q + delta_q
                
                # Apply joint limits
                for j, joint_idx in enumerate(arm_joint_indices):
                    if joint_idx < len(model.jnt_range):
                        joint_range = model.jnt_range[joint_idx]
                        if not (np.isinf(joint_range[0]) or np.isinf(joint_range[1])):
                            new_q[j] = np.clip(new_q[j], joint_range[0], joint_range[1])
                
                # Apply new configuration
                for j, joint_idx in enumerate(arm_joint_indices):
                    data.qpos[joint_idx] = new_q[j]
                mujoco.mj_forward(model, data)
                
                # Check if this is better
                new_distance = np.linalg.norm(target_pos - data.body(ee_body_id).xpos)
                if new_distance < best_distance:
                    best_distance = new_distance
                    best_q = new_q.copy()
            
            # Apply best solution
            for j, joint_idx in enumerate(arm_joint_indices):
                data.qpos[joint_idx] = best_q[j]
            mujoco.mj_forward(model, data)
            
            final_pos = data.body(ee_body_id).xpos.copy()
            final_error = np.linalg.norm(target_pos - final_pos)
            
            print(f"Direct IK result: error = {final_error*1000:.1f}mm")
            print(f"Target: {target_pos}")
            print(f"Achieved: {final_pos}")
            print(f"Movement: {np.linalg.norm(final_pos - current_pos)*100:.1f}cm toward target")
            
            # Check for collisions in final position
            collisions = self.check_arm_collisions(model, data, arm_joint_indices)
            if not collisions and final_error < 0.02:  # Accept if within 2cm and no collisions
                return best_q, final_pos, final_error
            
            print("Direct IK has collisions or insufficient accuracy, trying collision-aware approach...")
            
        except Exception as e:
            print(f"Direct IK failed: {e}")
        
        # Fallback: Use collision-aware IK if direct approach failed
        try:
            result_q = ik_solver.solve_with_obstacles(data, target_pos, None, ee_body_id, 
                                                    arm_joint_indices, arm_joint_indices)
            
            if result_q is not None:
                for i, joint_idx in enumerate(arm_joint_indices):
                    data.qpos[joint_idx] = result_q[i]
                mujoco.mj_forward(model, data)
                
                final_pos = data.body(ee_body_id).xpos.copy()
                error = np.linalg.norm(target_pos - final_pos)
                
                print(f"Collision-aware IK result: error = {error*1000:.1f}mm")
                return result_q, final_pos, error
            
        except Exception as e:
            print(f"Collision-aware IK failed: {e}")
        
        print("All IK methods failed")
        return None, np.array([0, 0, 0]), float('inf')
    
    def get_current_end_effector_position(self, data, ee_body_id, ee_site_id, use_site):
        """Get consistent end effector position - ALWAYS use this method"""
        if use_site and ee_site_id is not None:
            return data.site_xpos[ee_site_id].copy()
        else:
            # Fallback to body position but warn user
            print("WARNING: Using end effector body position instead of pinch site")
            return data.body(ee_body_id).xpos.copy()

    # Replace all instances of position getting with calls to this function:
    # OLD CODE:
    # if use_site and ee_site_id is not None:
    #     current_pos = data.site_xpos[ee_site_id].copy()
    # else:
    #     current_pos = data.body(ee_body_id).xpos.copy()

    # NEW CODE:
    # current_pos = self.get_current_end_effector_position(data, ee_body_id, ee_site_id, use_site)

    # UPDATE THE IK SOLVER TO TARGET PINCH SITE
    def move_arm_to_target_with_obstacles_pinch_site(self, model, data, target_position, arm_joint_indices, ee_body_id, ee_site_id, use_site, ik_solver):
        """Move arm so that PINCH SITE reaches target, not end effector body"""
        print(f"Moving PINCH SITE to target: {target_position}")
        
        # Get current pinch site position
        current_pos = self.get_current_end_effector_position(data, ee_body_id, ee_site_id, use_site)
        initial_distance = np.linalg.norm(np.array(target_position) - current_pos)
        print(f"Initial PINCH SITE distance to target: {initial_distance*100:.1f}cm")
        
        target_pos = np.array(target_position)
        
        # PINCH SITE TARGETED IK - More precise than end effector body
        max_iterations = 100
        step_size = 0.08
        tolerance = 0.003  # 3mm tolerance for pinch site
        
        best_q = np.array([data.qpos[idx] for idx in arm_joint_indices])
        best_distance = initial_distance
        
        for iteration in range(max_iterations):
            # Get current pinch site position
            current_ee_pos = self.get_current_end_effector_position(data, ee_body_id, ee_site_id, use_site)
            
            # Calculate error
            position_error = target_pos - current_ee_pos
            error_magnitude = np.linalg.norm(position_error)
            
            if error_magnitude < tolerance:
                print(f"PINCH SITE IK converged in {iteration} iterations, error: {error_magnitude*1000:.1f}mm")
                break
            
            # Calculate Jacobian for PINCH SITE position
            jacobian = np.zeros((3, len(arm_joint_indices)))
            delta = 0.001
            current_q = np.array([data.qpos[idx] for idx in arm_joint_indices])
            
            for j, joint_idx in enumerate(arm_joint_indices):
                # Positive perturbation
                data.qpos[joint_idx] += delta
                mujoco.mj_forward(model, data)
                pos_plus = self.get_current_end_effector_position(data, ee_body_id, ee_site_id, use_site)
                
                # Negative perturbation
                data.qpos[joint_idx] = current_q[j] - delta
                mujoco.mj_forward(model, data)
                pos_minus = self.get_current_end_effector_position(data, ee_body_id, ee_site_id, use_site)
                
                # Finite difference for PINCH SITE
                jacobian[:, j] = (pos_plus - pos_minus) / (2 * delta)
                
                # Restore position
                data.qpos[joint_idx] = current_q[j]
            
            mujoco.mj_forward(model, data)
            
            # Solve using damped least squares
            try:
                JTJ = jacobian.T @ jacobian
                damping = 0.005 * np.eye(JTJ.shape[0])
                delta_q = jacobian.T @ np.linalg.solve(JTJ + damping, position_error)
            except:
                delta_q = np.linalg.pinv(jacobian, rcond=1e-4) @ position_error
            
            # Limit step size
            delta_q_norm = np.linalg.norm(delta_q)
            if delta_q_norm > step_size:
                delta_q = delta_q * (step_size / delta_q_norm)
            
            # Update joints
            new_q = current_q + delta_q
            
            # Apply joint limits
            for j, joint_idx in enumerate(arm_joint_indices):
                if joint_idx < len(model.jnt_range):
                    joint_range = model.jnt_range[joint_idx]
                    if not (np.isinf(joint_range[0]) or np.isinf(joint_range[1])):
                        new_q[j] = np.clip(new_q[j], joint_range[0], joint_range[1])
            
            # Apply new configuration
            for j, joint_idx in enumerate(arm_joint_indices):
                data.qpos[joint_idx] = new_q[j]
            mujoco.mj_forward(model, data)
            
            # Check improvement
            new_pos = self.get_current_end_effector_position(data, ee_body_id, ee_site_id, use_site)
            new_distance = np.linalg.norm(target_pos - new_pos)
            
            if new_distance < best_distance:
                best_distance = new_distance
                best_q = new_q.copy()
        
        # Apply best solution
        for j, joint_idx in enumerate(arm_joint_indices):
            data.qpos[joint_idx] = best_q[j]
        mujoco.mj_forward(model, data)
        
        final_pos = self.get_current_end_effector_position(data, ee_body_id, ee_site_id, use_site)
        final_error = np.linalg.norm(target_pos - final_pos)
        
        print(f"PINCH SITE IK result: error = {final_error*1000:.1f}mm")
        print(f"Target: {target_pos}")
        print(f"Achieved: {final_pos}")
        print(f"PINCH SITE moved: {np.linalg.norm(final_pos - current_pos)*100:.1f}cm toward target")
        
        return best_q, final_pos, final_error

    def generate_ik_trajectory(self, episode_id, scene_path, scene_info, model, data, arm_joint_indices, ee_body_id, ee_site_id, 
        use_site, ik_solver, basic_ik_solver, query_pos_array, episode_dir, obs_dir, 
        distance_threshold, capture_images, query_object, use_timestep_alignment, timestep_alignment_type, visualize_trajectory):
        """Generate adaptive collision-free IK trajectory with proper obstacle avoidance"""
        
        # Get base link position for coordinate conversion
        base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        base_pos = data.body(base_body_id).xpos.copy() if base_body_id >= 0 else [0, 0, 0]
        print(f"Base link position: {base_pos}")
        
        # Convert all positions to base-relative coordinates
        query_pos_relative = self.convert_to_base_relative_coords(query_pos_array, base_pos)
        objects_center = self.calculate_objects_center(scene_info)
        objects_center_relative = self.convert_to_base_relative_coords(objects_center, base_pos)
        
        video_frames = []
        
        # Add debug visuals and external camera
        scene_root = ET.parse(scene_path).getroot()
        self.setup_external_camera(scene_root, scene_info)
        
        debug_scene_path = scene_path.replace('.xml', '_debug.xml')
        debug_tree = ET.ElementTree(scene_root)
        debug_tree.write(debug_scene_path, encoding="utf-8", xml_declaration=True)
        
        # Reload with debug scene
        model = mujoco.MjModel.from_xml_path(debug_scene_path)
        data = mujoco.MjData(model)
        arm_joint_indices, _ = self.find_arm_indices(model)
        ee_body_id = self.get_end_effector_body_id(model)
        ee_site_id = self.get_end_effector_site_id(model)
        
        # Set random arm pose and settle objects
        self.setup_random_arm_pose(model, data, arm_joint_indices)
        self.settle_objects(model, data, num_steps=1000)
        
        # STEP 1: Camera alignment for obs_0
        print("STEP 1: Camera alignment for obs_0")
        obs_0_q = self.align_camera_for_observation(model, data, arm_joint_indices, ee_body_id, objects_center)
        
        # Take obs_0 image and external frame
        if capture_images:
            self.capture_observation_image(model, data, obs_dir, 0)
        external_frame = self.capture_external_video_frame(model, data)
        if external_frame is not None:
            video_frames.append(external_frame)
        
        # STEP 2: Camera alignment for query image
        print("STEP 2: Camera alignment for query image")
        self.align_camera_for_observation(model, data, arm_joint_indices, ee_body_id, query_pos_array)
        
        if capture_images:
            self.capture_query_image(model, data, episode_dir, query_object, np.array([data.qpos[idx] for idx in arm_joint_indices]))
        
        # STEP 3: Return to obs_0 position
        print("STEP 3: Returning to obs_0 position")
        for j, joint_idx in enumerate(arm_joint_indices):
            data.qpos[joint_idx] = obs_0_q[j]
        mujoco.mj_forward(model, data)
        
        # STEP 4: Adaptive trajectory planning and execution
        print("STEP 4: Adaptive trajectory planning with obstacle avoidance")
        
        # Calculate target position with proper object boundary consideration
        object_radius = self.calculate_object_boundary_radius(model, data, query_object)
        safety_margin = max(distance_threshold, 0.025)  # At least 2.5cm
        final_target_pos = self.calculate_safe_target_position(query_pos_array, object_radius, safety_margin)
        
        # Initialize trajectory variables
        num_timesteps = 10
        action_data = []
        
        # Get initial position (pinch site)
        if use_site and ee_site_id is not None:
            current_pos = data.site_xpos[ee_site_id].copy()
        else:
            current_pos = data.body(ee_body_id).xpos.copy()
            print("Warning: Using end effector body instead of pinch site")
        
        prev_pos = current_pos.copy()
        
        # Execute adaptive trajectory
        for timestep in range(1, num_timesteps + 1):
            print(f"\n=== ADAPTIVE TIMESTEP {timestep}/{num_timesteps} ===")
            
            # Calculate remaining distance and plan next waypoint with monotonic progress
            remaining_distance = np.linalg.norm(final_target_pos - current_pos)
            remaining_steps = num_timesteps - timestep + 1

            print(f"Planning timestep {timestep}: {remaining_distance*100:.1f}cm remaining over {remaining_steps} steps")

            # CRITICAL: Ensure monotonic progress toward target
            if timestep > 1:
                prev_distance = np.linalg.norm(final_target_pos - prev_pos)
                if remaining_distance > prev_distance + 0.005:  # Allow 5mm tolerance
                    print(f"WARNING: Distance increased from {prev_distance*100:.1f}cm to {remaining_distance*100:.1f}cm")
                    # Force correction toward target
                    correction_factor = 1.5  # Be more aggressive
                    direction = (final_target_pos - current_pos)
                    direction = direction / np.linalg.norm(direction)
                    step_size = min(remaining_distance * 0.8, 0.08)  # Take up to 80% of remaining distance
                    next_waypoint = current_pos + (direction * step_size * correction_factor)
                else:
                    # Normal progression - ensure steady approach
                    if remaining_steps > 1:
                        # Calculate step size to ensure even progression
                        ideal_step = remaining_distance / remaining_steps
                        min_step = 0.02  # 2cm minimum progress per step
                        max_step = min(0.08, remaining_distance * 0.6)  # Max 8cm or 60% remaining
                        step_size = max(min_step, min(ideal_step * 1.1, max_step))  # 10% more aggressive
                    else:
                        # Final step - go directly to target
                        step_size = remaining_distance
                    
                    # Calculate next waypoint with guaranteed progress toward target
                    direction = (final_target_pos - current_pos)
                    if np.linalg.norm(direction) > 0.001:
                        direction = direction / np.linalg.norm(direction)
                        next_waypoint = current_pos + (direction * step_size)
                    else:
                        next_waypoint = final_target_pos.copy()
            else:
                # First timestep - calculate initial step
                total_distance = np.linalg.norm(final_target_pos - current_pos)
                step_size = min(total_distance / num_timesteps * 1.2, 0.06)  # Slightly aggressive start
                direction = (final_target_pos - current_pos) / np.linalg.norm(final_target_pos - current_pos)
                next_waypoint = current_pos + (direction * step_size)

            # Ensure waypoint is above table height with safety margin
            table_pos = scene_info['config']['table_position']
            min_safe_height = table_pos[2] + 0.08  # 8cm above table (increased safety)
            if next_waypoint[2] < min_safe_height:
                print(f"Adjusting waypoint height from {next_waypoint[2]:.3f} to {min_safe_height:.3f}")
                next_waypoint[2] = min_safe_height

            # Validate waypoint is actually closer to target than current position
            waypoint_distance = np.linalg.norm(final_target_pos - next_waypoint)
            current_distance = np.linalg.norm(final_target_pos - current_pos)

            if waypoint_distance >= current_distance:
                print("WARNING: Waypoint is not closer to target, forcing direct approach")
                # Force direct approach with slight offset to avoid collision
                direction = (final_target_pos - current_pos)
                direction = direction / np.linalg.norm(direction)
                next_waypoint = current_pos + direction * min(step_size, current_distance * 0.9)

            print(f"Current position: {current_pos}")
            print(f"Next waypoint: {next_waypoint}")
            print(f"Step size: {step_size*100:.1f}cm")
            print(f"Waypoint will be {waypoint_distance*100:.1f}cm from target (vs current {current_distance*100:.1f}cm)")
            
            # Plan collision-free path to next waypoint using PINCH SITE targeted IK
            print("Moving PINCH SITE to waypoint...")
            success_q, achieved_pos, ik_error = self.move_arm_to_target_with_obstacles_pinch_site(
                model, data, next_waypoint, arm_joint_indices, ee_body_id, ee_site_id, use_site, ik_solver
            )

            success = (success_q is not None) and (ik_error < 0.02)  # Accept up to 2cm error

            if not success:
                print(f"ERROR: Could not reach waypoint at timestep {timestep} (error: {ik_error*100:.1f}cm)")
                
                # IMPROVED RECOVERY: Try intermediate positions with different strategies
                recovery_strategies = [
                    # Strategy 1: Lift approach
                    {
                        'pos': current_pos + (next_waypoint - current_pos) * 0.7,
                        'pos_adjust': [0, 0, 0.03],  # Lift 3cm
                        'name': 'Lifted intermediate'
                    },
                    # Strategy 2: Side approach  
                    {
                        'pos': current_pos + (next_waypoint - current_pos) * 0.6,
                        'pos_adjust': [0.02, 0, 0.01],  # Slight side offset
                        'name': 'Side offset'
                    },
                    # Strategy 3: Conservative step
                    {
                        'pos': current_pos + (next_waypoint - current_pos) * 0.4,
                        'pos_adjust': [0, 0, 0.02],  # Small step with lift
                        'name': 'Conservative step'
                    }
                ]
                
                for strategy in recovery_strategies:
                    recovery_pos = strategy['pos'] + np.array(strategy['pos_adjust'])
                    print(f"Trying {strategy['name']}: {recovery_pos}")
                    
                    recovery_q, recovery_achieved, recovery_error = self.move_arm_to_target_with_obstacles_pinch_site(
                        model, data, recovery_pos, arm_joint_indices, ee_body_id, ee_site_id, use_site, ik_solver
                    )
                    
                    if recovery_q is not None and recovery_error < 0.03:
                        print(f"Recovery successful with {strategy['name']}")
                        success = True
                        achieved_pos = recovery_achieved
                        break
                
                if not success:
                    print(f"ERROR: All recovery strategies failed at timestep {timestep}")
                    break

            # Verify we actually moved toward the target
            final_distance_to_target = np.linalg.norm(query_pos_array - achieved_pos)
            if timestep > 1:
                prev_distance_to_target = np.linalg.norm(query_pos_array - prev_pos)
                if final_distance_to_target > prev_distance_to_target + 0.01:
                    print(f"ERROR: Moved away from target! Distance: {prev_distance_to_target*100:.1f}cm -> {final_distance_to_target*100:.1f}cm")
                    # Force correction
                    correction_target = current_pos + (query_pos_array - current_pos) * 0.3  # Move 30% closer
                    print(f"Applying correction toward: {correction_target}")
                    
                    correction_q, correction_pos, correction_error = self.move_arm_to_target_with_obstacles_pinch_site(
                        model, data, correction_target, arm_joint_indices, ee_body_id, ee_site_id, use_site, ik_solver
                    )
                    
                    if correction_q is not None:
                        achieved_pos = correction_pos
                        print(f"Correction applied, new distance: {np.linalg.norm(query_pos_array - achieved_pos)*100:.1f}cm")
            
            # Apply end effector alignment (joints 5-6 only, not joint 7)
            if use_timestep_alignment and timestep_alignment_type == 'end_effector':
                print("Applying end effector alignment (joints 5-6 only)")
                self.align_end_effector_orientation_adaptive(
                    model, data, arm_joint_indices, ee_body_id, ee_site_id, use_site, 
                    query_pos_array, query_object
                )
            
            # Get final position after alignment (MUST be pinch site)
            if use_site and ee_site_id is not None:
                final_pos = data.site_xpos[ee_site_id].copy()
            else:
                final_pos = data.body(ee_body_id).xpos.copy()
                print("Warning: Using end effector body position")
            
            # Validate no collisions occurred
            collision_resolved, _ = self.check_and_resolve_collisions(
                model, data, arm_joint_indices, ee_body_id, query_pos_array, f"Timestep {timestep}", ee_site_id
            )
            
            if not collision_resolved:
                print(f"ERROR: Unresolved collision at timestep {timestep}")
                break
            
            # Check distance constraint for final timestep
            final_distance_to_target = np.linalg.norm(query_pos_array - final_pos)
            if timestep == num_timesteps and final_distance_to_target > distance_threshold:
                print(f"WARNING: Final distance {final_distance_to_target*100:.1f}cm exceeds threshold {distance_threshold*100:.1f}cm")
                # Attempt final correction
                self.attempt_final_distance_correction(
                    model, data, arm_joint_indices, ee_body_id, ee_site_id, 
                    query_pos_array, distance_threshold, ik_solver
                )
                # Update final position after correction
                if use_site and ee_site_id is not None:
                    final_pos = data.site_xpos[ee_site_id].copy()
                else:
                    final_pos = data.body(ee_body_id).xpos.copy()
            
            # Convert positions to base-relative coordinates for dataset
            final_pos_relative = self.convert_to_base_relative_coords(final_pos, base_pos)
            prev_pos_relative = self.convert_to_base_relative_coords(prev_pos, base_pos)
            
            # Calculate action (displacement)
            current_action = query_pos_relative - final_pos_relative
            prev_action = query_pos_relative - prev_pos_relative if timestep > 1 else current_action
            
            # Store action data
            action_row = {
                'dx_t': current_action[0],
                'dy_t': current_action[1], 
                'dz_t': current_action[2],
                'dx_t-1': prev_action[0],
                'dy_t-1': prev_action[1],
                'dz_t-1': prev_action[2],
                'x_t': final_pos_relative[0],
                'y_t': final_pos_relative[1],
                'z_t': final_pos_relative[2],
                'x_t-1': prev_pos_relative[0],
                'y_t-1': prev_pos_relative[1],
                'z_t-1': prev_pos_relative[2]
            }
            action_data.append(action_row)
            
            # Take images
            if capture_images:
                self.capture_observation_image(model, data, obs_dir, timestep)
            
            external_frame = self.capture_external_video_frame(model, data)
            if external_frame is not None:
                video_frames.append(external_frame)
            
            # Interactive validation if enabled
            if self.validation:
                distance_to_target = np.linalg.norm(query_pos_array - final_pos)
                result = self.validate_timestep_interactively(model, data, timestep, final_pos, query_pos_array, distance_to_target)
                if result is None or not result:
                    break
            
            # Update positions for next iteration  
            movement_distance = np.linalg.norm(final_pos - current_pos)
            prev_pos = current_pos.copy()  # Store previous position before updating
            current_pos = final_pos.copy()  # Update current position
            print(f"Timestep {timestep} completed: moved {movement_distance*100:.1f}cm, distance to target: {np.linalg.norm(query_pos_array - final_pos)*100:.1f}cm")
        
        # Save data and video
        self.save_action_space_data(action_data, episode_dir)
        
        if visualize_trajectory and len(action_data) > 0:
            self.visualize_trajectory_with_spheres(model, data, scene_info, action_data, scene_root, debug_scene_path, arm_joint_indices)
        
        # Save video
        if len(video_frames) > 0:
            video_path = os.path.join(episode_dir, f"arm_movement_episode_{episode_id}.mp4")
            try:
                import imageio.v3 as iio
                iio.imwrite(video_path, video_frames, fps=2)
                print(f"Video saved: {video_path}")
            except Exception as e:
                print(f"Video save failed: {e}")
        
        # Success criteria
        final_distance = np.linalg.norm(query_pos_relative - final_pos_relative) if len(action_data) > 0 else float('inf')
        success = (len(action_data) == num_timesteps) and (final_distance <= distance_threshold)
        
        print(f"Episode completion: {len(action_data)}/{num_timesteps} timesteps, final distance: {final_distance*100:.1f}cm")
        print(f"Success: {success}")
        
        return success

    def validate_timestep_interactively(self, model, data, timestep, current_pos, target_pos, distance_to_target):
        """Interactive validation for each timestep during live simulation"""
        
        # Update viewer to show current state
        try:
            # Small delay to ensure viewer updates
            time.sleep(0.1)
        except:
            pass
        
        print(f"\n--- Timestep {timestep} ---")
        print(f"Current position: [{current_pos[0]:.3f}, {current_pos[1]:.3f}, {current_pos[2]:.3f}]")
        print(f"Target position: [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]")
        print(f"Distance to target: {distance_to_target:.4f}")
        
        while True:
            user_input = input("Continue to next step? (y/yes to continue, n/no to stop, q/quit): ").lower().strip()
            if user_input in ['y', 'yes']:
                return True
            elif user_input in ['n', 'no']:
                print("Stopping current trajectory...")
                return False
            elif user_input in ['q', 'quit']:
                print("Quitting dataset generation...")
                return None
            else:
                print("Please enter 'y' for yes, 'n' for no, or 'q' to quit")

    def generate_dataset(self, num_episodes=50, distance_threshold=0.02, capture_images=True, 
                    use_timestep_alignment=False, timestep_alignment_type='camera', visualize_trajectory=False):
        """Generate episodic dataset with IK trajectories"""
        print(f"Generating {'validation' if self.validation else 'training'} dataset...")
        print(f"Episodes: {num_episodes}")
        
        successful_episodes = 0
        failed_episodes = 0
        
        for episode_id in range(num_episodes):
            try:
                episode_success = self.generate_episode(
                    episode_id, distance_threshold, capture_images, use_timestep_alignment, timestep_alignment_type, visualize_trajectory
                )
                
                if episode_success:
                    successful_episodes += 1
                else:
                    failed_episodes += 1
                    
                if (episode_id + 1) % 10 == 0:
                    print(f"Completed {episode_id + 1}/{num_episodes} episodes...")
                    
            except Exception as e:
                print(f"Failed to generate episode {episode_id}: {e}")
                failed_episodes += 1
        
        print(f"\nDataset generation complete!")
        print(f"Successful episodes: {successful_episodes}")
        print(f"Failed episodes: {failed_episodes}")

        return {'successful_episodes': successful_episodes, 'failed_episodes': failed_episodes}

# Done!

# Notes:
# - Doesn't matter fi the pinch site is farther or closer to the end effector. It's ok if its unknwon. try to set it where the gripper owuld get (12 cm). Then reduce the safte margin  --- THIS HAS BEEN DONE
# - fix the whole timestep part n the generate_ik_trjaectory thign because its doign some wierd stuff to limit it to 5 tiemsteps. i just want the arm to smoothly move properly to the target. 
# - icnrrease tiemsteps to 9 or 10 by makign the movement in each timestep smaller. try to use some proper motion planning to get there in within 9 or 10 steps exactly, even accoutning for any errors along the path (liek in real-time) or adjustments due ot ocllisisons or antyhing
# - also make sure that the wrist (joint 7) is not rotating for end effector alignment. remmeebr to use the pinch site for the alignment
# - randomize between 2 valeus of n for generaizability of object sizes. Do n = 3 and n = 2
# - do 50 episodes for validation