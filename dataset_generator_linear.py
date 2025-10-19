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
import time
import threading


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
        self.table_height = 0.45
        
        # Find arm bodies for collision checking - MATCH post-movement collision checking exactly
        self.arm_body_ids = []
        arm_keywords = ['base_link', 'shoulder', 'arm', 'link', 'wrist', 'bracelet', 'hand', 'finger', 'gripper']

        # First add specific arm link names
        arm_link_names = ['base_link', 'shoulder_link', 'half_arm_1_link', 'half_arm_2_link', 
                        'forearm_link', 'spherical_wrist_1_link', 'spherical_wrist_2_link', 'bracelet_link']

        for name in arm_link_names:
            try:
                body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
                if body_id >= 0:
                    self.arm_body_ids.append(body_id)
            except:
                continue

        # Also add any bodies with arm keywords (to match post-movement checking)
        for body_id in range(model.nbody):
            try:
                body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
                if body_name:
                    body_name_lower = body_name.lower()
                    if any(keyword in body_name_lower for keyword in arm_keywords):
                        if body_id not in self.arm_body_ids:  # Avoid duplicates
                            self.arm_body_ids.append(body_id)
            except:
                continue

        print(f"Found {len(self.arm_body_ids)} arm bodies for collision checking (matching post-movement detection)")
    
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

    def ensure_goal_above_table(self, goal_q, arm_joint_indices):
        """Ensure goal configuration keeps pinch site above table"""
        # Set goal configuration and check resulting pinch site position
        original_q = np.array([self.data.qpos[idx] for idx in arm_joint_indices])
        
        for i, joint_idx in enumerate(arm_joint_indices):
            if i < len(goal_q):
                self.data.qpos[joint_idx] = goal_q[i]
        mujoco.mj_forward(self.model, self.data)
        
        pinch_pos = self.get_pinch_site_position(self.data)
        
        # If pinch site is below table, adjust goal upward
        if pinch_pos[2] < self.table_height:  
            print(f"Goal puts pinch site at {pinch_pos[2]:.3f}, adjusting upward...")
            
            # Try lifting joints 1 and 3 (shoulder and elbow typically)
            adjusted_goal = np.array(goal_q)
            if len(adjusted_goal) > 1:
                adjusted_goal[0] += 0.1  # Lift shoulder
            if len(adjusted_goal) > 3:
                adjusted_goal[2] -= 0.1  # Adjust elbow
                
            # Test the adjustment
            for i, joint_idx in enumerate(arm_joint_indices):
                if i < len(adjusted_goal):
                    self.data.qpos[joint_idx] = adjusted_goal[i]
            mujoco.mj_forward(self.model, self.data)
            
            new_pinch_pos = self.get_pinch_site_position(self.data)
            if new_pinch_pos[2] >= self.table_height:
                print(f"Adjusted goal: pinch site now at {new_pinch_pos[2]:.3f}")
                # Restore original and return adjusted
                for i, joint_idx in enumerate(arm_joint_indices):
                    self.data.qpos[joint_idx] = original_q[i]
                mujoco.mj_forward(self.model, self.data)
                return adjusted_goal
        
        # Restore original configuration
        for i, joint_idx in enumerate(arm_joint_indices):
            self.data.qpos[joint_idx] = original_q[i]
        mujoco.mj_forward(self.model, self.data)
        
        return goal_q
    
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
        """MuJoCo-consistent collision checking using actual contact detection"""
        # Store original configuration
        original_q = np.array([self.data.qpos[idx] for idx in arm_joint_indices])
        
        # Set test configuration
        for i, joint_idx in enumerate(arm_joint_indices):
            if i < len(joint_config):
                self.data.qpos[joint_idx] = joint_config[i]
        
        # Forward kinematics
        mujoco.mj_forward(self.model, self.data)
        
        collision_free = True
        
        # Check table collision (keep this simple check)
        for arm_body_id in self.arm_body_ids[-4:]:  # Only check end links
            arm_pos = self.data.body(arm_body_id).xpos
            if arm_pos[2] < self.table_height:
                collision_free = False
                break
        
        # CRITICAL FIX: Use MuJoCo contact detection instead of distance checking
        if collision_free:
            # Get arm body IDs for contact checking
            arm_body_ids_set = set(self.arm_body_ids)
            
            # Check for contacts using MuJoCo's built-in collision detection
            for i in range(self.data.ncon):
                contact = self.data.contact[i]
                geom1_id = contact.geom1
                geom2_id = contact.geom2
                
                # Get body IDs for these geoms
                body1_id = self.model.geom_bodyid[geom1_id]
                body2_id = self.model.geom_bodyid[geom2_id]
                
                # Check if this is an arm-object collision
                arm_involved = False
                object_involved = False
                
                if body1_id in arm_body_ids_set:
                    arm_involved = True
                    # Check if body2 is an object
                    try:
                        body2_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body2_id)
                        if body2_name and ('object_' in body2_name or 'table' in body2_name):
                            object_involved = True
                    except:
                        pass
                elif body2_id in arm_body_ids_set:
                    arm_involved = True
                    # Check if body1 is an object
                    try:
                        body1_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body1_id)
                        if body1_name and ('object_' in body1_name or 'table' in body1_name):
                            object_involved = True
                    except:
                        pass
                
                # If we found an arm-object collision, this configuration is invalid
                if arm_involved and object_involved:
                    collision_free = False
                    break
        
        # Restore original configuration
        for i, joint_idx in enumerate(arm_joint_indices):
            self.data.qpos[joint_idx] = original_q[i]
        mujoco.mj_forward(self.model, self.data)
        
        return collision_free

    def planning(self, start, goal, arm_joint_indices, expand_dis=0.1, path_resolution=0.02, 
            goal_sample_rate=20, max_iter=2000):
        """Main RRT planning function with improved goal reaching"""
        
        joint_limits = self.get_joint_limits(arm_joint_indices)
        
        # Initialize with start node
        start_node = self.Node(start)
        node_list = [start_node]
        
        print(f"Starting RRT planning with {max_iter} iterations...")
        
        for iteration in range(max_iter):
            # Sample random node with higher goal bias for better convergence
            if np.random.randint(0, 100) > goal_sample_rate:
                rand_q = []
                for limit in joint_limits:
                    rand_q.append(np.random.uniform(limit[0], limit[1]))
                rnd_node = self.Node(rand_q)
            else:
                # Ensure goal is above table height before using it
                safe_goal = self.ensure_goal_above_table(goal, arm_joint_indices)
                rnd_node = self.Node(safe_goal)
            
            # Find nearest node
            nearest_ind = self.get_nearest_node_index(node_list, rnd_node)
            nearest_node = node_list[nearest_ind]
            
            # Steer towards random node
            new_node = self.steer(nearest_node, rnd_node, expand_dis, path_resolution, joint_limits)
            
            # Check if new node is collision-free
            if self.check_collision(new_node.q, arm_joint_indices):
                node_list.append(new_node)
                
                # CRITICAL FIX: Check if we reached the goal in CARTESIAN SPACE, not joint space
                if self.reached_goal_cartesian(new_node.q, goal, arm_joint_indices):
                    print(f"Goal reached at iteration {iteration}!")
                    return self.generate_final_path(len(node_list) - 1, node_list)
            
            if iteration % 200 == 0:
                print(f"RRT iteration {iteration}, nodes: {len(node_list)}")
        
        print("RRT could not find complete path to goal")
        return None  # Return None instead of partial path
    
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
    
    def reached_goal_cartesian(self, current_q, goal_q, arm_joint_indices, tolerance=0.01):  # Tighter tolerance
        """Check if current configuration reaches goal using pinch site position"""
        try:
            # Set current configuration
            for i, joint_idx in enumerate(arm_joint_indices):
                if i < len(current_q):
                    self.data.qpos[joint_idx] = current_q[i]
            mujoco.mj_forward(self.model, self.data)
            
            # ALWAYS use pinch site position
            current_pinch_pos = self.get_pinch_site_position(self.data)
            
            # Set goal configuration
            for i, joint_idx in enumerate(arm_joint_indices):
                if i < len(goal_q):
                    self.data.qpos[joint_idx] = goal_q[i]
            mujoco.mj_forward(self.model, self.data)
            
            # ALWAYS use pinch site position
            goal_pinch_pos = self.get_pinch_site_position(self.data)
            
            # Check Cartesian distance
            cartesian_distance = np.linalg.norm(current_pinch_pos - goal_pinch_pos)
            return cartesian_distance <= tolerance
            
        except Exception as e:
            print(f"Error in Cartesian goal check: {e}")
            return False
         
    def get_end_effector_site_id(self, model):
        """Find the end effector site ID for pinch_site"""
        try:
            site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "pinch_site")
            if site_id >= 0:
                return site_id
        except:
            pass
        
        print("Warning: pinch_site not found, using end effector body as fallback")
        return None

    def get_pinch_site_position(self, data):
        """Get pinch site position using site data"""
        site_id = self.get_end_effector_site_id(self.model)
        if site_id is not None:
            return data.site(site_id).xpos.copy()
        else:
            # Fallback to bracelet_link body
            try:
                bracelet_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "bracelet_link")
                return data.body(bracelet_id).xpos
            except:
                return data.body(self.model.nbody - 1).xpos

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

    def get_end_effector_site_id(self, model):
        """Find the end effector site ID for pinch_site"""
        try:
            site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "pinch_site")
            if site_id >= 0:
                return site_id
        except:
            pass
        
        print("Warning: pinch_site not found, using end effector body as fallback")
        return None

    def get_pinch_site_position(self, data):
        """Get pinch site position using site data"""
        site_id = self.get_end_effector_site_id(self.model)
        if site_id is not None:
            return data.site(site_id).xpos.copy()
        else:
            # Fallback to bracelet_link body
            try:
                bracelet_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "bracelet_link")
                return data.body(bracelet_id).xpos
            except:
                return data.body(self.model.nbody - 1).xpos
        
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
        """Calculate position error using pinch site ALWAYS"""
        pinch_site_id = self.get_end_effector_site_id(self.model)
        if pinch_site_id is not None:
            current_pos = data.site_xpos[pinch_site_id].copy()
        else:
            current_pos = data.body(body_id).xpos.copy()
        return goal - current_pos
        
    def solve(self, data, goal, init_q, body_id, arm_joint_indices, arm_actuator_indices, table_height=0.45):
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

            # MUCH TIGHTER TOLERANCE
            if error_norm < 0.001:  # 0.1cm instead of 1cm
                self.converged = True
                break

            # Check if goal is below table and adjust if needed
            if goal[2] < table_height:
                print(f"WARNING: Goal {goal} is below table height {table_height}, adjusting upward")
                original_z = goal[2]
                goal = goal.copy()
                goal[2] = table_height + 0.015  # Minimal lift: 1.5cm above table
                print(f"Adjusted goal from z={original_z:.3f} to z={goal[2]:.3f}")

            # Calculate jacobian at pinch site position
            pinch_site_id = self.get_end_effector_site_id(self.model)
            if pinch_site_id is not None:
                pinch_pos = data.site_xpos[pinch_site_id].copy()
                mujoco.mj_jacSite(self.model, data, self.jacp, self.jacr, pinch_site_id)
            else:
                # Fallback to body
                mujoco.mj_jac(self.model, data, self.jacp, self.jacr, goal, body_id)

            # Extract jacobian columns for arm joints only
            arm_jacp = self.jacp[:, arm_joint_indices]

            # Check for singularity
            if np.linalg.norm(arm_jacp) < 1e-8:
                print(f"Warning: Near-singular jacobian at iteration {i}")
                break

            # Gauss-Newton update with stronger regularization
            JTJ = arm_jacp.T @ arm_jacp
            reg = 1e-3 * np.eye(JTJ.shape[0])  # Stronger regularization

            try:
                if np.linalg.det(JTJ + reg) > 1e-6:
                    j_inv = np.linalg.inv(JTJ + reg) @ arm_jacp.T
                else:
                    j_inv = np.linalg.pinv(arm_jacp)
            except np.linalg.LinAlgError:
                j_inv = np.linalg.pinv(arm_jacp)

            delta_q = j_inv @ error

            # CRITICAL FIX: Much smaller step size for precision
            max_step = 0.1  # Reduced from 0.3
            delta_q_norm = np.linalg.norm(delta_q)
            if delta_q_norm > max_step:
                delta_q = delta_q * (max_step / delta_q_norm)

            # Update arm joint positions with smaller step
            current_q = np.array([data.qpos[idx] for idx in arm_joint_indices])
            new_q = current_q + 0.3 * delta_q  # Smaller step size

            # Apply joint limits
            new_q = self.check_joint_limits(new_q, arm_joint_indices)

            # Set new joint positions
            for j, joint_idx in enumerate(arm_joint_indices):
                data.qpos[joint_idx] = new_q[j]

            # Forward kinematics
            mujoco.mj_forward(self.model, data)
            
            # Check if pinch site would go below table
            current_pos = self.get_pinch_site_position(data)
            if current_pos[2] < table_height:
                print(f"WARNING: Pinch site at {current_pos} would go below table height {table_height}")
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
        
    def solve_with_obstacles(self, data, goal, init_q, body_id, arm_joint_indices, arm_actuator_indices, table_height=0.45):
        """Solve IK with RRT path planning - FIXED parameter handling"""
        
        print(f"Solving IK with improved RRT for goal: {goal}")
        
        # CRITICAL: Ensure all parameters are correct types
        goal = np.array(goal)  # Ensure goal is numpy array
        if init_q is None:
            init_q = np.zeros(len(arm_joint_indices))
        else:
            init_q = np.array(init_q)  # Ensure init_q is numpy array
        
        # Check if goal is valid and adjust if needed
        original_goal = goal.copy()
        if goal[2] < table_height:
            print(f"Goal {goal} is below table height, adjusting upward")
            goal = goal.copy()
            # Only lift minimally - just enough to clear table
            goal[2] = table_height + 0.015  # Lift goal 1.5cm above table (minimal adjustment)
            print(f"Adjusted goal to: {goal}")
            
            # If the adjustment is too large (>2cm vertical change), try alternative approach
            if abs(goal[2] - original_goal[2]) > 0.02:
                print(f"Large vertical adjustment needed, trying intermediate target approach...")
                # Create intermediate target that's reachable
                intermediate_goal = original_goal.copy()
                intermediate_goal[2] += 0.01  # Small lift
                # Move slightly away from table in XY to gain clearance
                table_center = np.array([0.5, 0.0])  # Assuming table center
                current_xy = intermediate_goal[:2]
                direction_away = current_xy - table_center
                if np.linalg.norm(direction_away) > 0:
                    direction_away = direction_away / np.linalg.norm(direction_away)
                    intermediate_goal[:2] += direction_away * 0.01  # Move 1cm away from table center
                goal = intermediate_goal
                print(f"Using intermediate goal: {goal}")
                
        # Try direct IK first
        try:
            direct_result = self.solve(data, goal, init_q, body_id, arm_joint_indices, arm_actuator_indices, table_height)
            
            if direct_result is not None:
                # Set the result configuration to check error
                for i, joint_idx in enumerate(arm_joint_indices):
                    if i < len(direct_result):
                        data.qpos[joint_idx] = direct_result[i]
                mujoco.mj_forward(self.model, data)

                # Check if direct solution is good enough using pinch site
                achieved_pos = self.get_pinch_site_position(data)
                final_error = np.linalg.norm(goal - achieved_pos)
                if final_error < 0.03 and self.rrt_system.check_collision(direct_result, arm_joint_indices):
                    print(f"Direct IK solution is good (error: {final_error:.4f}) and collision-free")
                    return direct_result
        except Exception as e:
            print(f"Direct IK failed: {e}")
        
        print("Direct IK failed or has collisions, using improved RRT...")
        
        # FIXED: Generate better IK candidates with proper error handling
        try:
            ik_goals = self.generate_better_ik_candidates(data, goal, body_id, arm_joint_indices, arm_actuator_indices, table_height)
            
            if not ik_goals:
                print("No valid IK solutions found")
                return self.fallback_solution(data, goal, arm_joint_indices, table_height)
            
            # Get current configuration
            start_config = np.array([data.qpos[idx] for idx in arm_joint_indices])
            
            # Try RRT to BEST candidate goal only
            best_goal = ik_goals[0]
            print(f"Trying RRT to best candidate goal")
            
            path = self.rrt_system.planning(
                start=start_config,
                goal=best_goal,
                arm_joint_indices=arm_joint_indices,
                expand_dis=0.1,
                path_resolution=0.02,
                goal_sample_rate=20,
                max_iter=500
            )
            
            if path and len(path) > 1:
                print(f"RRT found path with {len(path)} waypoints")
                return path[-1]
            else:
                print("RRT failed - returning best direct IK attempt")
                return ik_goals[0]
                
        except Exception as e:
            print(f"RRT planning failed: {e}")
            return self.fallback_solution(data, goal, arm_joint_indices, table_height)
        
    def solve_precise_targeting(self, data, target_pos, object_pos, target_distance, arm_joint_indices, arm_actuator_indices, table_height=0.45):
        """Solve IK for precise object targeting with multiple strategies"""
        
        print(f"PRECISE TARGETING: Moving to {target_pos} to be {target_distance}m from object at {object_pos}")
        
        # Strategy 1: Direct targeting
        result1 = self.solve_with_obstacles(data, target_pos, None, None, arm_joint_indices, arm_actuator_indices, table_height)
        if result1 is not None:
            # Verify distance to object
            for i, joint_idx in enumerate(arm_joint_indices):
                data.qpos[joint_idx] = result1[i]
            mujoco.mj_forward(self.model, data)
            
            achieved_pos = self.get_pinch_site_position(data)
            distance_to_object = np.linalg.norm(achieved_pos - object_pos)
            
            if abs(distance_to_object - target_distance) < 0.005:  # Within 5mm
                print(f"Strategy 1 SUCCESS: Distance to object = {distance_to_object:.3f}m")
                return result1
        
        # Strategy 2: Vector-based targeting - calculate position that's exactly target_distance from object
        current_pos = self.get_pinch_site_position(data)
        direction_to_target = target_pos - object_pos
        if np.linalg.norm(direction_to_target) > 0:
            direction_to_target = direction_to_target / np.linalg.norm(direction_to_target)
            precise_target = object_pos + direction_to_target * target_distance
            
            # Ensure it's above table
            precise_target[2] = max(precise_target[2], table_height + 0.01)
            
            print(f"Strategy 2: Calculated precise target at {precise_target}")
            result2 = self.solve_with_obstacles(data, precise_target, None, None, arm_joint_indices, arm_actuator_indices, table_height)
            
            if result2 is not None:
                for i, joint_idx in enumerate(arm_joint_indices):
                    data.qpos[joint_idx] = result2[i]
                mujoco.mj_forward(self.model, data)
                
                achieved_pos = self.get_pinch_site_position(data)
                distance_to_object = np.linalg.norm(achieved_pos - object_pos)
                
                print(f"Strategy 2: Distance to object = {distance_to_object:.3f}m (target: {target_distance:.3f}m)")
                if abs(distance_to_object - target_distance) < 0.01:  # Within 1cm
                    return result2
        
        # Strategy 3: Current position adjustment - small moves from current position
        current_q = np.array([data.qpos[idx] for idx in arm_joint_indices])
        
        # Try small adjustments in joint space
        for scale in [0.1, 0.2, 0.05]:  # Different adjustment magnitudes
            direction = target_pos - current_pos
            if np.linalg.norm(direction) > 0:
                # Convert to joint space adjustment (simplified)
                adjusted_q = current_q.copy()
                
                # Simple heuristic adjustments based on direction
                if direction[0] > 0.01:  # Move +X
                    adjusted_q[0] += scale  # Base rotation
                elif direction[0] < -0.01:  # Move -X
                    adjusted_q[0] -= scale
                    
                if direction[2] > 0.01:  # Move +Z (up)
                    if len(adjusted_q) > 1:
                        adjusted_q[1] -= scale  # Shoulder up
                    if len(adjusted_q) > 3:
                        adjusted_q[3] -= scale * 0.5  # Elbow adjust
                elif direction[2] < -0.01:  # Move -Z (down)
                    if len(adjusted_q) > 1:
                        adjusted_q[1] += scale  # Shoulder down
                        
                # Check this adjustment
                if self.rrt_system.check_collision(adjusted_q, arm_joint_indices):
                    for i, joint_idx in enumerate(arm_joint_indices):
                        data.qpos[joint_idx] = adjusted_q[i]
                    mujoco.mj_forward(self.model, data)
                    
                    achieved_pos = self.get_pinch_site_position(data)
                    distance_to_object = np.linalg.norm(achieved_pos - object_pos)
                    
                    if distance_to_object < target_distance + 0.01:  # Close enough
                        print(f"Strategy 3 SUCCESS with scale {scale}: Distance = {distance_to_object:.3f}m")
                        return adjusted_q
        
        print("All precise targeting strategies failed")
        return None

    def fallback_solution(self, data, goal, arm_joint_indices, table_height):
        """Generate a safe fallback solution when IK/RRT fails"""
        try:
            # Get current position
            current_q = np.array([data.qpos[idx] for idx in arm_joint_indices])
            
            # Move slightly toward goal but not all the way
            current_pinch = self.get_pinch_site_position(data)
            direction = goal - current_pinch
            distance = np.linalg.norm(direction)
            
            if distance > 0.05:  # Move 50% of the way
                partial_goal = current_pinch + 0.5 * direction
                partial_goal[2] = max(partial_goal[2], table_height + 0.01)  # Stay above table
                
                # Try simple IK to partial goal
                result = self.solve(data, partial_goal, current_q, None, arm_joint_indices, arm_joint_indices, table_height)
                if result is not None:
                    return result
            
            return current_q  # Return current position as last resort
            
        except Exception as e:
            print(f"Fallback solution failed: {e}")
            return np.array([data.qpos[idx] for idx in arm_joint_indices])
    
    def generate_better_ik_candidates(self, data, goal, body_id, arm_joint_indices, arm_actuator_indices, table_height):
        """Generate fewer but higher-quality IK solutions"""
        
        candidates = []
        
        # Focused initialization strategies based on arm kinematics
        init_strategies = [
            # Neutral pose
            np.zeros(len(arm_joint_indices)),
            # Elbow up configuration
            np.array([0.0, 0.5, 0.0, 1.2, 0.0, 0.0, 0.0])[:len(arm_joint_indices)],
            # Side approach
            np.array([0.8, 0.3, 0.0, 1.0, 0.0, 0.3, 0.0])[:len(arm_joint_indices)],
            # Alternative side
            np.array([-0.8, 0.3, 0.0, 1.0, 0.0, 0.3, 0.0])[:len(arm_joint_indices)],
        ]
        
        for init_q in init_strategies:
            try:
                result_q = self.solve(data, goal, init_q, body_id, arm_joint_indices, arm_actuator_indices, table_height)
                
                # Verify this solution reaches the goal using pinch site
                for j, joint_idx in enumerate(arm_joint_indices):
                    data.qpos[joint_idx] = result_q[j]
                mujoco.mj_forward(self.model, data)

                achieved_pos = self.get_pinch_site_position(data)
                error = np.linalg.norm(goal - achieved_pos)
                                
                if error < 0.05:  # Accept solutions within 5cm
                    candidates.append((result_q, error))
                    print(f"Generated IK candidate with error {error:.4f}")
                        
            except:
                continue
        
        # Sort by error and return best configurations only
        candidates.sort(key=lambda x: x[1])
        return [candidate[0] for candidate in candidates[:3]]  # Return top 3 candidates
    
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
        
        # Available objects in assets folder
        self.available_objects = [
            "apple.stl", 
            "banana.stl", 
            "bowl.stl", 
            "computer_mouse.stl",
            "cup.stl", 
            "minion.stl",
            "tissue_box.stl",
            "cap.stl",
            "glasses.stl"
        ]
        
        n = random.choice([2, 3])  # Randomize between 2 and 3 for object size generalizability

        # Object scaling factors
        self.object_scales = {
            "apple.stl": 0.12 / n, 
            "banana.stl": 0.001 / n,  
            "bowl.stl": 0.0007 / n,  
            "computer_mouse.stl": 0.0013 / n, 
            "cup.stl": 0.001 / n,   
            "minion.stl": 0.0012 / n, 
            "tissue_box.stl": 0.001 / n,
            "cap.stl": 0.002 / n,
            "glasses.stl": 0.0025 / n,
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
                #print(f"Found end effector site: 'pinch_site' (ID: {site_id})")
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

    def generate_scene(self, scene_id, config=None, required_object=None):
        """Generate a single scene with specified or random configuration"""
        print(f"Generating scene {scene_id}...")
        
        # Create base scene
        root = self.create_base_scene_xml()

        # Apply configuration or randomize
        if config is None:
            config = self.generate_random_config(required_object=required_object)
        
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

    def generate_random_config(self, required_object=None):
        """Generate a random scene configuration with dynamic target positioning"""
        existing_objects = self.check_objects_exist()
    
        if existing_objects:
            # Remove the required_object from the pool if it's there (to avoid duplicates)
            remaining_objects = [obj for obj in existing_objects if obj != required_object]

            # Decide total number of objects (3 or 4)
            num_objects = random.randint(3, 4)

            # If a required object is given, reduce number of remaining objects accordingly
            num_to_sample = max(0, num_objects - (1 if required_object else 0))
            selected_objects = random.sample(remaining_objects, min(num_to_sample, len(remaining_objects)))

            # Include the required object if specified
            if required_object:
                selected_objects.append(required_object)
            
            # Optional: shuffle so required_object isn't always last
            random.shuffle(selected_objects)
        else:
            # Fallback: fill with generic objects
            num_objects = random.randint(3, 4)
            selected_objects = ["geometric"] * num_objects
            # Include required_object if specified
            if required_object:
                selected_objects[0] = required_object  # ensure it's included
        
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
                'main': [random.uniform(0.5, 1.0), random.uniform(0.5, 1.0), random.uniform(0.5, 1.0)],
                'aux1': [random.uniform(0.2, 0.6), random.uniform(0.2, 0.6), random.uniform(0.2, 0.6)],
                'aux2': [random.uniform(0.15, 0.5), random.uniform(0.15, 0.5), random.uniform(0.15, 0.5)]
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

        return model, data
        
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
        model, data = self.freeze_objects_after_settling(model, data)
        
        print("Object settling and freezing complete")

        return model, data

    def add_objects_to_scene(self, root, worldbody, assets, config):
        """Add objects to the scene and return object information"""
        objects_info = []
        table_pos = config['table_position']
        
        # Define table bounds - MUCH MORE SPREAD OUT
        scale_factor = 0.85  # Use 85% of table area instead of 60%
        
        # More varied random shifts for better coverage
        shift_x = random.uniform(-0.15, 0.10)  # Continuous range instead of fixed options
        shift_y = random.uniform(-0.12, 0.12)  # More Y-axis variation

        # Calculate scaled and shifted bounds - WIDER AREA
        base_width = 0.30 * scale_factor  # Increased from 0.25
        base_height = 0.22 * scale_factor  # Increased from 0.15
        table_x_min = table_pos[0] - base_width + shift_x
        table_x_max = table_pos[0] + base_width + shift_x
        table_y_min = table_pos[1] - base_height + shift_y
        table_y_max = table_pos[1] + base_height + shift_y

        # Ensure bounds stay within table limits (keep same)
        table_x_min = max(table_x_min, table_pos[0] - 0.4)
        table_x_max = min(table_x_max, table_pos[0] + 0.4)
        table_y_min = max(table_y_min, table_pos[1] - 0.3)
        table_y_max = min(table_y_max, table_pos[1] + 0.3)
        table_z = table_pos[2] + 0.04  # Slightly above table surface
        
        # Keep track of placed positions to avoid overlap
        placed_positions = []
        min_distance = 0.10  # Reduced slightly to allow more spread while avoiding overlap
        
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

        return model, data
    
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
                                      arm_joint_indices, arm_joint_indices, table_height=0.45)
                
                # Get final pinch site position
                site_id = self.get_end_effector_site_id(model)
                final_pos = data.site(site_id).xpos.copy()
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
            final_pos = data.site(site_id).xpos.copy()
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
        """Calculate alignment by simulating where the pinch site would 'look' if it moved forward"""
        try:
            # Get pinch site position
            if use_site and ee_site_id is not None:
                pinch_pos = data.site_xpos[ee_site_id].copy()
            else:
                pinch_pos = data.body(ee_body_id).xpos.copy()
            
            # Get end effector body position
            ee_pos = data.body(ee_body_id).xpos.copy()
            
            # CRITICAL INSIGHT: The pinch site pointing direction is the vector FROM end effector TO pinch site
            # This represents the direction the gripper/pinch mechanism extends
            pinch_direction_vector = pinch_pos - ee_pos
            pinch_direction_magnitude = np.linalg.norm(pinch_direction_vector)
            
            if pinch_direction_magnitude < 0.001:
                # Pinch site is at same location as end effector, fall back to orientation matrix
                ee_rotation_matrix = data.body(ee_body_id).xmat.reshape(3, 3)
                pinch_pointing_direction = -ee_rotation_matrix[:, 2]  # Try -Z first
            else:
                # Use the actual geometric relationship
                pinch_pointing_direction = pinch_direction_vector / pinch_direction_magnitude
            
            # Vector from pinch site to query object (where we want to point)
            to_query = np.array(query_pos) - pinch_pos
            distance_to_query = np.linalg.norm(to_query)
            
            if distance_to_query < 0.01:
                return 1.0
            
            to_query_normalized = to_query / distance_to_query
            
            # Calculate alignment
            alignment = np.dot(pinch_pointing_direction, to_query_normalized)
            
            # Simulate where pinch site would point if moved forward 5cm
            simulated_target = pinch_pos + pinch_pointing_direction * 0.05
            
            # Calculate how close this simulated target is to the actual query
            distance_to_simulated = np.linalg.norm(simulated_target - np.array(query_pos))
            
            # Alignment score based on both dot product and geometric proximity
            geometric_score = max(0, 1.0 - distance_to_simulated / 0.1)  # Good if within 10cm
            
            # Only accept if pointing generally toward object (positive alignment)
            if alignment > 0:
                combined_score = (alignment * 0.7) + (geometric_score * 0.3)
            else:
                combined_score = 0.0
            
            # DEBUG: Print comprehensive alignment info
            print(f"DEBUG ALIGNMENT: pinch_pos={pinch_pos}, query_pos={query_pos}")
            print(f"  ee_pos: {ee_pos}")
            print(f"  pinch_direction_vector (ee->pinch): {pinch_direction_vector}")
            print(f"  pinch_pointing_direction: {pinch_pointing_direction}")
            print(f"  to_query_normalized: {to_query_normalized}")
            print(f"  dot product alignment: {alignment:.3f}")
            print(f"  simulated_target (pinch + 5cm forward): {simulated_target}")
            print(f"  distance_to_simulated: {distance_to_simulated*100:.1f}cm")
            print(f"  geometric_score: {geometric_score:.3f}")
            print(f"  combined_score: {combined_score:.3f}")
            
            return max(0.0, min(1.0, combined_score))
            
        except Exception as e:
            print(f"Error calculating pinch orientation alignment: {e}")
            return 0.0

    def precise_pinch_site_alignment(self, model, data, arm_joint_indices, ee_body_id, ee_site_id, use_site, query_pos, query_object):
        """Precise pinch site alignment with explicit direction reversal"""
        
        print(f"PRECISE PINCH SITE ALIGNMENT using joints 5&6 with direction reversal")
        
        # Get actual query position
        actual_query_pos = self.get_actual_object_position_in_sim(model, data, query_object)
        if actual_query_pos is not None:
            query_pos = actual_query_pos
            print(f"Using actual object position: {query_pos}")
        
        # Get current positions
        if use_site and ee_site_id is not None:
            initial_pinch_pos = data.site_xpos[ee_site_id].copy()
        else:
            initial_pinch_pos = data.body(ee_body_id).xpos.copy()
        
        # Distance constraints
        initial_distance_to_target = np.linalg.norm(np.array(query_pos) - initial_pinch_pos)
        max_distance_reduction = 0.01
        min_allowed_distance = initial_distance_to_target - max_distance_reduction
        max_allowed_distance = initial_distance_to_target + max_distance_reduction
        
        print(f"Initial pinch site position: {initial_pinch_pos}")
        print(f"Target object position: {query_pos}")
        print(f"Distance constraints: {min_allowed_distance*100:.1f}cm to {max_allowed_distance*100:.1f}cm")
        
        # Only use joints 5 and 6
        if len(arm_joint_indices) < 6:
            print("Error: Need at least 6 joints for pinch site alignment")
            return np.array([data.qpos[idx] for idx in arm_joint_indices])
        
        joint_5_idx = arm_joint_indices[4]
        joint_6_idx = arm_joint_indices[5]
        
        original_j5 = data.qpos[joint_5_idx]
        original_j6 = data.qpos[joint_6_idx]
        current_q = np.array([data.qpos[idx] for idx in arm_joint_indices])
        
        best_score = -1.0
        best_j5, best_j6 = original_j5, original_j6
        best_config_info = None
        
        # Strategy 1: Try configurations that flip the pointing direction
        print("Strategy 1: Testing direction-flipping configurations...")
        
        # These are common joint combinations that tend to flip end effector orientation
        flip_configurations = [
            (0, 0),           # Original
            (np.pi, 0),       # Flip joint 5 by 180°
            (0, np.pi),       # Flip joint 6 by 180°
            (np.pi, np.pi),   # Flip both by 180°
            (-np.pi, 0),      # Flip joint 5 by -180°
            (0, -np.pi),      # Flip joint 6 by -180°
            (np.pi/2, 0),     # 90° rotations
            (0, np.pi/2),
            (-np.pi/2, 0),
            (0, -np.pi/2),
            (np.pi/2, np.pi/2),
            (-np.pi/2, -np.pi/2),
        ]
        
        valid_configs = []
        
        for base_j5_offset, base_j6_offset in flip_configurations:
            # Around each base configuration, try small adjustments
            for j5_fine in np.linspace(-0.2, 0.2, 9):
                for j6_fine in np.linspace(-0.2, 0.2, 9):
                    test_j5 = original_j5 + base_j5_offset + j5_fine
                    test_j6 = original_j6 + base_j6_offset + j6_fine
                    
                    # Handle joint limits with wraparound for rotational joints
                    if joint_5_idx < len(model.jnt_range):
                        joint_range = model.jnt_range[joint_5_idx]
                        if not (np.isinf(joint_range[0]) or np.isinf(joint_range[1])):
                            # Wrap around if necessary
                            while test_j5 > joint_range[1]:
                                test_j5 -= 2*np.pi
                            while test_j5 < joint_range[0]:
                                test_j5 += 2*np.pi
                            test_j5 = np.clip(test_j5, joint_range[0], joint_range[1])
                    
                    if joint_6_idx < len(model.jnt_range):
                        joint_range = model.jnt_range[joint_6_idx]
                        if not (np.isinf(joint_range[0]) or np.isinf(joint_range[1])):
                            # Wrap around if necessary
                            while test_j6 > joint_range[1]:
                                test_j6 -= 2*np.pi
                            while test_j6 < joint_range[0]:
                                test_j6 += 2*np.pi
                            test_j6 = np.clip(test_j6, joint_range[0], joint_range[1])
                    
                    # Apply test configuration
                    data.qpos[joint_5_idx] = test_j5
                    data.qpos[joint_6_idx] = test_j6
                    mujoco.mj_forward(model, data)
                    
                    # Get test positions
                    if use_site and ee_site_id is not None:
                        test_pinch_pos = data.site_xpos[ee_site_id].copy()
                    else:
                        test_pinch_pos = data.body(ee_body_id).xpos.copy()
                    
                    test_ee_pos = data.body(ee_body_id).xpos.copy()
                    
                    # Check distance constraint
                    test_distance = np.linalg.norm(np.array(query_pos) - test_pinch_pos)
                    if test_distance < min_allowed_distance or test_distance > max_allowed_distance:
                        continue
                    
                    # Test multiple pointing directions from end effector orientation matrix
                    ee_rotation_matrix = data.body(ee_body_id).xmat.reshape(3, 3)
                    
                    possible_directions = [
                        ee_rotation_matrix[:, 0],   # +X
                        -ee_rotation_matrix[:, 0],  # -X
                        ee_rotation_matrix[:, 1],   # +Y
                        -ee_rotation_matrix[:, 1],  # -Y
                        ee_rotation_matrix[:, 2],   # +Z
                        -ee_rotation_matrix[:, 2],  # -Z
                    ]
                    
                    direction_names = ["+X", "-X", "+Y", "-Y", "+Z", "-Z"]
                    
                    # Also try geometric direction if different from orientation matrix
                    if np.linalg.norm(test_pinch_pos - test_ee_pos) > 0.001:
                        geometric_direction = (test_pinch_pos - test_ee_pos) / np.linalg.norm(test_pinch_pos - test_ee_pos)
                        possible_directions.extend([geometric_direction, -geometric_direction])
                        direction_names.extend(["geom+", "geom-"])
                    
                    # Find the best pointing direction for this configuration
                    best_direction_score = -1.0
                    best_direction = None
                    best_direction_name = ""
                    
                    to_query = np.array(query_pos) - test_pinch_pos
                    to_query_normalized = to_query / np.linalg.norm(to_query)
                    
                    for direction, name in zip(possible_directions, direction_names):
                        # Alignment score
                        alignment = np.dot(direction, to_query_normalized)
                        
                        if alignment > 0:  # Only consider directions pointing toward object
                            # Test forward simulation
                            forward_pos = test_pinch_pos + direction * 0.03  # 3cm forward
                            forward_distance = np.linalg.norm(forward_pos - np.array(query_pos))
                            current_distance = np.linalg.norm(test_pinch_pos - np.array(query_pos))
                            
                            # Bonus if forward movement gets closer
                            forward_bonus = max(0, (current_distance - forward_distance) / current_distance)
                            
                            combined_score = alignment * 0.7 + forward_bonus * 0.3
                            
                            if combined_score > best_direction_score:
                                best_direction_score = combined_score
                                best_direction = direction
                                best_direction_name = name
                    
                    if best_direction_score > 0:  # Found a valid direction
                        config_info = {
                            'j5': test_j5,
                            'j6': test_j6,
                            'score': best_direction_score,
                            'direction': best_direction,
                            'direction_name': best_direction_name,
                            'pinch_pos': test_pinch_pos,
                            'distance_to_target': test_distance,
                            'base_config': f"j5+{base_j5_offset:.2f}, j6+{base_j6_offset:.2f}"
                        }
                        
                        valid_configs.append(config_info)
                        
                        if best_direction_score > best_score:
                            best_score = best_direction_score
                            best_j5, best_j6 = test_j5, test_j6
                            best_config_info = config_info
        
        print(f"Strategy 1 results: {len(valid_configs)} valid configurations found")
        
        if best_config_info is None or best_score < 0.1:
            print("Strategy 1 failed, trying Strategy 2: Inverse kinematics approach...")
            
            # Strategy 2: Try to calculate what joint angles would point toward target
            target_direction = np.array(query_pos) - initial_pinch_pos
            target_direction_normalized = target_direction / np.linalg.norm(target_direction)
            
            # Try different assumptions about which axis should point toward target
            for axis_to_align in range(3):  # X, Y, Z axes
                for flip in [1, -1]:  # Positive or negative direction
                    
                    # Calculate desired end effector orientation
                    desired_direction = target_direction_normalized * flip
                    
                    # Try small joint adjustments to achieve this orientation
                    for j5_step in np.linspace(-0.4, 0.4, 21):
                        for j6_step in np.linspace(-0.4, 0.4, 21):
                            test_j5 = original_j5 + j5_step
                            test_j6 = original_j6 + j6_step
                            
                            # Apply joint limits
                            if joint_5_idx < len(model.jnt_range):
                                joint_range = model.jnt_range[joint_5_idx]
                                if not (np.isinf(joint_range[0]) or np.isinf(joint_range[1])):
                                    test_j5 = np.clip(test_j5, joint_range[0], joint_range[1])
                            
                            if joint_6_idx < len(model.jnt_range):
                                joint_range = model.jnt_range[joint_6_idx]
                                if not (np.isinf(joint_range[0]) or np.isinf(joint_range[1])):
                                    test_j6 = np.clip(test_j6, joint_range[0], joint_range[1])
                            
                            # Test configuration
                            data.qpos[joint_5_idx] = test_j5
                            data.qpos[joint_6_idx] = test_j6
                            mujoco.mj_forward(model, data)
                            
                            # Check distance constraint
                            if use_site and ee_site_id is not None:
                                test_pinch_pos = data.site_xpos[ee_site_id].copy()
                            else:
                                test_pinch_pos = data.body(ee_body_id).xpos.copy()
                            
                            test_distance = np.linalg.norm(np.array(query_pos) - test_pinch_pos)
                            if test_distance < min_allowed_distance or test_distance > max_allowed_distance:
                                continue
                            
                            # Get the actual axis direction
                            ee_rotation_matrix = data.body(ee_body_id).xmat.reshape(3, 3)
                            actual_axis_direction = ee_rotation_matrix[:, axis_to_align] * flip
                            
                            # Check alignment with target
                            to_target = np.array(query_pos) - test_pinch_pos
                            to_target_normalized = to_target / np.linalg.norm(to_target)
                            
                            alignment_score = np.dot(actual_axis_direction, to_target_normalized)
                            
                            if alignment_score > best_score:
                                best_score = alignment_score
                                best_j5, best_j6 = test_j5, test_j6
                                best_config_info = {
                                    'j5': test_j5,
                                    'j6': test_j6,
                                    'score': alignment_score,
                                    'direction': actual_axis_direction,
                                    'direction_name': f"{'+-'[flip<0]}{['X','Y','Z'][axis_to_align]}",
                                    'pinch_pos': test_pinch_pos,
                                    'distance_to_target': test_distance,
                                    'base_config': f"Strategy2-axis{axis_to_align}-flip{flip}"
                                }
            
            print(f"Strategy 2 results: Best score {best_score:.4f}")
        
        # Apply best configuration
        if best_config_info is not None:
            data.qpos[joint_5_idx] = best_j5
            data.qpos[joint_6_idx] = best_j6
            mujoco.mj_forward(model, data)
            
            print(f"Best configuration found:")
            print(f"  Base config: {best_config_info.get('base_config', 'unknown')}")
            print(f"  Direction: {best_config_info.get('direction_name', 'unknown')}")
            print(f"  Score: {best_score:.4f}")
            
            # Verify the result
            final_verification_passed = self.verify_pointing_with_simulation(model, data, ee_body_id, ee_site_id, use_site, query_pos)
            
        else:
            print("ERROR: No valid configuration found!")
            data.qpos[joint_5_idx] = original_j5
            data.qpos[joint_6_idx] = original_j6
            mujoco.mj_forward(model, data)
        
        # Final results
        if use_site and ee_site_id is not None:
            final_pinch_pos = data.site_xpos[ee_site_id].copy()
        else:
            final_pinch_pos = data.body(ee_body_id).xpos.copy()
        
        final_distance_to_target = np.linalg.norm(np.array(query_pos) - final_pinch_pos)
        movement_distance = np.linalg.norm(final_pinch_pos - initial_pinch_pos)
        
        print(f"PINCH SITE ALIGNMENT RESULTS:")
        print(f"  Initial pinch position: {initial_pinch_pos}")
        print(f"  Final pinch position: {final_pinch_pos}")
        print(f"  Target object position: {query_pos}")
        print(f"  Movement distance: {movement_distance*100:.1f}cm")
        print(f"  Final distance to target: {final_distance_to_target*100:.1f}cm")
        print(f"  Final alignment score: {best_score:.4f}")
        print(f"  Joint 5 change: {best_j5 - original_j5:.4f} rad")
        print(f"  Joint 6 change: {best_j6 - original_j6:.4f} rad")
        
        # Update and return configuration
        final_q = current_q.copy()
        final_q[4] = best_j5
        final_q[5] = best_j6
        
        return final_q

    def verify_pointing_with_simulation(self, model, data, ee_body_id, ee_site_id, use_site, query_pos):
        """Simple verification using forward simulation"""
        try:
            if use_site and ee_site_id is not None:
                pinch_pos = data.site_xpos[ee_site_id].copy()
            else:
                pinch_pos = data.body(ee_body_id).xpos.copy()
            
            ee_rotation_matrix = data.body(ee_body_id).xmat.reshape(3, 3)
            
            # Test all 6 axis directions
            directions = [
                (ee_rotation_matrix[:, 0], "+X"),
                (-ee_rotation_matrix[:, 0], "-X"),
                (ee_rotation_matrix[:, 1], "+Y"),
                (-ee_rotation_matrix[:, 1], "-Y"),
                (ee_rotation_matrix[:, 2], "+Z"),
                (-ee_rotation_matrix[:, 2], "-Z"),
            ]
            
            current_distance = np.linalg.norm(np.array(query_pos) - pinch_pos)
            
            best_direction = None
            best_score = -1
            
            for direction, name in directions:
                # Simulate 3cm forward movement
                forward_pos = pinch_pos + direction * 0.03
                forward_distance = np.linalg.norm(forward_pos - np.array(query_pos))
                
                # Score based on getting closer
                if forward_distance < current_distance:
                    score = (current_distance - forward_distance) / current_distance
                    if score > best_score:
                        best_score = score
                        best_direction = name
            
            if best_direction:
                print(f"VERIFICATION: Best pointing direction is {best_direction} (improvement: {best_score:.3f})")
                return True
            else:
                print(f"VERIFICATION: No direction gets closer to target")
                return False
                
        except Exception as e:
            print(f"Verification error: {e}")
            return False
    
    def verify_pinch_site_pointing_direction(self, model, data, ee_body_id, ee_site_id, use_site, query_pos):
        """Verify pointing direction using geometric simulation"""
        try:
            # Get positions
            if use_site and ee_site_id is not None:
                pinch_pos = data.site_xpos[ee_site_id].copy()
            else:
                pinch_pos = data.body(ee_body_id).xpos.copy()
            
            ee_pos = data.body(ee_body_id).xpos.copy()
            
            # Calculate actual pointing direction from geometry
            if np.linalg.norm(pinch_pos - ee_pos) > 0.001:
                pointing_direction = (pinch_pos - ee_pos) / np.linalg.norm(pinch_pos - ee_pos)
                direction_source = "geometric (ee->pinch)"
            else:
                ee_rotation_matrix = data.body(ee_body_id).xmat.reshape(3, 3)
                pointing_direction = -ee_rotation_matrix[:, 2]
                direction_source = "orientation matrix (-Z)"
            
            # Simulate forward movement
            distances_to_test = [0.01, 0.02, 0.03, 0.05]  # 1cm, 2cm, 3cm, 5cm forward
            getting_closer_count = 0
            
            current_distance = np.linalg.norm(np.array(query_pos) - pinch_pos)
            
            for test_distance in distances_to_test:
                forward_pos = pinch_pos + pointing_direction * test_distance
                forward_distance = np.linalg.norm(np.array(query_pos) - forward_pos)
                
                if forward_distance < current_distance:
                    getting_closer_count += 1
                
                print(f"  Forward {test_distance*100:.0f}cm: distance to target {forward_distance*100:.1f}cm (vs current {current_distance*100:.1f}cm)")
            
            is_pointing_toward = getting_closer_count >= 2  # At least half the tests should get closer
            
            # Vector alignment check
            to_query = np.array(query_pos) - pinch_pos
            distance = np.linalg.norm(to_query)
            
            if distance > 0.01:
                to_query_normalized = to_query / distance
                vector_alignment = np.dot(pointing_direction, to_query_normalized)
            else:
                vector_alignment = 1.0
            
            direction_status = "TOWARD" if is_pointing_toward else "AWAY FROM"
            
            print(f"PINCH SITE DIRECTION VERIFICATION:")
            print(f"  Direction source: {direction_source}")
            print(f"  Pointing direction: {pointing_direction}")
            print(f"  Forward movement tests: {getting_closer_count}/{len(distances_to_test)} get closer to target")
            print(f"  Vector alignment score: {vector_alignment:.3f}")
            print(f"  Overall assessment: pointing {direction_status} the query object")
            print(f"  Distance to object: {current_distance*100:.1f}cm")
            
            return is_pointing_toward, vector_alignment, direction_status
            
        except Exception as e:
            print(f"Error verifying pinch site direction: {e}")
            return False, 0.0, "ERROR"

    def select_query_object(self, scene_info, required_object=None):
        """Select one target object from scene and get its ACTUAL position"""
        if 'objects' in scene_info and scene_info['objects']:
            # Try to find object with the required file
            selected_obj = None
            if required_object:
                for obj in scene_info['objects']:
                    if obj.get("file") == required_object:
                        selected_obj = obj
                        break
            
            # If not found, pick randomly
            if not selected_obj:
                selected_obj = random.choice(scene_info['objects'])
            
            print(f"SELECTED QUERY OBJECT: {selected_obj}")
            return selected_obj
        else:
            # Fallback to table center
            table_pos = scene_info['config']['table_position']
            fallback_obj = {'position': [table_pos[0], table_pos[1], table_pos[2] + 0.05]}
            print(f"DEBUG FALLBACK QUERY OBJECT: {fallback_obj}")
            return fallback_obj
    
    def get_actual_object_position_in_sim(self, model, data, query_object):
        """Get the actual geometric center of the object in the simulation"""
        try:
            if 'name' not in query_object:
                print("Query object has no 'name'")
                return None
            
            obj_name = query_object['name']
            obj_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, obj_name)
            if obj_body_id < 0:
                print(f"Could not find object '{obj_name}' in simulation")
                return None
            
            # Collect all geom positions (world coordinates)
            geom_positions = []
            for geom_id in range(model.ngeom):
                if model.geom_bodyid[geom_id] == obj_body_id:
                    geom_type = model.geom_type[geom_id]
                    
                    # Use geom world position
                    geom_pos = data.geom_xpos[geom_id].copy()
                    
                    # For meshes, optionally compute mesh vertex center
                    if geom_type == mujoco.mjtGeom.mjGEOM_MESH:
                        mesh_id = model.geom_dataid[geom_id]  # or model.geom_meshid[geom_id]
                        start = model.mesh_vertadr[mesh_id]
                        nvert = model.mesh_vertnum[mesh_id]
                        verts = model.mesh_vert[start:start+nvert] * model.geom_size[geom_id]
                        
                        # Transform to world coordinates
                        geom_rot = data.geom_xmat[geom_id].reshape(3,3)
                        world_verts = (geom_rot @ verts.T).T + geom_pos
                        # Center of this geom
                        geom_center = np.mean(world_verts, axis=0)
                    else:
                        # For primitive geoms, just use geom world position
                        geom_center = geom_pos
                    
                    geom_positions.append(geom_center)
            
            if not geom_positions:
                print(f"No geoms found for object '{obj_name}'")
                return data.body(obj_body_id).xpos.copy()
            
            # Average all geom centers to get approximate object center
            actual_center = np.mean(np.array(geom_positions), axis=0)
            print(f"Actual center of '{obj_name}' in simulation: {actual_center}")
            return actual_center
        
        except Exception as e:
            print(f"Error getting actual object center: {e}")
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
        
    def add_target_object_marker(self, scene_root, target_position, query_object, marker_name="target_marker"):           # USEFUL FODR EBUGGING. DO NOT REMOVE
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

    def find_safe_pinch_target(self, model, data, pinch_site_pos, query_object_center, query_object, distance_threshold):
        """
        Calculate the ideal target position for the pinch site along the line to the query object.
        
        Parameters:
            model, data        : MuJoCo model/data
            pinch_site_pos     : np.array([x, y, z]) current pinch site position
            query_object_center: np.array([x, y, z]) object center position
            query_object       : dict with mesh info, including 'name', 'scale'
            distance_threshold : float, desired distance from object surface
        
        Returns:
            np.array([x, y, z]) target point for pinch site
        """
        print("Attempting to perform safe pinch target calculation...")
        pinch_site_pos = np.array(pinch_site_pos)
        obj_center = np.array(query_object_center)
        
        # Unit vector from pinch site to object center
        line_vec = obj_center - pinch_site_pos
        line_len = np.linalg.norm(line_vec)
        
        if line_len < 1e-6:
            raise ValueError("Pinch site and object center are too close")
        
        direction = line_vec / line_len
        
        # Start: initial guess is twice the distance threshold from object center
        target_pos = obj_center - 2 * distance_threshold * direction
        
        # Helper: check if a point is inside the object mesh
        def is_point_inside_object(pos):
            try:
                # Get mesh geom id
                geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, query_object['name'])
                if geom_id < 0:
                    return False
                
                # Convert point to geom local coordinates
                geom_pos = data.geom_xpos[geom_id]  # world position
                geom_scale = np.array(query_object.get('scale', 1.0))
                rel_pos = (pos - geom_pos) / geom_scale  # un-scale
                
                # Use MuJoCo built-in distance check: negative = inside
                signed_dist = mujoco.mj_distance(model, data, geom_id, rel_pos)
                return signed_dist < 0
            except Exception:
                # fallback if mj_distance unavailable
                return False
        
        # If initial guess is inside, move it outward along line
        step_size = distance_threshold / 2.0  # start step
        max_iters = 20
        iter_count = 0
        
        while is_point_inside_object(target_pos) and iter_count < max_iters:
            target_pos = target_pos - step_size * direction  # move toward pinch site
            step_size *= 0.5  # reduce step for finer adjustment
            iter_count += 1
        
        # Now move exactly distance_threshold away from object surface along line
        # Approximate: move from target_pos outward until distance_threshold away
        # If we had a signed distance function, we could do precise calculation
        if hasattr(model, 'mj_distance'):  # or a real SDF if available
            dist_to_surface = distance_threshold
        else:
            # fallback: just add threshold along line
            target_pos = target_pos + distance_threshold * direction
        
        return np.array(target_pos)

    def simple_camera_pointing(self, model, data, arm_joint_indices, ee_body_id, target_pos, target_name):
        """Precise camera pointing using ONLY joints 5 and 6 with mathematical optimization"""
        
        print(f"Pointing camera at {target_name}: {target_pos}")

        # Check current distance to target using PINCH SITE position
        ee_site_id = self.get_end_effector_site_id(model)
        if ee_site_id >= 0:
            current_pinch_pos = data.site_xpos[ee_site_id].copy()
            print(f"Using pinch site position for distance calculation    ---- GOOD")
        else:
            current_pinch_pos = data.body(ee_body_id).xpos.copy()  # Fallback
            
        distance_to_target = np.linalg.norm(np.array(target_pos) - current_pinch_pos)
        
        print(f"Camera will adjust to point towards target from pinch site perspective")
        
        # Get current joint configuration
        current_q = np.array([data.qpos[idx] for idx in arm_joint_indices])
        
        # Only use joints 5 and 6 (spherical_wrist_1 and spherical_wrist_2)
        if len(arm_joint_indices) < 6:
            print("Warning: Need at least 6 joints for camera pointing")
            return current_q
        
        joint_5_idx = arm_joint_indices[4]  # joint_5 (spherical_wrist_1)
        joint_6_idx = arm_joint_indices[5]  # joint_6 (spherical_wrist_2)
        
        # Store original joint values
        original_j5 = data.qpos[joint_5_idx]
        original_j6 = data.qpos[joint_6_idx]
        
        best_alignment = -1.0
        best_j5, best_j6 = original_j5, original_j6
        
        # Multi-resolution search for optimal camera pointing
        if distance_to_target < 0.05:  # If closer than 5cm, use micro-adjustments only
            print(f"Target very close ({distance_to_target*100:.1f}cm), using micro-adjustments")
            search_resolutions = [
                (np.linspace(-0.03, 0.03, 7), np.linspace(-0.03, 0.03, 7)),  # ±1.7 degrees
                (np.linspace(-0.01, 0.01, 5), np.linspace(-0.01, 0.01, 5))   # ±0.6 degrees
            ]
        elif distance_to_target < 0.1:  # If closer than 10cm, use small adjustments
            print(f"Target close ({distance_to_target*100:.1f}cm), using small adjustments")
            search_resolutions = [
                (np.linspace(-0.08, 0.08, 9), np.linspace(-0.08, 0.08, 9)),  # ±4.6 degrees
                (np.linspace(-0.03, 0.03, 7), np.linspace(-0.03, 0.03, 7))   # ±1.7 degrees
            ]
        else:
            search_resolutions = [
                (np.linspace(-0.1, 0.1, 11), np.linspace(-0.1, 0.1, 11)),  # Coarse: ±6 degrees
                (np.linspace(-0.05, 0.05, 11), np.linspace(-0.05, 0.05, 11)),  # Medium: ±3 degrees  
                (np.linspace(-0.02, 0.02, 9), np.linspace(-0.02, 0.02, 9))   # Fine: ±1 degree
            ]
        
        for resolution_stage, (j5_range, j6_range) in enumerate(search_resolutions):
            print(f"Camera alignment stage {resolution_stage + 1}: testing {len(j5_range)}x{len(j6_range)} combinations")
            stage_best = best_alignment
            
            for j5_delta in j5_range:
                for j6_delta in j6_range:
                    # Calculate test joint values
                    if resolution_stage == 0:
                        # First stage: search around original position
                        test_j5 = original_j5 + j5_delta
                        test_j6 = original_j6 + j6_delta
                    else:
                        # Later stages: search around best found position
                        test_j5 = best_j5 + j5_delta
                        test_j6 = best_j6 + j6_delta
                    
                    # Apply joint limits
                    if joint_5_idx < len(model.jnt_range):
                        joint_range = model.jnt_range[joint_5_idx]
                        if not (np.isinf(joint_range[0]) or np.isinf(joint_range[1])):
                            test_j5 = np.clip(test_j5, joint_range[0], joint_range[1])
                    
                    if joint_6_idx < len(model.jnt_range):
                        joint_range = model.jnt_range[joint_6_idx]
                        if not (np.isinf(joint_range[0]) or np.isinf(joint_range[1])):
                            test_j6 = np.clip(test_j6, joint_range[0], joint_range[1])
                    
                    # Set test configuration (only modify joints 5 and 6)
                    data.qpos[joint_5_idx] = test_j5
                    data.qpos[joint_6_idx] = test_j6
                    mujoco.mj_forward(model, data)
                    
                    # Get pinch site ID once outside the loop for efficiency
                    if 'ee_site_id_for_alignment' not in locals():
                        ee_site_id_for_alignment = self.get_end_effector_site_id(model)

                    alignment_score = self.calculate_precise_camera_alignment(model, data, ee_body_id, target_pos, ee_site_id_for_alignment)
                                        
                    if alignment_score > best_alignment:
                        # Validate that the movement is reasonable (not too large)
                        j5_change = abs(test_j5 - original_j5)
                        j6_change = abs(test_j6 - original_j6)
                        max_allowed_change = 1.0  # About 60 degrees maximum change
                        
                        if j5_change <= max_allowed_change and j6_change <= max_allowed_change:
                            best_alignment = alignment_score
                            best_j5, best_j6 = test_j5, test_j6
                        else:
                            print(f"  Rejected large movement: J5={j5_change:.3f}, J6={j6_change:.3f}")
            
            print(f"  Stage {resolution_stage + 1} best alignment: {best_alignment:.4f}")
            
            # Early termination if we achieve very good alignment
            if best_alignment > 0.8:  # Lower threshold for "good enough"
                print(f"  Good alignment achieved ({best_alignment:.3f}), stopping early")
                break
            
            # If no improvement in this stage, we're likely at optimum
            if best_alignment <= stage_best + 0.001:
                print(f"  No significant improvement, optimization converged")
                break
        
        # Set final best configuration
        data.qpos[joint_5_idx] = best_j5
        data.qpos[joint_6_idx] = best_j6
        mujoco.mj_forward(model, data)
        
        # Update return configuration
        final_q = current_q.copy()
        final_q[4] = best_j5  # joint_5
        final_q[5] = best_j6  # joint_6
        
        print(f"Final {target_name} camera alignment: {best_alignment:.4f}")
        print(f"Joint 5 adjustment: {best_j5 - original_j5:.4f} rad")
        print(f"Joint 6 adjustment: {best_j6 - original_j6:.4f} rad")
        
        return final_q

    def calculate_precise_camera_alignment(self, model, data, ee_body_id, target_pos, ee_site_id=None):     # can be used as fallback
        """Calculate how well the camera is aligned to the target (0-1 score)."""
        try:
            # 1. Try to get wrist camera
            try:
                camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist")
                if camera_id < 0:
                    raise ValueError("Camera not found")

                camera_pos = data.cam_xpos[camera_id].copy()
                camera_mat = data.cam_xmat[camera_id].reshape(3, 3)

                # Camera axes
                camera_forward = camera_mat[:, 2]  # Z axis

                # Vector to target
                target_vector = np.array(target_pos) - camera_pos

                # Ensure forward is pointing toward target
                if np.dot(camera_forward, target_vector) < 0:
                    camera_forward = -camera_forward

            except Exception:
                # 2. Fallback to end effector pose
                camera_pos = data.body(ee_body_id).xpos.copy()
                ee_rotation_matrix = data.body(ee_body_id).xmat.reshape(3, 3)
                camera_forward = -ee_rotation_matrix[:, 2]  # -Z forward in EE frame
                target_vector = np.array(target_pos) - camera_pos

                if np.dot(camera_forward, target_vector) < 0:
                    camera_forward = -camera_forward

            # Normalize vectors
            camera_direction = camera_forward / np.linalg.norm(camera_forward)
            target_distance = np.linalg.norm(target_vector)

            if target_distance < 1e-6:
                return 0.0  # target coincides with camera

            target_direction = target_vector / target_distance

            # Alignment score from dot product (cosine similarity)
            dot_product = np.dot(camera_direction, target_direction)
            alignment_score = (dot_product + 1.0) / 2.0  # map [-1,1] → [0,1]

            # Distance penalty
            if target_distance < 0.1:        # Too close
                distance_penalty = target_distance / 0.1
            elif target_distance > 0.6:      # Too far
                distance_penalty = 0.6 / target_distance
            else:
                distance_penalty = 1.0

            final_score = alignment_score * distance_penalty
            return max(0.0, min(1.0, final_score))

        except Exception as e:
            print(f"Error in camera alignment calculation: {e}")
            return 0.0

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
        
        '''# Additional check: Pinch site collision with objects/environment
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
                            min_clearance = 0.01  # 2cm minimum clearance for pinch site
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
                        continue'''           # DONT REMOVE THIS --- It was commented because it causes too many false positives in cluttered scenes
        
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
    
    def collision_backtrack_recovery(self, model, data, arm_joint_indices, ee_site_id, 
                               start_q, target_q, collision_step=0.5):
        """Backtrack along path to find collision-free position"""
        print("Attempting collision backtrack recovery...")
        
        # Try positions at 90%, 80%, 70%, 60%, 50% of the way to target
        for backtrack_ratio in [0.9, 0.8, 0.7, 0.6, 0.5]:
            test_q = start_q + (target_q - start_q) * backtrack_ratio
            
            # Apply test configuration
            for i, joint_idx in enumerate(arm_joint_indices):
                data.qpos[joint_idx] = test_q[i]
            mujoco.mj_forward(model, data)
            
            # Check collisions
            collisions = self.check_arm_collisions(model, data, arm_joint_indices, ee_site_id)
            
            if not collisions:
                print(f"✓ Collision-free position found at {backtrack_ratio:.1%} progress")
                return test_q, True
        
        print("✗ Could not find collision-free backtrack position")
        return start_q, False

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

    def generate_episode(self, episode_id, distance_threshold, capture_images, use_timestep_alignment, 
                         timestep_alignment_type, required_object=None):
        """Generate one episode with multiple simulations"""
        episode_dir = os.path.join(self.output_dir, f"episode_{episode_id}")
        os.makedirs(episode_dir, exist_ok=True)
        
        try:
            # Generate scene for this simulation
            scene_path, scene_info = self.generate_scene(f"{episode_id}", required_object=required_object)
            
            # Process simulation with IK trajectory
            ideal_distance_reached, sim_success = self.process_simulation(
                scene_path, scene_info, episode_id, 
                distance_threshold, capture_images, use_timestep_alignment, 
                timestep_alignment_type, viewer_instance=self.validation,
                required_object=required_object
            )
            
            return ideal_distance_reached, sim_success
                
        except Exception as e:
            print(f"Failed simulation episode {episode_id}: {e}")
            return False, False
    
    def process_simulation(self, scene_path, scene_info, episode_id, 
                  distance_threshold, capture_images, use_timestep_alignment, timestep_alignment_type, 
                  viewer_instance=False, required_object=None):
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
            basic_ik_solver = GaussNewtonIK(model, step_size=0.5, tol=0.005, max_iter=1000)  # Tighter tolerance
            ik_solver = CollisionAwareIK(model, self.rrt_system, step_size=0.5, tol=0.005, max_iter=1000)

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
            model, data = self.setup_random_arm_pose(model, data, arm_joint_indices)
            model, data = self.settle_objects(model, data, num_steps=500)
            
            # Select query object
            query_object = self.select_query_object(scene_info, required_object=required_object)
            query_pos = self.get_actual_object_position_in_sim(model, data, query_object)

            # Create episode directory structure
            episode_dir = os.path.join(self.output_dir, f"episode_{episode_id}")
            obs_dir = os.path.join(episode_dir, "observation_space")
            os.makedirs(obs_dir, exist_ok=True)
            
            # Convert query_pos to array here to avoid variable scope issues
            query_pos_array = np.array(query_pos)

            # Generate IK trajectory
            ideal_distance_reached, trajectory_success = self.generate_ik_trajectory(
                episode_id, scene_path, scene_info, model, data, arm_joint_indices, ee_body_id, ee_site_id, use_site,
                ik_solver, basic_ik_solver, query_pos_array, episode_dir, obs_dir, distance_threshold, capture_images, 
                query_object, use_timestep_alignment, timestep_alignment_type, live_viewer=viewer_instance
            )

            if viewer_instance is not None:
                viewer_instance.close()
            
            return ideal_distance_reached, trajectory_success
            
        except Exception as e:
            print(f"Error processing simulation {episode_id}: {e}")
            stop = input("Press Enter to continue...")
            return False, False
        
    def move_arm_to_target_with_obstacles(self, model, data, base_pos, target_position, arm_joint_indices, ee_body_id, ik_solver, ee_site_id=None):
        """Move arm using improved RRT with proper error checking"""
        print(f"Moving arm with improved RRT to: {target_position}")
        
        try:
            # Validate target is reasonable before RRT
            target_distance_from_base = np.linalg.norm(target_position - base_pos[:3])
            if target_distance_from_base > 0.85:  # Beyond reach
                print(f"Target too far from base ({target_distance_from_base:.3f}m), adjusting...")
                direction_to_target = (target_position - base_pos[:3]) / target_distance_from_base
                target_position = base_pos[:3] + direction_to_target * 0.8  # 80cm max reach

            current_q = np.array([data.qpos[idx] for idx in arm_joint_indices])
            try:
                if hasattr(ik_solver, 'solve_with_obstacles'):
                    result_q = ik_solver.solve_with_obstacles(data, target_position, current_q, ee_body_id, 
                                                            arm_joint_indices, arm_joint_indices)
                else:
                    result_q = ik_solver.solve(data, target_position, current_q, ee_body_id, 
                                            arm_joint_indices, arm_joint_indices, table_height=0.45)
            except Exception as e:
                print(f"IK solver failed: {e}")
                result_q = None

            if result_q is None:
                print("Improved RRT could not find solution")
                return None, np.array([0, 0, 0]), float('inf')
            
            # Set final configuration
            for i, joint_idx in enumerate(arm_joint_indices):
                if i < len(result_q):
                    data.qpos[joint_idx] = result_q[i]
            mujoco.mj_forward(model, data)
            
            # Verify collision-free
            collisions = self.check_arm_collisions(model, data, arm_joint_indices)
            if collisions:
                print(f"WARNING: Solution has {len(collisions)} collisions!")
            else:
                print("✓ Solution confirmed collision-free")
            
            final_pos = data.body(ee_site_id).xpos.copy()
            error = np.linalg.norm(np.array(target_position) - final_pos)
            
            print(f"Improved RRT result: error = {error:.4f}")
            
            # CRITICAL: Only accept solutions with reasonable error
            if error > 0.15:  # 15cm maximum acceptable error
                print(f"ERROR: Solution error {error:.4f} exceeds acceptable threshold")
                return None, final_pos, error
            
            return result_q, final_pos, error
            
        except Exception as e:
            print(f"Improved RRT failed: {e}")
            return None, np.array([0, 0, 0]), float('inf')

    def move_arm_with_precise_targeting(self, model, data, base_pos, target_waypoint, arm_joint_indices, ee_body_id, 
                               ik_solver, query_pos, target_distance_to_object, timestep, ee_site_id=None):
        """Enhanced movement with precise distance control and pinch site targeting"""
        
        print(f"=== PRECISE TARGETING for Timestep {timestep} ===")
        print(f"Target waypoint: {target_waypoint}")
        print(f"Required distance to object: {target_distance_to_object*100:.1f}cm")
        
        # ALWAYS get pinch site position for consistency
        def get_current_pinch_position():
            if ee_site_id is not None and ee_site_id >= 0:
                return data.site_xpos[ee_site_id].copy()
            else:
                site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "pinch_site")
                if site_id >= 0:
                    return data.site_xpos[site_id].copy()
                else:
                    print("ERROR: Cannot find pinch site!")
                    return data.body(ee_body_id).xpos.copy()

        current_pinch_pos = get_current_pinch_position()
        print(f"Current pinch site position: {current_pinch_pos}")
                
        current_distance_to_object = np.linalg.norm(query_pos - current_pinch_pos)
        
        # Calculate precise target position to achieve exact distance
        direction_to_object = (query_pos - current_pinch_pos)
        distance_to_object = np.linalg.norm(direction_to_object)
        
        if distance_to_object > 0.01:  # Valid direction
            direction_normalized = direction_to_object / distance_to_object
            # Calculate where pinch site should be to achieve target distance
            precise_target = query_pos - direction_normalized * target_distance_to_object
            print(f"Calculated precise target: {precise_target}")
            print(f"Expected distance after move: {target_distance_to_object*100:.1f}cm")
        else:
            precise_target = target_waypoint
            print(f"Using original waypoint (too close to calculate direction)")
        
        # Try direct IK with precise target
        try:
            current_q = np.array([data.qpos[idx] for idx in arm_joint_indices])
            # CRITICAL FIX: Ensure all parameters are correct types
            try:
                result_q = ik_solver.solve_with_obstacles(
                    data, 
                    np.array(precise_target),  # Ensure numpy array
                    current_q,  # Current configuration
                    ee_body_id, 
                    arm_joint_indices, 
                    arm_joint_indices, 
                    table_height=0.45
                )
            except Exception as e:
                print(f"RRT failed with error: {e}")
                result_q = None
            
            # Set configuration and verify
            for i, joint_idx in enumerate(arm_joint_indices):
                if i < len(result_q):
                    data.qpos[joint_idx] = result_q[i]
            mujoco.mj_forward(model, data)
            
            # Get achieved pinch site position
            if ee_site_id is not None:
                achieved_pinch_pos = data.site_xpos[ee_site_id].copy()
            else:
                achieved_pinch_pos = data.body(ee_body_id).xpos.copy()
            
            achieved_distance_to_object = np.linalg.norm(query_pos - achieved_pinch_pos)
            target_error = np.linalg.norm(precise_target - achieved_pinch_pos)
            distance_error = abs(achieved_distance_to_object - target_distance_to_object)
            
            print(f"IK ATTEMPT RESULTS:")
            print(f"  Target pinch position: {precise_target}")
            print(f"  Achieved pinch position: {achieved_pinch_pos}")
            print(f"  Query object position: {query_pos}")
            print(f"  Target error: {target_error*100:.1f}cm")
            print(f"  Target distance to object: {target_distance_to_object*100:.1f}cm")
            print(f"  Achieved distance to object: {achieved_distance_to_object*100:.1f}cm")
            print(f"  Distance error: {distance_error*100:.1f}cm")
            
            # Check collisions
            collisions = self.check_arm_collisions(model, data, arm_joint_indices, ee_site_id)
            
            if not collisions and target_error < 0.08 and distance_error < 0.04:  # 8cm position error, 4cm distance error
                print(f"✓ Direct IK successful with precise targeting")
                return result_q, achieved_pinch_pos, target_error
            elif collisions:
                print(f"✗ Direct IK has {len(collisions)} collisions")
            else:
                print(f"✗ Direct IK errors too large (pos: {target_error*100:.1f}cm, dist: {distance_error*100:.1f}cm)")
                
        except Exception as e:
            print(f"✗ Direct IK failed: {e}")
        
        # If direct IK failed, try RRT with obstacles
        print("Attempting RRT with obstacle avoidance...")
        try:
            # Validate target is reasonable before RRT
            target_distance_from_base = np.linalg.norm(precise_target - base_pos[:3])
            if target_distance_from_base > 0.85:  # Beyond reach
                print(f"Target too far from base ({target_distance_from_base:.3f}m), adjusting...")
                direction_to_target = (precise_target - base_pos[:3]) / target_distance_from_base
                precise_target = base_pos[:3] + direction_to_target * 0.8  # 80cm max reach
            
            result_q = ik_solver.solve_with_obstacles(data, precise_target, current_q, ee_body_id, 
                                                    arm_joint_indices, arm_joint_indices)
            
            if result_q is not None:
                # Set and verify RRT result
                for i, joint_idx in enumerate(arm_joint_indices):
                    if i < len(result_q):
                        data.qpos[joint_idx] = result_q[i]
                mujoco.mj_forward(model, data)
                
                # Get achieved position
                if ee_site_id is not None:
                    achieved_pinch_pos = data.site_xpos[ee_site_id].copy()
                else:
                    achieved_pinch_pos = data.body(ee_body_id).xpos.copy()
                
                achieved_distance_to_object = np.linalg.norm(query_pos - achieved_pinch_pos)
                target_error = np.linalg.norm(precise_target - achieved_pinch_pos)
                distance_error = abs(achieved_distance_to_object - target_distance_to_object)
                
                print(f"RRT ATTEMPT RESULTS:")
                print(f"  Target pinch position: {precise_target}")
                print(f"  Achieved pinch position: {achieved_pinch_pos}")
                print(f"  Target error: {target_error*100:.1f}cm")
                print(f"  Target distance to object: {target_distance_to_object*100:.1f}cm")
                print(f"  Achieved distance to object: {achieved_distance_to_object*100:.1f}cm")
                print(f"  Distance error: {distance_error*100:.1f}cm")
                
                # Verify no collisions
                collisions = self.check_arm_collisions(model, data, arm_joint_indices, ee_site_id)
                if not collisions:
                    print(f"✓ RRT successful with collision avoidance")
                    return result_q, achieved_pinch_pos, target_error
                else:
                    print(f"✗ RRT result has {len(collisions)} collisions")
            else:
                print(f"✗ RRT could not find solution")
                
        except Exception as e:
            print(f"✗ RRT failed: {e}")

        if result_q is None:
            print(f"✗ RRT could not find solution")
            # Try backtrack recovery as last resort
            start_q = np.array([data.qpos[idx] for idx in arm_joint_indices])
            recovery_q, recovery_success = self.collision_backtrack_recovery(
                model, data, arm_joint_indices, ee_site_id, start_q, start_q
            )
            if recovery_success:
                return recovery_q, current_pinch_pos, 0.1  # Accept with moderate error
        
        print(f"✗ All targeting methods failed for timestep {timestep}")
        return None, current_pinch_pos, float('inf')
        
    def generate_ik_trajectory(self, episode_id, scene_path, scene_info, model, data, arm_joint_indices, ee_body_id, ee_site_id, 
          use_site, ik_solver, basic_ik_solver, query_pos_array, episode_dir, obs_dir, 
          distance_threshold, capture_images, query_object, use_timestep_alignment, timestep_alignment_type, live_viewer=False):
        """Generate simplified IK trajectory with live MuJoCo viewer"""
        
        # Initialize MuJoCo viewer for live visualization (optional)
        viewer = None
        viewer_thread = None
        viewer_running = False

        if live_viewer:
            def start_viewer():
                """Start MuJoCo viewer in separate thread"""
                nonlocal viewer, viewer_running
                try:
                    viewer = mujoco.viewer.launch_passive(model, data)
                    viewer_running = True
                    print("🎥 MuJoCo viewer started - you should see the simulation window!")
                    
                    # Keep viewer alive
                    while viewer_running and viewer.is_running():
                        time.sleep(0.01)
                        
                except Exception as e:
                    print(f"Viewer error: {e}")
                    viewer_running = False
            
            def update_viewer():
                """Update viewer display"""
                if viewer and viewer_running and viewer.is_running():
                    try:
                        viewer.sync()
                        time.sleep(0.1)  # Small delay to see movement
                    except:
                        pass
            
            def close_viewer():
                """Close viewer safely"""
                nonlocal viewer_running
                viewer_running = False
                if viewer and viewer.is_running():
                    try:
                        viewer.close()
                    except:
                        pass
            
            # Start viewer in background thread
            viewer_thread = threading.Thread(target=start_viewer, daemon=True)
            viewer_thread.start()
            time.sleep(1)  # Give viewer time to initialize
            print("🎥 Live viewer enabled - simulation window should appear!")
        else:
            # Dummy functions when viewer is disabled
            def update_viewer():
                pass
            def close_viewer():
                pass
            print("📱 Live viewer disabled - running headless")
        
        try:
            def get_pinch_site_position():
                if use_site and ee_site_id is not None and ee_site_id >= 0:
                    return data.site_xpos[ee_site_id].copy()
                else:
                    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "pinch_site")
                    if site_id >= 0:
                        return data.site_xpos[site_id].copy()
                    else:
                        return data.body(ee_body_id).xpos.copy()
            
            # Get base link position for coordinate conversion
            base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
            base_pos = data.body(base_body_id).xpos.copy() if base_body_id >= 0 else [0, 0, 0]
            print(f"Base link position: {base_pos}")
            
            # Convert all positions to base-relative coordinates
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

            # Set arm to initial position
            model, data = self.setup_random_arm_pose(model, data, arm_joint_indices)
            model, data = self.settle_objects(model, data, num_steps=500)

            # For debugging, re-verify query object position
            print(f"Previous query pos: {query_pos_array}")
            query_pos_array = np.array(self.get_actual_object_position_in_sim(model, data, query_object))      # We need to update it because it gets shifted for some reason
            print(f"Verified query pos: {query_pos_array}")
            
            # Update viewer with new model/data
            if viewer and viewer_running:
                try:
                    viewer.close()
                    if live_viewer:
                        time.sleep(0.5)
                    viewer = mujoco.viewer.launch_passive(model, data)
                    print("🔄 Viewer updated with debug scene")
                except Exception as e:
                    print(f"Viewer update error: {e}")
            
            # STEP 1: Precise camera alignment for obs_0 using joints 5&6 only
            print("🎯 STEP 1: Precise camera alignment for obs_0 using joints 5&6")
            obs_0_q = self.simple_camera_pointing(model, data, arm_joint_indices, ee_body_id, objects_center, "obs_0")
            
            # Update physics and viewer
            mujoco.mj_forward(model, data)
            update_viewer()
            if live_viewer:
                time.sleep(3)  # Pause to see the movement
            
            # Take obs_0 image and external frame
            if capture_images:
                self.capture_observation_image(model, data, obs_dir, 0)
            external_frame = self.capture_external_video_frame(model, data)
            if external_frame is not None:
                video_frames.append(external_frame)

            obs_image_path = os.path.join(obs_dir, f"obs_0.png")
            user_validation = input(f"{obs_image_path} has been taken. Is it satisfactory? (y/n): ")
            if user_validation.lower() != 'y':
                print("User indicated obs_0 is not satisfactory. Aborting trajectory generation.")
                close_viewer()
                return False, False
            
            # STEP 2: Precise camera alignment for query_image using joints 5&6 only
            print("🎯 STEP 2: Precise camera alignment for query_image using joints 5&6")
            query_q = self.simple_camera_pointing(model, data, arm_joint_indices, ee_body_id, query_pos_array, "query_image")  # Use query pos because it lets us capture the whole object
            
            # Update physics and viewer
            mujoco.mj_forward(model, data)
            update_viewer()
            if live_viewer:
                time.sleep(0.5)
            
            if capture_images:
                self.capture_query_image(model, data, episode_dir, query_object, query_q)
            
            # STEP 3: Return to obs_0 position
            print("🔄 STEP 3: Returning to obs_0 position")
            for j, joint_idx in enumerate(arm_joint_indices):
                data.qpos[joint_idx] = obs_0_q[j]
            mujoco.mj_forward(model, data)
            update_viewer()
            if live_viewer:
                time.sleep(0.5)

            # STEP 4: Generate exactly waypoints with guaranteed reach
            print("🚀 STEP 4: Planning waypoint trajectory with guaranteed object reach")
            # Get current pinch site position
            if use_site and ee_site_id is not None:
                start_pos = data.site_xpos[ee_site_id].copy()
                print(f"Using pinch site '{model.site(ee_site_id).name}' for trajectory start position   ----- GOOD")
            else:
                start_pos = data.body(ee_body_id).xpos.copy()
                print(f"WARNING: Using end effector body '{model.body(ee_body_id).name}' for trajectory start position ----- LESS IDEAL")
            
            # Update the query pos again since it migth have shifted
            query_pos_array = np.array(self.get_actual_object_position_in_sim(model, data, query_object))     # Just in case the object moves somehow
            query_pos_relative = self.convert_to_base_relative_coords(query_pos_array, base_pos)
            # The query pos is simply the center of the object which is not realsitic for the arm to go towards (since that would mean going inside of the object), but we can adjust a find a point that is "just-outside" but on the way of the trajectory
            real_query_pos = self.find_safe_pinch_target(model, data, start_pos, query_pos_array, query_object, distance_threshold)
            distance_to_object = np.linalg.norm(real_query_pos - start_pos)
            real_query_pos_relative = self.convert_to_base_relative_coords(real_query_pos, base_pos)

            # Calculate FINAL target position (object center + safety margin)
            object_height = self.get_actual_object_height(model, data, query_object)
            print(f"Start position: {start_pos}")
            print(f"Final target: {real_query_pos}")
            print(f"Total distance: {np.linalg.norm(real_query_pos - start_pos):.4f}m")
            
            # Pre-validate that target is reachable
            distance_to_target = np.linalg.norm(real_query_pos - start_pos)
            if distance_to_target > 0.8:  # Gen3 max reach ~90cm
                print(f"WARNING: Target may be outside arm reach ({distance_to_target:.3f}m)")
            
            # Adaptive waypoint generation with RRT integration
            action_data = []
            prev_pos = start_pos.copy()

            # Record initial action (zero movement) --- initial position (timestep 0)
            current_action = real_query_pos_relative - self.convert_to_base_relative_coords(start_pos, base_pos)
            prev_action = real_query_pos_relative - self.convert_to_base_relative_coords(start_pos, base_pos)    # Keep it same (because its initial positon)
            final_pos_relative = self.convert_to_base_relative_coords(start_pos, base_pos)
            prev_pos_relative = self.convert_to_base_relative_coords(start_pos, base_pos)    # Keep it same (because its initial positon)
            action_row = {
                'dx_t': current_action[0],
                'dy_t': current_action[1], 
                'dz_t': current_action[2],
                #'dx_t-1': prev_action[0],
                #'dy_t-1': prev_action[1],
                #'dz_t-1': prev_action[2],
                #'x_t': final_pos_relative[0],
                #'y_t': final_pos_relative[1],
                #'z_t': final_pos_relative[2],
                #'x_t-1': prev_pos_relative[0],
                #'y_t-1': prev_pos_relative[1],
                #'z_t-1': prev_pos_relative[2]
            }
            action_data.append(action_row)
            
            # --- Non-linear Smoothing Formula (Ease-Out) ---
            # Formula: P(i) = 1 - (1 - i/N)^k
            # This curve starts fast and slows down dramatically near the end (approaching 1.0)
            num_timesteps = 9
            smoothing_factor_k = 3  # Higher values mean more extreme "easing". k=2 is quadratic, k=3 is cubic, etc.
            start_progress = 0.1
            progress_points = [start_progress + (1 - start_progress) * (1 - (1 - (i - 1) / (num_timesteps - 1))**1.5) for i in range(1, num_timesteps + 1)]
            print(f"Using progression: {[f'{p:.2f}' for p in progress_points]}")
            
            # Use larger steps to ensure visible movement - minimum 4cm per step
            total_distance = np.linalg.norm(real_query_pos - start_pos)
            ideal_step_size = total_distance / num_timesteps
            step_size = ideal_step_size
            print(f"Total distance to target: {total_distance*100:.1f}cm")
            print(f"Step size: {step_size*100:.1f}cm per timestep")
            
            # Calculate object boundaries for safe distance calculation
            object_radius = max(0.02, object_height * 0.3)  # Estimate object radius
            print(f"Object boundaries: height={object_height*100:.1f}cm, radius={object_radius*100:.1f}cm")
            
            # Start adaptive trajectory generation
            current_pos = start_pos.copy()
            for timestep in range(1, num_timesteps + 1):
                print(f"\n🎯 === ADAPTIVE TIMESTEP {timestep}/{num_timesteps} ===")

                # Update it
                #print(f"Previous query pos: {query_pos_array}")
                query_pos_array = np.array(self.get_actual_object_position_in_sim(model, data, query_object))     # Just in case the object moves somehow
                query_pos_relative = self.convert_to_base_relative_coords(query_pos_array, base_pos)
                #print(f"Verified query pos: {query_pos_array}") 

                # Recalculate linear path from CURRENT position
                progress = progress_points[timestep - 1]
                current_pos = get_pinch_site_position()

                # The query pos is simply the center of the object which is not realsitic for the arm to go towards (since that would mean going inside of the object), but we can adjust a find a point that is "just-outside" but on the way of the trajectory
                real_query_pos = self.find_safe_pinch_target(model, data, current_pos, query_pos_array, query_object, distance_threshold)
                distance_to_object = np.linalg.norm(real_query_pos - current_pos)
                real_query_pos_relative = self.convert_to_base_relative_coords(real_query_pos, base_pos)
                distance_to_object = np.linalg.norm(real_query_pos - current_pos) 

                target_distance_to_object = distance_threshold + (distance_to_object - distance_threshold) * (1.0 - progress)
                
                # CRITICAL: Ensure we don't get closer than object radius + safety margin
                #min_safe_distance = object_radius + distance_threshold   # Original
                min_safe_distance = distance_threshold
                if target_distance_to_object < min_safe_distance and progress < 1.0:
                    target_distance_to_object = min_safe_distance
                    print(f"Adjusted target distance for object boundaries: {target_distance_to_object*100:.1f}cm")
                
                # Calculate waypoint position on line from current position to object center
                if distance_to_object > min_safe_distance:
                    unit_vector_to_object = (real_query_pos - current_pos) / distance_to_object
                    waypoint_pos = real_query_pos - target_distance_to_object * unit_vector_to_object
                else:
                    waypoint_pos = current_pos.copy()  # Stay put if already close enough
                
                # ENSURE waypoint_pos is always numpy array
                waypoint_pos = np.array(waypoint_pos)
                print(f"Progress: {progress:.1%}")
                print(f"Current distance to object: {distance_to_object*100:.1f}cm")
                print(f"Target distance to object: {target_distance_to_object*100:.1f}cm")
                print(f"Adaptive waypoint: {waypoint_pos}")

                # Track all valid positions throughout this timestep
                timestep_positions = []

                # Add initial position (before any movement)
                initial_pos = get_pinch_site_position()
                initial_distance = np.linalg.norm(real_query_pos - initial_pos)
                timestep_positions.append({
                    'position': initial_pos.copy(),
                    'distance_to_object': initial_distance,
                    'joint_config': np.array([data.qpos[idx] for idx in arm_joint_indices]),
                    'source': 'initial'
                })
                
                # Enhanced movement with precise distance control
                try:
                    result_q, achieved_pos, error = self.move_arm_with_precise_targeting(
                        model, data, base_pos, waypoint_pos, arm_joint_indices, ee_body_id, ik_solver, 
                        real_query_pos, target_distance_to_object, timestep, ee_site_id
                    )

                    # Track this position as a candidate
                    if result_q is not None:
                        post_ik_pos = get_pinch_site_position()
                        post_ik_distance = np.linalg.norm(real_query_pos - post_ik_pos)
                        timestep_positions.append({
                            'position': post_ik_pos.copy(),
                            'distance_to_object': post_ik_distance,
                            'joint_config': np.array([data.qpos[idx] for idx in arm_joint_indices]),
                            'source': 'post_ik'
                        })
                        print(f"Tracked post-IK position: {post_ik_distance*100:.1f}cm from object")
                    
                        # Update physics and viewer after movement
                        mujoco.mj_forward(model, data)
                        update_viewer()
                        if live_viewer:
                            time.sleep(0.8)  # Longer pause to see each movement clearly
                    
                except Exception as e:
                    print(f"ERROR in move_arm_with_precise_targeting: {e}")
                    result_q = None
                    achieved_pos = current_pos
                    error = float('inf')
                
                if result_q is None:
                    print(f"🚨 CRITICAL: Could not reach waypoint {timestep} - attempting emergency recovery")
                    
                    # Emergency recovery: minimal movement toward target
                    try:
                        emergency_target = current_pos + 0.02 * (waypoint_pos - current_pos) / np.linalg.norm(waypoint_pos - current_pos)
                        emergency_target[2] = max(emergency_target[2], 0.45 + 0.01)  # Stay above table
                        
                        current_q = np.array([data.qpos[idx] for idx in arm_joint_indices])
                        result_q = basic_ik_solver.solve(data, emergency_target, current_q, ee_body_id, 
                                                    arm_joint_indices, arm_joint_indices, table_height=0.45)
                        
                        if result_q is not None:
                            achieved_pos = get_pinch_site_position()
                            emergency_distance = np.linalg.norm(real_query_pos - achieved_pos)
                            print(f"Emergency recovery successful: moved {emergency_distance*100:.1f}cm")

                            timestep_positions.append({
                                'position': achieved_pos.copy(),
                                'distance_to_object':  emergency_distance,
                                'joint_config': np.array([data.qpos[idx] for idx in arm_joint_indices]),
                                'source': 'emergency'
                            })
                            print(f"Tracked emergency position: {emergency_distance*100:.1f}cm from object")

                            # Update physics and viewer after emergency recovery
                            mujoco.mj_forward(model, data)
                            update_viewer()
                            if live_viewer:
                                time.sleep(0.5)
                        else:
                            print("Emergency recovery failed - stopping trajectory")
                            break
                            
                    except Exception as recovery_error:
                        print(f"Emergency recovery failed: {recovery_error}")
                        break
                
                # Only check for significant distance regression (getting much farther from target)
                if timestep > 1:
                    prev_distance_to_target = np.linalg.norm(real_query_pos - prev_pos)
                    current_distance_to_target = np.linalg.norm(real_query_pos - achieved_pos)
                    distance_regression = current_distance_to_target - prev_distance_to_target
                    
                    if distance_regression > 0.005:  # Only warn if getting 0.5cm farther
                        print(f"⚠️ Distance regression: {distance_regression*100:.1f}cm farther from target")
                
                if result_q is None:
                    print(f"ERROR: Could not reach waypoint {timestep}")
                    print("All adaptive methods failed - stopping trajectory")
                    break
                
                # Update current position for next iteration
                current_pos = achieved_pos.copy()
        
                # Get actual achieved position (MUST use pinch site for consistency)
                if use_site and ee_site_id is not None:
                    actual_pos = data.site_xpos[ee_site_id].copy()
                    print(f"Using pinch site position: {actual_pos}    ---- GOOD")
                else:
                    actual_pos = data.body(ee_body_id).xpos.copy()
                    print(f"Warning: Using end effector body position: {actual_pos}")
                
                # Apply collision checking AFTER alignment, focusing on PINCH SITE distance
                collision_resolved, final_q = self.check_and_resolve_collisions(
                    model, data, arm_joint_indices, ee_body_id, real_query_pos, f"Timestep {timestep}", ee_site_id
                )
                
                # Update viewer after collision resolution
                if collision_resolved:
                    mujoco.mj_forward(model, data)
                    update_viewer()
                    if live_viewer:
                        time.sleep(0.3)
                
                # CRITICAL: Verify PINCH SITE is within distance threshold of target
                if use_site and ee_site_id is not None:
                    pinch_pos = data.site_xpos[ee_site_id].copy()
                else:
                    print("WARNING: No pinch site - using end effector body position")
                    pinch_pos = data.body(ee_body_id).xpos.copy()
                
                pinch_distance_to_target = np.linalg.norm(real_query_pos - pinch_pos)
                print(f"PINCH SITE distance to target: {pinch_distance_to_target*100:.1f}cm (threshold: {distance_threshold*100:.1f}cm)")

                if timestep == num_timesteps and pinch_distance_to_target > distance_threshold:
                    print(f"ERROR: Final pinch site distance {pinch_distance_to_target*100:.1f}cm exceeds threshold {distance_threshold*100:.1f}cm")
                    
                    # Add current position to candidates
                    timestep_positions.append({
                        'position': pinch_pos.copy(),
                        'distance_to_object': pinch_distance_to_target,
                        'joint_config': np.array([data.qpos[idx] for idx in arm_joint_indices]),
                        'source': 'current'
                    })
                    
                    print(f"POSITION RECOVERY: Evaluating {len(timestep_positions)} candidate positions...")
                    
                    # Find the best position (closest to target but collision-free)
                    best_candidate = None
                    best_distance = float('inf')
                    
                    for candidate in timestep_positions:
                        # Test this position for collisions
                        original_q = np.array([data.qpos[idx] for idx in arm_joint_indices])
                        
                        # Set candidate configuration
                        for i, joint_idx in enumerate(arm_joint_indices):
                            data.qpos[joint_idx] = candidate['joint_config'][i]
                        mujoco.mj_forward(model, data)
                        
                        # Check collisions
                        collision_detected = False
                        try:
                            collision_detected = not ik_solver.rrt_system.check_collision(candidate['joint_config'], arm_joint_indices)
                        except:
                            collision_detected = True
                        
                        if not collision_detected and candidate['distance_to_object'] < best_distance:
                            best_candidate = candidate
                            best_distance = candidate['distance_to_object']
                            print(f"  {candidate['source']}: {candidate['distance_to_object']*100:.1f}cm ✓")
                        else:
                            status = "collision" if collision_detected else f"{candidate['distance_to_object']*100:.1f}cm"
                            print(f"  {candidate['source']}: {status} ✗")
                        
                        # Restore original configuration
                        for i, joint_idx in enumerate(arm_joint_indices):
                            data.qpos[joint_idx] = original_q[i]
                        mujoco.mj_forward(model, data)
                    
                    if best_candidate is not None:
                        print(f"RECOVERY SUCCESS: Using {best_candidate['source']} position with {best_distance*100:.1f}cm distance")
                        
                        # Set the best configuration
                        for i, joint_idx in enumerate(arm_joint_indices):
                            data.qpos[joint_idx] = best_candidate['joint_config'][i]
                        mujoco.mj_forward(model, data)
                        update_viewer()
                        
                        # Update final_q and pinch_pos
                        final_q = best_candidate['joint_config']
                        pinch_pos = best_candidate['position']
                        pinch_distance_to_target = best_distance
                        
                        print(f"Final distance after recovery: {pinch_distance_to_target*100:.1f}cm")
                    else:
                        print(f"RECOVERY FAILED: No collision-free positions found, keeping current position")
                
                if not collision_resolved:
                    print(f"ERROR: Collision at timestep {timestep}")
                    break
                
                # Optional alignment between camera and end effector
                if use_timestep_alignment:
                    print(f"🔧 Applying optional {timestep_alignment_type} alignment")
                    if timestep_alignment_type == 'camera':
                        # Pure camera alignment (joints 6-7 only)
                        self.optimize_camera_alignment_wrist_only(
                            model, data, arm_joint_indices, ee_body_id, objects_center, final_q
                        )
                    elif timestep_alignment_type == 'end_effector':
                        # Skip alignment if we're already close enough
                        current_distance = np.linalg.norm(real_query_pos - get_pinch_site_position())
                        if current_distance <= distance_threshold * 1.2:  # Within 20% of target
                            print(f"Skipping alignment - already close enough ({current_distance*100:.1f}cm)")
                        else:
                            print(f"Applying distance-constrained pinch site alignment")
                            pre_align_q = np.array([data.qpos[idx] for idx in arm_joint_indices])
                            pre_align_pos = get_pinch_site_position()
                            
                            final_q = self.precise_pinch_site_alignment(
                                model, data, arm_joint_indices, ee_body_id, ee_site_id, use_site, real_query_pos, query_object
                            )
                            
                            # Check if alignment made things worse
                            post_align_pos = get_pinch_site_position()
                            post_align_distance = np.linalg.norm(real_query_pos - post_align_pos)
                            pre_align_distance = np.linalg.norm(real_query_pos - pre_align_pos)
                            
                            if post_align_distance > pre_align_distance * 1.5:  # Got 50% worse
                                print(f"WARNING: Alignment made distance worse ({pre_align_distance*100:.1f}cm → {post_align_distance*100:.1f}cm)")
                                print("Reverting to pre-alignment position")
                                
                                # Revert to pre-alignment
                                for i, joint_idx in enumerate(arm_joint_indices):
                                    data.qpos[joint_idx] = pre_align_q[i]
                                mujoco.mj_forward(model, data)
                                final_q = pre_align_q
                        
                        # VERIFY the alignment actually worked
                        is_pointing_toward, alignment_score, direction_status = self.verify_pinch_site_pointing_direction(
                            model, data, ee_body_id, ee_site_id, use_site, real_query_pos
                        )
                        
                        if not is_pointing_toward:
                            print(f"WARNING: Pinch site alignment failed - pointing {direction_status} object!")
                            print(f"Consider increasing search range or checking end effector orientation convention")
                    
                    # Update viewer after alignment
                    mujoco.mj_forward(model, data)
                    update_viewer()
                    if live_viewer:
                        time.sleep(0.3)
                
                # Get final position after alignment 
                if use_site and ee_site_id is not None:
                    final_pos = data.site_xpos[ee_site_id].copy()
                    print(f"Timestep {timestep}: Using pinch site position: {final_pos}   ---- GOOD")
                else:
                    final_pos = data.body(ee_body_id).xpos.copy()
                    print(f"Timestep {timestep}: WARNING - Using body position instead of pinch site: {final_pos}")
                
                # Convert to base-relative coordinates for dataset 
                final_pos_relative = self.convert_to_base_relative_coords(final_pos, base_pos)
                prev_pos_relative = self.convert_to_base_relative_coords(prev_pos, base_pos)
                print(f"PINCH SITE - World: {final_pos}, Base-relative (dataset values are in base-relative): {final_pos_relative}")
                print(f"REAL QUERY OBJ POS - World: {real_query_pos}, Base-relative (dataset values are in base-relative): {real_query_pos_relative}")
                
                # Calculate action (we use query_pos instead of real_query_pos, because the latter changes depending on EE position, and we don't want that)
                current_action = real_query_pos_relative - final_pos_relative   # WE USE QUERY_POS HERE, NOT real_query_pos (see above)
                prev_action = real_query_pos_relative - prev_pos_relative if timestep > 1 else current_action
                
                # Store action data
                action_row = {
                    'dx_t': current_action[0],
                    'dy_t': current_action[1], 
                    'dz_t': current_action[2],
                    #'dx_t-1': prev_action[0],
                    #'dy_t-1': prev_action[1],
                    #'dz_t-1': prev_action[2],
                    #'x_t': final_pos_relative[0],
                    #'y_t': final_pos_relative[1],
                    #'z_t': final_pos_relative[2],
                    #'x_t-1': prev_pos_relative[0],
                    #'y_t-1': prev_pos_relative[1],
                    #'z_t-1': prev_pos_relative[2]
                }
                action_data.append(action_row)
                print(f"Dataset ONLY uses the base-relative positions and actions   --- GOOD")
                
                # Take images
                if capture_images:
                    self.capture_observation_image(model, data, obs_dir, timestep)
                
                external_frame = self.capture_external_video_frame(model, data)
                if external_frame is not None:
                    video_frames.append(external_frame)
                
                # Interactive validation if enabled
                if self.validation:
                    distance_to_target = np.linalg.norm(real_query_pos - final_pos)
                    result = self.validate_timestep_interactively(model, data, timestep, final_pos, real_query_pos, distance_to_target)
                    if result is None or not result:
                        break
                
                # Track progress and detect stagnation
                current_distance = np.linalg.norm(real_query_pos_relative - final_pos_relative)
                # Initialize tracking variables on first timestep
                if timestep == 1:
                    previous_distances = []
                    stagnant_count = 0
                    
                previous_distances.append(current_distance)
                
                prev_pos = final_pos.copy()
                print(f"✅ Completed timestep {timestep}, distance to target: {current_distance:.4f}m ({current_distance*100:.1f}cm)")
                
                # Give user time to observe the movement
                print(f"👀 Observe the arm movement in the MuJoCo viewer...")
                if live_viewer:
                    time.sleep(1.0)
            
            if live_viewer:
                print(f"🏁 Trajectory complete! Keeping viewer open for 3 seconds...")
                time.sleep(3.0)
            else:
                print(f"🏁 Trajectory complete!")
            
            # Save data and video
            self.save_action_space_data(action_data, episode_dir)
            
            # Save video
            if len(video_frames) > 0:
                video_path = os.path.join(episode_dir, f"arm_movement_episode_{episode_id}.mp4")
                try:
                    import imageio.v3 as iio
                    iio.imwrite(video_path, video_frames, fps=2)
                    print(f"Video saved: {video_path}")
                except Exception as e:
                    print(f"Video save failed: {e}")
            
            # Success criteria: completed timesteps AND final distance <= threshold
            final_distance = np.linalg.norm(real_query_pos_relative - final_pos_relative) if 'final_pos_relative' in locals() else float('inf')
            success = (num_timesteps - 1 <= len(action_data) <= num_timesteps) and (final_distance <= distance_threshold)
            print(f"Episode completion: {len(action_data)} timesteps, final distance: {final_distance*100:.1f}cm")
            print(f"Success: {success} (need total number of timesteps AND distance <= {distance_threshold*100:.1f}cm)")
            
            return success, True          # Return whether ideal distance was reached and whether episode completed fine (based on user's obs_0 vlaidation)
            
        except Exception as e:
            print(f"Error in trajectory generation: {e}")
            return False, False
            
        finally:
            # Always close viewer when done
            print("🔚 Closing MuJoCo viewer...")
            close_viewer()
            if viewer_thread and viewer_thread.is_alive():
                viewer_thread.join(timeout=2)

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
                    use_timestep_alignment=False, timestep_alignment_type='camera', required_object=None):
        """Generate episodic dataset with IK trajectories"""
        print(f"Generating {'validation' if self.validation else 'training'} dataset...")
        print(f"Episodes: {num_episodes}")
        
        successful_episodes = 0
        failed_episodes = 0

        existing_objects = self.check_objects_exist()
        eps_per_object = num_episodes // len(existing_objects) if len(existing_objects) > 0 else 0
        object_chosen_indx = 0
        
        for episode_id in range(num_episodes):
            try:
                episode_success = False
                while not episode_success:
                    required_object = existing_objects[object_chosen_indx]
                    if (episode_id + 1) % eps_per_object == 0:
                        object_chosen_indx = (object_chosen_indx + 1) % len(existing_objects)
                    print(f"\n=== Generating episode {episode_id} with object '{required_object}' ===")

                    ideal_distance_reached, episode_success = self.generate_episode(
                        episode_id, distance_threshold, capture_images, use_timestep_alignment, 
                        timestep_alignment_type, required_object=required_object
                    )
                    
                if ideal_distance_reached:
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

# Notes:3
# - A potential issue is that in the IK and alignmnet stuff, it may rotate a bearing joint (lke joint 5), and this should be fine (since we ban joint 7, which is the camera ring, movement), but when the joints surrounding it are aligned straightly, then rotating this joint will NOT (or extremely small) change the pinch site position, which will be confusing future models that parse through the created dataset
# - To "patch" this, we can add a check to see if the joints surrounding the bearing joint are aligned straightly (within some small threshold), and if so, we can either skip the rotation of the bearing joint.
# - For a robust solution, the best idea is to just record the joint movements within the dataset, and then the model can learn to perform all motions including bearing joints (this also allows for joint 7 to be used!)