import numpy as np
import mujoco
import mujoco.viewer as viewer
import os
import json
import time
from datetime import datetime
import threading
import cv2

# Try to import mediapy, fallback to cv2 if not available or ffmpeg missing
try:
    import mediapy as media
    MEDIAPY_AVAILABLE = True
except ImportError:
    MEDIAPY_AVAILABLE = False
    print("Warning: mediapy not available, using cv2 for video writing")

class BaseIK:
    """Base class for inverse kinematics solvers"""
    
    def __init__(self, model, step_size=0.5, tol=0.01, max_iter=1000):
        self.model = model
        self.step_size = step_size
        self.tol = tol
        self.max_iter = max_iter
        self.iterations = 0
        self.converged = False
        
        # Pre-allocate jacobians
        self.jacp = np.zeros((3, model.nv))
        self.jacr = np.zeros((3, model.nv))
    
    def check_joint_limits(self, q, joint_indices):
        """Check and clamp joint limits for specific joints"""
        q_limited = q.copy()
        for i, joint_idx in enumerate(joint_indices):
            if joint_idx < len(self.model.jnt_range):
                joint_range = self.model.jnt_range[joint_idx]
                # Check if joint has finite limits
                if not (np.isinf(joint_range[0]) or np.isinf(joint_range[1])):
                    q_limited[i] = np.clip(q[i], joint_range[0], joint_range[1])
        return q_limited
    
    def calculate_error(self, data, goal, body_id):
        """Calculate position error"""
        current_pos = data.body(body_id).xpos
        return goal - current_pos
    
    def solve(self, data, goal, init_q, body_id, arm_joint_indices, arm_actuator_indices):
        """Base solve method - to be overridden"""
        raise NotImplementedError

class GradientDescentIK(BaseIK):
    def __init__(self, model, step_size=0.5, tol=0.01, alpha=0.5, max_iter=1000):
        super().__init__(model, step_size, tol, max_iter)
        self.alpha = alpha
    
    def solve(self, data, goal, init_q, body_id, arm_joint_indices, arm_actuator_indices):
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
            
            # Calculate jacobian
            mujoco.mj_jac(self.model, data, self.jacp, self.jacr, goal, body_id)
            
            # Extract jacobian columns for arm joints only
            arm_jacp = self.jacp[:, arm_joint_indices]
            
            # Gradient descent update
            grad = self.alpha * arm_jacp.T @ error
            
            # Update arm joint positions
            current_q = np.array([data.qpos[idx] for idx in arm_joint_indices])
            new_q = current_q + self.step_size * grad
            
            # Apply joint limits
            new_q = self.check_joint_limits(new_q, arm_joint_indices)
            
            # Set new joint positions
            for i, joint_idx in enumerate(arm_joint_indices):
                data.qpos[joint_idx] = new_q[i]
            
            # Forward kinematics
            mujoco.mj_forward(self.model, data)
            self.iterations = i + 1
        
        return np.array([data.qpos[idx] for idx in arm_joint_indices])

class GaussNewtonIK(BaseIK):
    def solve(self, data, goal, init_q, body_id, arm_joint_indices, arm_actuator_indices):
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
            
            # Calculate jacobian
            mujoco.mj_jac(self.model, data, self.jacp, self.jacr, goal, body_id)
            
            # Extract jacobian columns for arm joints only
            arm_jacp = self.jacp[:, arm_joint_indices]
            
            # Gauss-Newton update with regularization
            JTJ = arm_jacp.T @ arm_jacp
            reg = 1e-4 * np.eye(JTJ.shape[0])  # Regularization
            
            try:
                if np.linalg.det(JTJ + reg) > 1e-6:
                    j_inv = np.linalg.inv(JTJ + reg) @ arm_jacp.T
                else:
                    j_inv = np.linalg.pinv(arm_jacp)
            except np.linalg.LinAlgError:
                j_inv = np.linalg.pinv(arm_jacp)
            
            delta_q = j_inv @ error
            
            # Update arm joint positions
            current_q = np.array([data.qpos[idx] for idx in arm_joint_indices])
            new_q = current_q + self.step_size * delta_q
            
            # Apply joint limits
            new_q = self.check_joint_limits(new_q, arm_joint_indices)
            
            # Set new joint positions
            for i, joint_idx in enumerate(arm_joint_indices):
                data.qpos[joint_idx] = new_q[i]
            
            # Forward kinematics
            mujoco.mj_forward(self.model, data)
            self.iterations = i + 1
        
        return np.array([data.qpos[idx] for idx in arm_joint_indices])

class LevenbergMarquardtIK(BaseIK):
    def __init__(self, model, step_size=0.5, tol=0.01, damping=0.1, max_iter=1000):
        super().__init__(model, step_size, tol, max_iter)
        self.damping = damping
    
    def solve(self, data, goal, init_q, body_id, arm_joint_indices, arm_actuator_indices):
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
            
            # Calculate jacobian
            mujoco.mj_jac(self.model, data, self.jacp, self.jacr, goal, body_id)
            
            # Extract jacobian columns for arm joints only
            arm_jacp = self.jacp[:, arm_joint_indices]
            
            # Levenberg-Marquardt update
            n = arm_jacp.shape[1]
            I = np.eye(n)
            A = arm_jacp.T @ arm_jacp + self.damping * I
            
            try:
                delta_q = np.linalg.solve(A, arm_jacp.T @ error)
            except np.linalg.LinAlgError:
                delta_q = np.linalg.pinv(A) @ arm_jacp.T @ error
            
            # Update arm joint positions
            current_q = np.array([data.qpos[idx] for idx in arm_joint_indices])
            new_q = current_q + self.step_size * delta_q
            
            # Apply joint limits
            new_q = self.check_joint_limits(new_q, arm_joint_indices)
            
            # Set new joint positions
            for i, joint_idx in enumerate(arm_joint_indices):
                data.qpos[joint_idx] = new_q[i]
            
            # Forward kinematics
            mujoco.mj_forward(self.model, data)
            self.iterations = i + 1
        
        return np.array([data.qpos[idx] for idx in arm_joint_indices])

class MuJoCoIKSimulator:
    def __init__(self, xml_path, output_dir="output"):
        self.xml_path = xml_path
        self.output_dir = output_dir
        self.setup_output_directory()
        
        # Load model
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model)
        
        # Setup camera
        self.camera = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(self.model, self.camera)
        self.camera.distance = 3.0
        self.camera.azimuth = 45
        self.camera.elevation = -20
        
        # Find arm joint and actuator indices dynamically
        self.arm_joint_indices, self.arm_actuator_indices = self.find_arm_indices()
        
        # Initialize solvers
        self.solvers = {
            'gradient_descent': GradientDescentIK(self.model, step_size=0.5, alpha=0.5),
            'gauss_newton': GaussNewtonIK(self.model, step_size=0.5),
            'levenberg_marquardt': LevenbergMarquardtIK(self.model, step_size=0.5, damping=0.1)
        }
        
        # Define home position for arm joints
        self.home_position = np.zeros(len(self.arm_joint_indices))
        
        # Define target objects with their positions
        self.target_objects = [
            {'name': 'ground_target', 'position': np.array([0.5, 0.3, 0.15]), 'color': [1, 0, 0, 1]},
            {'name': 'floating_target_1', 'position': np.array([0.3, -0.4, 0.5]), 'color': [0, 1, 0, 1]},
            {'name': 'floating_target_2', 'position': np.array([0.6, 0.2, 0.7]), 'color': [0, 0, 1, 1]}
        ]
        
        self.results = {}
        
        # Print model information for debugging
        self.print_model_info()
    
    def find_arm_indices(self):
        """Find arm joint and actuator indices dynamically"""
        arm_joint_indices = []
        arm_actuator_indices = []
        
        print("Finding arm joints and actuators...")
        
        # Look for joints with "Actuator" in their name or typical arm joint names
        arm_joint_patterns = ['Actuator', 'joint_', 'shoulder', 'elbow', 'wrist', 'forearm']
        
        # Find arm joints
        for i in range(self.model.njnt):
            try:
                joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
                if joint_name:
                    # Skip wheelchair joints and free joints
                    if any(pattern in joint_name for pattern in ['wheel', 'steering', 'throttle', 'free']):
                        continue
                    # Look for typical arm joint patterns
                    if any(pattern in joint_name for pattern in arm_joint_patterns):
                        arm_joint_indices.append(i)
                        print(f"Found arm joint {len(arm_joint_indices)}: {joint_name} (index {i})")
            except:
                continue
        
        # Find corresponding actuators
        for i in range(self.model.nu):
            try:
                actuator_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                if actuator_name:
                    # Skip wheelchair actuators
                    if any(pattern in actuator_name for pattern in ['wheel', 'steering', 'throttle']):
                        continue
                    # Look for arm actuator patterns
                    if any(pattern in actuator_name for pattern in arm_joint_patterns):
                        arm_actuator_indices.append(i)
                        print(f"Found arm actuator {len(arm_actuator_indices)}: {actuator_name} (index {i})")
            except:
                continue
        
        # If we can't find by name patterns, try by excluding known wheelchair components
        if not arm_joint_indices:
            print("Warning: No arm joints found by name pattern, using fallback method...")
            # Skip the first few joints which are likely wheelchair/free joints
            start_idx = 0
            for i in range(self.model.njnt):
                try:
                    joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
                    if joint_name and not any(pattern in joint_name.lower() for pattern in ['free', 'wheel', 'steering', 'throttle']):
                        arm_joint_indices.append(i)
                        if len(arm_joint_indices) >= 7:  # Assume 7 DOF arm
                            break
                except:
                    continue
        
        if not arm_actuator_indices:
            print("Warning: No arm actuators found by name pattern, using fallback method...")
            start_idx = 0
            for i in range(self.model.nu):
                try:
                    actuator_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                    if actuator_name and not any(pattern in actuator_name.lower() for pattern in ['wheel', 'steering', 'throttle']):
                        arm_actuator_indices.append(i)
                        if len(arm_actuator_indices) >= 7:  # Assume 7 DOF arm
                            break
                except:
                    continue
        
        print(f"Found {len(arm_joint_indices)} arm joints: {arm_joint_indices}")
        print(f"Found {len(arm_actuator_indices)} arm actuators: {arm_actuator_indices}")
        
        return arm_joint_indices, arm_actuator_indices
    
    def setup_output_directory(self):
        """Create output directory with timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"{self.output_dir}_{timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(f"{self.output_dir}/images", exist_ok=True)
        os.makedirs(f"{self.output_dir}/videos", exist_ok=True)
    
    def print_model_info(self):
        """Print model information for debugging"""
        print(f"Model: {self.xml_path}")
        print(f"Number of bodies: {self.model.nbody}")
        print(f"Number of joints: {self.model.njnt}")
        print(f"Number of DOFs: {self.model.nv}")
        print(f"Number of generalized coordinates: {self.model.nq}")
        print(f"Number of actuators: {self.model.nu}")
        
        print("\nBodies:")
        for i in range(self.model.nbody):
            try:
                body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
                if body_name:
                    print(f"  {i}: {body_name}")
            except:
                pass
        
        print("\nJoints:")
        for i in range(self.model.njnt):
            try:
                joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
                if joint_name:
                    print(f"  {i}: {joint_name}")
            except:
                pass
        
        print("\nActuators:")
        for i in range(self.model.nu):
            try:
                actuator_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                if actuator_name:
                    print(f"  {i}: {actuator_name}")
            except:
                pass
    
    def save_image(self, image, filename):
        """Save image to file"""
        filepath = f"{self.output_dir}/images/{filename}"
        if MEDIAPY_AVAILABLE:
            try:
                media.write_image(filepath, image)
            except RuntimeError:
                cv2.imwrite(filepath, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        else:
            cv2.imwrite(filepath, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    
    def save_results_json(self, results, filename="results.json"):
        """Save results to JSON file"""
        filepath = f"{self.output_dir}/{filename}"
        
        json_results = {}
        for target_name, methods in results.items():
            json_results[target_name] = {}
            for method, data in methods.items():
                joint_angles = data.get('joint_angles', None)
                if isinstance(joint_angles, np.ndarray):
                    joint_angles = joint_angles.tolist()
                end_effector_pos = data.get('end_effector_pos', None)
                if isinstance(end_effector_pos, np.ndarray):
                    end_effector_pos = end_effector_pos.tolist()
                target_position = data.get('target_position', None)
                if isinstance(target_position, np.ndarray):
                    target_position = target_position.tolist()
                json_results[target_name][method] = {
                    'joint_angles': joint_angles,
                    'end_effector_pos': end_effector_pos,
                    'error': float(data.get('error', 0.0)),
                    'iterations': int(data.get('iterations', 0)),
                    'converged': bool(data.get('converged', False)),
                    'solve_time': float(data.get('solve_time', 0.0)),
                    'target_name': data.get('target_name', ''),
                    'target_position': target_position if target_position is not None else []
                }
        
        with open(filepath, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        return filepath
    
    def save_video_cv2(self, frames, filename, fps=60):
        """Save video using OpenCV"""
        if not frames:
            return None
            
        filepath = f"{self.output_dir}/videos/{filename}"
        height, width = frames[0].shape[:2]
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filepath, fourcc, fps, (width, height))
        
        for frame in frames:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        
        out.release()
        return filepath
    
    def set_home_position(self):
        """Set the robot arm to home position"""
        for i, joint_idx in enumerate(self.arm_joint_indices):
            self.data.qpos[joint_idx] = self.home_position[i]
        mujoco.mj_forward(self.model, self.data)
    
    def test_ik_methods(self, goal_pos, init_q=None, body_name='ee_link', target_name="target"):
        """Test all IK methods and compare results"""
        if init_q is None:
            init_q = self.home_position.copy()
        
        # Get body ID
        try:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        except:
            print(f"Warning: Body '{body_name}' not found, trying alternatives...")
            # Try common end effector names
            for alt_name in ['end_effector', 'gripper', 'tool0', 'ee', 'end_effector_link']:
                try:
                    body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, alt_name)
                    print(f"Using body: {alt_name}")
                    break
                except:
                    continue
            else:
                # Use the last body as fallback
                body_id = self.model.nbody - 1
                body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id)
                print(f"Using fallback body: {body_name} (id: {body_id})")
        
        results = {}
        
        print(f"\nTesting IK methods for {target_name}: {goal_pos}")
        print("-" * 60)
        
        for method_name, solver in self.solvers.items():
            print(f"Testing {method_name}...")
            
            # Reset to home position
            self.set_home_position()
            
            # Time the solving
            start_time = time.time()
            result_q = solver.solve(self.data, goal_pos, init_q, body_id, 
                                  self.arm_joint_indices, self.arm_actuator_indices)
            solve_time = time.time() - start_time
            
            # Get final end effector position
            final_pos = self.data.body(body_id).xpos.copy()
            error = np.linalg.norm(goal_pos - final_pos)
            
            # Store results
            results[method_name] = {
                'joint_angles': result_q,
                'end_effector_pos': final_pos,
                'error': error,
                'iterations': solver.iterations,
                'converged': solver.converged,
                'solve_time': solve_time,
                'target_name': target_name,
                'target_position': goal_pos
            }
            
            # Render and save image
            self.renderer.update_scene(self.data, self.camera)
            image = self.renderer.render()
            self.save_image(image, f"{target_name}_{method_name}_result.png")
            
            print(f"  Result: {final_pos}")
            print(f"  Error: {error:.6f}")
            print(f"  Iterations: {solver.iterations}")
            print(f"  Converged: {solver.converged}")
            print(f"  Time: {solve_time:.4f}s\n")
        
        return results

    def test_all_targets(self):
        """Test IK for all target objects"""
        all_results = {}
        
        print("Testing IK for all target objects...")
        print("=" * 80)
        
        for target in self.target_objects:
            target_name = target['name']
            target_pos = target['position']
            
            target_results = self.test_ik_methods(
                goal_pos=target_pos,
                init_q=self.home_position,
                target_name=target_name
            )
            
            if target_results:
                all_results[target_name] = target_results
        
        self.results = all_results
        return all_results
    
    def run_sequential_simulation(self, duration_per_target=5, method='levenberg_marquardt'):
        """Run simulation visiting each target sequentially"""
        print(f"Starting sequential simulation with {method}...")
        print("Close the viewer window to end simulation early.")
        
        # Get body ID for end effector
        try:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'ee_link')
        except:
            try:
                body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'end_effector')
            except:
                body_id = self.model.nbody - 1
        
        solver = self.solvers[method]
        
        # For video recording
        frames = []
        framerate = 30
        
        try:
            with mujoco.viewer.launch_passive(self.model, self.data) as viewer_instance:
                target_index = 0
                start_time = time.time()
                last_target_switch = start_time
                last_frame_time = 0
                
                # Start at home position
                self.set_home_position()
                
                print(f"Starting at home position")
                
                while viewer_instance.is_running() and target_index < len(self.target_objects):
                    step_start = time.time()
                    current_time = time.time()

                    # Switch to next target if enough time has passed
                    if current_time - last_target_switch > duration_per_target:
                        target_index += 1
                        last_target_switch = current_time
                        if target_index < len(self.target_objects):
                            print(f"Switching to target: {self.target_objects[target_index]['name']}")
                    
                    # Get current target
                    if target_index < len(self.target_objects):
                        current_goal = self.target_objects[target_index]['position']
                    else:
                        current_goal = np.array([0.0, 0.0, 1.0])  # High above home
                    
                    # Calculate error
                    current_pos = self.data.body(body_id).xpos
                    error = current_goal - current_pos
                    
                    # Only solve IK if error is significant
                    if np.linalg.norm(error) > solver.tol:
                        # Calculate jacobian
                        mujoco.mj_jac(self.model, self.data, solver.jacp, solver.jacr, 
                                    current_goal, body_id)
                        
                        # Extract jacobian columns for arm joints only
                        arm_jacp = solver.jacp[:, self.arm_joint_indices]
                        
                        # Use Levenberg-Marquardt method for real-time
                        n = arm_jacp.shape[1]
                        I = np.eye(n)
                        damping = 0.1
                        A = arm_jacp.T @ arm_jacp + damping * I
                        
                        try:
                            delta_q = np.linalg.solve(A, arm_jacp.T @ error)
                        except np.linalg.LinAlgError:
                            delta_q = np.linalg.pinv(A) @ arm_jacp.T @ error
                        
                        # Update joint positions
                        current_arm_q = np.array([self.data.qpos[idx] for idx in self.arm_joint_indices])
                        new_arm_q = current_arm_q + 0.5 * delta_q
                        new_arm_q = solver.check_joint_limits(new_arm_q, self.arm_joint_indices)
                        
                        # Set control to actuators
                        for i, actuator_idx in enumerate(self.arm_actuator_indices):
                            if i < len(new_arm_q):
                                self.data.ctrl[actuator_idx] = new_arm_q[i]
                    
                    # Step simulation
                    mujoco.mj_step(self.model, self.data)
                    
                    # Record frame for video at reduced rate
                    if self.data.time - last_frame_time >= 1.0/framerate:
                        self.renderer.update_scene(self.data, self.camera)
                        frame = self.renderer.render()
                        frames.append(frame)
                        last_frame_time = self.data.time
                    
                    # Update viewer
                    viewer_instance.sync()
                    
                    # Control simulation speed
                    elapsed = time.time() - step_start
                    if elapsed < self.model.opt.timestep:
                        time.sleep(self.model.opt.timestep - elapsed)
                        
        except Exception as e:
            print(f"Viewer error: {e}")
            print("Continuing without real-time visualization...")
        
        # Save video
        if frames:
            try:
                if MEDIAPY_AVAILABLE:
                    video_path = f"{self.output_dir}/videos/sequential_simulation.mp4"
                    media.write_video(video_path, frames, fps=framerate)
                    print(f"Video saved to: {video_path}")
                else:
                    video_path = self.save_video_cv2(frames, "sequential_simulation.mp4", framerate)
                    if video_path:
                        print(f"Video saved to: {video_path}")
            except Exception as e:
                print(f"Error saving video: {e}")
                try:
                    video_path = self.save_video_cv2(frames, "sequential_simulation_fallback.mp4", framerate)
                    if video_path:
                        print(f"Video saved with fallback method to: {video_path}")
                except Exception as e2:
                    print(f"Could not save video: {e2}")
        
        print("Sequential simulation completed!")

def main():
    # Configuration
    XML_PATH = r"all_gen3_scene.xml"  # Update this path to your scene file
    
    # Create simulator
    sim = MuJoCoIKSimulator(XML_PATH)
    
    # Get joint information
    #joint_info = sim.get_joint_info()
    
    # Print target object information
    print("\nTarget Objects:")
    print("=" * 50)
    for i, target in enumerate(sim.target_objects):
        print(f"{i+1}. {target['name']}: {target['position']} (Color: {target['color'][:3]})")

    # Test all targets with all IK methods
    all_results = sim.test_all_targets()
    
    if all_results:
        # Save results to JSON
        results_file = sim.save_results_json(all_results, "all_targets_results.json")
        print(f"Results saved to: {results_file}")
        
        # Print comparison for each target
        for target_name, target_results in all_results.items():
            print(f"\nMethod Comparison for {target_name}:")
            print("-" * 80)
            print(f"{'Method':<20} {'Error':<12} {'Iterations':<12} {'Time (s)':<10} {'Converged'}")
            print("-" * 80)
            
            for method, data in target_results.items():
                print(f"{method:<20} {data['error']:<12.6f} {data['iterations']:<12} "
                      f"{data['solve_time']:<10.4f} {data['converged']}")
        
        # Run sequential simulation
        print("\nStarting sequential simulation...")
        print("The arm will visit each target in order, starting from home position.")
        sim.run_sequential_simulation(
            duration_per_target=8,  # 8 seconds per target
            method='gauss_newton'  # Change to 'gradient_descent' or 'levenberg_marquardt' as needed
        )
        
        print(f"\nAll outputs saved to: {sim.output_dir}")
    else:
        print("IK testing failed. Check the model and body names.")

if __name__ == "__main__":
    main()