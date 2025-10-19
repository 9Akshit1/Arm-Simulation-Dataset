import mujoco
import mujoco.viewer
import numpy as np
import os
from dataset_generator_linear import DatasetGenerator, GaussNewtonIK
import random

def test_scene_loading():
    """Test if the scene loads correctly"""
    try:
        model = mujoco.MjModel.from_xml_path("dataset/scenes/gen3_scene.xml")
        data = mujoco.MjData(model)
        print("✓ Scene loaded successfully!")
        
        # Print some basic info
        print(f"  - Number of bodies: {model.nbody}")
        print(f"  - Number of joints: {model.njnt}")
        print(f"  - Number of actuators: {model.nu}")
        
        # Check if camera exists
        camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist")
        if camera_id >= 0:
            print("✓ Wrist camera found!")
        else:
            print("✗ Wrist camera not found!")
            
        return model, data
        
    except Exception as e:
        print(f"✗ Error loading scene: {e}")
        return None, None

def test_arm_configurations_ik():
    """Test different arm configurations using IK with dynamic target generation"""
    model, data = test_scene_loading()
    if model is None:
        return
    
    generator = DatasetGenerator()
    
    print("\nTesting arm configurations with GaussNewtonIK and dynamic targets:")
    
    # Find arm components
    arm_joint_indices, arm_actuator_indices = generator.find_arm_indices(model)
    ee_body_id = generator.get_end_effector_body_id(model)
    
    # Initialize IK solver with proper parameters from reference
    ik_solver = GaussNewtonIK(model, step_size=0.5, tol=0.01, max_iter=1000)
    
    # Generate some test positions dynamically using the new system
    test_table_pos = [0.6, 0, 0.43]  # Default table position
    test_objects_center = [0.6, 0, 0.48]  # Simulated objects center
    
    print("  Generating dynamic target positions...")
    test_positions = []
    for i in range(5):
        target_pos = generator.generate_valid_target_position(test_table_pos, test_objects_center)
        test_positions.append(target_pos)
        print(f"    Generated target {i+1}: {target_pos}")
    
    for i, target_pos in enumerate(test_positions):
        print(f"  Config {i+1}: Moving to target {target_pos}")
        
        try:
            result_q, final_pos, error = generator.move_arm_to_target_ik(
                model, data, target_pos, arm_joint_indices, ee_body_id, ik_solver
            )
            print(f"    - Joint angles: {[f'{x:.2f}' for x in result_q]}")
            print(f"    - Final position: {[f'{x:.3f}' for x in final_pos]}")
            print(f"    - Error: {error:.4f}")
            print(f"    - Converged: {ik_solver.converged}")
        except Exception as e:
            print(f"    - Error: {e}")

def test_camera_view():
    """Test camera view and positioning using IK"""
    model, data = test_scene_loading()
    if model is None:
        return
    
    print("\nTesting camera view with IK positioning...")
    
    # Set up IK components
    generator = DatasetGenerator()
    arm_joint_indices, arm_actuator_indices = generator.find_arm_indices(model)
    ee_body_id = generator.get_end_effector_body_id(model)
    ik_solver = GaussNewtonIK(model, step_size=0.5, tol=0.01, max_iter=1000)
    
    # Use a target position that should give a good camera view
    target_position = [0.6, 0.0, 0.8]  # Center overhead
    
    try:
        result_q, final_pos, error = generator.move_arm_to_target_ik(
            model, data, target_position, arm_joint_indices, ee_body_id, ik_solver
        )
        
        print(f"✓ Arm moved to target position")
        print(f"  - Target: {target_position}")
        print(f"  - Achieved: {[f'{x:.3f}' for x in final_pos]}")
        print(f"  - Error: {error:.4f}")
        print(f"  - Converged: {ik_solver.converged}")
        print(f"  - Iterations: {ik_solver.iterations}")
        
        # Check camera
        camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist")
        if camera_id >= 0:
            print(f"✓ Camera position: {data.cam_xpos[camera_id]}")
            print(f"✓ Camera orientation matrix shape: {data.cam_xmat[camera_id].shape}")
            
            # Try to render from camera
            try:
                renderer = mujoco.Renderer(model, height=480, width=640)
                renderer.update_scene(data, camera=camera_id)
                image = renderer.render()
                print(f"✓ Camera rendering successful! Image shape: {image.shape}")
            except Exception as e:
                print(f"✗ Camera rendering failed: {e}")
        else:
            print("✗ Camera not found!")
            
    except Exception as e:
        print(f"✗ IK positioning failed: {e}")

def test_interactive_viewer():
    """Launch interactive viewer to check the scene with dynamic IK positioning"""
    model, data = test_scene_loading()
    if model is None:
        return
    
    print("\nLaunching interactive viewer with dynamic IK positioning...")
    print("Instructions:")
    print("- Check if objects are properly sized (home-sized objects)")
    print("- Verify 3-4 objects are on the table")
    print("- Look at the camera view (should be on bracelet, facing upward)")
    print("- The arm will be positioned using dynamic target generation")
    print("- Press ESC to close the viewer")
    
    # Set up IK components
    generator = DatasetGenerator()
    arm_joint_indices, arm_actuator_indices = generator.find_arm_indices(model)
    ee_body_id = generator.get_end_effector_body_id(model)
    ik_solver = GaussNewtonIK(model, step_size=0.5, tol=0.01, max_iter=1000)
    
    # Generate a dynamic target position
    test_table_pos = [0.6, 0, 0.43]
    test_objects_center = [0.6, 0, 0.48]
    target_position = generator.generate_valid_target_position(test_table_pos, test_objects_center)
    print(f"Using dynamically generated target position: {target_position}")
    
    try:
        # Move arm to target position
        result_q, final_pos, error = generator.move_arm_to_target_ik(
            model, data, target_position, arm_joint_indices, ee_body_id, ik_solver
        )
        print(f"Arm positioned with error: {error:.4f}, converged: {ik_solver.converged}")
        
        # Launch viewer
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                mujoco.mj_step(model, data)
                viewer.sync()
                
    except Exception as e:
        print(f"Error positioning arm: {e}")
        # Launch viewer anyway with default position
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                mujoco.mj_step(model, data)
                viewer.sync()

def test_object_scaling():
    """Test if objects are properly scaled"""
    print("\nTesting object scaling...")
    
    generator = DatasetGenerator()
    print("Object scaling factors:")
    for obj_name, scale in generator.object_scales.items():
        print(f"  - {obj_name}: {scale}x")
    
    print("\nThese scales should make objects home-sized when placed on the table")

def test_dynamic_target_generation():
    """Test the dynamic target position generation system"""
    print("\nTesting dynamic target position generation...")
    
    generator = DatasetGenerator()
    
    print("Workspace limits:")
    for key, value in generator.workspace_limits.items():
        print(f"  - {key}: {value}")
    
    print(f"\nArm base position: {generator.arm_base_position}")
    print(f"Arm reach: {generator.arm_reach}m")
    
    # Test generating positions for different table configurations
    test_scenarios = [
        ([0.6, 0, 0.43], [0.6, 0, 0.48], "Default table"),
        ([0.65, 0.05, 0.43], [0.65, 0.05, 0.48], "Table right-forward"),
        ([0.55, -0.05, 0.43], [0.55, -0.05, 0.48], "Table left-back"),
    ]
    
    for table_pos, objects_center, description in test_scenarios:
        print(f"\n{description}:")
        print(f"  Table: {table_pos}, Objects: {objects_center}")
        
        for i in range(3):
            target = generator.generate_valid_target_position(table_pos, objects_center)
            score = generator.score_target_position(target, generator.arm_base_position, objects_center, table_pos)
            print(f"    {i+1}. {target} (score: {score:.3f})")

def test_dataset_generation():
    """Test generating a small validation dataset with episodic IK"""
    
    # Check if assets directory exists
    if not os.path.exists("dataset/scenes/assets"):
        print("Warning: 'assets' directory not found. Creating placeholder objects...")
        create_placeholder_objects()
    
    try:
        generator = DatasetGenerator(validation=True)  # validation mode
        existing_objects = generator.check_objects_exist()
        print(f"Automatically will do {len(existing_objects)} episodes (1 per object)")

        use_timestep_alignment = input("Use optional timestep alignment? (y/n): ").lower().startswith('y')
        alignment_type = 'camera'
        if use_timestep_alignment:
            alignment_type = input("Alignment type (camera/end_effector): ").strip().lower()
            if alignment_type not in ['camera', 'end_effector']:
                alignment_type = 'camera'
                
        dataset_info = generator.generate_dataset(
            num_episodes=len(existing_objects),
            distance_threshold=0.01,
            capture_images=True,
            use_timestep_alignment=use_timestep_alignment,
            timestep_alignment_type=alignment_type,
        )
        
        print("✓ Dataset generation test completed!")
        print(f"✓ Successful episodes: {dataset_info['successful_episodes']}")
        print(f"✓ Failed episodes: {dataset_info['failed_episodes']}")
        
    except Exception as e:
        print(f"✗ Dataset generation failed: {e}")

def real_dataset_generation():
    """Generate a real training dataset without validation"""
    print("\nReal dataset generation (episodic IK)...")
    
    # Check if assets directory exists
    if not os.path.exists("dataset/scenes/assets"):
        print("Warning: 'assets' directory not found. Creating placeholder objects...")
        create_placeholder_objects()
    
    try:
        generator = DatasetGenerator(validation=False)  # training mode
        
        existing_objects = generator.check_objects_exist()
        eps_per_object = 8

        num_episodes = int(input(f"Enter number of episodes to generate (default {len(existing_objects)*eps_per_object}): ") or len(existing_objects)*8)
        use_timestep_alignment = input("Use optional timestep alignment? (y/n): ").lower().startswith('y')
        alignment_type = 'camera'
        if use_timestep_alignment:
            alignment_type = input("Alignment type (camera/end_effector): ").strip().lower()
            if alignment_type not in ['camera', 'end_effector']:
                alignment_type = 'camera'
        
        dataset_info = generator.generate_dataset(
            num_episodes=num_episodes,
            distance_threshold=0.01,
            capture_images=True,
            use_timestep_alignment=use_timestep_alignment,
            timestep_alignment_type=alignment_type
        )
        
        print("✓ Dataset generation completed!")
        print(f"✓ Successful episodes: {dataset_info['successful_episodes']}")
        print(f"✓ Failed episodes: {dataset_info['failed_episodes']}")
        
    except Exception as e:
        print(f"✗ Dataset generation failed: {e}")

def test_ik_solver_performance():
    """Test IK solver performance with dynamically generated targets"""
    print("\nTesting GaussNewtonIK solver performance with dynamic targets...")
    
    model, data = test_scene_loading()
    if model is None:
        return
    
    generator = DatasetGenerator()
    arm_joint_indices, arm_actuator_indices = generator.find_arm_indices(model)
    ee_body_id = generator.get_end_effector_body_id(model)
    ik_solver = GaussNewtonIK(model, step_size=0.5, tol=0.01, max_iter=1000)
    
    # Generate test positions dynamically
    test_table_pos = [0.6, 0, 0.43]
    test_objects_center = [0.6, 0, 0.48]
    
    print("Generating dynamic target positions for performance test...")
    test_targets = []
    for i in range(10):
        target = generator.generate_valid_target_position(test_table_pos, test_objects_center)
        test_targets.append(target)
    
    print("Testing convergence for dynamically generated targets:")
    success_count = 0
    total_iterations = 0
    total_error = 0.0
    
    for i, target_pos in enumerate(test_targets):
        try:
            result_q, final_pos, error = generator.move_arm_to_target_ik(
                model, data, target_pos, arm_joint_indices, ee_body_id, ik_solver
            )
            
            if ik_solver.converged and error < 0.05:  # 5cm tolerance
                success_count += 1
                status = "✓"
            else:
                status = "✗"
            
            total_iterations += ik_solver.iterations
            total_error += error
            
            print(f"  {i+1:2d}. {status} Target: {[f'{x:.2f}' for x in target_pos]}, Error: {error:.4f}, "
                  f"Iter: {ik_solver.iterations}, Conv: {ik_solver.converged}")
                  
        except Exception as e:
            print(f"  {i+1:2d}. ✗ Target: {target_pos}, Error: {e}")
    
    print(f"\nGaussNewtonIK Performance Summary (Dynamic Targets):")
    print(f"  - Success rate: {success_count}/{len(test_targets)} "
          f"({100*success_count/len(test_targets):.1f}%)")
    print(f"  - Average iterations: {total_iterations/len(test_targets):.1f}")
    print(f"  - Average error: {total_error/len(test_targets):.4f}")

def create_placeholder_objects():
    """Create simple geometric objects if STL files aren't available"""
    print("Note: For full functionality, place your STL object files in 'dataset/scenes/assets' directory")
    print("The following objects are expected:")
    expected_objects = [
        "apple.stl", "banana.stl", "book.stl", "bottle.stl", "bowl.stl",
        "computer_mouse.stl", "cup.stl", "dinner_plate.stl", "minion.stl", 
        "robot.stl", "teddy_bear.stl", "vase.stl"
    ]
    for obj in expected_objects:
        print(f"  - {obj}")

def validate_file_structure():
    """Validate the expected file structure"""
    print("Validating file structure:")
    
    required_files = [
        "dataset/scenes/gen3.xml",
        "dataset/scenes/gen3_scene.xml",
        "dataset/scenes/assets/"
    ]
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} - MISSING!")
    
    # Check for STL files in assets
    assets_dir = "dataset/scenes/assets"
    if os.path.exists(assets_dir):
        stl_files = [f for f in os.listdir(assets_dir) if f.endswith('.stl')]
        print(f"Found {len(stl_files)} STL files in assets directory")
        if stl_files:
            print("Available objects:")
            for stl_file in sorted(stl_files):
                print(f"  - {stl_file}")

def test_workspace_visualization():
    """Test and visualize the workspace constraints"""
    print("\nTesting workspace constraints...")
    
    generator = DatasetGenerator()
    import math
    
    print("Testing hemisphere constraint (no left reaching):")
    
    # Generate many positions and check they're all in valid hemisphere
    valid_positions = []
    invalid_positions = []
    
    test_table_pos = [0.6, 0, 0.43]
    test_objects_center = [0.6, 0, 0.48]
    
    for i in range(20):
        target = generator.generate_valid_target_position(test_table_pos, test_objects_center)
        
        # Check if position is to the left of arm (negative Y relative to arm base)
        arm_base = generator.arm_base_position
        relative_y = target[1] - arm_base[1]
        
        if relative_y < -0.5:  # Too far left
            invalid_positions.append(target)
        else:
            valid_positions.append(target)
        
        # Calculate angle from forward direction
        relative_x = target[0] - arm_base[0]
        angle_deg = math.degrees(math.atan2(relative_y, relative_x))
        
        print(f"  {i+1:2d}. {target} -> angle: {angle_deg:5.1f}°, y_rel: {relative_y:5.2f}")
    
    print(f"\nResults:")
    print(f"  Valid positions (not too far left): {len(valid_positions)}")
    print(f"  Invalid positions (too far left): {len(invalid_positions)}")
    
    if len(invalid_positions) == 0:
        print("  ✓ All positions respect hemisphere constraint!")
    else:
        print("  ✗ Some positions violate hemisphere constraint")

def test_detailed_ik_comparison():
    """Compare the new implementation with expected results"""
    print("\nTesting detailed IK comparison with reference implementation...")
    
    model, data = test_scene_loading()
    if model is None:
        return
    
    generator = DatasetGenerator()
    arm_joint_indices, arm_actuator_indices = generator.find_arm_indices(model)
    ee_body_id = generator.get_end_effector_body_id(model)
    
    # Test the same positions that were failing in your original output
    problem_targets = [
        [0.6, 0.0, 0.8],   # Should achieve ~0.025 error
        [0.4, -0.3, 0.7],  # Was getting 0.3337 error - should be much better
        [0.8, 0.0, 0.7],   # Should achieve ~0.025 error
    ]
    
    print("Testing problematic targets with GaussNewtonIK:")
    print("-" * 60)
    
    for i, target_pos in enumerate(problem_targets, 1):
        # Initialize IK solver with exact parameters from reference
        ik_solver = GaussNewtonIK(model, step_size=0.5, tol=0.01, max_iter=1000)
        
        try:
            result_q, final_pos, error = generator.move_arm_to_target_ik(
                model, data, target_pos, arm_joint_indices, ee_body_id, ik_solver
            )
            
            print(f"Config {i}: Moving to target {target_pos}")
            print(f"    - Joint angles: {[f'{x:.2f}' for x in result_q]}")
            print(f"    - Final position: {[f'{x:.3f}' for x in final_pos]}")
            print(f"    - Error: {error:.4f}")
            print(f"    - Converged: {ik_solver.converged}")
            print(f"    - Iterations: {ik_solver.iterations}")
            
            # Compare with expected performance
            if error < 0.03:  # Should be much better than 0.3337
                print(f"    ✓ MUCH IMPROVED from original implementation!")
            elif error < 0.1:
                print(f"    ⚠ Better but still needs improvement")
            else:
                print(f"    ✗ Still having issues")
            print()
                
        except Exception as e:
            print(f"Config {i}: ✗ Failed with error: {e}")

def main():
    print("Testing MuJoCo Dataset Generation Setup (Dynamic Target Generation)")
    print("=" * 80)
    
    # Test 0: File structure validation
    validate_file_structure()
    print()
    
    # Test 1: Scene loading
    test_scene_loading()
    print()
    
    # Test 2: Object scaling
    test_object_scaling()
    print()
    
    # Test 3: Dynamic target generation (UPDATED)
    test_dynamic_target_generation()
    print()
    
    # Test 4: Workspace visualization (NEW)
    test_workspace_visualization()
    print()
    
    # Test 5: Camera view test with IK
    test_camera_view()
    print()
    
    # Test 6: Arm configurations with dynamic IK (UPDATED)
    test_arm_configurations_ik()
    print()
    
    '''# Test 7: Detailed IK comparison (keep existing)
    response = input("Would you like to test the specific failing cases from your original output? (y/n): ")
    if response.lower().startswith('y'):
        test_detailed_ik_comparison()
        print()
    
    # Test 8: IK solver performance with dynamic targets (UPDATED)
    response = input("Would you like to test IK solver performance with dynamic targets? (y/n): ")
    if response.lower().startswith('y'):
        test_ik_solver_performance()
        print()
    
    # Test 9: Interactive viewer (UPDATED)
    response = input("Would you like to launch the interactive viewer to check the scene? (y/n): ")
    if response.lower().startswith('y'):
        test_interactive_viewer()
        print()'''      # DO NOT DELETE THESE, BECAUSE THEYRE VALAUBLE TESTS
    
    # Ask user if they want to test dataset generation
    response = input("Would you like to test dataset generation with validation? (y/n): ")
    if response.lower().startswith('y'):
        test_dataset_generation()
    
    print("\nSetup testing complete!")
    print("\nKey improvements in the Dynamic Target Generation system:")
    print("1. ✓ No more predefined target positions - all generated dynamically")
    print("2. ✓ Hemisphere constraint - arm never reaches too far left")
    print("3. ✓ Workspace-aware positioning within arm's actual capabilities")
    print("4. ✓ Object-aware targeting for better camera alignment")
    print("5. ✓ Smart scoring system for optimal viewpoints")
    print("6. ✓ Table-aware positioning that adapts to different table locations")
    print("\nExpected improvements:")
    print("- All targets guaranteed to be within arm's physical reach")
    print("- Better camera alignment with objects on table")
    print("- More diverse and realistic arm poses")
    print("- No more failed IK due to unreachable targets")
    print("\nNext steps:")
    print("1. Add your STL object files to 'dataset/scenes/assets' directory")
    print("2. Run: python dataset_generator.py --num_scenes=400")
    print("3. Use --no_validation flag to skip interactive validation")
    print("4. Monitor workspace constraint compliance")

    # Ask user if they want to actually start dataset generation
    response = input("Would you like to start real dataset generation without validation? (y/n): ")
    if response.lower().startswith('y'):
        real_dataset_generation()

if __name__ == "__main__":
    main()


'''
action = displacement vector (x, y z - separate columns) --> basically the state
use pinch site for IK
directly put image into dataset (idk try svg, or array or something)
avoid self-collision with arm
use separate datasets for each episode/trajectory
make the function allow for multiple timesteps
each data row should be (timestep, image (current), new action, current/past action)
'''