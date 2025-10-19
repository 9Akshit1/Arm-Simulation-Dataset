import mujoco
import mujoco.viewer
import numpy as np
import time

def main():
    # Load the model (assuming the XML file is named 'gen3.xml')
    model = mujoco.MjModel.from_xml_path('dataset/scenes/gen3_scene.xml')
    data = mujoco.MjData(model)
    
    # Get joint names and their ranges
    joint_names = []
    joint_ranges = []
    
    for i in range(model.njnt):
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if joint_name:  # Skip unnamed joints
            joint_names.append(joint_name)
            # Get joint range from the model
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if model.jnt_limited[joint_id]:
                range_min = model.jnt_range[joint_id][0]
                range_max = model.jnt_range[joint_id][1]
                joint_ranges.append((range_min, range_max))
            else:
                joint_ranges.append((-3.14, 3.14))  # Default range for unlimited joints
    
    print(f"Found joints: {joint_names}")
    
    # Start the viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Reset to home position
        mujoco.mj_resetDataKeyframe(model, data, 0)  # Use the "home" keyframe
        mujoco.mj_step(model, data)
        viewer.sync()
        
        print("Starting joint movement demonstration...")
        print("Each joint will move for 3 seconds, then return to home position")
        
        # Move each joint one by one
        for i, (joint_name, (range_min, range_max)) in enumerate(zip(joint_names, joint_ranges)):
            print(f"\nMoving Joint {i+1}: {joint_name}")
            print(f"Range: [{range_min:.3f}, {range_max:.3f}] radians")
            
            # Reset to home position first
            mujoco.mj_resetDataKeyframe(model, data, 0)
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.5)  # Brief pause at home position
            
            # Move the joint through its range
            start_time = time.time()
            duration = 3.0  # 3 seconds per joint
            
            while time.time() - start_time < duration:
                # Calculate current time within the cycle
                t = (time.time() - start_time) / duration
                
                # Create a sinusoidal movement within the joint's range
                center = (range_min + range_max) / 2
                amplitude = (range_max - range_min) / 3  # Use 1/3 of the range for safety
                
                # Sinusoidal motion
                joint_angle = center + amplitude * np.sin(2 * np.pi * t)
                
                # Set the joint position
                data.ctrl[i] = joint_angle
                
                # Step the simulation
                mujoco.mj_step(model, data)
                viewer.sync()
                
                # Small delay to control animation speed
                time.sleep(0.01)
            
            # Return to home position
            mujoco.mj_resetDataKeyframe(model, data, 0)
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.5)  # Pause at home position
        
        print("\nDemo complete! Press Ctrl+C to exit or close the viewer window")
        
        # Keep the viewer open
        try:
            while viewer.is_running():
                mujoco.mj_step(model, data)
                viewer.sync()
                time.sleep(0.01)
        except KeyboardInterrupt:
            print("Exiting...")

if __name__ == "__main__":
    main()