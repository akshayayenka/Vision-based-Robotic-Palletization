import mujoco
import numpy as np
from inverse_kinematics import compute_inverse_kinematics  # Assuming this function is defined in ik.py

# Load Mujoco model
def load_model(xml_path):
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    return model, data

# Function to calculate Euclidean distance
def euclidean_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

# Nearest Neighbor Search
def find_nearest_object(ee_position, objects):
    distances = [euclidean_distance(ee_position, obj) for obj in objects]
    nearest_idx = np.argmin(distances)
    return nearest_idx, objects[nearest_idx]

# Pick and Place Sequence
def pick_and_place(model, data, objects, pallets):
    ee_position = data.site_xpos[model.site_name2id("end_effector")]  # Get end-effector position
    
    while objects:
        nearest_idx, nearest_object = find_nearest_object(ee_position, objects)
        
        # Compute IK to reach the object
        joint_positions = compute_inverse_kinematics(model, data, nearest_object)
        
        if joint_positions is not None:
            data.qpos[:len(joint_positions)] = joint_positions  # Apply IK result
            mujoco.mj_forward(model, data)
            
            print(f"Grasping object at {nearest_object}")
            
            # Determine the corresponding pallet
            pallet = pallets[len(objects) % len(pallets)]
            joint_positions_place = compute_inverse_kinematics(model, data, pallet)
            
            if joint_positions_place is not None:
                data.qpos[:len(joint_positions_place)] = joint_positions_place
                mujoco.mj_forward(model, data)
                print(f"Placing object at {pallet}")
        
        # Remove the picked object from the list
        objects.pop(nearest_idx)
        
        # Update end-effector position
        ee_position = data.site_xpos[model.site_name2id("end_effector")]

