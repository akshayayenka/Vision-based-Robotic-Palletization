import mujoco
import numpy as np

def compute_inverse_kinematics(model, data, target_position, alpha=0.1, tolerance=0.01, max_iterations=100):
    """
    Compute inverse kinematics using the Gradient Descent method.
    
    Parameters:
    - model: MuJoCo model object
    - data: MuJoCo data object
    - target_position: Desired end-effector position
    - alpha: Learning rate (step size)
    - tolerance: Positioning tolerance
    - max_iterations: Maximum iterations for convergence
    
    Returns:
    - Updated joint angles if successful, None otherwise
    """
    site_id = model.site_name2id("end_effector")
    q = np.copy(data.qpos[:7])  # Assuming a 7-DOF arm
    
    for _ in range(max_iterations):
        current_position = data.site_xpos[site_id]
        error = target_position - current_position
        
        if np.linalg.norm(error) < tolerance:
            return q
        
        jac = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jac, None, site_id)
        jac_reduced = jac[:, :7]
        
        gradient = alpha * jac_reduced.T @ error
        q += gradient
        
        # Ensure joint limits are respected
        q = np.clip(q, model.jnt_range[:, 0], model.jnt_range[:, 1])
        
        data.qpos[:7] = q
        mujoco.mj_forward(model, data)
    
    return None  # IK failed to converge

def pick_and_place(model, data, objects, pallets):
    ee_position = data.site_xpos[model.site_name2id("end_effector")]  # Get end-effector position
    
    while objects:
        nearest_idx, nearest_object = find_nearest_object(ee_position, objects)
        
        # Compute IK to reach the object using Gradient Descent
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

if __name__ == "__main__":
    xml_path = "ufactory_xarm7/scene.xml"
    model, data = load_model(xml_path)
    
    # Define scattered object positions
    objects = np.random.rand(15, 3) * [1.0, 1.0, 0]  # Random (x, y) positions, z=0
    
    # Define pallet positions
    pallets = np.array([[0.5, 0.5, 0], [0.5, -0.5, 0], [-0.5, 0.5, 0]])
    
    pick_and_place(model, data, objects.tolist(), pallets.tolist())
