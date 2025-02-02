import mujoco
import numpy as np
import os
from PIL import Image

# Directory to save depth maps
DEPTH_MAP_DIR = "depth_maps"
os.makedirs(DEPTH_MAP_DIR, exist_ok=True)

def save_depth_map(physics, step, filename):
    """
    Save the depth map of the current MuJoCo physics state as an image.

    Parameters:
    - physics: The MuJoCo physics object.
    - step: Current step index (used for naming files).
    - filename: Name of the output depth map file.
    """
    # Render depth map
    depth = physics.render(depth=True)

    # Normalize depth values to [0, 255]
    depth -= depth.min()
    depth /= 2 * depth[depth <= 1].mean()
    pixels = 255 * np.clip(depth, 0, 1)

    # Convert to uint8 image and save
    depth_image = Image.fromarray(pixels.astype(np.uint8))
    depth_image.save(os.path.join(DEPTH_MAP_DIR, filename))

def generate_depth_maps(model_path, actuator_index, actuator_min, actuator_max, actuator_step):
    """
    Simulates a MuJoCo environment and saves depth maps for each actuator position.

    Parameters:
    - model_path: Path to the MuJoCo XML model.
    - actuator_index: Index of the actuator to control.
    - actuator_min: Minimum actuator value.
    - actuator_max: Maximum actuator value.
    - actuator_step: Step size for the actuator values.
    """
    # Load MuJoCo model and data
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    # Iterate over actuator values and save depth maps
    step_index = 0
    for actuator_value in np.arange(actuator_min, actuator_max + actuator_step, actuator_step):
        print(f"Processing actuator value: {actuator_value:.2f}")
        
        # Set actuator position
        data.ctrl[actuator_index] = actuator_value
        mujoco.mj_step(model, data)

        # Save depth map
        save_depth_map(data, step_index, f"depth_step_{step_index:03d}.png")
        step_index += 1

    print("Depth map generation completed!")

# Example usage
if __name__ == "__main__":
    generate_depth_maps(
        model_path="ufactory_xarm7/scene.xml",
        actuator_index=0,
        actuator_min=-1.12,
        actuator_max=-0.9,
        actuator_step=0.01
    )
