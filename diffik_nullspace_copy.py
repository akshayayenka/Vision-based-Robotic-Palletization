import mujoco
import mujoco.viewer
import numpy as np
import time
#from scipy.integrate import simps
# Integration timestep in seconds. This corresponds to the amount of time the joint
# velocities will be integrated for to obtain the desired joint positions.
integration_dt: float = 0.1

# Damping term for the pseudoinverse. This is used to prevent joint velocities from
# becoming too large when the Jacobian is close to singular.
damping: float = 1e-4

# Gains for the twist computation. These should be between 0 and 1. 0 means no
# movement, 1 means move the end-effector to the target in one integration step. 
Kpos: float = 0.95
Kori: float = 0.95

# Whether to enable gravity compensation.
gravity_compensation: bool = True

# Simulation timestep in seconds.
dt: float = 0.002

# Nullspace P gain.
Kn = np.asarray([10.0, 10.0, 10.0, 10.0, 5.0, 5.0, 5.0])

# Maximum allowable joint velocity in rad/s.
max_angvel = 0.785


def main() -> None:
    assert mujoco.__version__ >= "3.1.0", "Please upgrade to mujoco 3.1.0 or later."

    # Load the model and data.
    model = mujoco.MjModel.from_xml_path("ufactory_xarm7/scene.xml")
    data = mujoco.MjData(model)

    # Enable gravity compensation. Set to 0.0 to disable.
    model.body_gravcomp[:17] = float(gravity_compensation)
    model.opt.timestep = dt

    # End-effector site we wish to control.
    site_name = "link_tcp"
    site_id = model.site(site_name).id
    print("site_id: ", site_id)

    # Get the dof and actuator ids for the joints we wish to control. These are copied
    # from the XML file. Feel free to comment out some joints to see the effect on
    # the controller.
    joint_names = [
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
        "joint7",
    ]
    dof_ids = np.array([model.joint(name).id for name in joint_names])
    actuator_ids = np.array([model.actuator(name).id for name in joint_names])
    print("dof_ids: ", dof_ids)

    # Initial joint configuration saved as a keyframe in the XML file.
    key_name = "home"
    key_id = model.key(key_name).id
    q0 = model.key(key_name).qpos
    #only till joints
    q01 = q0[:7]
    print("keyid:", key_id)
    print("qo:", q0)

    # Mocap body we will control with our mouse. *** commented testing purposes***
    #mocap_name = "target"
    #mocap_id = model.body(mocap_name).mocapid[0]
    # Define waypoints
    waypoints = [
    np.array([0.4, 0.3, 0.30]),   # Initial position (or another suitable starting point)
    np.array([0.3, 0.6, 0.30]), # Move 10 cm along the x-axis
    '''np.array([0.5, 0, 0.30]), # Move another 10 cm along the x-axis
    np.array([0.6, 0, 0.30])  # Final target along the x-axis'''
    ]
    current_waypoint_index = 0

    # Pre-allocate numpy arrays.
    jac = np.zeros((6, model.nv))
    diag = damping * np.eye(7)
    eye = np.eye(7)
    twist = np.zeros(6)
    site_quat = np.zeros(4)
    site_quat_conj = np.zeros(4)
    error_quat = np.zeros(4)
    Kvel= 0.5
    target_velocity = np.array([2.0, 0.0, 0.0])

    with mujoco.viewer.launch_passive(
        model=model,
        data=data,
        show_left_ui=False,
        show_right_ui=False,
    ) as viewer:
        # Reset the simulation.
        mujoco.mj_resetDataKeyframe(model, data, key_id)

        # Reset the free camera.
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        # Enable site frame visualization.
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE
        q11=[]
        while viewer.is_running():
            step_start = time.time()

            # Spatial velocity (aka twist).
            #dx = data.mocap_pos[mocap_id] - data.site(site_id).xpos
            current_waypoint = waypoints[current_waypoint_index]
            dx = current_waypoint - data.site(site_id).xpos

            # result = np.zeros((6, 1))
            # mujoco.mj_objectVelocity(model, data,mujoco.mjtObj.mjOBJ_GEOM, site_id, result, 0)
            # print("striker vel:", result)
            
            #dv = target_velocity - result[:3].flatten()
            if np.linalg.norm(dx) < 0.01:  # Tolerance of 1 cm to consider as reached
                current_waypoint_index += 1
                print("Reached waypoint", current_waypoint_index)
                if current_waypoint_index >= len(waypoints):
                    #current_waypoint_index = 0
                    return #if wave points are over then return 
            twist[:3] = Kpos * dx / integration_dt  #10.0 * dx + 1.0 * dv  #
            mujoco.mju_mat2Quat(site_quat, data.site(site_id).xmat)
            mujoco.mju_negQuat(site_quat_conj, site_quat)
            #mujoco.mju_mulQuat(error_quat, data.mocap_quat[mocap_id], site_quat_conj)
            mujoco.mju_quat2Vel(twist[3:], error_quat, 1.0)
            twist[3:] *= Kori / integration_dt

            # Jacobian.
            mujoco.mj_jacSite(model, data, jac[:3], jac[3:], site_id)

            # Damped least squares.
            jac1 = jac[:,:7]
            dq = np.linalg.solve(jac1.T @ jac1 + diag, jac1.T @ twist)
            
            
            #dq1= dq[:7]
            # print("dq", dq.shape)
            # print('jac size:', jac.shape)
            # print("q01", q0.shape)
            # print("dataqpos:", data.qpos[dof_ids])
            # Nullspace control biasing joint velocities towards the home configuration.
            
            # Add velocity control term##################################################################################
           # desired_velocity = np.array([0.5, 0.0, 0.0])  # Replace with your desired velocity
            # dq += Kvel * (desired_velocity - np.linalg.pinv(jac1) @ jac1 @ dq)

            # ##another 
           

            # Assuming desired_velocity, jac1, and dq are properly defined
            # Compute the current velocity using the Jacobian and joint velocities
            #current_velocity = jac1 @ dq

            # Compute the difference between the desired and current velocities
            #velocity_difference = desired_velocity - current_velocity

            # Multiply the difference by the transpose of the Jacobian to get the joint velocity change
            #joint_velocity_change = np.linalg.pinv(jac1) @ velocity_difference

            # Scale the joint velocity change by a factor Kvel and add it to the current joint velocities dq
            #dq += Kvel * joint_velocity_change

            dq += (eye - np.linalg.pinv(jac1) @ jac1) @ (Kn * (q0[:7] - data.qpos[dof_ids]))

            # Clamp maximum joint velocity.
            dq_abs_max = np.abs(dq).max()
            if dq_abs_max > max_angvel:
                dq *= max_angvel / dq_abs_max

            # Integrate joint velocities to obtain joint positions.
            q = data.qpos.copy()  # Note the copy here is important.
            #print("qpos shape:", q)
            #print("nv", model.nv)
            # print("dq", dq.shape)
            # adding zeros in dq to match the size of nv
            #print("dq shape:", dq.shape)
            dq1 = np.append(dq, np.zeros(12))
            #print("dq1 shape:", dq1.shape)
            mujoco.mj_integratePos(model, q, dq1, integration_dt)
            np.clip(q[:14], *model.jnt_range.T, out=q[:14])

            # Set the control signal and step the simulation.
            data.ctrl[7] = 248
            #position     :: to uncommnet this you need to enable position actuators in xml file       
            #data.ctrl[actuator_ids] = q[dof_ids]
            #velocity
            data.ctrl[actuator_ids] = dq1[dof_ids]     # to uncomment this you need to enable velcoity actuators in xml file.
            mujoco.mj_step(model, data)
           #print("vel of striker:", data.site(site_id).cvel)
            #print("striker vel:", data.cvel[7])
            # result = np.zeros((6, 1))
            # mujoco.mj_objectVelocity(model, data,mujoco.mjtObj.mjOBJ_GEOM, 3, result, 0)
            # print(result)
            # get the mass of site
            print('After action: ',data.site(site_id).xpos)
            print('Expected position: ',waypoints[current_waypoint_index])


            viewer.sync()
            time_until_next_step = dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()
