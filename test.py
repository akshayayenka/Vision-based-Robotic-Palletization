
import numpy as np
from src.mujoco_parser import MuJoCoParserClass
from src.PID import PID_ControllerClass
from src.ik_module import solve_IK
from scipy.spatial.transform import Rotation as R

def set_gripper(desired_q, option="open"):
    if option == "open":
        desired_q[7], desired_q[8] = np.pi, -np.pi
    elif option == "close":
        desired_q[7], desired_q[8] = 0.0, 0.0
    return desired_q

def get_z_rot_mat(z_deg):
    z_rad = np.deg2rad(z_deg)
    y_rad = np.deg2rad(3.0)
    x_rad = np.deg2rad(180.0)
    rpy = np.array([z_rad, y_rad, x_rad])
    rot = R.from_euler('zyx', rpy)
    return rot.as_matrix()

def get_q_from_ik(env):
    body_name = 'panda_eef'
    pre_grasp_p = np.array([0.3, 0.0, 0.2])  # Fixed object position with offset for pre-grasp
    pre_grasp_R = np.array([
        [6.123e-17, 0.9848, 0.1736],
        [1.0, 0.0, 0.0],
        [0.0, 0.1736, -0.9848]
    ])

    to_rot_angle = np.linspace(-90.0, 90.0, num=10, endpoint=True)
    rotate_eef_p_lst = []
    rotate_eef_R_lst = []
    move_down_p_1_lst = []
    move_down_R_1_lst = []
    move_down_p_2_lst = []
    move_down_R_2_lst = []

    for angle in to_rot_angle:
        rotate_eef_p = pre_grasp_p
        rotate_eef_R = get_z_rot_mat(angle)
        rotate_eef_p_lst.append(rotate_eef_p)
        rotate_eef_R_lst.append(rotate_eef_R)
        
        move_down_p_1 = rotate_eef_p - np.array([0.0, 0.0, 0.1])
        move_down_R_1 = rotate_eef_R
        move_down_p_1_lst.append(move_down_p_1)
        move_down_R_1_lst.append(move_down_R_1)

        move_down_p_2 = rotate_eef_p - np.array([0.0, 0.0, 0.12])
        move_down_R_2 = rotate_eef_R
        move_down_p_2_lst.append(move_down_p_2)
        move_down_R_2_lst.append(move_down_R_2)

    pre_grasp_q = solve_IK(env, max_tick=1000, p_trgt=pre_grasp_p, R_trgt=pre_grasp_R, body_name=body_name, is_render=False)
    rotate_eef_q_lst = [solve_IK(env, max_tick=1000, p_trgt=p, R_trgt=R, body_name=body_name, curr_q=pre_grasp_q, is_render=False) 
                        for p, R in zip(rotate_eef_p_lst, rotate_eef_R_lst)]
    move_down_q_1_lst = [solve_IK(env, max_tick=1000, p_trgt=p, R_trgt=R, body_name=body_name, curr_q=rotate_eef_q_lst[i], is_render=False) 
                         for i, (p, R) in enumerate(zip(move_down_p_1_lst, move_down_R_1_lst))]
    move_down_q_2_lst = [solve_IK(env, max_tick=1000, p_trgt=p, R_trgt=R, body_name=body_name, curr_q=rotate_eef_q_lst[i], is_render=False) 
                         for i, (p, R) in enumerate(zip(move_down_p_2_lst, move_down_R_2_lst))]
    return pre_grasp_q, rotate_eef_q_lst, move_down_q_1_lst, move_down_q_2_lst

def main():
    xml_path = 'ufactory_xarm7/scene.xml'
    env = MuJoCoParserClass(name='Xarm7', rel_xml_path=xml_path, VERBOSE=False)
    env.forward()

    env.init_viewer(viewer_title="Grasp Fixed Object", viewer_width=1600, viewer_height=900, viewer_hide_menus=False)
    env.update_viewer(cam_id=0)
    env.reset()

    PID = PID_ControllerClass(
        name='PID', dim=env.n_ctrl,
        k_p=800.0, k_i=20.0, k_d=100.0,
        out_min=env.ctrl_ranges[env.ctrl_joint_idxs, 0],
        out_max=env.ctrl_ranges[env.ctrl_joint_idxs, 1],
        ANTIWU=True
    )
    PID.reset()

    pre_grasp_q, rotate_eef_q_lst, move_down_q_1_lst, move_down_q_2_lst = get_q_from_ik(env)
    max_tick = 1000000
    task_sequence = [
        "pre_grasp", "rotate_eef_i0", "move_down_1", "grasp", 
        "pre_grasp_with_close", "rotate_eef_i1_with_close", 
        "move_down_2_with_close", "release"
    ]
    task_idx = 0
    rot_idx = 0

    while env.tick < max_tick:
        if env.tick % 1500 == 0:
            if task_idx >= len(task_sequence):
                task_idx = 0
                rot_idx = (rot_idx + 1) % len(rotate_eef_q_lst)
                if rot_idx == 0: break
            current_task = task_sequence[task_idx]
            task_idx += 1

        if current_task == "pre_grasp":
            desired_q = set_gripper(pre_grasp_q, option="open")
        elif current_task == "rotate_eef_i0":
            desired_q = set_gripper(rotate_eef_q_lst[rot_idx], option="open")
        elif current_task == "move_down_1":
            desired_q = move_down_q_1_lst[rot_idx]
        elif current_task == "grasp":
            desired_q = set_gripper(desired_q, option="close")
        elif current_task == "pre_grasp_with_close":
            desired_q = set_gripper(pre_grasp_q, option="close")
        elif current_task == "rotate_eef_i1_with_close":
            desired_q = set_gripper(rotate_eef_q_lst[(rot_idx + 1) % len(rotate_eef_q_lst)], option="close")
        elif current_task == "move_down_2_with_close":
            desired_q = set_gripper(move_down_q_2_lst[(rot_idx + 1) % len(move_down_q_2_lst)], option="close")
        elif current_task == "release":
            desired_q = set_gripper(desired_q, option="open")

        PID.update(x_trgt=desired_q)
        PID.update(t_curr=env.get_sim_time(), x_curr=env.get_q(joint_idxs=env.ctrl_joint_idxs), VERBOSE=False)
        torque = PID.out()
        env.step(ctrl=torque, ctrl_idxs=env.ctrl_joint_idxs)

        if env.tick % 3 == 0:
            env.render()

    env.close_viewer()
    print("Grasping task completed.")

if __name__ == "__main__":
    main()
