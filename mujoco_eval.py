import time
import mujoco
import mujoco.viewer
import numpy as np
import torch
import os

from model import GaussianActor

# PD control function
def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd

# Function to transform a vector from world frame to body frame
def transform_to_body_frame(vector, quat):
    """
    Transforms a vector from the world frame to the body frame using a quaternion.
    """
    rotated_vector = np.zeros(3)
    quat_conj = np.zeros(4)
    # Get the conjugate of the quaternion for world-to-body frame transformation
    mujoco.mju_negQuat(quat_conj, quat)
    # Rotate the vector using the conjugate quaternion
    mujoco.mju_rotVecQuat(rotated_vector, vector, quat_conj)
    return rotated_vector

if __name__ == "__main__":
    # The path to the scene file you provided.
    MODEL_PATH = "go2_assets/scene.xml"

    # Load MuJoCo model and data
    if not os.path.exists(MODEL_PATH):
        print("Error: Model file not found. Please ensure you have placed both 'scene.xml' and 'go2.xml' in the same directory as this script.")
        exit()

    try:
        m = mujoco.MjModel.from_xml_path(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")
        exit()
        
    d = mujoco.MjData(m)

    # Set simulation parameters
    SIMULATION_DT = 1/200
    CONTROL_DECIMATION = 4
    m.opt.timestep = SIMULATION_DT

    # PD gains - tuned for Go2
    KP = 80.0
    KD = 1.5
    KPS = np.full(12, KP, dtype=np.float32)
    KDS = np.full(12, KD, dtype=np.float32)

    # Default joint angles
    DEFAULT_JOINT_POS = np.array([0.0, 0.67, -1.3, 0.0, 0.67, -1.3, 0.0, 0.67, -1.3, 0.0, 0.67, -1.3])

    # Observation and action scaling
    DOF_POS_SCALE = 1.0
    DOF_VEL_SCALE = 1.0
    ACTION_SCALE = 0.25

    # Initialize state variables
    num_actions = 12
    num_obs = 45
    obs = np.zeros(num_obs, dtype=np.float32)
    previous_actions = np.zeros(num_actions, dtype=np.float32)
    commands = np.array([0.5, 0.0, 0.0], dtype=np.float32)

    # Load policy
    try:
        actor = GaussianActor(num_obs, num_actions, [128, 128])
        actor.load_state_dict(torch.load("ppo_actor.pth"))
        actor.eval()
    except FileNotFoundError:
        print("Error: ppo_actor.pth file not found. Please provide the correct path.")
        exit()

    with mujoco.viewer.launch_passive(m, d) as viewer:
        start_time = time.time()
        counter = 0

        while viewer.is_running() and (time.time() - start_time) < 30:
            step_start = time.time()
            
            mujoco.mj_step(m, d)
            counter += 1

            if counter % CONTROL_DECIMATION == 0:
                quat = d.qpos[3:7]
                
                base_lin_vel_b = transform_to_body_frame(d.qvel[0:3], quat)
                base_ang_vel_b = transform_to_body_frame(d.qvel[3:6], quat)

                qj = d.qpos[7:]
                dqj = d.qvel[6:]

                qj_rel = (qj - DEFAULT_JOINT_POS) * DOF_POS_SCALE
                dqj_scaled = dqj * DOF_VEL_SCALE

                obs_parts = [
                    commands,
                    base_lin_vel_b,
                    base_ang_vel_b,
                    qj_rel,
                    dqj_scaled,
                    previous_actions
                ]
                obs = np.concatenate(obs_parts)
                obs_tensor = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0)

                with torch.no_grad():
                    actions = actor(obs_tensor).action.detach().numpy().squeeze()
                
                previous_actions = actions.copy()

                target_dof_pos = actions * ACTION_SCALE + DEFAULT_JOINT_POS

                torques = pd_control(target_dof_pos, qj, KPS, np.zeros_like(KDS), dqj, KDS)
                
                d.ctrl[:] = torques

            viewer.sync()
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)