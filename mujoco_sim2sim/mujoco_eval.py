#!/usr/bin/env python3
import os
import sys
import time
import numpy as np
import torch
import mujoco
import mujoco.viewer

# Ensure parent directory is on path so that model, RLAlg imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from model import GaussianActor
from RLAlg.nn.steps import StochasticContinuousPolicyStep

# ——— CONFIG matching UNITREE_GO2_CFG ———
KP = 25.0
KD = 0.5
EFFORT_LIMIT = 23.5

OBS_DIM = 45
ACTION_DIM = 12
ACTION_SCALE = 0.25

# Isaac ordering for joints used in config:
DEFAULT_JOINT_POS_ISAAC = np.array([
    0.1, 0.8, -1.5,
   -0.1, 0.8, -1.5,
    0.1, 1.0, -1.5,
   -0.1, 1.0, -1.5,
], dtype=np.float64)

ISAAC_TO_MJ = np.array([0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11], dtype=int)
MJ_TO_ISAAC = np.argsort(ISAAC_TO_MJ)

def build_obs(data, prev_actions_isaac, commands, default_joint_pos_isaac):
    commands_obs = np.array(commands, dtype=np.float32)

    base_cvel = np.array(data.body('base_link').cvel, dtype=np.float64)
    omega_w = base_cvel[0:3]
    v_w = base_cvel[3:6]

    quat_b_w = np.array(data.body('base_link').xmat.reshape(3, 3).T, dtype=np.float64)
    base_lin_vel_b = quat_b_w @ v_w
    base_ang_vel_b = quat_b_w @ omega_w

    qpos_mj = np.array(data.qpos[7:], dtype=np.float64)
    qvel_mj = np.array(data.qvel[6:], dtype=np.float64)

    qpos_isaac = qpos_mj[MJ_TO_ISAAC]
    qvel_isaac = qvel_mj[MJ_TO_ISAAC]

    joint_pos_rel = (qpos_isaac - default_joint_pos_isaac).astype(np.float32)
    joint_vel = qvel_isaac.astype(np.float32)

    prev_actions = np.array(prev_actions_isaac, dtype=np.float32)

    obs = np.concatenate([
        commands_obs,
        base_lin_vel_b.astype(np.float32),
        base_ang_vel_b.astype(np.float32),
        joint_pos_rel,
        joint_vel,
        prev_actions
    ], axis=0)

    return obs

def main():
    model = mujoco.MjModel.from_xml_path("mujoco_sim2sim/go2_assets/scene.xml")
    data = mujoco.MjData(model)
    model.opt.timestep = 1/200

    actor = GaussianActor(obs_dim=OBS_DIM, action_dim=ACTION_DIM, hidden_dims=[128,128])
    actor.load_state_dict(torch.load("policies/ppo_actor_latest_16h_10_10_0-7_2_0-6_2-45_3-55.pth", map_location=torch.device("cpu")))
    actor.eval()

    # set joint positions and velocities to proper starting values
    default_mj = DEFAULT_JOINT_POS_ISAAC[ISAAC_TO_MJ]
    data.qpos[7:] = default_mj.copy()
    data.qvel[6:] = np.full(12, 0, dtype = np.float64)
    # mujoco.mj_forward(model, data)

    prev_actions_isaac = np.zeros(ACTION_DIM, dtype=np.float32)
    commands = np.random.uniform(-1.0, 1.0, size=(3,)).astype(np.float32)

    commands = np.array([-0.3016,  0.8262,  0.1271]).astype(np.float32)
    print("Using command:", commands)

    kp_mj = np.full((ACTION_DIM,), KP, dtype=np.float64)[ISAAC_TO_MJ]
    kd_mj = np.full((ACTION_DIM,), KD, dtype=np.float64)[ISAAC_TO_MJ]

    try:
        act_min = np.array([model.actuator_ctrlrange[i, 0] for i in range(model.nu)], dtype=np.float64)
        act_max = np.array([model.actuator_ctrlrange[i, 1] for i in range(model.nu)], dtype=np.float64)
        has_ctrlrange = True
    except Exception:
        has_ctrlrange = False


    with mujoco.viewer.launch_passive(model, data) as viewer:

        for i in range(30000):

            obs = build_obs(data, prev_actions_isaac, commands, DEFAULT_JOINT_POS_ISAAC)
            obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)
            print(f"obs: {obs_tensor}")
            with torch.no_grad():
                step: StochasticContinuousPolicyStep = actor.forward(obs_tensor)
                action_raw = step.action.cpu().numpy().squeeze(0)

            print(f"action: {step.action}")
            action_raw = torch.rand_like(step.action).cpu().numpy().squeeze(0)
            target_pos_isaac = DEFAULT_JOINT_POS_ISAAC + ACTION_SCALE * action_raw
            target_pos_mj = target_pos_isaac[ISAAC_TO_MJ]

            print(f"processed action: {target_pos_isaac}")
            qpos_mj = np.array(data.qpos[7:], dtype=np.float64)
            qvel_mj = np.array(data.qvel[6:], dtype=np.float64)

            tau = KP * (target_pos_mj - qpos_mj) - KD * qvel_mj

            if has_ctrlrange and tau.shape[0] == act_min.shape[0]:
                tau = np.minimum(np.maximum(tau, act_min), act_max)
            else:
                tau = np.clip(tau, -EFFORT_LIMIT, EFFORT_LIMIT)

            # data.ctrl[:] = tau
            data.qpos[7:] = target_pos_mj

            for i in range(4):
                mujoco.mj_step(model, data)
            viewer.sync()

            prev_actions_isaac = action_raw.astype(np.float32)


if __name__ == "__main__":
    main()
