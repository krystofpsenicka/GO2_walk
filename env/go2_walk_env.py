import gymnasium
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import euler_xyz_from_quat

from .go2_walk_cfg import Go2EnvCfg

class Go2Env(DirectRLEnv):

    cfg: Go2EnvCfg

    def __init__(self, cfg, render_mode = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._actions = torch.zeros(self.num_envs, gymnasium.spaces.flatdim(self.single_action_space), device=self.device)
        self._previous_actions = torch.zeros_like(self._actions)
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)

        self.reward_weights = {
            "tracking_lin_vel": 10.0,      # Most important - command tracking
            "tracking_ang_vel": 10.0,      # Most important - command tracking
            "height_penalty": 0.7,
            "lin_vel_z_penalty": 2.0,
            "orientation_penalty": 0.7,
            "pose_similarity": 2.85,
            "action_rate_penalty": 3.95,
        }

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        self.scene.clone_environments(copy_from_source=False)

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def sample_commands(self, env_ids):
        self._commands[env_ids, 0] = torch.zeros_like(self._commands[env_ids, 0]).uniform_(-1.0, 1.0)
        self._commands[env_ids, 1] = torch.zeros_like(self._commands[env_ids, 0]).uniform_(-1.0, 1.0)
        self._commands[env_ids, 2] = torch.zeros_like(self._commands[env_ids, 0]).uniform_(-1.0, 1.0)

    def _pre_physics_step(self, actions):
        self._actions = actions.clone()
        self._processed_actions = self.cfg.action_scale * self._actions + self._robot.data.default_joint_pos

    def _apply_action(self):
        self._robot.set_joint_position_target(self._processed_actions)

    def _get_observations(self):
        self._previous_actions = self._actions.clone()
        
        # Commands
        commands_obs = self._commands

        # Base Linear Velocity (in base frame)
        base_lin_vel_obs = self._robot.data.root_lin_vel_b

        # Base Angular Velocity (in base frame)
        base_ang_vel_obs = self._robot.data.root_ang_vel_b

        # Joint Positions (relative to default)
        joint_pos_obs = self._robot.data.joint_pos - self._robot.data.default_joint_pos

        # Joint Velocities
        joint_vel_obs = self._robot.data.joint_vel

        # Previous Actions
        previous_actions_obs = self._previous_actions
        
        # Concatenate all observations
        observations = torch.cat(
            [
                commands_obs,
                base_lin_vel_obs,
                base_ang_vel_obs,
                joint_pos_obs,
                joint_vel_obs,
                previous_actions_obs,
            ],
            dim=-1,
        )

        return {"policy": observations}
        
    def _get_dones(self):
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # Get Euler angles from quaternion (assumed in radians)
        roll, pitch, yaw = euler_xyz_from_quat(self._robot.data.root_com_quat_w)

        roll_deg = torch.rad2deg(roll)
        roll_deg = (roll_deg + 180) % 360 - 180

        pitch_deg = torch.rad2deg(pitch)
        pitch_deg = (pitch_deg + 180) % 360 - 180
        
        # Element-wise termination condition
        terminate = (torch.abs(roll_deg) > 40) | (torch.abs(pitch_deg) > 40)
        

        return terminate, time_out
    
    def _reset_idx(self, env_ids):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        if len(env_ids) == self.num_envs:
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))
        
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0

        self.sample_commands(env_ids)

        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]

        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        extras = dict()

    def _get_rewards(self):
        
        # Initialize rewards tensor
        rewards = torch.zeros(self.num_envs, device=self.device)

        # 1. Command Tracking Rewards (most important)
        # Reward for tracking linear velocity in x and y
        lin_vel_error = torch.sum(torch.square(self._commands[:, :2] - self._robot.data.root_lin_vel_b[:, :2]), dim=1)
        # reward_tracking_lin_vel = torch.exp(-lin_vel_error / 0.25)  # Normalize error scale
        reward_tracking_lin_vel = torch.exp(-lin_vel_error)
        rewards += self.reward_weights["tracking_lin_vel"] * reward_tracking_lin_vel

        # Reward for tracking angular velocity in z
        ang_vel_error = torch.square(self._commands[:, 2] - self._robot.data.root_ang_vel_b[:, 2])
        reward_tracking_ang_vel = torch.exp(-ang_vel_error)  # Normalize error scale
        rewards += self.reward_weights["tracking_ang_vel"] * reward_tracking_ang_vel

        # # 2. Height penalty
        ref_height = 0.3
        height_error = torch.square(ref_height - self._robot.data.root_state_w[:, 2])
        height_penalty = -height_error
        rewards += self.reward_weights["height_penalty"] * height_penalty

        # # 3. Penalize linear velocity in z (vertical velocity)
        lin_vel_z_error = torch.square(self._robot.data.root_lin_vel_b[:, 2])
        lin_vel_z_penalty = -lin_vel_z_error
        rewards += self.reward_weights["lin_vel_z_penalty"] * lin_vel_z_penalty

        # # 5. Stability Reward - Penalize deviations from upright orientation
        roll, pitch, yaw = euler_xyz_from_quat(self._robot.data.root_link_quat_w)
        orientation_error = torch.square(roll) + torch.square(pitch)
        orientation_penalty = -orientation_error
        rewards += self.reward_weights["orientation_penalty"] * orientation_penalty

        # # 6. Pose similarity reward
        joint_pos_error = torch.sum(torch.square(self._robot.data.joint_pos - self._robot.data.default_joint_pos), dim=1)
        pose_similarity = -joint_pos_error
        rewards += self.reward_weights["pose_similarity"] * pose_similarity

        # 8. Action Regularization - Penalize large changes in actions
        action_rate_error = torch.sum(torch.square(self._actions - self._previous_actions), dim=1)
        action_rate_penalty = -action_rate_error
        rewards += self.reward_weights["action_rate_penalty"] * action_rate_penalty

        return rewards