# GO2 Walk - Reinforcement Learning for Unitree Go2 Quadruped Locomotion

A reinforcement learning project for training and evaluating locomotion policies on the Unitree Go2 quadruped robot using NVIDIA Isaac Lab and MuJoCo simulation environments.

## Overview

This project implements Proximal Policy Optimization (PPO) to train a neural network policy that enables the Unitree Go2 robot to walk and follow velocity commands. The trained policies can be evaluated both in Isaac Lab (NVIDIA Omniverse) and transferred to MuJoCo for sim-to-sim validation.

## Project Structure

```
GO2_walk/
├── train.py                 # PPO training script for Isaac Lab
├── eval.py                  # Evaluation script for Isaac Lab
├── model.py                 # Neural network architecture (Actor-Critic)
├── env/                     # Custom Gymnasium environment
│   ├── __init__.py          # Environment registration
│   ├── go2_walk_cfg.py      # Environment configuration
│   └── go2_walk_env.py      # DirectRL environment implementation
├── policies/                # Trained policy checkpoints (.pth files)
├── mujoco_sim2sim/          # MuJoCo sim-to-sim transfer
│   ├── mujoco_eval.py       # MuJoCo evaluation script
│   ├── isaaclab_go2_manual_control.py
│   └── go2_assets/          # MuJoCo robot model files
│       ├── go2.xml
│       └── scene.xml
└── source/                  # Isaac Lab source (isaaclab, isaaclab_assets, etc.)
```

## Dependencies

- **NVIDIA Isaac Lab** (with Omniverse)
- **PyTorch**
- **Gymnasium**
- **MuJoCo** (for sim-to-sim transfer)
- **RLAlg** - Custom RL library providing:
  - `ReplayBuffer` and `compute_gae` for experience storage
  - `PPO` loss calculator
  - Neural network layers (`make_mlp_layers`, `GaussianHead`, `CriticHead`)

## Environment Details

### Observation Space (45 dimensions)

| Component | Dimensions | Description |
|-----------|------------|-------------|
| Commands | 3 | Target velocities (lin_vel_x, lin_vel_y, ang_vel_z) |
| Base Linear Velocity | 3 | Robot base linear velocity in body frame |
| Base Angular Velocity | 3 | Robot base angular velocity in body frame |
| Joint Positions | 12 | Relative joint positions (current - default) |
| Joint Velocities | 12 | Joint angular velocities |
| Previous Actions | 12 | Actions from the previous timestep |

### Action Space (12 dimensions)

- **Type**: Continuous joint position targets
- **Scaling**: Actions are scaled by `action_scale=0.25` and added to default joint positions
- **Control**: Position control via `set_joint_position_target()`

### Reward Function

The reward function consists of multiple weighted components and is largely based on [Federico Sarrocco's blog](https://federicosarrocco.com/blog/Making-Quadrupeds-Learning-To-Walk):

| Reward Component | Weight | Description |
|------------------|--------|-------------|
| `tracking_lin_vel` | 10.0 | Reward for tracking commanded linear velocity (x, y) |
| `tracking_ang_vel` | 10.0 | Reward for tracking commanded angular velocity (z) |
| `height_penalty` | 0.7 | Penalty for deviation from reference height (0.3m) |
| `lin_vel_z_penalty` | 2.0 | Penalty for vertical velocity |
| `orientation_penalty` | 0.7 | Penalty for roll/pitch deviation from upright |
| `pose_similarity` | 2.85 | Penalty for deviation from default joint pose |
| `action_rate_penalty` | 3.95 | Penalty for rapid action changes (smoothness) |

### Termination Conditions

- Roll or pitch angle exceeds ±40 degrees
- Episode timeout (20 seconds)

## Neural Network Architecture

### GaussianActor

```
Input (45) → MLP [128, 128] (SiLU) → GaussianHead → Action (12)
```

- Encoder: 2-layer MLP with SiLU activation and layer normalization
- Output: Gaussian distribution with learnable log-std (clamped to [-20, 2])

### Critic

```
Input (45) → MLP [128, 128] (SiLU) → CriticHead → Value (1)
```

## Training

### Configuration

Key hyperparameters in `train.py`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `num_envs` | 4096 | Number of parallel environments |
| `steps_per_rollout` | 25 | Steps collected before each update |
| `ppo_epoch` | 10 | PPO update epochs per rollout |
| `clip_param` | 0.2 | PPO clipping parameter |
| `lr` | 3e-4 | Learning rate |
| `gamma` | 0.99 | Discount factor |
| `gae_lambda` | 0.95 | GAE lambda |
| `entropy_coef` | 0.01 | Entropy bonus coefficient |

### Running Training

```bash
# Basic training
python train.py

# With headless mode (no rendering)
python train.py --headless

# Additional Isaac Lab launcher arguments
python train.py --help
```

The trained actor model is saved to `ppo_actor.pth` after training completes.

## Evaluation

### Isaac Lab Evaluation

```bash
# Run evaluation with 12 parallel environments
python eval.py
```

The evaluation script loads a pre-trained policy from `policies/`.

### MuJoCo Sim-to-Sim Transfer

```bash
python mujoco_sim2sim/mujoco_eval.py
```

This script:
1. Loads the trained Isaac Lab policy
2. Constructs observations matching the Isaac Lab format
3. Applies PD control with matching gains (Kp=25, Kd=0.5)
4. Handles joint ordering differences between Isaac Lab and MuJoCo

**Note**: Joint ordering differs between simulators. The mapping arrays `ISAAC_TO_MJ` and `MJ_TO_ISAAC` handle the conversion.

## Environment Configuration

Key settings in `env/go2_walk_cfg.py`:

```python
episode_length_s = 20.0      # Episode duration
decimation = 4               # Physics substeps per control step
action_scale = 0.25          # Action scaling factor
sim.dt = 1/200               # Physics timestep (200 Hz)
scene.num_envs = 4096        # Parallel environments
scene.env_spacing = 4.0      # Spacing between environments
```

### Policy Naming

Saved policies follow the naming convention indicating training duration and reward weights:
```
{hours}h_{param1}_{param2}_...pth
```

## References

- [Federico Sarrocco's blog](https://federicosarrocco.com/blog/Making-Quadrupeds-Learning-To-Walk)
- [NVIDIA Isaac Lab](https://isaac-sim.github.io/IsaacLab/)
- [Unitree Go2](https://www.unitree.com/go2/)
- [Proximal Policy Optimization (PPO)](https://arxiv.org/abs/1707.06347)
