import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Random agent for Isaac Lab environments.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium
import torch

from env.go2_walk_cfg import Go2EnvCfg
from RLAlg.nn.steps import StochasticContinuousPolicyStep

from model import GaussianActor

class Trainer:
    def __init__(self):
        self.cfg = Go2EnvCfg()
        # ensure the number of envs is set low for testing
        self.cfg.scene.num_envs = 12
        self.env = gymnasium.make("Go2Walk-v0", cfg=self.cfg)
        self.observation_dim = self.cfg.observation_space
        self.action_dim = self.cfg.action_space

        self.device = self.env.unwrapped.device

        self.actor = GaussianActor(self.observation_dim, self.action_dim, [128, 128]).to(self.device)
        self.actor.load_state_dict(torch.load("ppo_actor.pth", map_location=self.device))
        self.actor.eval()

    
    def rollout(self):
        obs, info = self.env.reset()
        obs = obs["policy"].to(self.device)

        print(obs)

        for i in range(1000):
            with torch.no_grad():
                policy_step: StochasticContinuousPolicyStep = self.actor.forward(obs)
                action = policy_step.action
                #action = torch.rand_like(action, device=self.device)
            next_obs, reward, terminate, timeout, _ = self.env.step(action)

            print(reward)

            obs = next_obs["policy"].to(self.device)


def main():
    trainer = Trainer()

    trainer.rollout()

    trainer.env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()