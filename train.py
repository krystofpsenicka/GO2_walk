import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="PPO agent for Isaac Lab environments using RLAlg components.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Import from RLAlg library
from RLAlg.buffer.replay_buffer import ReplayBuffer, compute_gae
from RLAlg.alg.ppo import PPO as RLAlg_PPO_Loss_Calculator

from env.go2_walk_cfg import Go2EnvCfg
from model import GaussianActor, Critic

class PPOAgent:
    def __init__(self,
                 actor_critic_dims,
                 clip_param,
                 value_loss_coef,
                 entropy_coef,
                 lr,
                 epsilon,
                 max_grad_norm,
                 device):

        self.actor = GaussianActor(actor_critic_dims["obs_dim"], actor_critic_dims["action_dim"], actor_critic_dims["hidden_dims"]).to(device)
        self.critic = Critic(actor_critic_dims["obs_dim"], actor_critic_dims["hidden_dims"]).to(device)

        self.optimizer = optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr, eps=epsilon)

        self.clip_param = clip_param
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.device = device

    def get_action_and_value(self, obs):
        with torch.no_grad():
            pi, action, log_prob = self.actor(obs)
            value = self.critic(obs)
        return action, log_prob, value

    def update(self, rollout_buffer, ppo_epoch, num_mini_batch):
        batch_keys = ["observations", "actions", "log_probs", "rewards", "values", "returns", "advantages"]

        value_loss_epoch = 0
        action_loss_epoch = 0
        entropy_loss_epoch = 0

        for e in range(ppo_epoch):
            for batch in rollout_buffer.sample_batchs(batch_keys, num_mini_batch):
                states = batch["observations"].to(self.device)
                actions = batch["actions"].to(self.device)
                log_probs = batch["log_probs"].to(self.device)
                values = batch["values"].to(self.device)
                returns = batch["returns"].to(self.device)
                advantages = batch["advantages"].to(self.device)

                actor_loss, entropy_loss, _ = RLAlg_PPO_Loss_Calculator.compute_policy_loss(self.actor, log_probs, states, actions, advantages, self.clip_param)
                critic_loss = RLAlg_PPO_Loss_Calculator.compute_value_loss(self.critic, states, returns)
                
                loss = actor_loss + self.value_loss_coef * critic_loss - self.entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                value_loss_epoch += critic_loss.item()
                action_loss_epoch += actor_loss.item()
                entropy_loss_epoch += entropy_loss.item()


class Trainer:
    def __init__(self, observation_dim=45, action_dim=12, rollout_steps=25):
        self.cfg = Go2EnvCfg()

        self.env = gymnasium.make("Go2Walk-v0", cfg=self.cfg)

        self.device = self.env.unwrapped.device

        self.num_envs = self.cfg.scene.num_envs

        self.observation_dim = observation_dim
        self.action_dim = action_dim
        
        # PPO Hyperparameters
        self.steps_per_rollout = rollout_steps
        self.num_mini_batch = 5 * 4096 
        self.ppo_epoch = 5     
        self.clip_param = 0.2   
        self.value_loss_coef = 0.5 
        self.entropy_coef = 0.01   
        self.lr = 3e-4          
        self.eps = 1e-5         
        self.max_grad_norm = 1.0 
        self.gamma = 0.99       
        self.gae_lambda = 0.95  

        actor_critic_dims = {
            "obs_dim": observation_dim,
            "action_dim": action_dim,
            "hidden_dims": [128, 128] 
        }
        self.agent = PPOAgent(actor_critic_dims,
                              self.clip_param, self.value_loss_coef, self.entropy_coef,
                              self.lr, self.eps, self.max_grad_norm, self.device)

        # Initialize ReplayBuffer
        self.rollout_buffer = ReplayBuffer(self.num_envs, self.steps_per_rollout)
        self.rollout_buffer.create_storage_space("observations", [self.observation_dim])
        self.rollout_buffer.create_storage_space("actions", [self.action_dim])
        self.rollout_buffer.create_storage_space("log_probs")
        self.rollout_buffer.create_storage_space("rewards")
        self.rollout_buffer.create_storage_space("values")
        self.rollout_buffer.create_storage_space("dones")

    def train(self, num_iterations=1000):
        obs, _ = self.env.reset()

        for j in range(num_iterations):
            self.rollout_buffer.reset() # Clear buffer for new rollout

            # Collect data
            for step in range(self.steps_per_rollout):
                obs_policy = obs["policy"].to(self.device)
                action, log_prob, value = self.agent.get_action_and_value(obs_policy)

                next_obs, reward, terminate, timeout, info = self.env.step(action)
                
                done = (terminate | timeout)

                record = {
                    "observations": obs["policy"],
                    "actions": action,
                    "log_probs": log_prob.squeeze(-1),
                    "rewards": reward,
                    "values": value.squeeze(-1),
                    "dones": done.float() # Convert bool to float for consistency
                }

                self.rollout_buffer.add_records(record)

                obs = next_obs
                
                if "episode" in info:
                    rewards_finished_episodes = [r for r in info['episode']['r'] if r != 0]
                    if rewards_finished_episodes:
                        print(f"Iteration: {j}, Step: {step}, Mean Episode Reward: {np.mean(rewards_finished_episodes):.2f}")


            # After collecting rollout, compute returns and advantages
            with torch.no_grad():
                # Get the value for the last observation in the rollout
                last_obs_policy = obs["policy"].to(self.device)
                _, _, last_value = self.agent.get_action_and_value(last_obs_policy)

            returns, advantages = compute_gae(
                self.rollout_buffer.data["rewards"],
                self.rollout_buffer.data["values"],
                self.rollout_buffer.data["dones"],
                last_value.squeeze(-1), # last_value is (num_envs, 1), make it (num_envs,)
                self.gamma,
                self.gae_lambda
            )
                
            self.rollout_buffer.add_storage("returns", returns)
            self.rollout_buffer.add_storage("advantages", advantages)

            # Update the agent
            self.agent.update(self.rollout_buffer, self.ppo_epoch, self.num_mini_batch)

            print(f"Iteration {j+1}/{num_iterations} completed.")
        
        torch.save(self.agent.actor.state_dict(), "ppo_actor.pth")

def main():
    trainer = Trainer()
    trainer.train(500)
    trainer.env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()