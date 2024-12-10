import gymnasium as gym
from gymnasium.spaces import Box, Discrete
import numpy as np
import pygame
import math
import random
import matplotlib as plt
import os

class Minion:
    def __init__(self, x, y, thrust=0.01):
        self.pos = pygame.math.Vector2(x, y)
        # Initialize velocity with random values between -.1 and .1
        self.vel = pygame.math.Vector2(
            random.uniform(-.1, .1),
            random.uniform(-.1, .1)
        )
        self.thrust = thrust
        self.radius = 10

    def apply_action(self, angle):
        acceleration = pygame.math.Vector2(
            self.thrust * math.cos(angle),
            self.thrust * math.sin(angle),
        )
        self.vel += acceleration

    def update_position(self):
        self.pos += self.vel

    def enforce_bounds(self, width, height):
        if self.pos.x > width:
            self.pos.x = width
            self.vel.x *= -0.5
        elif self.pos.x < 0:
            self.pos.x = 0
            self.vel.x *= -0.5
        if self.pos.y > height:
            self.pos.y = height
            self.vel.y *= -0.5
        elif self.pos.y < 0:
            self.pos.y = 0
            self.vel.y *= -0.5


class Goal:
    def __init__(self, width, height):
        self.radius = 15
        self.pos = self._generate_random_position(width, height)

    def _generate_random_position(self, width, height):
        return pygame.math.Vector2(random.uniform(0, width), random.uniform(0, height))

    def regenerate(self, width, height):
        self.pos = self._generate_random_position(width, height)


class RocketMinionEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, width=800, height=600):
        super().__init__()
        self.width = width
        self.height = height
        self.max_distance = math.sqrt(width**2 + height**2)

        # Initialize Minion and Goal
        self.minion = Minion(x=random.uniform(0, width), y=random.uniform(0, height))
        self.goal = Goal(width, height)

        # Observation: [minion_x, minion_y, vel_x, vel_y, xdist_to_goal, ydist_to_goal]
        self.observation_space = Box(
            low=np.array([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf]),
            high=np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]),
            dtype=np.float32,
        )
        # Action space: Discrete angles
        self.action_space = Discrete(20)

    def _get_state(self):
        xdist_to_goal = self.goal.pos.x - self.minion.pos.x
        ydist_to_goal = self.goal.pos.y - self.minion.pos.y
        return np.array(
            [
                self.minion.pos.x,
                self.minion.pos.y,
                self.minion.vel.x,
                self.minion.vel.y,
                xdist_to_goal,
                ydist_to_goal,
            ],
            dtype=np.float32,
        )

    def step(self, action):
        self.last_action = action
        angle = action * (2 * math.pi / 20)
        self.minion.apply_action(angle)
        self.minion.update_position()
        self.minion.enforce_bounds(self.width, self.height)

        # Distance to goal
        distance = np.linalg.norm([
            self.goal.pos.x - self.minion.pos.x,
            self.goal.pos.y - self.minion.pos.y
        ])

        # Compute reward
        # No termination in base class - will be handled in subclass
        # Just a placeholder here: no termination, no truncation.
        # Reward is small negative based on distance
        reward = -distance * 0.01

        return self._get_state(), reward, False, False, {}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self.minion = Minion(x=self.width / 2, y=self.height / 2)
        self.goal = Goal(self.width, self.height)

        return self._get_state(), {}

    def render(self):
        if not hasattr(self, "_window"):
            pygame.init()
            self._window = pygame.display.set_mode((self.width * 2, self.height * 2))
            pygame.display.set_caption("Rocket Minion Simulation")

            self._render_surface = pygame.Surface((self.width, self.height))
            self._background_image = pygame.image.load("D:/last/background.png").convert()
            self._background_image = pygame.transform.scale(self._background_image, (self.width, self.height))
            self._minion_image = pygame.image.load("D:/last/minion.png").convert_alpha()
            self._minion_image = pygame.transform.scale(self._minion_image, (50, 60))
            self._coin_image = pygame.image.load("D:/last/coin.png").convert_alpha()
            self._coin_image = pygame.transform.scale(self._coin_image, (40, 40))

        # Draw background and goal
        self._render_surface.blit(self._background_image, (0, 0))
        coin_center = self.goal.pos - pygame.math.Vector2(15, 15)
        self._render_surface.blit(self._coin_image, (int(coin_center.x), int(coin_center.y)))

        # Smooth rotation for minion
        if not hasattr(self, "_current_angle"):
            self._current_angle = 0

        target_angle = self.last_action * (360 / self.action_space.n)  # Discrete action angle
        angle_difference = (target_angle - self._current_angle + 180) % 360 - 90  # Smallest rotation path
        self._current_angle += angle_difference * 0.1  # Smooth rotation with interpolation

        rotated_image = pygame.transform.rotate(self._minion_image, -self._current_angle)
        rotated_rect = rotated_image.get_rect(center=(int(self.minion.pos.x), int(self.minion.pos.y)))

        self._render_surface.blit(rotated_image, rotated_rect.topleft)

        scaled_surface = pygame.transform.scale(self._render_surface, (self.width * 2, self.height * 2))
        self._window.blit(scaled_surface, (0, 0))
        pygame.display.flip()


class CustomRocketMinionEnv(RocketMinionEnv):
    def __init__(self, width=800, height=600, max_steps=10000):
        super().__init__(width, height)
        self.goals_reached = 0
        self.max_distance = math.sqrt(width**2 + height**2)
        self.max_steps = max_steps
        self.current_step = 0

    def step(self, action):
        self.current_step += 1

        # Parent step gives base reward and no terminal flags
        state, base_reward, _, _, info = super().step(action)

        # Recompute distance after step
        distance = np.linalg.norm([
            self.goal.pos.x - self.minion.pos.x,
            self.goal.pos.y - self.minion.pos.y
        ])

        # Check for goal condition
        if distance < (self.minion.radius + self.goal.radius):
            # Goal reached
            self.goals_reached += 1
            reward = 100.0  # More modest reward
            terminated = False # CHANGE TO TRUE FOR TRAINING
            truncated = False
            #print(f"Minion reached the goal with velocity: {self.minion.vel}")
            # Regenerate goal for next episode
            self.goal.regenerate(self.width, self.height)
        else:
            # No goal yet, penalize distance
            # Use a small negative reward per step plus base distance penalty
            # so that moving closer to the goal is relatively less negative.
            step_penalty = -1
            reward = base_reward - step_penalty
            terminated = False
            truncated = (self.current_step >= self.max_steps)

        # Add episode info if ending
        if terminated or truncated:
            info["episode"] = {
                "r": reward,
                "l": self.current_step,
                "goals_reached": self.goals_reached
            }

        return state, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        self.goals_reached = 0
        self.current_step = 0
        return super().reset(seed=seed, options=options)


from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
import numpy as np

class ModelSaverCallback(BaseCallback):
    def __init__(self, save_path, save_freq, verbose=0):
        """
        Custom callback for saving the model at regular intervals.
        
        :param save_path: (str) The directory where models will be saved.
        :param save_freq: (int) Save the model every `save_freq` steps.
        :param verbose: (int) Verbosity level (0 or 1).
        """
        super(ModelSaverCallback, self).__init__(verbose)
        self.save_path = save_path
        self.save_freq = save_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            model_save_path = f"{self.save_path}/model_step_{self.n_calls}.zip"
            self.model.save(model_save_path)
            if self.verbose > 0:
                print(f"[INFO] Saved model at step {self.n_calls} to {model_save_path}")
        return True

class TensorBoardProgressCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(TensorBoardProgressCallback, self).__init__(verbose)
        self.episode_rewards = []  # Store cumulative episode rewards
        self.episode_lengths = []  # Store episode lengths
        self.all_rewards = []  # Store all rewards for plotting over all episodes
        self.all_lengths = []  # Store all lengths for plotting over all episodes

    def _save_plot(self, data, ylabel, filename):
        import os
        from matplotlib import pyplot as plt
        import numpy as np

        # Define the output directory
        output_dir = "D:/last"
        os.makedirs(output_dir, exist_ok=True)

        # Compute moving average
        window_size = 1000
        moving_avg = np.convolve(data, np.ones(window_size) / window_size, mode='valid')

        # Construct the full file path
        filepath = os.path.join(output_dir, filename)

        plt.figure()
        plt.plot(range(len(data)), data, alpha=0.5, label='Raw Data')
        plt.plot(range(len(moving_avg)), moving_avg, label='100-Point Moving Avg')
        plt.xlabel('Episodes')
        plt.ylabel(ylabel)
        plt.title(f'{ylabel} Over Episodes')
        plt.legend()
        plt.grid(True)
        plt.savefig(filepath)
        plt.close()

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
                self.episode_lengths.append(info["episode"]["l"])
                self.all_rewards.append(info["episode"]["r"])
                self.all_lengths.append(info["episode"]["l"])
                self.logger.record("rollout/episode_reward", info["episode"]["r"])
                self.logger.record("rollout/episode_length", info["episode"]["l"])

        if self.n_calls % 5000 == 0 and len(self.all_rewards) > 0:
            # Save plots for the current progress
            print(f"[INFO] Saving plots at step {self.n_calls}")
            self._save_plot(self.all_rewards, "Mean Reward", "mean_reward_full.png")
            self._save_plot(self.all_lengths, "Mean Length", "mean_length_full.png")

        return True


if __name__ == "__main__":
    # Check environment before training
    env = CustomRocketMinionEnv()
    check_env(env)

    def make_env():
        return CustomRocketMinionEnv()

    # Create multiple parallel environments for faster training
    n_envs = 16
    vec_env = DummyVecEnv([make_env for _ in range(n_envs)])

    # Configure TensorBoard logger
    log_dir = "./tensorboard_logs/"
    new_logger = configure(log_dir, ["stdout", "tensorboard"])

    # Define the PPO model
    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        device="cpu",
        learning_rate=1e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.99,
        ent_coef=0.01,
        tensorboard_log=log_dir
    )

    # Set the custom logger
    model.set_logger(new_logger)

    # Instantiate the progress callback
    progress_callback = TensorBoardProgressCallback(verbose=1)

    # Define the folder where models will be saved
    model_save_dir = "D:/last/saved_models"
    os.makedirs(model_save_dir, exist_ok=True)

    # Create the model saver callback
    save_freq = 50000  # Save the model every 5000 steps
    model_saver_callback = ModelSaverCallback(save_path=model_save_dir, save_freq=save_freq, verbose=1)

    # Combine your progress and model saver callbacks
    callback = [progress_callback, model_saver_callback]

    # Start training with the combined callback
    print("Starting PPO training with TensorBoard logging and periodic model saving...")
    model.learn(total_timesteps=20000000, log_interval=5, callback=callback)
