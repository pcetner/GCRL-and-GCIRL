from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.logger import configure
import os

# Path to the pre-trained model
model_path = "D:/discrete/ppo_rocket_minion.zip"

# Reduced learning rate for retraining
reduced_learning_rate = 1e-4

# Create or load the environment
from last import CustomRocketMinionEnv  # Ensure `CustomRocketMinionEnv` is imported correctly from your resave file

def make_env():
    return CustomRocketMinionEnv()

# Create multiple parallel environments for faster training
n_envs = 4
vec_env = DummyVecEnv([make_env for _ in range(n_envs)])

# Load the pre-trained model
print(f"Loading pre-trained model from: {model_path}")
model = PPO.load(model_path, env=vec_env)

# Set up TensorBoard logging
log_dir = "./tensorboard_logs_retrain/"
os.makedirs(log_dir, exist_ok=True)
new_logger = configure(log_dir, ["stdout", "tensorboard"])
model.set_logger(new_logger)

# Update learning rate
model.learning_rate = 1e-4  

# Retraining with the new learning rate
print("Starting retraining with a reduced learning rate...")
model.learn(total_timesteps=2000000, log_interval=5)

# Save the retrained model
retrained_model_path = "D:/discrete/ppo_rocket_minion_retrained"
model.save(retrained_model_path)
print(f"Retrained model saved to: {retrained_model_path}.zip")
