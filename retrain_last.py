import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import BaseCallback
from last import CustomRocketMinionEnv, TensorBoardProgressCallback, ModelSaverCallback

if __name__ == "__main__":
    # Check environment before training
    def make_env():
        return CustomRocketMinionEnv()

    # Create multiple parallel environments for faster training
    n_envs = 16
    vec_env = DummyVecEnv([make_env for _ in range(n_envs)])

    # Configure TensorBoard logger
    log_dir = "./tensorboard_logs/"
    new_logger = configure(log_dir, ["stdout", "tensorboard"])

    # Path to the pre-trained model
    pretrained_model_path = "D:/last/saved_models/retrain_no_init_2_vel/model_step_850000.zip"

    # Load the saved model
    print(f"[INFO] Loading model from {pretrained_model_path}")
    model = PPO.load(pretrained_model_path, env=vec_env, device="cpu")
    model.set_env(vec_env)

    # Set the custom logger
    model.set_logger(new_logger)

    # Instantiate the progress callback
    progress_callback = TensorBoardProgressCallback(verbose=1)

    # Define the folder where models will be saved
    model_save_dir = "D:/last/saved_models/lower_rand_vel/"
    os.makedirs(model_save_dir, exist_ok=True)

    # Create the model saver callback
    save_freq = 50000  # Save the model every 50000 steps
    model_saver_callback = ModelSaverCallback(save_path=model_save_dir, save_freq=save_freq, verbose=1)

    # Combine your progress and model saver callbacks
    callback = [progress_callback, model_saver_callback]

    # Start retraining the model with the combined callback
    print("Starting retraining of PPO model from step 2500000...")
    model.learn(total_timesteps=100000000, log_interval=5, callback=callback)
