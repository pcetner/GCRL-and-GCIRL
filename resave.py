import os
from stable_baselines3 import PPO

# Path to existing model file
existing_model_path = "ppo_rocket_minion.zip"

# Load the existing model
print(f"Loading model from: {existing_model_path}")
model = PPO.load(existing_model_path)

# Specify the folder to save the model
model_folder = "D:/last"
os.makedirs(model_folder, exist_ok=True)  # Create the folder if it doesn't exist

# Save the model to the specified folder
model_path = os.path.join(model_folder, "ppo_rocket_minion")
print(f"Saving the model to: {model_path}.zip")
model.save(model_path)

print(f"Model successfully saved at: {model_path}.zip")
