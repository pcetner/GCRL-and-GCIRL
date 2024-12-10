import time
import pygame
import numpy as np
from stable_baselines3 import PPO
from last import CustomRocketMinionEnv
import cv2  # For saving the video

if __name__ == "__main__":
    # Create the environment
    env = CustomRocketMinionEnv()

    # Load the trained model from the specified path
    model_path = "D:/last/saved_models/model_step_1250000.zip"
    model = PPO.load(model_path)

    # Reset the environment
    obs, info = env.reset()

    # Video parameters
    video_filename = "verification.mp4"
    frame_rate = 175
    frame_width = env.width * 2  # Scaled width (based on rendering)
    frame_height = env.height * 2  # Scaled height (based on rendering)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
    out = cv2.VideoWriter(video_filename, fourcc, frame_rate, (frame_width, frame_height))

    try:
        for step in range(8000):
            # Predict the action
            action, _states = model.predict(obs, deterministic=True)
            
            # Take the action in the environment
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Render the environment
            env.render()
            
            # Capture the rendered frame
            frame = pygame.surfarray.array3d(env._window)
            frame = np.transpose(frame, (1, 0, 2))  # Convert from (width, height, channels) to (height, width, channels)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV
            out.write(frame)  # Write the frame to the video file

            # Slow down the rendering so you can see the movement
            #time.sleep(1 / frame_rate)

            # If the episode is terminated or truncated, reset the environment
            if terminated or truncated:
                obs, info = env.reset()

    except KeyboardInterrupt:
        print("Stopped by user.")
    
    finally:
        # Release video writer and close environment
        out.release()
        env.close()
        print(f"Video saved as {video_filename}")
