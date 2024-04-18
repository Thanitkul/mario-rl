import os
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common import env_checker
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym.wrappers import FrameStack 
import cv2
from pathlib import Path
import datetime
import numpy as np
from gym.spaces import Box


class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        return np.dot(observation[..., :3], [0.2989, 0.5870, 0.1140])

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        return self.observation(observation)

class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        self.shape = (shape, shape) if isinstance(shape, int) else tuple(shape)
        self.observation_space = Box(low=0, high=255, shape=(1,) + self.shape, dtype=np.uint8)

    def observation(self, observation):
        from PIL import Image
        observation = Image.fromarray(observation)
        observation = observation.resize(self.shape)
        return np.array(observation, dtype=np.uint8)[np.newaxis, ...]

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        return self.observation(observation)



def create_environment():
    env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
    env = JoypadSpace(env, [["right"], ["right", "A"], ["NOOP"], ["down"], ["right", "A", "B"]])
    env = SkipFrame(env, skip=4)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=84)
    env = FrameStack(env, num_stack=4)
    return env

np.random.seed(np.random.randint(0, 2**31 - 1))
# Initialize environment with wrappers
env = make_vec_env(lambda: create_environment(), n_envs=1)

# Check the environment for compatibility with Stable Baselines
#check = env_checker.check_env(env, warn=True, skip_render_check=True)

# Define save directory
save_dir = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
save_dir.mkdir(parents=True, exist_ok=True)

# Initialize PPO model
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_supermario_tensorboard/")

# Callback for saving models
checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=str(save_dir),
                                         name_prefix='ppo_supermario')

# Function to save video of episodes using the environment's render method
def record_video(env, model, video_length=500, prefix='', video_folder='videos/'):
    os.makedirs(video_folder, exist_ok=True)
    images = []
    obs = env.reset()
    img = env.render(mode='rgb_array')
    for i in range(video_length):
        images.append(img)
        action, _ = model.predict(obs)
        obs, _, _, _ = env.step(action)
        img = env.render(mode='rgb_array')

    height, width, layers = images[0].shape
    video_name = f"{video_folder}/{prefix}_episode.mp4"
    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))
    for image in images:
        video.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    video.release()
    print(f"Video saved as {video_name}")

# Callback for evaluating and recording videos
eval_callback = EvalCallback(env, best_model_save_path=str(save_dir),
                             log_path=str(save_dir), eval_freq=10000,
                             deterministic=True, render=False)

# Training the model
model.learn(total_timesteps=250000, callback=[checkpoint_callback, eval_callback])

# Save the final model
model.save(save_dir / "final_ppo_supermario")

# Optionally, record a video from the final model
record_video(env, model, video_folder=str(save_dir))

# Close the environment
env.close()
