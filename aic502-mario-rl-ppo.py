import gym
import gym_super_mario_bros
import gym.version
from gym.wrappers import FrameStack, GrayScaleObservation, ResizeObservation
from nes_py.wrappers import JoypadSpace
import numpy as np
from gym import spaces
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv, SubprocVecEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecVideoRecorder



import cv2

print(gym.__version__)


class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        """Return only every `skip`-th frame"""
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        """Repeat action, and sum reward"""
        total_reward = 0.0
        for i in range(self._skip):
            # Accumulate reward and repeat the same action
            obs, reward, done, trunk, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, trunk, info
    
class RewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.previous_score = 0
        self.coins_collected = 0

    def step(self, action):
        obs, reward, done, trunk ,info = self.env.step(action)
        current_score = info.get('score', 0)
        current_coins = info.get('coins', 0)

        
        # Add to reward for score increase
        if current_score > self.previous_score:
            reward += 10
        # Add to reward for collecting coins
        if current_coins > self.coins_collected:
            reward += 5
        
        # Update the previous values
        self.previous_score = current_score
        self.coins_collected = current_coins

        return obs, reward, done, trunk, info
    
class ReshapeObservations(gym.ObservationWrapper):
    def __init__(self, env, new_shape):
        super().__init__(env)
        self.new_shape = new_shape
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=new_shape, dtype=np.uint8)

    def observation(self, observation):
        return observation.reshape(self.new_shape)


class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        self.logger.record('reward', np.mean(self.training_env.get_attr('reward')))
        return True 
    





def record_video(env_id, model, video_length=500, video_folder='videos/', video_name='mario_video'):
    env = DummyVecEnv([lambda: gym.make(env_id)])
    env = VecVideoRecorder(env, video_folder,
                           record_video_trigger=lambda x: x == 0, video_length=video_length,
                           name_prefix=video_name)

    obs = env.reset()
    for _ in range(video_length):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _ = env.step(action)
        if done:
            obs = env.reset()

    env.close()


if __name__ == '__main__':
    '''
    if gym.__version__ < '0.26':
        env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0")
        print(1)
    else:
        env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0", render_mode='rgb_array', apply_api_compatibility=True)
        print(2)
    '''

    #env = JoypadSpace(env, [["right"], ["right", "A"],  ['NOOP'], ['down'], ['right', 'A', 'B']])

    #env = SkipFrame(env, skip=4)
    #env = GrayScaleObservation(env, keep_dim=True)
    #env = ResizeObservation(env, shape=84)
    #env = RewardWrapper(env)
    
    env = SubprocVecEnv([lambda: gym_super_mario_bros.make("SuperMarioBros-1-1-v0", render_mode='rgb_array', apply_api_compatibility=True) for i in range(4)])

    print(env.observation_space.shape)

    #env = VecFrameStack(env, n_stack=4, channels_order='last')

    print(env.observation_space.shape)

    env.reset()
    print(env.observation_space.shape)
    obs, reward, done, info = env.step(actions=0)
    print(f"{obs.shape},\n {reward},\n ,\n {info}")
    

    model = PPO('CnnPolicy', env, verbose=1, learning_rate=1e-5, n_steps=512) 
    model.learn(total_timesteps=2000000, callback=TensorboardCallback())
    model.save("ppo_mario")

    env = VecFrameStack(env, n_stack=4)
    record_video("SuperMarioBros-1-1-v0", model, video_length=500, video_name='ppo_mario')