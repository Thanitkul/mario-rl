import gym.version
import torch
print(torch.__version__) # LOG: runnable with version 2.1.0+cu121 / 2.1.0+cpu / 2.1.1+cu121
from torch import nn
from torchvision import transforms as T
from PIL import Image
import numpy as np
from pathlib import Path
from collections import deque
import random, datetime, os, copy

# Gym is an OpenAI toolkit for RL
import gym
from gym.spaces import Box
from gym.wrappers import FrameStack

# NES Emulator for OpenAI Gym
from nes_py.wrappers import JoypadSpace

from gym.wrappers.monitoring.video_recorder import VideoRecorder

# Super Mario environment for OpenAI Gym
import gym_super_mario_bros

from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage

import matplotlib.pyplot as plt
print(gym.__version__)



if gym.__version__ < '0.26':
    env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0", new_step_api=True)
    print(1)
else:
    env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0", render_mode='rgb_array', apply_api_compatibility=True)
    print(2)

SAVE_PATH = '/mario_net_last.chkpt'

# Limit the action-space to
#   0. walk right
#   1. jump rightp
#   2. go down 
#   3. jump-run right 
env = JoypadSpace(env, [["right"], ["right", "A"], ['down'], ['right', 'A', 'B']])

env.reset()
next_state, reward, done, trunc, info = env.step(action=0)
print(f"{next_state.shape},\n {reward},\n {done},\n {info}")


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

    
class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def permute_orientation(self, observation):
        # permute [H, W, C] array to [C, H, W] tensor
        observation = np.transpose(observation, (2, 0, 1))
        observation = torch.tensor(observation.copy(), dtype=torch.float)
        return observation

    def observation(self, observation):
        observation = self.permute_orientation(observation)
        transform = T.Grayscale()
        observation = transform(observation)
        return observation


class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)

        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        transforms = T.Compose(
            [T.Resize(self.shape, antialias=True), T.Normalize(0, 255)]
        )
        observation = transforms(observation).squeeze(0)
        return observation
    
class RewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.previous_score = 0
        self.coins_collected = 0
        self.previous_x_position = 0
        self.progress_normalization_factor = 20  # Example normalization factor


    def step(self, action):
        obs, reward, done, trunk ,info = self.env.step(action)
        current_score = info.get('score', 0)
        current_coins = info.get('coins', 0)
        current_x_position = info.get('x_pos', 0)

        # Normalize the reward for moving forward
        forward_progress = current_x_position - self.previous_x_position
        normalized_progress_reward = forward_progress / self.progress_normalization_factor
        reward += normalized_progress_reward
        
        # Add to reward for score increase
        if current_score > self.previous_score:
            reward += 10
        # Add to reward for collecting coins
        if current_coins > self.coins_collected:
            reward += 5

        # Penalty for stagnation (no progress)
        if forward_progress == 0:
            reward -= 0.1  # Small penalty for no movement forward
                
        # Update the previous values
        self.previous_score = current_score
        self.coins_collected = current_coins
        self.previous_x_position = current_x_position

        return obs, reward, done, trunk, info


# Apply Wrappers to environment
env = SkipFrame(env, skip=4)
env = GrayScaleObservation(env)
env = ResizeObservation(env, shape=84)
env = RewardWrapper(env)
if gym.__version__ < '0.26':
    env = FrameStack(env, num_stack=4, new_step_api=True)
else:
    env = FrameStack(env, num_stack=4)  

state = env.reset()

# if gym.__version__ > '0.26': render got no rgb_array arg
# img = env.render("rgb_array").copy() # IMPORTANT: render keep modifying the same object
img = env.render().copy() # IMPORTANT: render keep modifying the same object

print(img.shape)

plt.imshow(img)


def first_if_tuple(x):
    return x[0] if isinstance(x, tuple) else x
print(env.action_space)
print(env.action_space.n)

print(first_if_tuple(state).shape)
print(env.action_space.n)

print("HERE")


state = env.reset()
next_state, reward, done, trunc, info = env.step(action=1)
next_state, reward, done, trunc, info = env.step(action=1)
next_state, reward, done, trunc, info = env.step(action=1)
print(f"{first_if_tuple(next_state).shape},\n {reward},\n {done},\n {info}")
img2 = env.render()
print(img2.shape)
plt.imshow(img2)

np.any(img != img2)

class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.n_entries = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def add(self, p, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[data_idx])

    def total_priority(self):
        return self.tree[0]  # the root
    
class PERBuffer:
    epsilon = 0.01  # small amount to avoid zero priority
    alpha = 0.6     # [0..1] convert the importance of TD error to priority
    beta = 0.4      # importance-sampling, from initial value increasing to 1
    beta_increment_per_sampling = 0.001

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def _get_priority(self, error):
        # Ensure error is a scalar by taking the absolute value and raising it to the power of alpha
        # If error is a tensor, make sure it is moved to CPU and converted to numpy
        if isinstance(error, torch.Tensor):
            error = error.cpu().detach().numpy()
            if error.size == 1:
                error = error.item()  # Convert single-element tensor to scalar
            else:
                error = error.mean().item()  # Reduce tensor to mean and then convert to scalar
        return (abs(error) + self.epsilon) ** self.alpha

    def add(self, error, sample):
        p = self._get_priority(error)
        self.tree.add(p, sample)

    def sample(self, n):
        batch = []
        segment = self.tree.total_priority() / n
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)
            s = np.random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            sampling_probabilities = p / self.tree.total_priority()
            is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
            batch.append((idx, data, is_weight))
        return batch

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)


class Mario:
    def __init__():
        pass

    def act(self, state):
        """Given a state, choose an epsilon-greedy action"""
        pass

    def cache(self, experience):
        """Add the experience to memory"""
        pass

    def recall(self):
        """Sample experiences from memory"""
        pass

    def learn(self):
        """Update online action value (Q) function with a batch of experiences"""
        pass

class Mario:
    def __init__(self, state_dim, action_dim, save_dir):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.save_dir = save_dir

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Mario's DNN to predict the most optimal action - we implement this in the Learn section
        self.net = MarioNet(self.state_dim, self.action_dim).float()
        self.net = self.net.to(device=self.device)

        self.exploration_rate = 1
        self.exploration_rate_decay = 0.99999975
        self.no_flag_hit_min_exploration_rate_min = 0.3
        self.flag_hit_min_exploration_rate_min = 0.2
        self.flag_hit = False  # Flag to check if flag has been reached
        self.curr_step = 0

        self.save_every = 50000  # no. of steps of experiences between saving Mario Net

        self.use_cuda = torch.cuda.is_available()

    def act(self, state):
        """
    Given a state, choose an epsilon-greedy action and update value of step.

    Inputs:
    state(``LazyFrame``): A single observation of the current state, dimension is (state_dim)
    Outputs:
    ``action_idx`` (``int``): An integer representing which action Mario will perform
    """
        # EXPLORE
        if np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(self.action_dim)

        # EXPLOIT
        else:
            state = state[0].__array__() if isinstance(state, tuple) else state.__array__()
            state = torch.tensor(state, device=self.device).unsqueeze(0)
            action_values = self.net(state, model="online")
            action_idx = torch.argmax(action_values, axis=1).item()

        if self.flag_hit:
            min_rate = self.flag_hit_min_exploration_rate_min
        else:
            min_rate = self.no_flag_hit_min_exploration_rate_min
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(min_rate, self.exploration_rate)


        # increment step
        self.curr_step += 1
        return action_idx
    
class Mario(Mario):  # subclassing for continuity
    def __init__(self, state_dim, action_dim, save_dir):
        super().__init__(state_dim, action_dim, save_dir)
        self.memory = PERBuffer(100000)  # Using PERBuffer instead of simple ReplayBuffer
        self.max_priority = 1.0  # Start with a high priority for the first experience
        self.batch_size = 32

    def cache(self, state, next_state, action, reward, done):
        """
        Store the experience to self.memory (replay buffer)

        Inputs:
        state (``LazyFrame``),
        next_state (``LazyFrame``),
        action (``int``),
        reward (``float``),
        done(``bool``))
        """
        
        state = first_if_tuple(state).__array__()
        next_state = first_if_tuple(next_state).__array__()

        if info.get("flag_get"):  # Check if flag is reached
            print("Flag reached")
            if not self.flag_hit:
                self.flag_hit = True  # Update flag_hit status on first successful flag hit


        state = torch.tensor(state)
        next_state = torch.tensor(next_state)
        action = torch.tensor([action])
        reward = torch.tensor([reward])
        done = torch.tensor([done])

        experience = (state, next_state, action, reward, done)
        self.memory.add(self.max_priority, experience)  # Store with max priority initially


    def recall(self):
        """
        Retrieve a batch of experiences from memory
        """

        # batch = random.sample(self.memory, self.batch_size)
        # state, next_state, action, reward, done = map(torch.stack, zip(*batch))

        batch = self.memory.sample(self.batch_size)
        state, next_state, action, reward, done, weights, indices = map(np.array, zip(*[(s[1][0], s[1][1], s[1][2], s[1][3], s[1][4], s[2], s[0]) for s in batch]))

        weights = torch.tensor(weights, dtype=torch.float, device=self.device)

        return state, next_state, action, reward, done, weights, indices

class MarioNet(nn.Module):
    """mini CNN structure
  input -> (conv2d + relu) x 3 -> flatten -> (dense + relu) x 2 -> output
  """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        c, h, w = input_dim

        if h != 84:
            raise ValueError(f"Expecting input height: 84, got: {h}")
        if w != 84:
            raise ValueError(f"Expecting input width: 84, got: {w}")

        self.online = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )

        self.target = copy.deepcopy(self.online)

        # Q_target parameters are frozen.
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, input, model):
        if not isinstance(input, torch.Tensor):
            input = torch.tensor(input, device=self.device, dtype=torch.float)
        else:
            input = input.to(device=self.device, dtype=torch.float)
        if model == "online":
            return self.online(input)
        elif model == "target":
            return self.target(input)
        
class Mario(Mario):
    def __init__(self, state_dim, action_dim, save_dir):
        super().__init__(state_dim, action_dim, save_dir)
        self.gamma = 0.9

    def td_estimate(self, state, action):
        current_Q = self.net(state, model="online")[
            np.arange(0, self.batch_size), action
        ]  # Q_online(s,a)
        return current_Q

    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        reward = torch.tensor(reward, device=self.device).float()
        done = torch.tensor(done, device=self.device).float()
        next_state = torch.tensor(next_state, device=self.device).float()
        next_state_Q = self.net(next_state, model="online")
        best_action = torch.argmax(next_state_Q, axis=1)
        next_Q = self.net(next_state, model="target")[
            np.arange(0, self.batch_size), best_action
        ]
        return (reward + (1 - done.float()) * self.gamma * next_Q).float()
    
class Mario(Mario):
    def __init__(self, state_dim, action_dim, save_dir):
        super().__init__(state_dim, action_dim, save_dir)
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=0.00025)
        self.loss_fn = torch.nn.SmoothL1Loss()

    def update_Q_online(self, td_estimate, td_target):
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_Q_target(self):
        self.net.target.load_state_dict(self.net.online.state_dict())


class Mario(Mario):
    def save(self, save_path):
        if save_path is None:
            try:
                save_path = os.path.join(self.save_dir, f"mario_net_{self.curr_step}.chkpt")
            except AttributeError:
                save_path = os.path.join(self.save_dir, f"mario_net_{self.curr_step}.chkpt")
        else:
            save_path = os.path.join(save_path, f"mario_net_{self.curr_step}.chkpt")
        print(f"Saving MarioNet at {save_path}")
        # Only save the serializable parts
        torch.save({
            'model': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'exploration_rate': self.exploration_rate
        }, save_path)
        print(f"MarioNet saved to {save_path} at step {self.curr_step}")



    def load(self, load_path):
        if not os.path.exists(load_path):
            print(f"Load path {load_path} does not exist. Starting from scratch.")
            return

        ckp = torch.load(load_path, map_location=('cuda' if torch.cuda.is_available() else 'cpu'))
        exploration_rate = ckp.get('exploration_rate')
        state_dict = ckp.get('model')

        print(f"Loading model at {load_path} with exploration rate {exploration_rate}")
        self.net.load_state_dict(state_dict)
        self.exploration_rate = exploration_rate
        self.optimizer.load_state_dict(ckp.get('optimizer_state_dict'))
        # self.memory = ckp.get('memory')  # Uncomment if you want to load memory as well


class Mario(Mario):
    def __init__(self, state_dim, action_dim, save_dir, load_path=None):
        super().__init__(state_dim, action_dim, save_dir)
        self.burnin = 1e4  # min. experiences before training
        self.learn_every = 3  # no. of experiences between updates to Q_online
        self.sync_every = 1e4  # no. of experiences between Q_target & Q_online sync

         # Only attempt to load if a valid load_path is provided
        if load_path and os.path.exists(load_path):
            self.load(load_path)
            
    def learn(self, save_path):
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()

        if self.curr_step % self.save_every == 0:
            self.save(save_path=save_path)

        if self.curr_step < self.burnin:
            return None, None

        if self.curr_step % self.learn_every != 0:
            return None, None

        # Sample from memory
        state, next_state, action, reward, done, weights, indices = self.recall()

        # Get TD Estimate
        td_est = self.td_estimate(state, action)

        # Get TD Target
        td_tgt = self.td_target(reward, next_state, done)

        # Update Q-online
        loss = self.update_Q_online(td_est, td_tgt, weights)

        # Update priorities in the PER buffer
        errors = td_tgt - td_est
        for i in range(len(indices)):
            self.memory.update(indices[i], errors[i])

        return (td_est.mean().item(), loss)
    
    def update_Q_online(self, td_estimate, td_target, weights):
        td_estimate = td_estimate.cpu()  # move td_estimate to the CPU
        td_target = td_target.cpu()  # move td_target to the CPU
        loss = self.loss_fn(td_estimate, td_target) * weights
        loss = loss.mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()


import numpy as np
import time, datetime
import matplotlib.pyplot as plt


class MetricLogger:
    def __init__(self, save_dir):
        self.save_log = save_dir / "log"
        with open(self.save_log, "w") as f:
            f.write(
                f"{'Episode':>8}{'Step':>8}{'Epsilon':>10}{'MeanReward':>15}"
                f"{'MeanLength':>15}{'MeanLoss':>15}{'MeanQValue':>15}"
                f"{'TimeDelta':>15}{'Time':>20}\n"
            )
        self.ep_rewards_plot = save_dir / "reward_plot.jpg"
        self.ep_lengths_plot = save_dir / "length_plot.jpg"
        self.ep_avg_losses_plot = save_dir / "loss_plot.jpg"
        self.ep_avg_qs_plot = save_dir / "q_plot.jpg"

        # History metrics
        self.ep_rewards = []
        self.ep_lengths = []
        self.ep_avg_losses = []
        self.ep_avg_qs = []

        # Moving averages, added for every call to record()
        self.moving_avg_ep_rewards = []
        self.moving_avg_ep_lengths = []
        self.moving_avg_ep_avg_losses = []
        self.moving_avg_ep_avg_qs = []

        # Current episode metric
        self.init_episode()

        # Timing
        self.record_time = time.time()

    def log_step(self, reward, loss, q):
        self.curr_ep_reward += reward
        self.curr_ep_length += 1
        if loss:
            self.curr_ep_loss += loss
            self.curr_ep_q += q
            self.curr_ep_loss_length += 1

    def log_episode(self):
        "Mark end of episode"
        self.ep_rewards.append(self.curr_ep_reward)
        self.ep_lengths.append(self.curr_ep_length)
        if self.curr_ep_loss_length == 0:
            ep_avg_loss = 0
            ep_avg_q = 0
        else:
            ep_avg_loss = np.round(self.curr_ep_loss / self.curr_ep_loss_length, 5)
            ep_avg_q = np.round(self.curr_ep_q / self.curr_ep_loss_length, 5)
        self.ep_avg_losses.append(ep_avg_loss)
        self.ep_avg_qs.append(ep_avg_q)

        self.init_episode()

    def init_episode(self):
        self.curr_ep_reward = 0.0
        self.curr_ep_length = 0
        self.curr_ep_loss = 0.0
        self.curr_ep_q = 0.0
        self.curr_ep_loss_length = 0

    def record(self, episode, epsilon, step):
        mean_ep_reward = np.round(np.mean(self.ep_rewards[-100:]), 3)
        mean_ep_length = np.round(np.mean(self.ep_lengths[-100:]), 3)
        mean_ep_loss = np.round(np.mean(self.ep_avg_losses[-100:]), 3)
        mean_ep_q = np.round(np.mean(self.ep_avg_qs[-100:]), 3)
        self.moving_avg_ep_rewards.append(mean_ep_reward)
        self.moving_avg_ep_lengths.append(mean_ep_length)
        self.moving_avg_ep_avg_losses.append(mean_ep_loss)
        self.moving_avg_ep_avg_qs.append(mean_ep_q)

        last_record_time = self.record_time
        self.record_time = time.time()
        time_since_last_record = np.round(self.record_time - last_record_time, 3)

        print(
            f"Episode {episode} - "
            f"Step {step} - "
            f"Epsilon {epsilon} - "
            f"Mean Reward {mean_ep_reward} - "
            f"Mean Length {mean_ep_length} - "
            f"Mean Loss {mean_ep_loss} - "
            f"Mean Q Value {mean_ep_q} - "
            f"Time Delta {time_since_last_record} - "
            f"Time {datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}"
        )

        with open(self.save_log, "a") as f:
            f.write(
                f"{episode:8d}{step:8d}{epsilon:10.3f}"
                f"{mean_ep_reward:15.3f}{mean_ep_length:15.3f}{mean_ep_loss:15.3f}{mean_ep_q:15.3f}"
                f"{time_since_last_record:15.3f}"
                f"{datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'):>20}\n"
            )

        for metric in ["ep_lengths", "ep_avg_losses", "ep_avg_qs", "ep_rewards"]:
            plt.clf()
            plt.plot(getattr(self, f"moving_avg_{metric}"), label=f"moving_avg_{metric}")
            plt.legend()
            plt.savefig(getattr(self, f"{metric}_plot"))

save_dir = Path("checkpoints") #/ datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
if not os.path.exists(save_dir):
  save_dir.mkdir(parents=True)
print(f'All weights will saved at {save_dir}')

# Initialize mario, this will re-init the weight
mario = Mario(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir)
use_cuda = torch.cuda.is_available()
print(f"Using CUDA: {use_cuda}")

print()
import warnings
save_dir = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
save_dir.mkdir(parents=True)


load_path = r"checkpoints/120000ep/episode_1000/mario_net_116638.chkpt"



mario = Mario(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir)
mario.load(load_path)

'''Training Mario'''
logger = MetricLogger(save_dir)

mario.optimizer = torch.optim.Adam(mario.net.parameters(), lr=0.001)

import cv2

def animate(imgs, video_name):
    height, width, layers = imgs[0].shape
    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'XVID'), 20, (width, height))
    for img in imgs:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV
        video.write(img)
    video.release()
    print(f"Video saved as {video_name}")

# Run training episodes
episodes = 500
save_freq = 1



for e in range(episodes):
    state = env.reset()
    imgs = []  # Initialize the frame list for this episode

    #print(f"Episode {e}")

    while True:
        action = mario.act(state)
        next_state, reward, done, trunc, info = env.step(action)


        # Collect the current frame
        img = env.render().copy()
        imgs.append(img)

        mario.cache(state, next_state, action, reward, done)
        q, loss = mario.learn(save_path=save_dir)
        logger.log_step(reward, loss, q)
        state = next_state

        if done or info["flag_get"]:
            break

    logger.log_episode()

    # record every 20 episodes
    if (e) % 20 == 0:
        logger.record(episode=e, epsilon=mario.exploration_rate, step=mario.curr_step)

    # Check if it's time to save
    if (e + 1) % save_freq == 0 or e == episodes - 1:
        # Add another subdirectory to save_dir, being number of episodes
        original_save_dir = save_dir
        save_dir = save_dir / f"episode_{e + 1}"
        save_dir.mkdir(parents=True, exist_ok=True)  # Create the directory if it does not exist

        # Create video from collected images
        video_path = save_dir / f"episode_{e + 1}.mp4"
        video_path = str(video_path)
        animate(imgs, video_name=video_path)

        # Save the model
        mario.save(save_path=save_dir)

        # Reset the save_dir to the original for the next episode
        save_dir = original_save_dir

# Final save if not already saved on the last episode
if episodes % save_freq != 0:
    video_path = save_dir / f"episode_{episodes}.mp4"
    video_path = str(video_path)
    animate(imgs, video_name=video_path)
    mario.save(save_path=save_dir)
