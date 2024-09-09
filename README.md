# Mario Reinforcement Learning Project -- mario-rl for AIC-502 Reinforcement Learning competency

## Project Overview
This project implements a Mario reinforcement learning agent using Deep Q-Learning techniques. The goal is to train the agent to successfully navigate through levels in the Super Mario environment by optimizing its policy for maximum reward using advanced reinforcement learning methods.

The project consists of two models trained for different numbers of episodes, each incorporating dynamic exploration rates, reward shaping, and advanced optimizers for better learning outcomes. It also implements Prioritized Experience Replay to improve learning efficiency.

## Code Description
The repository includes the following key components:

- **aic502-mario-rl-1.py & aic502-mario-rl-2.py**: These Python scripts include the entire training setup, from environment initialization to model training. The agent is trained using Deep Q-Learning with two models:
  
  - **Model 1** (`aic502-mario-rl-1.py`):
    - Trained for 40,000 episodes
    - Exploration rate decay: 0.9999998
    - Custom reward shaping based on x-coordinate progress, score increases, and coin collection
    - Mean Squared Error (MSE) Loss Function
    - AdamW Optimizer
  
  - **Model 2** (`aic502-mario-rl-2.py`):
    - Trained for 120,000 episodes
    - Exploration rate decay: 0.99999975
    - Enhanced reward structure with normalized progress reward and penalties for inactivity
    - SmoothL1Loss Function
    - AdamW Optimizer

- **Checkpoints**: The script automatically saves model checkpoints every 1,000 episodes, allowing for incremental training and visualizing progress through gameplay video capture using OpenCV.

## Features
- **Dynamic Exploration Rates**: Adjusts based on whether the agent finishes the level, ensuring sufficient exploration.
- **Custom Reward Shaping**: Incentivizes forward progress, score improvement, and coin collection, with penalties for inactivity.
- **Prioritized Experience Replay**: Focuses on important experiences for faster convergence and better learning.
- **AdamW Optimizer**: Applied to improve generalization and model stability.

## How to Run
1. Clone the repository: 
   ```bash
   git clone https://github.com/Thanitkul/mario-rl
   ```
2. Install the required dependencies: 
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Python scripts to start training the agent:
   ```bash
   python aic502-mario-rl-1.py
   ```
   or
   ```bash
   python aic502-mario-rl-2.py
   ```

Model checkpoints will be saved every 1,000 episodes.

## Future Improvements
- Implement more sophisticated reward shaping strategies to balance risk and reward.
- Use Huber Loss for improved handling of outliers in the environment.
- Fine-tune Prioritized Experience Replay to increase diversity in sampled experiences.

## References
- Super Mario environment: [gym-super-mario-bros](https://github.com/Kautenja/gym-super-mario-bros)
- AdamW Optimizer: [AdamW Matters](https://towardsdatascience.com/why-adamw-matters-736223f31b5d)
- Prioritized Experience Replay: [Howuhh PER](https://github.com/Howuhh/prioritized_experience_replay)

## Video Links
- Training Results: [Mario Agent Training](https://www.youtube.com/watch?v=ktsdrrRtK1M)
