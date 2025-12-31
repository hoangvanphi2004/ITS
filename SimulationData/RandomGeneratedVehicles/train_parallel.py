```python
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
import os
import argparse
import time
from typing import Callable
import os
import argparse
import time

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.
    :param initial_value: Initial learning rate.
    :return: schedule that computes current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.
        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value
    return func

def make_env(rank, mode='delta_time', gui=False):
    """
    Utility function for multiprocessed env.
    
    :param rank: Index of the subprocess
    """
    def _init():
        env = SumoGymEnv(mode=mode, gui=gui, rank=rank)
        # Wrap the environment with Monitor to log custom metrics
        # allowing us to track queue_pcu, loss_pcu, and delta in Tensorboard/Logs
        env = Monitor(env, info_keywords=("queue_pcu", "loss_pcu", "delta"))
        return env
    return _init # Added return _init

def train(steps=100000, num_cpu=4, mode='delta_time', gui=False):
    print(f"Starting PPO training with {num_cpu} CPUs, mode={mode}...")
    
    # Create the vectorized environment
    # Using SubprocVecEnv for true parallelism
    if num_cpu > 1:
        env = SubprocVecEnv([make_env(i, mode=mode, gui=gui) for i in range(num_cpu)])
    else:
        env = DummyVecEnv([make_env(0, mode=mode, gui=gui)])

    # [NEW] Normalize Observation and Reward
    # This is critical for convergence (Explained Variance fix)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10., gamma=0.99)

    # Initialize the PPO agent with Optimized Hyperparameters
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1,
        learning_rate=linear_schedule(3e-5), # Lower LR with decay
        n_steps=2048, # More steps per update
        batch_size=64, # Larger batch size
        n_epochs=10, 
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01, # Encourage exploration
        tensorboard_log="./ppo_traffic_tensorboard/"
    )
    
    print("Training started...")
    start_time = time.time()
    
    # Train
    model.learn(total_timesteps=steps)
    
    end_time = time.time()
    print(f"Training finished in {end_time - start_time:.2f} seconds")
    
    # Save the final model
    model.save("ppo_traffic_final")
    print("Model saved to ppo_traffic_final.zip")
    
    # Close environment
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100000, help="Total training steps")
    parser.add_argument("--cpu", type=int, default=4, help="Number of CPU cores to use")
    parser.add_argument("--mode", type=str, default="delta_time", help="Simulation mode (default: delta_time)")
    parser.add_argument("--gui", action="store_true", help="Enable GUI (not recommended for parallel)")
    
    args = parser.parse_args()
    
    train(steps=args.steps, num_cpu=args.cpu, mode=args.mode, gui=args.gui)
