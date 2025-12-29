
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from sumo_env import SumoGymEnv
import os
import argparse
import time

def make_env(rank, mode='phase_selection', gui=False):
    """
    Utility function for multiprocessed env.
    
    :param rank: (int) index of the subprocess
    :param mode: (str) simulation mode
    :param gui: (bool) whether to run with GUI (usually False for parallel)
    """
    def _init():
        env = SumoGymEnv(mode=mode, gui=gui, rank=rank)
        return env
    return _init

def train(steps=100000, num_cpu=4, mode='phase_selection', gui=False):
    print(f"Starting PPO training with {num_cpu} CPUs, mode={mode}...")
    
    # Create the vectorized environment
    # Using SubprocVecEnv for true parallelism
    if num_cpu > 1:
        env = SubprocVecEnv([make_env(i, mode=mode, gui=gui) for i in range(num_cpu)])
    else:
        env = DummyVecEnv([make_env(0, mode=mode, gui=gui)])

    # Initialize the PPO agent
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        gamma=0.99
    )
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=10000, 
        save_path='./logs/',
        name_prefix='ppo_traffic'
    )

    # Train
    start_time = time.time()
    model.learn(total_timesteps=steps, callback=checkpoint_callback)
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
    parser.add_argument("--mode", type=str, default="phase_selection", choices=["time_adjust", "phase_selection"])
    parser.add_argument("--gui", action="store_true", help="Enable GUI (not recommended for parallel)")
    
    args = parser.parse_args()
    
    train(steps=args.steps, num_cpu=args.cpu, mode=args.mode, gui=args.gui)
