
import gymnasium as gym
from stable_baselines3 import PPO
from sumo_env import SumoGymEnv
import os
import argparse
import time
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

def evaluate(model_path="ppo_traffic_final", steps=1000, mode='delta_time', delay=0.05):
    print(f"Loading model from {model_path}...")
    
    # 1. Create Environment (GUI enabled for visualization)
    # Note: For evaluation, we ideally need to wrap this in DummyVecEnv and VecNormalize
    # loading the statistics from training.
    
    env = SumoGymEnv(mode=mode, gui=True, max_steps=steps)
    env = DummyVecEnv([lambda: env]) # Wrap for compatibility
    
    # Load Normalization Stats if they exist
    stats_path = "vec_normalize.pkl"
    if os.path.exists(stats_path):
        print(f"Loading normalization stats from {stats_path}...")
        env = VecNormalize.load(stats_path, env)
        env.training = False # Disable update of stats during evaluation
        env.norm_reward = False # Don't normalize reward for evaluation visualization
    else:
        print("Warning: vec_normalize.pkl not found. Running without normalization (performance might be poor).")

    # 2. Load Model
    try:
        model = PPO.load(model_path, env=env)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 3. Run Evaluation Episode
    obs = env.reset() # VecEnv reset returns obs directly
    done = False
    total_reward = 0
    step = 0
    
    print("Starting evaluation...")
    while not done:
        # Predict action using the trained model
        # deterministic=True means take the best action, no randomness
        action, _states = model.predict(obs, deterministic=True)
        
        # Step environment
        # VecEnv step: obs, rewards, dones, infos
        obs, rewards, dones, infos = env.step(action)
        
        # Accumulate reward (scalar from VecEnv is list of rewards)
        total_reward += rewards[0]
        
        # VecEnv auto-resets on done, so we don't need manual reset
        if dones[0]:
            print(f"Episode finished at step {step+1}")
            # If you want to stop after one episode:
            # break
            done = True # Set done to true to exit the loop after the episode finishes
        
        time.sleep(delay)
        
        step += 1 # Increment step counter
        
        if step % 100 == 0:
            print(f"Step {step}, Reward: {rewards[0]:.2f}, Info: {infos[0]}")

    print(f"Evaluation finished. Total Reward (Normalized?): {total_reward:.2f}")
    print("Press Enter to close SUMO...")
    input()
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="ppo_traffic_final", help="Path to trained model (without .zip)")
    parser.add_argument("--steps", type=int, default=1000, help="Number of steps to evaluate")
    parser.add_argument("--mode", type=str, default="delta_time", help="Simulation mode")
    parser.add_argument("--delay", type=float, default=0.05, help="Delay between steps for visualization (seconds)")
    
    args = parser.parse_args()
    
    evaluate(model_path=args.model, steps=args.steps, mode=args.mode, delay=args.delay)
