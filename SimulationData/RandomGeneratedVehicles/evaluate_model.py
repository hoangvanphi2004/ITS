
import gymnasium as gym
from stable_baselines3 import PPO
from sumo_env import SumoGymEnv
import os
import argparse
import time

def evaluate(model_path="ppo_traffic_final", steps=1000, mode='phase_selection', delay=0.05):
    print(f"Loading model from {model_path}...")
    
    # 1. Create Environment (GUI enabled for visualization)
    env = SumoGymEnv(mode=mode, gui=True, max_steps=steps)
    
    # 2. Load Model
    try:
        model = PPO.load(model_path)
    except FileNotFoundError:
        print(f"Error: Model file '{model_path}.zip' not found. Have you run training yet?")
        return

    # 3. Run Evaluation Episode
    obs, info = env.reset()
    done = False
    total_reward = 0
    step = 0
    
    print("Starting evaluation...")
    while not done:
        # Predict action using the trained model
        # deterministic=True means take the best action, no randomness
        action, _states = model.predict(obs, deterministic=True)
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        total_reward += reward
        step += 1
        
        # Slow down for visualization
        time.sleep(delay)
        
        if step % 100 == 0:
            print(f"Step {step}, Reward: {reward:.2f}, Info: {info}")

    print(f"Evaluation finished. Total Reward: {total_reward:.2f}")
    print("Press Enter to close SUMO...")
    input()
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="ppo_traffic_final", help="Path to trained model (without .zip)")
    parser.add_argument("--steps", type=int, default=1000, help="Number of steps to evaluate")
    parser.add_argument("--mode", type=str, default="phase_selection", choices=["time_adjust", "phase_selection"])
    parser.add_argument("--delay", type=float, default=0.05, help="Delay between steps for visualization (seconds)")
    
    args = parser.parse_args()
    
    evaluate(model_path=args.model, steps=args.steps, mode=args.mode, delay=args.delay)
