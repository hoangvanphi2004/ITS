
import os
import argparse
from sumo_env import SumoGymEnv
from rl_agent import TrafficAgent
import matplotlib.pyplot as plt

def train(episodes=10, mode='phase_selection', gui=False):
    """
    Main training loop
    """
    print(f"Starting training with mode={mode}, episodes={episodes}, gui={gui}")
    
    # 1. Initialize Environment
    env = SumoGymEnv(mode=mode, gui=gui, max_steps=500)
    
    # 2. Initialize Agent
    # Action space size depends on mode
    action_space_size = env.action_space.n
    agent = TrafficAgent(action_space_size)
    
    rewards_history = []
    
    # 3. Training Loop
    for episode in range(episodes):
        state, info = env.reset()
        total_reward = 0
        done = False
        step = 0
        
        print(f"Episode {episode+1}/{episodes} started...")
        
        while not done:
            # Agent chooses action
            action = agent.act(state)
            
            # Environment step
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Agent learns
            agent.learn(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            step += 1
            
        rewards_history.append(total_reward)
        print(f"Episode {episode+1} finished. Total Reward: {total_reward:.2f}, Steps: {step}")
        
    env.close()
    
    # Plot results
    plt.plot(rewards_history)
    plt.title("Training Progress")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward (Negative Wait Time)")
    plt.savefig("training_results.png")
    print("Training finished. Results saved to training_results.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to run")
    parser.add_argument("--mode", type=str, default="phase_selection", choices=["time_adjust", "phase_selection"], help="Control mode")
    parser.add_argument("--gui", action="store_true", help="Run with SUMO GUI")
    
    args = parser.parse_args()
    
    train(episodes=args.episodes, mode=args.mode, gui=args.gui)
