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
    if resume_path and not os.path.isfile(resume_path):
        raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
    run_dir = resolve_run_dir(log_dir, run_dir, resume_path)
    print(f"Logging to: {run_dir}")
    if resume_path:
        print(f"Resuming from checkpoint: {resume_path}")
    resolved_device = resolve_device(device)
    cuda_info = get_cuda_info()
    
    # Create the vectorized environment
    # Using SubprocVecEnv for true parallelism
    if num_cpu > 1:
        vec_env_name = "SubprocVecEnv"
        env = SubprocVecEnv([
            make_env(
                i,
                mode=mode,
                gui=gui,
                port_base=port_base,
                simulation_steps=simulation_steps,
                num_vehicles=num_vehicles,
                curriculum_end=curriculum_end,
                curriculum_episodes=curriculum_episodes,
                scenario_seed=scenario_seed,
                scenario_seed_increment=scenario_seed_increment,
                throughput_weight=throughput_weight,
                sumo_no_step_log=sumo_no_step_log,
                sumo_no_warnings=sumo_no_warnings,
            )
            for i in range(num_cpu)
        ])
    else:
        vec_env_name = "DummyVecEnv"
        env = DummyVecEnv([
            make_env(
                0,
                mode=mode,
                gui=gui,
                port_base=port_base,
                simulation_steps=simulation_steps,
                num_vehicles=num_vehicles,
                curriculum_end=curriculum_end,
                curriculum_episodes=curriculum_episodes,
                scenario_seed=scenario_seed,
                scenario_seed_increment=scenario_seed_increment,
                throughput_weight=throughput_weight,
                sumo_no_step_log=sumo_no_step_log,
                sumo_no_warnings=sumo_no_warnings,
            )
        ])
    env = VecMonitor(env)
    vecnorm_path = None
    if normalize:
        vecnorm_path = os.path.join(run_dir, "vecnormalize.pkl")
        if resume_path:
            resume_dir = os.path.dirname(os.path.abspath(resume_path))
            resume_vecnorm_path = os.path.join(resume_dir, "vecnormalize.pkl")
            if os.path.isfile(resume_vecnorm_path):
                env = VecNormalize.load(resume_vecnorm_path, env)
                env.training = True
                env.norm_reward = True
                if resume_dir != run_dir:
                    vecnorm_path = os.path.join(run_dir, "vecnormalize.pkl")
            else:
                print(
                    "WARNING: VecNormalize stats not found; "
                    "starting normalization from scratch."
                )
                env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.99)
        else:
            env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.99)

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
    parser.add_argument("--steps", type=int, default=100000, help="Total training steps (ignored if --iterations set)")
    parser.add_argument("--iterations", type=int, default=None, help="Total policy iterations (overrides --steps)")
    parser.add_argument("--cpu", type=int, default=4, help="Number of CPU cores to use")
    parser.add_argument("--mode", type=str, default="delta_time", help="Simulation mode (default: delta_time)")
    parser.add_argument("--gui", action="store_true", help="Enable GUI (not recommended for parallel)")
    parser.add_argument("--log-dir", type=str, default="./logs", help="Base directory for training logs")
    parser.add_argument("--run-dir", type=str, default=None, help="Explicit run directory (overrides --log-dir)")
    parser.add_argument("--log-interval", type=int, default=1, help="Log every N updates")
    parser.add_argument("--port-base", type=int, default=None, help="Base port for SUMO TraCI (unique per env)")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Training device")
    parser.add_argument("--n-steps", type=int, default=128, help="PPO rollout steps per env")
    parser.add_argument("--batch-size", type=int, default=256, help="PPO batch size")
    parser.add_argument("--ent-coef", type=float, default=0.05, help="PPO entropy coefficient")
    parser.add_argument("--sim-steps", type=int, default=1, help="SUMO steps per env step")
    parser.add_argument("--vehicles", type=int, default=50, help="Vehicles per scenario")
    parser.add_argument("--vehicles-end", type=int, default=200, help="Curriculum end vehicles")
    parser.add_argument("--curriculum-episodes", type=int, default=2000, help="Curriculum duration in episodes")
    parser.add_argument("--scenario-seed", type=int, default=None, help="Base seed for scenario generation")
    parser.add_argument("--fixed-scenario", action="store_true", help="Use a fixed scenario seed (no increment)")
    parser.add_argument("--throughput-weight", type=float, default=0.01, help="Reward weight for moving vehicles")
    parser.add_argument("--sumo-step-log", action="store_true", help="Enable SUMO step log")
    parser.add_argument("--sumo-warnings", action="store_true", help="Enable SUMO warnings")
    parser.add_argument("--checkpoint-freq", type=int, default=10000, help="Checkpoint frequency in timesteps")
    parser.add_argument("--normalize", dest="normalize", action="store_true", help="Enable VecNormalize")
    parser.add_argument("--no-normalize", dest="normalize", action="store_false", help="Disable VecNormalize")
    parser.add_argument("--target-iterations", type=int, default=None, help="Target policy iterations for ETA")
    parser.add_argument("--estimate-interval", type=int, default=1, help="Log ETA every N iterations")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint .zip to resume")
    parser.set_defaults(normalize=True)
    
    args = parser.parse_args()

    train(
        steps=args.steps,
        num_cpu=args.cpu,
        mode=args.mode,
        gui=args.gui,
        log_dir=args.log_dir,
        run_dir=args.run_dir,
        log_interval=args.log_interval,
        port_base=args.port_base,
        device=args.device,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        ent_coef=args.ent_coef,
        simulation_steps=args.sim_steps,
        num_vehicles=args.vehicles,
        curriculum_end=args.vehicles_end,
        curriculum_episodes=args.curriculum_episodes,
        scenario_seed=args.scenario_seed,
        scenario_seed_increment=not args.fixed_scenario,
        throughput_weight=args.throughput_weight,
        sumo_no_step_log=not args.sumo_step_log,
        sumo_no_warnings=not args.sumo_warnings,
        normalize=args.normalize,
        checkpoint_freq=args.checkpoint_freq,
        target_iterations=args.target_iterations,
        estimate_interval=args.estimate_interval,
        requested_iterations=args.iterations,
        resume_path=args.resume,
    )
