
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.logger import configure
from sumo_env import SumoGymEnv
import torch
import os
import argparse
import time

def make_env(
    rank,
    mode='phase_selection',
    gui=False,
    port_base=None,
    simulation_steps=5,
    num_vehicles=200,
    curriculum_end=None,
    curriculum_episodes=1,
    scenario_seed=None,
    scenario_seed_increment=True,
    throughput_weight=0.01,
    sumo_no_step_log=False,
    sumo_no_warnings=False,
):
    """
    Utility function for multiprocessed env.
    
    :param rank: (int) index of the subprocess
    :param mode: (str) simulation mode
    :param gui: (bool) whether to run with GUI (usually False for parallel)
    """
    def _init():
        env = SumoGymEnv(
            mode=mode,
            gui=gui,
            rank=rank,
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
        return env
    return _init

def make_log_dir(base_dir):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

def resolve_run_dir(log_dir, run_dir, resume_path):
    if run_dir:
        os.makedirs(run_dir, exist_ok=True)
        return run_dir
    if resume_path:
        resume_dir = os.path.dirname(os.path.abspath(resume_path))
        os.makedirs(resume_dir, exist_ok=True)
        return resume_dir
    return make_log_dir(log_dir)

def get_run_info_path(run_dir, resume_path):
    if resume_path:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        return os.path.join(run_dir, f"run_info_resume_{timestamp}.txt")
    return os.path.join(run_dir, "run_info.txt")

def get_iteration_log_path(run_dir, resume_path):
    if resume_path:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        return os.path.join(run_dir, f"iteration_estimate_resume_{timestamp}.log")
    return os.path.join(run_dir, "iteration_estimate.log")

def get_log_formats():
    formats = ["stdout", "csv", "json"]
    try:
        from torch.utils.tensorboard import SummaryWriter  # noqa: F401
        formats.append("tensorboard")
    except Exception:
        pass
    return formats

def format_duration(seconds):
    seconds = int(max(seconds, 0))
    days, rem = divmod(seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    if days > 0:
        return f"{days}d {hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

class IterationTimeCallback(BaseCallback):
    def __init__(self, target_iterations, estimate_interval, log_path=None, verbose=0):
        super().__init__(verbose)
        self.target_iterations = target_iterations
        self.estimate_interval = max(1, estimate_interval)
        self.log_path = log_path
        self.start_time = None
        self.last_logged_iteration = 0

    def _on_training_start(self):
        self.start_time = time.time()
        if self.log_path:
            with open(self.log_path, "w", encoding="utf-8") as f:
                f.write(
                    "iter,elapsed_sec,avg_iter_sec,eta_target_sec,total_est_sec\n"
                )

    def _on_step(self):
        return True

    def _on_rollout_end(self):
        n_steps = getattr(self.model, "n_steps", 0)
        n_envs = getattr(self.model, "n_envs", 0)
        denom = n_steps * n_envs
        if denom <= 0:
            return True
        iteration = int(self.model.num_timesteps / denom)
        if iteration <= 0:
            return True
        if iteration - self.last_logged_iteration < self.estimate_interval:
            return True

        elapsed = time.time() - self.start_time
        avg_iter_sec = elapsed / iteration
        remaining_iters = max(self.target_iterations - iteration, 0)
        eta_sec = remaining_iters * avg_iter_sec
        total_est_sec = self.target_iterations * avg_iter_sec

        print(
            "[IterEst] "
            f"iter={iteration} "
            f"avg_iter={avg_iter_sec:.3f}s "
            f"eta_to_{self.target_iterations}="
            f"{format_duration(eta_sec)} "
            f"elapsed={format_duration(elapsed)}"
        )
        if self.log_path:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(
                    f"{iteration},{elapsed:.2f},{avg_iter_sec:.4f},"
                    f"{eta_sec:.2f},{total_est_sec:.2f}\n"
                )

        self.logger.record("time/iter_sec", avg_iter_sec)
        self.logger.record("time/eta_target_sec", eta_sec)
        self.logger.record("time/eta_target_hours", eta_sec / 3600.0)
        self.logger.record("time/target_iterations", self.target_iterations)

        self.last_logged_iteration = iteration
        return True

class TotalRewardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.total_reward = 0.0
        self.total_reward_raw = 0.0
        self.num_steps = 0
        self.has_raw = False

    def _on_rollout_start(self):
        self.total_reward = 0.0
        self.total_reward_raw = 0.0
        self.num_steps = 0
        self.has_raw = False

    def _on_step(self):
        rewards = self.locals.get("rewards")
        if rewards is None:
            return True
        raw_rewards = None
        env = self.training_env
        if isinstance(env, VecNormalize):
            try:
                raw_rewards = env.get_original_reward()
            except Exception:
                raw_rewards = None
        if hasattr(rewards, "sum"):
            total = rewards.sum()
            count = rewards.size if hasattr(rewards, "size") else len(rewards)
        else:
            total = sum(rewards)
            count = len(rewards)
        self.total_reward += float(total)
        self.num_steps += int(count)
        if raw_rewards is not None:
            self.has_raw = True
            if hasattr(raw_rewards, "sum"):
                raw_total = raw_rewards.sum()
            else:
                raw_total = sum(raw_rewards)
            self.total_reward_raw += float(raw_total)
        return True

    def _on_rollout_end(self):
        self.logger.record("rollout/total_reward", self.total_reward)
        if self.num_steps > 0:
            self.logger.record(
                "rollout/avg_reward_per_step",
                self.total_reward / self.num_steps
            )
        if self.has_raw:
            self.logger.record("rollout/total_reward_raw", self.total_reward_raw)
            self.logger.record(
                "rollout/avg_reward_per_step_raw",
                self.total_reward_raw / self.num_steps
            )
        return True

class VecNormalizeSaveCallback(BaseCallback):
    def __init__(self, save_freq, save_path, verbose=0):
        super().__init__(verbose)
        self.save_freq = max(1, save_freq)
        self.save_path = save_path

    def _on_step(self):
        if self.n_calls % self.save_freq == 0:
            env = self.training_env
            if isinstance(env, VecNormalize):
                env.save(self.save_path)
        return True

    def _on_training_end(self):
        env = self.training_env
        if isinstance(env, VecNormalize):
            env.save(self.save_path)

def resolve_device(requested_device):
    if requested_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested_device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA requested but not available; falling back to CPU.")
        return "cpu"
    return requested_device

def get_cuda_info():
    cuda_available = torch.cuda.is_available()
    if not cuda_available:
        return {
            "available": False,
            "name": "N/A",
            "capability": "N/A",
            "cuda_version": torch.version.cuda or "N/A",
            "torch_version": torch.__version__,
        }
    capability = torch.cuda.get_device_capability(0)
    return {
        "available": True,
        "name": torch.cuda.get_device_name(0),
        "capability": f"{capability[0]}.{capability[1]}",
        "cuda_version": torch.version.cuda or "N/A",
        "torch_version": torch.__version__,
    }

def write_run_info(
    run_dir,
    num_cpu,
    mode,
    gui,
    log_interval,
    port_base,
    device,
    actual_device,
    cuda_info,
    vec_env_name,
    n_steps,
    batch_size,
    ent_coef,
    simulation_steps,
    num_vehicles,
    curriculum_end,
    curriculum_episodes,
    scenario_seed,
    scenario_seed_increment,
    throughput_weight,
    sumo_no_step_log,
    sumo_no_warnings,
    normalize,
    vecnorm_path,
    checkpoint_freq,
    target_iterations,
    estimate_interval,
    total_timesteps,
    requested_iterations,
    resume_path,
    resume_timesteps,
    info_path,
):
    lines = [
        f"mode: {mode}",
        f"gui: {gui}",
        f"num_cpu: {num_cpu}",
        f"vec_env: {vec_env_name}",
        f"n_steps: {n_steps}",
        f"batch_size: {batch_size}",
        f"ent_coef: {ent_coef}",
        f"simulation_steps: {simulation_steps}",
        f"num_vehicles: {num_vehicles}",
        f"curriculum_end: {curriculum_end}",
        f"curriculum_episodes: {curriculum_episodes}",
        f"scenario_seed: {scenario_seed}",
        f"scenario_seed_increment: {scenario_seed_increment}",
        f"throughput_weight: {throughput_weight}",
        f"sumo_no_step_log: {sumo_no_step_log}",
        f"sumo_no_warnings: {sumo_no_warnings}",
        f"normalize: {normalize}",
        f"vecnorm_path: {vecnorm_path}",
        f"checkpoint_freq: {checkpoint_freq}",
        f"total_timesteps: {total_timesteps}",
        f"requested_iterations: {requested_iterations}",
        f"target_iterations: {target_iterations}",
        f"estimate_interval: {estimate_interval}",
        f"resume_path: {resume_path}",
        f"resume_timesteps: {resume_timesteps}",
        f"log_interval: {log_interval}",
        f"port_base: {port_base}",
        "env_device: cpu",
        f"train_device_requested: {device}",
        f"train_device_actual: {actual_device}",
        f"torch_version: {cuda_info['torch_version']}",
        f"cuda_available: {cuda_info['available']}",
        f"cuda_version: {cuda_info['cuda_version']}",
        f"cuda_device_name: {cuda_info['name']}",
        f"cuda_capability: {cuda_info['capability']}",
    ]
    with open(info_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

def train(
    steps=100000,
    num_cpu=4,
    mode='phase_selection',
    gui=False,
    log_dir="./logs",
    run_dir=None,
    log_interval=1,
    port_base=None,
    device="auto",
    n_steps=128,
    batch_size=256,
    ent_coef=0.05,
    simulation_steps=1,
    num_vehicles=50,
    curriculum_end=200,
    curriculum_episodes=2000,
    scenario_seed=None,
    scenario_seed_increment=True,
    throughput_weight=0.01,
    sumo_no_step_log=True,
    sumo_no_warnings=True,
    normalize=True,
    checkpoint_freq=10000,
    target_iterations=None,
    estimate_interval=1,
    requested_iterations=None,
    resume_path=None,
):
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

    # Initialize or resume the PPO agent
    if resume_path:
        model = PPO.load(resume_path, env=env, device=resolved_device)
        model.tensorboard_log = run_dir
        if model.n_steps != n_steps:
            print(
                "WARNING: n_steps differs from checkpoint "
                f"({n_steps} -> {model.n_steps}). Using checkpoint value."
            )
            n_steps = model.n_steps
        if model.batch_size != batch_size:
            print(
                "WARNING: batch_size differs from checkpoint "
                f"({batch_size} -> {model.batch_size}). Using checkpoint value."
            )
            batch_size = model.batch_size
        if model.ent_coef != ent_coef:
            print(
                "WARNING: ent_coef differs from checkpoint "
                f"({ent_coef} -> {model.ent_coef}). Using new value."
            )
            model.ent_coef = ent_coef
    else:
        model = PPO(
            "MlpPolicy", 
            env, 
            verbose=1, 
            learning_rate=0.0003,
            n_steps=n_steps,
            batch_size=batch_size,
            ent_coef=ent_coef,
            gamma=0.99,
            tensorboard_log=run_dir,
            device=resolved_device
        )
    model.set_logger(configure(run_dir, get_log_formats()))
    
    actual_device = str(model.device)
    print(f"Env runs on CPU ({vec_env_name}, {num_cpu} workers)")
    print(
        "Train device requested: "
        f"{device} -> resolved {resolved_device} -> actual {actual_device}"
    )
    print(
        "Torch/CUDA: "
        f"torch={cuda_info['torch_version']}, "
        f"cuda={cuda_info['cuda_version']}, "
        f"available={cuda_info['available']}, "
        f"gpu={cuda_info['name']}, "
        f"cap={cuda_info['capability']}"
    )
    if target_iterations is None:
        target_iterations = requested_iterations if requested_iterations is not None else 1_000_000

    if requested_iterations is not None:
        target_steps = requested_iterations * model.n_steps * model.n_envs
        if resume_path:
            current_steps = int(model.num_timesteps)
            if current_steps >= target_steps:
                print(
                    "Already at or beyond requested iterations "
                    f"({current_steps} >= {target_steps}). Nothing to do."
                )
                env.close()
                return
            print(
                "Using iterations mode: "
                f"{requested_iterations} iterations -> {target_steps} timesteps "
                f"(current {current_steps})"
            )
        else:
            print(
                "Using iterations mode: "
                f"{requested_iterations} iterations -> {target_steps} timesteps"
            )
        steps = target_steps
    elif resume_path:
        current_steps = int(model.num_timesteps)
        if steps <= current_steps:
            print(
                "Total timesteps already reached or exceeded "
                f"({current_steps} >= {steps}). Nothing to do."
            )
            env.close()
            return
        print(
            f"Resuming from {current_steps} timesteps to target {steps} timesteps"
        )

    iter_log_path = get_iteration_log_path(run_dir, resume_path)
    run_info_path = get_run_info_path(run_dir, resume_path)
    resume_timesteps = int(model.num_timesteps) if resume_path else None
    write_run_info(
        run_dir,
        num_cpu,
        mode,
        gui,
        log_interval,
        port_base,
        device,
        actual_device,
        cuda_info,
        vec_env_name,
        n_steps,
        batch_size,
        ent_coef,
        simulation_steps,
        num_vehicles,
        curriculum_end,
        curriculum_episodes,
        scenario_seed,
        scenario_seed_increment,
        throughput_weight,
        sumo_no_step_log,
        sumo_no_warnings,
        normalize,
        vecnorm_path,
        checkpoint_freq,
        target_iterations,
        estimate_interval,
        steps,
        requested_iterations,
        resume_path,
        resume_timesteps,
        run_info_path,
    )
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq, 
        save_path=run_dir,
        name_prefix='ppo_traffic'
    )
    iter_callback = IterationTimeCallback(
        target_iterations=target_iterations,
        estimate_interval=estimate_interval,
        log_path=iter_log_path,
    )
    reward_callback = TotalRewardCallback()
    vecnorm_callback = None
    if normalize and vecnorm_path:
        vecnorm_callback = VecNormalizeSaveCallback(
            save_freq=checkpoint_freq,
            save_path=vecnorm_path,
        )

    # Train
    start_time = time.time()
    callbacks = [checkpoint_callback, iter_callback, reward_callback]
    if vecnorm_callback:
        callbacks.append(vecnorm_callback)
    model.learn(
        total_timesteps=steps,
        callback=callbacks,
        log_interval=log_interval,
        reset_num_timesteps=resume_path is None,
    )
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
    parser.add_argument("--mode", type=str, default="phase_selection", choices=["time_adjust", "phase_selection"])
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
