
import gymnasium as gym
from gymnasium import spaces
import traci
import numpy as np
import sys
import os

# Import generate_traffic to create new scenarios on reset
try:
    import generate_traffic
except ImportError:
    generate_traffic = None

# PCU Mapping
PCU_MAPPING = {
    'passenger': 1.0,
    'truck': 2.0,
    'bus': 2.5,
    'motorcycle': 0.5
}

try:
    from traffic_rules import TrafficRules
except ImportError:
    TrafficRules = None


class SumoGymEnv(gym.Env):
    """
    SUMO Gymnasium Environment for Traffic Signal Control
    
    Action Space:
        Depends on mode:
        - Mode 1 (Time Adjustment): 0 = Decrease Green, 1 = Keep, 2 = Increase Green
        - Mode 2 (Phase Selection): Discrete(num_phases) - Select specific phase index
        
    Observation Space:
        [queue_length_n, queue_length_e, queue_length_s, queue_length_w,
         wait_time_n, wait_time_e, wait_time_s, wait_time_w,
         current_phase_index]
    """
    
    def __init__(
        self,
        mode='time_adjust',
        gui=False,
        max_steps=500,
        rank=0,
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
        super(SumoGymEnv, self).__init__()
        
        self.mode = mode
        self.gui = gui
        self.max_steps = max_steps
        self.step_counter = 0
        self.rank = rank
        self.simulation_steps = simulation_steps
        self.base_num_vehicles = num_vehicles
        self.curriculum_end = curriculum_end
        self.curriculum_episodes = max(curriculum_episodes, 1)
        self.scenario_seed = scenario_seed
        self.scenario_seed_increment = scenario_seed_increment
        self.throughput_weight = throughput_weight
        self.episode_count = 0
        self.last_throughput = 0.0
        if port_base is None:
            env_port_base = os.getenv("SUMO_PORT_BASE")
            try:
                port_base = int(env_port_base) if env_port_base is not None else 8813
            except ValueError:
                port_base = 8813
        self.sumo_port = port_base + self.rank
        
        # Unique route file for this parallel instance to prevent conflicts
        # We need to go up two levels: 
        # 1. out of RandomGeneratedVehicles
        # 2. out of SimulationData
        # but wait, SimulationData/Intersect_Test/ is where we want to write.
        # let's check current working directory during execution.
        # Assuming we run from SimulationData/RandomGeneratedVehicles
        
        self.route_file = f"../Intersect_Test/routes_{self.rank}.rou.xml"
        
        self.sumo_binary = "sumo-gui" if gui else "sumo"
        self.sumo_cmd = [
            self.sumo_binary,
            "-c", "../Intersect_Test/config.sumocfg",
            "--route-files", self.route_file, # Override route file
            "--start",
            "--quit-on-end"
        ]
        if sumo_no_step_log:
            self.sumo_cmd.append("--no-step-log")
        if sumo_no_warnings:
            self.sumo_cmd.append("--no-warnings")
        
        # Define Action Space
        if self.mode == 'time_adjust':
            # 0: Decrease (-5s), 1: Keep, 2: Increase (+5s)
            self.action_space = spaces.Discrete(3)
        elif self.mode == 'phase_selection':
            # Select phase index (assuming 8 phases as seen in original code)
            self.action_space = spaces.Discrete(8)
            
        # Define Observation Space
        # 4 directions * 2 metrics (queue, wait) + 1 current_phase = 9 metrics
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(9,), dtype=np.float32
        )
        
        self.tls_id = "J1" # Focusing on J1 for now as primary agent target
        self.current_phase_index = 0
        self.metric_collector = None
        
        # Initialize Safety Layer
        if TrafficRules:
            self.rules = TrafficRules(min_green_time=15)
            self.last_phase_change_step = 0
        else:
            self.rules = None
            
    def _get_start_vehicle_type_pcu(self, vtype):
         # Try exact match first
        if vtype in PCU_MAPPING:
            return PCU_MAPPING[vtype]
        # Try partial match (e.g. 'bus_1' -> 'bus')
        vtype_lower = vtype.lower()
        for key, pcu in PCU_MAPPING.items():
            if key in vtype_lower:
                return pcu
        return 1.0 # Default
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.episode_count += 1
        
        # 1. Generate new traffic scenario (Adaptive)
        if generate_traffic:
            if self.curriculum_end is None or self.curriculum_end <= self.base_num_vehicles:
                current_vehicles = self.base_num_vehicles
            else:
                progress = min(self.episode_count / self.curriculum_episodes, 1.0)
                current_vehicles = int(
                    round(self.base_num_vehicles + progress * (self.curriculum_end - self.base_num_vehicles))
                )
            scenario_seed = None
            if self.scenario_seed is not None:
                scenario_seed = self.scenario_seed
                if self.scenario_seed_increment:
                    scenario_seed += self.episode_count + (self.rank * 100000)
            # Only print for rank 0 to avoid clutter
            if self.rank == 0:
                print(
                    "Generating new traffic scenario for rank "
                    f"{self.rank} (vehicles={current_vehicles}, seed={scenario_seed})..."
                )
            generate_traffic.generate_route_file(
                self.route_file,
                num_vehicles=current_vehicles,
                seed=scenario_seed,
            )
            
        # 2. Start SUMO
        try:
            traci.close()
        except:
            pass
            
        try:
            traci.start(self.sumo_cmd, port=self.sumo_port)
        except Exception as e:
            print(f"Error starting SUMO: {e}")
            # Consider raising error or retrying
        
        self.step_counter = 0
        self.current_phase_index = 0
        self.last_phase_change_step = 0 # Assume started at 0
        
        return self._get_observation(), {}
        
    def step(self, action):
        self.step_counter += 1
        
        # 1. Apply Action
        self._apply_action(action)
        
        # 2. Run Simulation Steps (e.g., 5 seconds)
        # We simulate multiple steps to let the action have effect
        for _ in range(self.simulation_steps):
            traci.simulationStep()
            
        # 3. Get Observation
        observation = self._get_observation()
        
        # 4. Calculate Reward
        # Reward = - (1.0 * Total Queue PCU + 0.1 * Total Wait PCU)
        # We want to minimize queue and wait, so maximize negative sum
        queue_len_pcu = np.sum(observation[:4])
        wait_time_pcu = np.sum(observation[4:8])
        
        # Weighted Reward
        throughput_reward = self.throughput_weight * self.last_throughput
        reward = - (1.0 * queue_len_pcu + 0.1 * wait_time_pcu) + throughput_reward
        
        # 5. Check Done
        terminated = self.step_counter >= self.max_steps
        truncated = False
        
        if terminated:
            # Don't close here, let the main loop call env.close() or env.reset()
            pass
            
        info = {
            "queue_length": queue_len_pcu,
            "wait_time": wait_time_pcu,
            "throughput_pcu": self.last_throughput,
            "throughput_reward": throughput_reward,
        }
        
        return observation, reward, terminated, truncated, info
        
    def _apply_action(self, action):
        if self.mode == 'time_adjust':
            # Adjust duration of current phase
            # This is tricky in SUMO continuously. 
            # Simplified approach: If action is extend, we just hold phase longer in next step loop.
            # Real implementation might need to modify logic phase duration.
            # For this prototype: 
            # 0: Force switch to next phase immediately
            # 1: Do nothing (follow logic)
            # 2: Extend current phase
            pass # TODO: Refine this logic for SUMO integration
            
        elif self.mode == 'phase_selection':
            # 1. Get current state
            current_phase = traci.trafficlight.getPhase(self.tls_id)
            
            # Calculate how long we've been in this phase
            # current simulation time - time of last change
            # Note: Simulation time in SUMO is in seconds, but we track steps.
            # Ideally we track simulation seconds.
            current_sim_time = traci.simulation.getTime()
            current_duration = current_sim_time - self.last_phase_change_step
            
            # 2. Safety Check (Rule-Based Supervisor)
            final_action = action
            if self.rules:
                final_action = self.rules.get_safe_action(
                    self.tls_id, 
                    action, 
                    current_phase, 
                    current_duration
                )
                
            # 3. Apply Action
            if current_phase != final_action:
                traci.trafficlight.setPhase(self.tls_id, final_action)
                self.current_phase_index = final_action
                # Update timestamp of change
                self.last_phase_change_step = traci.simulation.getTime()
                
    def _get_observation(self):
        # Collect metrics from SUMO
        # This is a simplified extraction. 
        # In a full implementation, we would inspect lanes for J1.
        
        # Dummy mapping for J1 lanes (from original code analysis)
        # North: -E1, East: -E2, South: -E3, West: -E0
        incoming_lanes = ["-E1_0", "-E1_1", "-E2_0", "-E2_1", "-E3_0", "-E3_1", "-E0_0", "-E0_1"]
        
        queues = []
        waits = []
        moving_pcu = 0.0
        
        # Rough estimation by aggregating lanes
        # North, East, South, West
        directions = [["-E1_0", "-E1_1"], ["-E2_0", "-E2_1"], ["-E3_0", "-E3_1"], ["-E0_0", "-E0_1"]]
        
        for lanes in directions:
            q = 0.0
            w = 0.0
            for lane in lanes:
                try:
                    # Get all vehicles in the lane
                    vehicles = traci.lane.getLastStepVehicleIDs(lane)
                    for veh in vehicles:
                        vtype = traci.vehicle.getTypeID(veh)
                        pcu = self._get_start_vehicle_type_pcu(vtype)
                        
                        # Calculate PCU-weighted Queue
                        # Queue: vehicles with speed < 0.1 m/s (approx)
                        speed = traci.vehicle.getSpeed(veh)
                        if speed < 0.1:
                            q += pcu
                        else:
                            moving_pcu += pcu
                            
                        # Calculate PCU-weighted Wait Time
                        # Note: getWaitingTime returns accumulated waiting seconds
                        # We weight the waiting seconds by PCU
                        waiting_seconds = traci.vehicle.getWaitingTime(veh)
                        w += waiting_seconds * pcu
                        
                except:
                    pass
            queues.append(q)
            waits.append(w)
            
        obs = np.array(queues + waits + [self.current_phase_index], dtype=np.float32)
        self.last_throughput = moving_pcu
        return obs
        
    def close(self):
        try:
            traci.close()
        except:
            pass

