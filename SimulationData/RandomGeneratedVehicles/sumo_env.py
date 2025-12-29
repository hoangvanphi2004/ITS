
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
    
    def __init__(self, mode='time_adjust', gui=False, max_steps=500, rank=0):
        super(SumoGymEnv, self).__init__()
        
        self.mode = mode
        self.gui = gui
        self.max_steps = max_steps
        self.step_counter = 0
        self.rank = rank
        
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
        
        # 1. Generate new traffic scenario (Adaptive)
        if generate_traffic:
            # Only print for rank 0 to avoid clutter
            if self.rank == 0:
                print(f"Generating new traffic scenario for rank {self.rank}...")
            generate_traffic.generate_route_file(self.route_file)
            
        # 2. Start SUMO
        try:
            traci.close()
        except:
            pass
            
        try:
            traci.start(self.sumo_cmd)
        except Exception as e:
            print(f"Error starting SUMO: {e}")
            # Consider raising error or retrying
        
        self.step_counter = 0
        self.current_phase_index = 0
        
        return self._get_observation(), {}
        
    def step(self, action):
        self.step_counter += 1
        
        # 1. Apply Action
        self._apply_action(action)
        
        # 2. Run Simulation Steps (e.g., 5 seconds)
        # We simulate multiple steps to let the action have effect
        simulation_steps = 5
        for _ in range(simulation_steps):
            traci.simulationStep()
            
        # 3. Get Observation
        observation = self._get_observation()
        
        # 4. Calculate Reward
        # Reward = - (1.0 * Total Queue PCU + 0.1 * Total Wait PCU)
        # We want to minimize queue and wait, so maximize negative sum
        queue_len_pcu = np.sum(observation[:4])
        wait_time_pcu = np.sum(observation[4:8])
        
        # Weighted Reward
        reward = - (1.0 * queue_len_pcu + 0.1 * wait_time_pcu)
        
        # 5. Check Done
        terminated = self.step_counter >= self.max_steps
        truncated = False
        
        if terminated:
            # Don't close here, let the main loop call env.close() or env.reset()
            pass
            
        info = {
            "queue_length": queue_len,
            "wait_time": wait_time
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
            # Switch to the selected phase
            target_phase = action
            current = traci.trafficlight.getPhase(self.tls_id)
            if current != target_phase:
                traci.trafficlight.setPhase(self.tls_id, target_phase)
                self.current_phase_index = target_phase
                
    def _get_observation(self):
        # Collect metrics from SUMO
        # This is a simplified extraction. 
        # In a full implementation, we would inspect lanes for J1.
        
        # Dummy mapping for J1 lanes (from original code analysis)
        # North: -E1, East: -E2, South: -E3, West: -E0
        incoming_lanes = ["-E1_0", "-E1_1", "-E2_0", "-E2_1", "-E3_0", "-E3_1", "-E0_0", "-E0_1"]
        
        queues = []
        waits = []
        
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
                        if traci.vehicle.getSpeed(veh) < 0.1:
                            q += pcu
                            
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
        return obs
        
    def close(self):
        try:
            traci.close()
        except:
            pass

