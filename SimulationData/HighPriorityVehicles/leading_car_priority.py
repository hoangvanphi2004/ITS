"""
SUMO Traffic Light Priority Control for LEADING CAR
- Detects when LEADING_CAR approaches an intersection
- Sets green light only for LEADING_CAR's direction for 100 seconds
- Returns to fixed-time control after that
"""

import traci
import sys
import json
from collections import defaultdict
from datetime import datetime

# SUMO configuration
SUMO_BINARY = "C:\\Phi\\Work\\Simulation\\SUMO\\bin\\sumo-gui.exe"
SUMO_CMD = [SUMO_BINARY, "-c", "./config.sumocfg"]

# Priority control parameters
LEADING_CAR_ID = "t_0"  # LEADING_CAR vehicle ID from routes file
PRIORITY_DURATION = 100  # Duration to keep priority (seconds)
DETECTION_DISTANCE = 50  # Distance to detect approaching vehicle (meters)

# Speed threshold for traffic metrics (when vehicle is considered stopped/queued)
VEHICLE_STOP_THRESHOLD = 0.1  # Speed threshold to detect vehicle stopping (m/s)

# Vehicle type to PCU mapping (Passenger Car Units)
PCU_MAPPING = {
    'passenger': 1.0,
    'truck': 2.0,
    'bus': 2.5,
    'motorcycle': 0.5,
    'bicycle': 0.2,
}

# Controlled area edges for travel time calculation
CONTROLLED_EDGES = ["-E0", "-E1", "-E2", "-E3", "-E4", "-E5", "-E6", "E2"]

# Traffic light states mapping
# Format: {tls_id: {edge_id: ['state_char_indices']}}
# For J1 intersection - IMPORTANT: Use incoming edge names with "-" prefix
TL_STATES = {
    "J1": {
        "-E2": [0, 1, 2],      # From -E2 (East): indices 0,1,2
        "-E3": [3, 4, 5],      # From -E3 (South): indices 3,4,5
        "-E0": [6, 7, 8],      # From -E0 (West): indices 6,7,8
        "-E1": [9, 10, 11],    # From -E1 (North): indices 9,10,11
    },
    "J3": {
        "-E5": [0, 1, 2],      # From -E5 (East): indices 0,1,2
        "-E6": [3, 4, 5],      # From -E6 (South): indices 3,4,5
        "E2": [6, 7, 8],       # From E2 (West - no minus!): indices 6,7,8
        "-E4": [9, 10, 11],    # From -E4 (North): indices 9,10,11
    }
}

# Fixed-time signal plans for each intersection
FIXED_TIME_PLANS = {
    "J1": [
        (33, "GrrGGgGrrGGg"),
        (3,  "yrryygyrryyg"),
        (6,  "rrrrrGrrrrrG"),
        (3,  "rrrrryrrrrry"),
        (33, "GGgGrrGGgGrr"),
        (3,  "yygyrryygyrr"),
        (6,  "rrGrrrrrGrrr"),
        (3,  "rryrrrrryrrr"),
    ],
    "J3": [
        (33, "GGgGrrGGgGrr"),
        (3,  "yygyrryygyrr"),
        (6,  "rrGrrrrrGrrr"),
        (3,  "rryrrrrryrrr"),
        (33, "GrrGGgGrrGGg"),
        (3,  "yrryygyrryyg"),
        (6,  "rrrrrGrrrrrG"),
        (3,  "rrrrryrrrrry"),
    ]
}


class MetricsCollector:
    """Collects and calculates traffic metrics during simulation"""
    
    def __init__(self):
        # Vehicle lane tracking: {vehicle_id: {'in_lane': True/False, 'entry_time': time or None, 'lane': lane_id, 'pcu': pcu_value}}
        # in_lane = True when vehicle is in controlled lane
        # entry_time = time when vehicle FIRST STOPS (speed < 0.1) after entering lane, None until first stop
        self.vehicles_in_intersection = {}
        # Completed wait times: {vehicle_id: {'wait_time': time, 'pcu': pcu_value}}
        # wait_time = exit_time - entry_time (first stop time)
        self.completed_wait_times = []
        
        # Queue length tracking
        # Previous traffic light states: {tls_id: state_string}
        self.previous_tls_states = {tls_id: None for tls_id in FIXED_TIME_PLANS}
        # Queue lengths recorded at red to green transitions: []
        self.queue_lengths = []
        
        # Travel time tracking
        # Vehicles in controlled area: {vehicle_id: {'entry_time': time, 'pcu': pcu_value}}
        self.vehicles_in_network = {}
        # Completed travel times: {vehicle_id: {'travel_time': time, 'pcu': pcu_value}}
        self.completed_travel_times = []
    
    def get_pcu_value(self, vehicle_id):
        """Get PCU value for a vehicle based on its type"""
        try:
            vtype = traci.vehicle.getTypeID(vehicle_id)
            # Try exact match first
            if vtype in PCU_MAPPING:
                return PCU_MAPPING[vtype]
            # Try partial match
            vtype_lower = vtype.lower()
            for vehicle_type, pcu in PCU_MAPPING.items():
                if vehicle_type in vtype_lower:
                    return pcu
            # Default to passenger car
            return PCU_MAPPING['passenger']
        except:
            return PCU_MAPPING['passenger']
    
    def update_metrics(self, current_time):
        """
        Update all metrics during simulation
        Called at each simulation step
        """
        self.update_wait_times(current_time)
        self.update_queue_lengths(current_time)
        self.update_travel_times(current_time)
    
    def update_wait_times(self, current_time):
        """
        Track vehicles entering and leaving intersections
        Entry: when vehicle FIRST STOPS (speed < 0.1 m/s) after entering controlled lane
        Exit: when vehicle leaves the controlled lane (exits intersection)
        Wait time = exit_time - entry_time (first stop time)
        """
        current_vehicles = traci.vehicle.getIDList()
        
        for vehicle_id in current_vehicles:
            try:
                speed = traci.vehicle.getSpeed(vehicle_id)
                lane = traci.vehicle.getLaneID(vehicle_id)
                
                # Extract edge from lane ID (format: edgeID_laneIndex)
                if "_" in lane:
                    edge = lane.rsplit("_", 1)[0]
                else:
                    edge = lane
                
                # Check if vehicle is in a controlled lane (approaching intersection)
                is_in_controlled_lane = False
                for tls_id in FIXED_TIME_PLANS:
                    controlled_lanes = traci.trafficlight.getControlledLanes(tls_id)
                    if lane in controlled_lanes:
                        is_in_controlled_lane = True
                        break
                
                # Vehicle enters intersection lane (first time detected in controlled lane)
                if is_in_controlled_lane and vehicle_id not in self.vehicles_in_intersection:
                    self.vehicles_in_intersection[vehicle_id] = {
                        'in_lane': True,
                        'entry_time': None,  # Will be set when first stops
                        'lane': lane,
                        'edge': edge,
                        'pcu': self.get_pcu_value(vehicle_id)
                    }
                
                # Vehicle is in lane but hasn't stopped yet -> capture first stop time
                elif is_in_controlled_lane and vehicle_id in self.vehicles_in_intersection:
                    vehicle_data = self.vehicles_in_intersection[vehicle_id]
                    # First stop detected (speed < threshold and entry_time not yet set)
                    if speed < VEHICLE_STOP_THRESHOLD and vehicle_data['entry_time'] is None:
                        vehicle_data['entry_time'] = current_time
                
                # Vehicle leaves intersection (was in controlled lane, now not)
                elif not is_in_controlled_lane and vehicle_id in self.vehicles_in_intersection:
                    entry_data = self.vehicles_in_intersection.pop(vehicle_id)
                    
                    # Only count if vehicle actually stopped in the lane
                    if entry_data['entry_time'] is not None:
                        wait_time = current_time - entry_data['entry_time']
                        
                        self.completed_wait_times.append({
                            'vehicle_id': vehicle_id,
                            'entry_time': entry_data['entry_time'],
                            'exit_time': current_time,
                            'wait_time': wait_time,
                            'pcu': entry_data['pcu']
                        })
                    
            except Exception as e:
                print(f"Error in update_wait_times for {vehicle_id}: {e}")
    
    def get_average_wait_time(self):
        """
        Calculate PCU-weighted average wait time
        Formula: (sum of wait_time * PCU) / sum of PCU
        """
        if not self.completed_wait_times:
            return 0.0
        
        total_weighted_wait = sum(wt['wait_time'] * wt['pcu'] for wt in self.completed_wait_times)
        total_pcu = sum(wt['pcu'] for wt in self.completed_wait_times)
        
        if total_pcu == 0:
            return 0.0
        
        return total_weighted_wait / total_pcu
    
    def update_queue_lengths(self, current_time):
        """
        Track queue lengths at moments when traffic lights transition from red to green
        Queue length = PCU-weighted sum of vehicles waiting (speed < 0.5 m/s)
        """
        for tls_id in FIXED_TIME_PLANS:
            try:
                current_state = traci.trafficlight.getRedYellowGreenState(tls_id)
                previous_state = self.previous_tls_states[tls_id]
                
                # Check for red to green transition (any 'r' -> 'G')
                is_red_to_green = False
                if previous_state is not None:
                    # Check if any light changed from red to green
                    for prev_char, curr_char in zip(previous_state, current_state):
                        if prev_char == 'r' and curr_char == 'G':
                            is_red_to_green = True
                            break
                
                # If transition detected, calculate queue length
                if is_red_to_green:
                    queue_length_pcu = self._calculate_queue_length_for_tls(tls_id)
                    self.queue_lengths.append({
                        'tls_id': tls_id,
                        'time': current_time,
                        'queue_length_pcu': queue_length_pcu
                    })
                
                # Update previous state
                self.previous_tls_states[tls_id] = current_state
                
            except Exception as e:
                print(f"Error updating queue lengths for {tls_id}: {e}")
    
    def _calculate_queue_length_for_tls(self, tls_id):
        """
        Calculate PCU-weighted queue length at a specific intersection
        Queue = vehicles in controlled lanes with speed < 0.5 m/s
        """
        try:
            controlled_lanes = traci.trafficlight.getControlledLanes(tls_id)
            queue_length_pcu = 0.0
            
            for lane in controlled_lanes:
                # Get all vehicles currently in this lane
                vehicles_in_lane = traci.lane.getLastStepVehicleIDs(lane)
                
                for vehicle_id in vehicles_in_lane:
                    try:
                        speed = traci.vehicle.getSpeed(vehicle_id)
                        # Vehicles with speed below threshold are considered queued
                        if speed < VEHICLE_STOP_THRESHOLD:
                            pcu = self.get_pcu_value(vehicle_id)
                            queue_length_pcu += pcu
                    except Exception as e:
                        print(f"Error calculating queue for vehicle {vehicle_id}: {e}")
            
            return queue_length_pcu
        except Exception as e:
            print(f"Error calculating queue length for {tls_id}: {e}")
            return 0.0
    
    def get_average_queue_length(self):
        """
        Calculate average queue length across all red to green transitions
        Formula: (sum of queue_length_pcu) / count of measurements
        """
        if not self.queue_lengths:
            return 0.0
        
        total_queue = sum(ql['queue_length_pcu'] for ql in self.queue_lengths)
        return total_queue / len(self.queue_lengths)
    
    def get_max_queue_length(self):
        """Get maximum queue length recorded"""
        if not self.queue_lengths:
            return 0.0
        return max(ql['queue_length_pcu'] for ql in self.queue_lengths)
    
    def update_travel_times(self, current_time):
        """
        Track vehicles' travel time in the controlled network
        Entry: when vehicle first appears in controlled area
        Exit: when vehicle leaves controlled area
        Travel time = exit_time - entry_time
        """
        current_vehicles = traci.vehicle.getIDList()
        
        # Track vehicles currently in network
        for vehicle_id in current_vehicles:
            if vehicle_id not in self.vehicles_in_network:
                # Vehicle just entered controlled area
                try:
                    self.vehicles_in_network[vehicle_id] = {
                        'entry_time': current_time,
                        'pcu': self.get_pcu_value(vehicle_id)
                    }
                except Exception as e:
                    print(f"Error tracking vehicle {vehicle_id} entry: {e}")
        
        # Check for vehicles that left
        vehicles_to_remove = []
        for vehicle_id in self.vehicles_in_network:
            if vehicle_id not in current_vehicles:
                travel_data = self.vehicles_in_network[vehicle_id]
                travel_time = current_time - travel_data['entry_time']
                
                self.completed_travel_times.append({
                    'vehicle_id': vehicle_id,
                    'entry_time': travel_data['entry_time'],
                    'exit_time': current_time,
                    'travel_time': travel_time,
                    'pcu': travel_data['pcu']
                })
                
                vehicles_to_remove.append(vehicle_id)
        
        for vehicle_id in vehicles_to_remove:
            del self.vehicles_in_network[vehicle_id]
    
    def get_average_travel_time(self):
        """
        Calculate average travel time
        Formula: (sum of travel_time) / count
        """
        if not self.completed_travel_times:
            return 0.0
        
        total_travel = sum(tt['travel_time'] for tt in self.completed_travel_times)
        return total_travel / len(self.completed_travel_times)
    
    def get_max_travel_time(self):
        """Get maximum travel time recorded"""
        if not self.completed_travel_times:
            return 0.0
        return max(tt['travel_time'] for tt in self.completed_travel_times)
    
    def get_min_travel_time(self):
        """Get minimum travel time recorded"""
        if not self.completed_travel_times:
            return 0.0
        return min(tt['travel_time'] for tt in self.completed_travel_times)


class TrafficLightController:
    def __init__(self):
        # Track priority status per traffic light (independent for each TLS)
        self.priority_status = {
            tls_id: {
                'active': False,
                'start_time': 0,
                'direction': None
            }
            for tls_id in FIXED_TIME_PLANS
        }
        # Track fixed-time cycle for smooth transitions
        self.fixed_time_cycle_start = {tls_id: 0 for tls_id in FIXED_TIME_PLANS}
        self.cycle_time = {tls_id: sum(duration for duration, _ in FIXED_TIME_PLANS[tls_id]) 
                          for tls_id in FIXED_TIME_PLANS}

    def get_green_state_for_direction(self, tls_id, incoming_edge):
        """
        Create a green state for the given direction at a traffic light
        All incoming edges get red, only the specified direction gets green
        """
        # Get the reference state (first green state)
        reference_state = FIXED_TIME_PLANS[tls_id][0][1]
        
        # Create all-red state
        new_state = list("r" * len(reference_state))
        
        # Set green for the priority direction
        if tls_id in TL_STATES and incoming_edge in TL_STATES[tls_id]:
            indices = TL_STATES[tls_id][incoming_edge]
            for idx in indices:
                new_state[idx] = "G"
            print(f"DEBUG: Setting green for {tls_id} from {incoming_edge}, indices: {indices}, state: {''.join(new_state)}")
        else:
            print(f"WARNING: No mapping found for {tls_id} from {incoming_edge}")
            print(f"Available edges: {list(TL_STATES.get(tls_id, {}).keys())}")
        
        return "".join(new_state)

    def detect_leading_car_at_intersection(self, current_time):
        """
        Detect if LEADING_CAR is approaching any intersection
        Returns (tls_id, incoming_edge) or (None, None)
        """
        try:
            # Check if vehicle exists
            if LEADING_CAR_ID not in traci.vehicle.getIDList():
                return None, None
            
            # Get vehicle position
            vehicle_lane = traci.vehicle.getLaneID(LEADING_CAR_ID)
            vehicle_pos = traci.vehicle.getLanePosition(LEADING_CAR_ID)
            
            # Extract edge from lane ID (format: edgeID_laneIndex)
            if "_" in vehicle_lane:
                current_edge = vehicle_lane.rsplit("_", 1)[0]
            else:
                current_edge = vehicle_lane
            
            # Get distance to end of lane
            lane_length = traci.lane.getLength(vehicle_lane)
            distance_to_end = lane_length - vehicle_pos
            
            # Check each intersection
            for tls_id in FIXED_TIME_PLANS:
                # Check which edges lead to this intersection
                incoming_lanes = traci.trafficlight.getControlledLanes(tls_id)
                
                if not incoming_lanes:
                    continue
                
                for lane in incoming_lanes:
                    if "_" in lane:
                        edge_of_lane = lane.rsplit("_", 1)[0]
                    else:
                        edge_of_lane = lane
                    
                    # Check if vehicle is on an incoming edge and close enough
                    if current_edge == edge_of_lane and distance_to_end <= DETECTION_DISTANCE:
                        print(f"[{current_time:.1f}s] LEADING_CAR detected approaching {tls_id} from {current_edge}")
                        return tls_id, current_edge
            
            return None, None
            
        except Exception as e:
            print(f"Error detecting vehicle: {e}")
            return None, None

    def activate_priority(self, tls_id, incoming_edge, current_time):
        """Activate priority mode for LEADING_CAR at specific traffic light"""
        self.priority_status[tls_id]['active'] = True
        self.priority_status[tls_id]['start_time'] = current_time
        self.priority_status[tls_id]['direction'] = incoming_edge
        print(f"[{current_time:.1f}s] PRIORITY ACTIVATED for {tls_id} from {incoming_edge}")

    def deactivate_priority(self, tls_id, current_time):
        """Deactivate priority mode for specific traffic light"""
        self.priority_status[tls_id]['active'] = False
        # Reset cycle start time to ensure smooth transition back to fixed-time
        self.fixed_time_cycle_start[tls_id] = current_time
        print(f"[{current_time:.1f}s] PRIORITY DEACTIVATED for {tls_id}, returning to fixed-time control")

    def apply_fixed_time_control(self, tls_id, current_time):
        """
        Apply fixed-time signal control based on predefined plan
        Uses absolute time to calculate which phase should be active
        """
        plan = FIXED_TIME_PLANS[tls_id]
        
        # Calculate elapsed time since cycle start
        cycle_duration = self.cycle_time[tls_id]
        elapsed = (current_time - self.fixed_time_cycle_start[tls_id]) % cycle_duration
        
        # Find which phase should be active
        current_position = 0
        for phase_idx, (duration, state) in enumerate(plan):
            if current_position + duration > elapsed:
                try:
                    traci.trafficlight.setRedYellowGreenState(tls_id, state)
                except Exception as e:
                    print(f"Error setting state for {tls_id}: {e}")
                return
            current_position += duration

    def control_traffic_lights(self, current_time):
        """Main control logic - manage each traffic light independently"""
        
        # Check for approaching LEADING_CAR and detect at which intersections
        for tls_id in FIXED_TIME_PLANS:
            # Check if this traffic light currently has active priority
            if self.priority_status[tls_id]['active']:
                # Check if priority duration has expired
                elapsed = current_time - self.priority_status[tls_id]['start_time']
                
                if elapsed >= PRIORITY_DURATION:
                    # Priority timeout - deactivate and return to fixed-time
                    self.deactivate_priority(tls_id, current_time)
                    # Apply fixed-time control immediately after deactivation
                    self.apply_fixed_time_control(tls_id, current_time)
                else:
                    # Keep priority active - apply green light for priority direction
                    try:
                        priority_direction = self.priority_status[tls_id]['direction']
                        green_state = self.get_green_state_for_direction(tls_id, priority_direction)
                        traci.trafficlight.setRedYellowGreenState(tls_id, green_state)
                    except Exception as e:
                        print(f"Error applying priority for {tls_id}: {e}")
                    continue
            
            # If no priority active, check if LEADING_CAR is approaching this intersection
            detected_edge = self.detect_leading_car_at_intersection_for_tls(tls_id, current_time)
            
            if detected_edge:
                # Activate priority for this traffic light
                self.activate_priority(tls_id, detected_edge, current_time)
                try:
                    green_state = self.get_green_state_for_direction(tls_id, detected_edge)
                    traci.trafficlight.setRedYellowGreenState(tls_id, green_state)
                except Exception as e:
                    print(f"Error applying priority for {tls_id}: {e}")
            else:
                # No priority - apply fixed-time control
                self.apply_fixed_time_control(tls_id, current_time)

    def detect_leading_car_at_intersection_for_tls(self, tls_id, current_time):
        """
        Detect if LEADING_CAR is approaching a specific traffic light
        Returns incoming_edge or None
        """
        try:
            # Check if vehicle exists
            if LEADING_CAR_ID not in traci.vehicle.getIDList():
                return None
            
            # Get vehicle position
            vehicle_lane = traci.vehicle.getLaneID(LEADING_CAR_ID)
            vehicle_pos = traci.vehicle.getLanePosition(LEADING_CAR_ID)
            
            # Extract edge from lane ID (format: edgeID_laneIndex)
            if "_" in vehicle_lane:
                current_edge = vehicle_lane.rsplit("_", 1)[0]
            else:
                current_edge = vehicle_lane
            
            # Get distance to end of lane
            lane_length = traci.lane.getLength(vehicle_lane)
            distance_to_end = lane_length - vehicle_pos
            
            # Check which edges lead to this specific intersection
            incoming_lanes = traci.trafficlight.getControlledLanes(tls_id)
            
            if not incoming_lanes:
                return None
            
            for lane in incoming_lanes:
                if "_" in lane:
                    edge_of_lane = lane.rsplit("_", 1)[0]
                else:
                    edge_of_lane = lane
                
                # Check if vehicle is on an incoming edge and close enough
                if current_edge == edge_of_lane and distance_to_end <= DETECTION_DISTANCE:
                    if not self.priority_status[tls_id]['active']:  # Only detect if not already active
                        print(f"[{current_time:.1f}s] LEADING_CAR detected approaching {tls_id} from {current_edge}")
                    return current_edge
            
            return None
            
        except Exception as e:
            print(f"Error detecting vehicle for {tls_id}: {e}")
            return None


def main():
    print("Starting SUMO simulation with LEADING_CAR priority control...")
    print(f"LEADING_CAR ID: {LEADING_CAR_ID}")
    print(f"Priority duration: {PRIORITY_DURATION} seconds")
    print(f"Detection distance: {DETECTION_DISTANCE} meters")
    print("-" * 60)
    
    try:
        traci.start(SUMO_CMD)
        controller = TrafficLightController()
        metrics_collector = MetricsCollector()
        
        step = 0
        max_steps = 500  # Run for more steps
        
        while step < max_steps:
            traci.simulationStep()
            current_time = traci.simulation.getTime()  # Get actual simulation time
            
            # Control traffic lights
            controller.control_traffic_lights(current_time)
            
            # Update metrics
            metrics_collector.update_metrics(current_time)
            
            # Print status periodically
            if step % 100 == 0:
                vehicle_list = traci.vehicle.getIDList()
                leading_car_status = "ACTIVE" if LEADING_CAR_ID in vehicle_list else "NOT IN NETWORK"
                priority_status = ", ".join([f"{tls}:{'ON' if controller.priority_status[tls]['active'] else 'OFF'}" 
                                            for tls in FIXED_TIME_PLANS])
                print(f"[Step {step:4d}] Time: {current_time:6.1f}s | Leading Car: {leading_car_status} | "
                      f"Priority: {priority_status}")
            
            step += 1
        
        print("-" * 60)
        print("Simulation completed")
        
        # Calculate and display metrics
        avg_wait_time = metrics_collector.get_average_wait_time()
        avg_queue_length = metrics_collector.get_average_queue_length()
        max_queue_length = metrics_collector.get_max_queue_length()
        avg_travel_time = metrics_collector.get_average_travel_time()
        max_travel_time = metrics_collector.get_max_travel_time()
        min_travel_time = metrics_collector.get_min_travel_time()
        
        print(f"\n===== METRICS RESULTS =====")
        print(f"Average Wait Time (PCU-weighted): {avg_wait_time:.2f} seconds")
        print(f"Total vehicles waited: {len(metrics_collector.completed_wait_times)}")
        print(f"\nAverage Queue Length (PCU-weighted): {avg_queue_length:.2f}")
        print(f"Maximum Queue Length (PCU): {max_queue_length:.2f}")
        print(f"Queue measurements taken: {len(metrics_collector.queue_lengths)}")
        print(f"\nAverage Travel Time: {avg_travel_time:.2f} seconds")
        print(f"Minimum Travel Time: {min_travel_time:.2f} seconds")
        print(f"Maximum Travel Time: {max_travel_time:.2f} seconds")
        print(f"Total vehicles traveled: {len(metrics_collector.completed_travel_times)}")
        
        # Save metrics to JSON file
        metrics_file = "traffic_metrics.json"
        metrics_data = {
            'timestamp': datetime.now().isoformat(),
            'average_wait_time': avg_wait_time,
            'total_vehicles_waited': len(metrics_collector.completed_wait_times),
            'average_queue_length': avg_queue_length,
            'max_queue_length': max_queue_length,
            'queue_measurements': len(metrics_collector.queue_lengths),
            'average_travel_time': avg_travel_time,
            'min_travel_time': min_travel_time,
            'max_travel_time': max_travel_time,
            'total_vehicles_traveled': len(metrics_collector.completed_travel_times),
            'wait_times_detail': metrics_collector.completed_wait_times,
            'queue_lengths_detail': metrics_collector.queue_lengths,
            'travel_times_detail': metrics_collector.completed_travel_times
        }
        with open(metrics_file, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        print(f"\nMetrics saved to {metrics_file}")
        
        traci.close()
        
    except Exception as e:
        print(f"Error occurred: {e}")
        if traci.isLoaded():
            traci.close()
        sys.exit(1)


if __name__ == "__main__":
    main()
