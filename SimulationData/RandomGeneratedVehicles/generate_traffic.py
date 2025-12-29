
"""
Adaptive Traffic Generator for SUMO Simulation
Generates routes.rou.xml with custom vehicle types and random traffic flows.
"""

import random
import os

# Configuration
OUTPUT_FILE = "../SimulationData/Intersect_Test/routes.rou.xml"
NET_FILE = "network.net.xml"  # Required for some generators, but here we use simple edge lists if known
SIMULATION_STEPS = 500  # Total duration to generate traffic for

# Define Vehicle Types with custom physical and movement properties
VEHICLE_TYPES = {
    "passenger": {
        "accel": 2.6, "decel": 4.5, "sigma": 0.5, "length": 5.0, "minGap": 2.5, "maxSpeed": 50.0,
        "color": "1,0,0", "probability": 0.7  # 70% chance
    },
    "bus": {
        "accel": 1.2, "decel": 4.0, "sigma": 0.5, "length": 12.0, "minGap": 3.0, "maxSpeed": 30.0,
        "color": "0,0,1", "probability": 0.1  # 10% chance
    },
    "truck": {
        "accel": 1.5, "decel": 3.5, "sigma": 0.5, "length": 10.0, "minGap": 3.0, "maxSpeed": 40.0,
        "color": "0,1,0", "probability": 0.1  # 10% chance
    },
    "motorcycle": {
        "accel": 3.0, "decel": 5.0, "sigma": 0.5, "length": 2.5, "minGap": 1.5, "maxSpeed": 55.0,
        "color": "1,1,0", "probability": 0.1  # 10% chance
    }
}

# Define potential routes (sequence of edges)
# Ideally this should be dynamic, but for now we use known routes from previous analysis
# Format: [list of edges]
ROUTES = [
    ["-E6", "-E2", "-E0"],
    ["-E6", "E4"],
    ["-E4", "-E2", "-E0"],
    ["-E2", "E3"],
    ["-E6", "-E2", "E3"],
    ["E0", "E2", "E4"],
    ["-E1", "-E0"],
    ["E0", "E3"],
    ["-E5", "E6"],
    ["-E4", "-E2", "E3"],
    ["-E2", "E1"],
    ["-E3", "E2", "E6"],
    ["E2", "E4"],
    ["-E5", "E4"],
    ["E0", "E2", "E5"],
    ["E2"],
    ["-E5", "-E2", "E1"],
    ["-E3", "E2", "E4"],
    ["E2", "E5"],
    ["-E1", "E2", "E6"],
    ["E2", "E6"],
    ["-E1", "E2", "E5"]
]

def get_random_vehicle_type():
    """Select a vehicle type based on probability"""
    r = random.random()
    cumulative_prob = 0.0
    for v_type, data in VEHICLE_TYPES.items():
        cumulative_prob += data["probability"]
        if r <= cumulative_prob:
            return v_type
    return "passenger"  # Fallback

def generate_route_file(output_path, num_vehicles=200):
    """Generate the routes.rou.xml file"""
    
    with open(output_path, "w") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<routes>\n')
        
        # 1. Define Vehicle Types
        for v_type, specs in VEHICLE_TYPES.items():
            f.write(f'    <vType id="{v_type}" '
                    f'accel="{specs["accel"]}" decel="{specs["decel"]}" '
                    f'sigma="{specs["sigma"]}" length="{specs["length"]}" '
                    f'minGap="{specs["minGap"]}" maxSpeed="{specs["maxSpeed"]}" '
                    f'color="{specs["color"]}"/>\n')
        
        
        # 2. Generate Vehicles
        vehicles = []
        for i in range(num_vehicles):
            veh_id = f"veh_{i}"
            depart_time = random.uniform(0, SIMULATION_STEPS * 0.8) 
            depart_time = round(depart_time, 2)
            
            v_type = get_random_vehicle_type()
            route_edges = random.choice(ROUTES)
            edges_str = " ".join(route_edges)
            
            vehicles.append({
                "id": veh_id,
                "type": v_type,
                "depart": depart_time,
                "route": edges_str
            })
            
        # Sort by departure time (SUMO requires sorted depart times)
        vehicles.sort(key=lambda x: x["depart"])
        
        for v in vehicles:
            f.write(f'    <vehicle id="{v["id"]}" type="{v["type"]}" depart="{v["depart"]}">\n')
            f.write(f'        <route edges="{v["route"]}"/>\n')
            f.write(f'    </vehicle>\n')
            
        f.write('</routes>\n')
    
    print(f"Generated {num_vehicles} vehicles in {output_path}")

if __name__ == "__main__":
    generate_route_file(OUTPUT_FILE, num_vehicles=300)
