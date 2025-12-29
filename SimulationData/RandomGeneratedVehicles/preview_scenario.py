
"""
Preview Scenario Script
Generates a random traffic scenario and runs it in SUMO-GUI for viewing.
"""
import time
import traci
import generate_traffic
import os

# Configuration
CONFIG_FILE = "../SimulationData/Intersect_Test/config.sumocfg"
SUMO_CMD = ["sumo-gui", "-c", CONFIG_FILE, "--start"]

def preview():
    print("WARNING: This will close any existing SUMO windows.")
    try:
        traci.close()
    except:
        pass

    print("1. Generating new random traffic scenario...")
    # Generate traffic to the file expected by config.sumocfg
    # Note: config points to routes.rou.xml in the same folder usually, 
    # but our generate_traffic writes to "SimulationData/Intersect_Test/routes.rou.xml" by default in the previous runs?
    # Let's check generate_traffic.py defaults. 
    # It defaults to "routes.rou.xml" in current dir if not specified, but in train.py we called it with specific path.
    
    route_file_path = "../SimulationData/Intersect_Test/routes.rou.xml"
    generate_traffic.generate_route_file(route_file_path, num_vehicles=100)
    print(f"   Scenario saved to: {route_file_path}")

    print("2. Starting SUMO-GUI...")
    traci.start(SUMO_CMD)
    
    print("3. Running simulation...")
    step = 0
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        # Slow down to make it visible (50ms delay)
        time.sleep(0.05) 
        step += 1
        
    print(f"Simulation finished after {step} steps.")
    
    print("Simulation finished. Press Enter to close the SUMO window...")
    input()
    
    try:
        traci.close()
    except:
        pass

if __name__ == "__main__":
    preview()
