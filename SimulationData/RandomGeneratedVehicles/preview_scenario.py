
"""
Preview Scenario Script
Generates a random traffic scenario and runs it in SUMO-GUI for viewing.
"""
import time
import traci
import generate_traffic
import os

# Configuration
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INTERSECT_DIR = os.path.join(BASE_DIR, "Intersect_Test")
CONFIG_FILE = os.path.join(INTERSECT_DIR, "config.sumocfg")
SUMO_CMD = ["sumo-gui", "-c", CONFIG_FILE, "--start"]

def preview():
    print("WARNING: This will close any existing SUMO windows.")
    try:
        traci.close()
    except:
        pass

    print("1. Generating new random traffic scenario...")
    # Generate traffic to the file expected by config.sumocfg
    route_file_path = os.path.join(INTERSECT_DIR, "routes.rou.xml")
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
