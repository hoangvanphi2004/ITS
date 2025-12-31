import gymnasium as gym
from gymnasium import spaces
import traci
import numpy as np
import sys
import os

# Import generate_traffic để tạo kịch bản mới khi reset (nếu có file này)
try:
    import generate_traffic
except ImportError:
    generate_traffic = None

# PCU Mapping: Quy đổi các loại xe ra đơn vị xe con
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
    Môi trường SUMO Gymnasium cho bài toán điều khiển đèn giao thông.
    Đã được tối ưu hóa Reward Function để tránh lỗi 'Time Bias'.
    """
    
    def __init__(self, mode='delta_time', gui=False, max_steps=1000, rank=0):
        super(SumoGymEnv, self).__init__()
        
        self.mode = mode
        self.gui = gui
        self.max_steps = max_steps 
        self.step_counter = 0
        self.rank = rank
        
        # Thời gian xanh cơ sở
        self.baseline_green_time = 35.0
        
        # File route riêng biệt cho từng process (nếu chạy song song)
        self.route_file = f"../Intersect_Test/routes_{self.rank}.rou.xml"
        
        self.sumo_binary = "sumo-gui" if gui else "sumo"
        self.sumo_cmd = [
            self.sumo_binary,
            "-c", "../Intersect_Test/config.sumocfg",
            "--route-files", self.route_file,
            "--start",
            "--quit-on-end"
        ]
        
        self.tls_id = "J1"
        self.green_phases = [0, 2] # Các phase xanh chính
        self.current_green_phase_index = 0 
        
        # Khởi tạo lớp luật giao thông (Safety Layer)
        if TrafficRules:
            self.rules = TrafficRules(min_green_time=15)
        else:
            self.rules = None

        # Không gian hành động: Delta time [-45s, +45s]
        self.action_space = spaces.Box(low=np.array([-45.0]), high=np.array([45.0]), dtype=np.float32)
            
        # Không gian quan sát: 8 chỉ số (Queue & Loss cho 4 hướng) + 1 chỉ số Phase hiện tại
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(9,), dtype=np.float32
        )
        
        # Biến lưu trữ danh sách làn đường theo hướng (được khởi tạo khi reset)
        self.direction_lanes = [] 

    def _get_start_vehicle_type_pcu(self, vtype):
        """Lấy hệ số PCU dựa trên loại xe"""
        if vtype in PCU_MAPPING:
            return PCU_MAPPING[vtype]
        # Tìm kiếm gần đúng (ví dụ 'bus_1' -> 'bus')
        vtype_lower = vtype.lower()
        for key, pcu in PCU_MAPPING.items():
            if key in vtype_lower:
                return pcu
        return 1.0 # Mặc định

    def _map_lanes_dynamic(self):
        """
        Tự động phát hiện các làn đường được điều khiển bởi đèn tín hiệu
        và gom nhóm chúng theo hướng (Edge ID).
        """
        controlled_lanes = list(set(traci.trafficlight.getControlledLanes(self.tls_id)))
        lane_groups = {}
        
        for lane in controlled_lanes:
            edge_id = traci.lane.getEdgeID(lane)
            if edge_id not in lane_groups:
                lane_groups[edge_id] = []
            lane_groups[edge_id].append(lane)
        
        # Sắp xếp theo tên Edge ID để đảm bảo thứ tự N-E-S-W nhất quán giữa các lần chạy
        sorted_keys = sorted(lane_groups.keys())
        self.direction_lanes = [lane_groups[k] for k in sorted_keys]
        
        # Đảm bảo luôn có 4 hướng (padding list rỗng nếu map ít hơn 4 hướng)
        while len(self.direction_lanes) < 4:
            self.direction_lanes.append([])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # 1. Tạo kịch bản giao thông mới (nếu có)
        if generate_traffic:
            if self.rank == 0:
                print(f"Generating new traffic scenario for rank {self.rank}...")
            generate_traffic.generate_route_file(self.route_file)
            
        # 2. Khởi động SUMO
        try:
            traci.close()
        except:
            pass
            
        try:
            traci.start(self.sumo_cmd)
        except Exception as e:
            print(f"Error starting SUMO: {e}")
        
        # 3. Map làn đường động (Chỉ làm 1 lần khi reset)
        self._map_lanes_dynamic()

        self.step_counter = 0
        self.current_green_phase_index = 0
        
        return self._get_observation(), {}
        
    def step(self, action):
        self.step_counter += 1
        terminated = False
        truncated = False
        
        # Lấy hành động (delta)
        delta = float(action[0])
        
        # 1. Tính thời gian đèn xanh mục tiêu
        target_duration = self.baseline_green_time + delta
        
        # 2. Áp dụng giới hạn an toàn (Min/Max Green)
        min_g = 15.0
        max_g = 90.0
        if self.rules:
            min_g = self.rules.min_green_time
            if hasattr(self.rules, 'max_green_time'):
                max_g = self.rules.max_green_time
        
        target_duration = np.clip(target_duration, min_g, max_g)
        
        # 3. Đặt Phase Xanh
        phase_id = self.green_phases[self.current_green_phase_index]
        traci.trafficlight.setPhase(self.tls_id, phase_id)
        
        # Biến để tính Reward trung bình
        accumulated_queue = 0
        accumulated_loss = 0
        steps_run = 0
        
        # 4. Chạy mô phỏng cho Phase Xanh
        steps_green = int(target_duration)
        for _ in range(steps_green):
            traci.simulationStep()
            steps_run += 1
            
            # Lấy quan sát tức thời để cộng dồn
            obs = self._get_observation()
            q_pcu = np.sum(obs[:4])
            l_pcu = np.sum(obs[4:8])
            
            accumulated_queue += q_pcu
            accumulated_loss += l_pcu
            
            if traci.simulation.getMinExpectedNumber() <= 0:
                terminated = True
                break
        
        # 5. Chuyển sang Phase Vàng (Cố định 3s)
        if not terminated:
            yellow_phase = phase_id + 1
            # Kiểm tra xem phase vàng có tồn tại trong file net không
            all_programs = traci.trafficlight.getAllProgramLogics(self.tls_id)
            if len(all_programs) > 0:
                num_phases = len(all_programs[0].phases)
                if yellow_phase < num_phases:
                    traci.trafficlight.setPhase(self.tls_id, yellow_phase)
            
            for _ in range(3): # 3 giây vàng
                traci.simulationStep()
                steps_run += 1
                
                obs = self._get_observation()
                accumulated_queue += np.sum(obs[:4])
                accumulated_loss += np.sum(obs[4:8])
        
        # --- QUAN TRỌNG: TÍNH REWARD ĐÃ CHUẨN HÓA ---
        # Chia tổng phạt cho số giây đã chạy -> Ra mức phạt trung bình/giây
        # Điều này ngăn Agent chọn phase ngắn chỉ để kết thúc sớm.
        if steps_run > 0:
            avg_queue = accumulated_queue / steps_run
            avg_loss = accumulated_loss / steps_run
            # Công thức Reward: Ưu tiên giảm hàng đợi, phụ là giảm thời gian chờ
            total_reward = - (1.0 * avg_queue + 0.1 * avg_loss)
        else:
            total_reward = 0
        
        # 6. Chuẩn bị cho bước tiếp theo
        self.current_green_phase_index = (self.current_green_phase_index + 1) % len(self.green_phases)

        # Kiểm tra giới hạn thời gian (Max Steps)
        if self.step_counter >= self.max_steps:
            truncated = True
            
        observation = self._get_observation()
        
        # Thông tin bổ sung để debug
        info = {
            "avg_queue_pcu": accumulated_queue / max(1, steps_run),
            "avg_loss_pcu": accumulated_loss / max(1, steps_run),
            "actual_duration": target_duration
        }
        
        return observation, total_reward, terminated, truncated, info

    def _get_observation(self):
        """
        Thu thập thông tin quan sát.
        Sử dụng danh sách làn đường đã được map tự động trong self.direction_lanes
        """
        queues = []
        waits = []
        
        # Duyệt qua 4 nhóm làn đường (tương ứng 4 hướng)
        for lanes in self.direction_lanes:
            q_val = 0.0
            w_val = 0.0
            
            for lane in lanes:
                try:
                    # Lấy danh sách xe trên làn
                    vehicles = traci.lane.getLastStepVehicleIDs(lane)
                    
                    for veh in vehicles:
                        vtype = traci.vehicle.getTypeID(veh)
                        pcu = self._get_start_vehicle_type_pcu(vtype)
                        
                        # Queue: Xe di chuyển chậm dưới 0.1 m/s
                        if traci.vehicle.getSpeed(veh) < 0.1:
                            q_val += pcu
                            
                        # TimeLoss: Thời gian mất đi do đi chậm hơn tốc độ tối đa
                        t_loss = traci.vehicle.getTimeLoss(veh)
                        w_val += t_loss * pcu
                except:
                    pass
            
            queues.append(q_val)
            waits.append(w_val)
            
        # Ghép lại thành vector [Q1, Q2, Q3, Q4, L1, L2, L3, L4, PhaseIndex]
        # Padding nếu thiếu hướng đã được xử lý ở _map_lanes_dynamic
        # Chúng ta chỉ lấy 4 phần tử đầu (nếu map có > 4 cạnh tới, ta chỉ lấy 4 cái chính)
        queues = queues[:4]
        waits = waits[:4]
        
        obs = np.array(queues + waits + [self.current_green_phase_index], dtype=np.float32)
        return obs
        
    def close(self):
        try:
            traci.close()
        except:
            pass