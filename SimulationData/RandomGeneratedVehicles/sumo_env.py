import gymnasium as gym
from gymnasium import spaces
import traci
import numpy as np
import sys
import os

# Cố gắng import module tạo traffic (nếu có)
try:
    import generate_traffic
except ImportError:
    generate_traffic = None

# Bảng quy đổi hệ số xe (Passenger Car Unit - PCU)
# Dùng để tính toán tải trọng thực tế của dòng xe
PCU_MAPPING = {
    'passenger': 1.0,   # Xe con
    'truck': 2.0,       # Xe tải (chiếm chỗ gấp đôi)
    'bus': 2.5,         # Xe buýt
    'motorcycle': 0.5   # Xe máy
}

# Import luật giao thông an toàn (nếu có)
try:
    from traffic_rules import TrafficRules
except ImportError:
    TrafficRules = None

class SumoGymEnv(gym.Env):
    """
    Môi trường SUMO Gymnasium cho bài toán điều khiển đèn giao thông.
    Tích hợp hệ thống theo dõi 3 chỉ số KPI chính:
    1. Average Wait Time: Thời gian chờ trung bình (tính khi xe rời hệ thống).
    2. Average Queue Length: Độ dài hàng chờ trung bình (đo tại thời điểm đèn chuyển xanh).
    3. Average Travel Time: Thời gian hành trình trung bình.
    """
    
    def __init__(self, mode='delta_time', gui=False, max_steps=1000, rank=0, num_vehicles=1000, seed=42):
        super(SumoGymEnv, self).__init__()
        
        # --- Cấu hình Môi trường ---
        self.mode = mode
        self.gui = gui
        self.max_steps = max_steps      # Số bước tối đa cho mỗi episode
        self.rank = rank                # ID của môi trường (dùng khi chạy song song)
        self.num_vehicles = num_vehicles # Số lượng xe cho kịch bản
        self.scenario_seed = seed       # Seed ngẫu nhiên để tái tạo kịch bản
        
        self.step_counter = 0
        self.episode_count = 0
        
        # --- Cấu hình Đèn tín hiệu ---
        self.baseline_green_time = 35.0 # Thời gian xanh cơ sở
        self.seconds_range = 45.0       # Biên độ điều chỉnh (+/- 45s)
        self.tls_id = "J1"              # ID của đèn giao thông trong SUMO
        self.green_phases = [0, 2]      # Index của các pha đèn xanh trong file net.xml
        self.current_green_phase_index = 0 
        
        # Đường dẫn file config
        self.route_file = f"../Intersect_Test/routes_{self.rank}.rou.xml"
        self.sumo_binary = "sumo-gui" if gui else "sumo"
        self.sumo_cmd = [
            self.sumo_binary,
            "-c", "../Intersect_Test/config.sumocfg",
            "--route-files", self.route_file,
            "--start",
            "--quit-on-end"
        ]
        
        # --- Lớp An toàn (Safety Layer) ---
        if TrafficRules:
            self.rules = TrafficRules(min_green_time=15)
        else:
            self.rules = None

        # --- Không gian Hành động & Quan sát ---
        # Action: 1 giá trị liên tục trong khoảng [-1, 1]
        # Giá trị này sẽ được nhân với seconds_range để ra delta time thực tế.
        self.action_space = spaces.Box(low=np.array([-1.0]), high=np.array([1.0]), dtype=np.float32)
        
        # Observation: 9 giá trị [Q_N, Q_E, Q_S, Q_W, L_N, L_E, L_S, L_W, Phase]
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(9,), dtype=np.float32)
        
        self.direction_lanes = [] # Sẽ được điền tự động khi chạy
        
        # --- Biến theo dõi KPI (KPI Tracking) ---
        self.vehicle_db = {} # Lưu: {veh_id: {entry_time, last_wait, pcu}}
        self.kpi_history = {
            "wait_times": [],       # Lưu (wait_time, pcu) của xe đã hoàn thành
            "queue_lengths": [],    # Lưu queue length tại thời điểm chuyển xanh
            "travel_times": []      # Lưu (travel_time, pcu) của xe đã hoàn thành
        }
        self.last_phase = -1 # Để phát hiện thời điểm chuyển pha

    def _get_start_vehicle_type_pcu(self, vtype):
        """Chuyển đổi loại xe sang hệ số PCU."""
        if vtype in PCU_MAPPING: return PCU_MAPPING[vtype]
        vtype_lower = vtype.lower()
        for key, pcu in PCU_MAPPING.items():
            if key in vtype_lower: return pcu
        return 1.0 # Mặc định là 1.0 nếu không tìm thấy

    def _map_lanes_dynamic(self):
        """
        Tự động phát hiện và gom nhóm các làn đường kết nối tới đèn tín hiệu.
        Giúp code chạy được trên nhiều map khác nhau mà không cần sửa cứng ID làn.
        """
        try:
            controlled_lanes = list(set(traci.trafficlight.getControlledLanes(self.tls_id)))
        except traci.TraCIException:
            return

        lane_groups = {}
        for lane in controlled_lanes:
            try:
                # Gom nhóm theo Edge ID (Ví dụ: lane E1_0, E1_1 thuộc edge E1)
                edge_id = traci.lane.getEdgeID(lane)
                if edge_id not in lane_groups: lane_groups[edge_id] = []
                lane_groups[edge_id].append(lane)
            except: continue
        
        # Sắp xếp để đảm bảo thứ tự quan sát cố định (Bắc-Đông-Nam-Tây)
        sorted_keys = sorted(lane_groups.keys())
        self.direction_lanes = [lane_groups[k] for k in sorted_keys]
        
        # Đảm bảo luôn có đủ 4 hướng (padding list rỗng nếu thiếu)
        while len(self.direction_lanes) < 4: 
            self.direction_lanes.append([])

    def reset(self, seed=None, options=None):
        """Khởi động lại môi trường cho episode mới."""
        super().reset(seed=seed)
        self.episode_count += 1
        
        # Reset các biến theo dõi KPI
        self.vehicle_db = {}
        self.kpi_history = {"wait_times": [], "queue_lengths": [], "travel_times": []}
        self.last_phase = -1
        
        # 1. Tạo file giao thông mới (nếu module có sẵn)
        if generate_traffic:
            try:
                if self.rank == 0: 
                    print(f"Generating traffic (Rank {self.rank}, Seed {self.scenario_seed})...")
                # Gọi hàm generate với tham số đã lưu trong __init__
                generate_traffic.generate_route_file(
                    self.route_file,
                    num_vehicles=self.num_vehicles,
                    seed=self.scenario_seed
                )
            except Exception as e:
                # Fallback nếu hàm generate có signature khác
                pass 
            
        # 2. Khởi động SUMO
        try: traci.close()
        except: pass
            
        try: 
            # Không truyền port để traci tự tìm port rảnh
            traci.start(self.sumo_cmd)
        except Exception as e: 
            print(f"Error starting SUMO: {e}")
            return self.observation_space.sample(), {}
        
        # 3. Map làn đường (Chạy 1 lần sau khi SUMO start)
        self._map_lanes_dynamic()
        self.step_counter = 0
        self.current_green_phase_index = 0
        
        return self._get_observation(), {}
        
    def step(self, action):
        """Thực hiện một bước hành động (Action -> Next State, Reward)."""
        self.step_counter += 1
        terminated = False
        truncated = False
        
        # --- Xử lý Action ---
        # Action [-1, 1] -> Delta Time [-45s, +45s]
        raw_action = float(action[0])
        delta = raw_action * self.seconds_range
        target_duration = self.baseline_green_time + delta
        
        # Áp dụng giới hạn an toàn (Min/Max Green Time)
        min_g = 15.0
        max_g = 90.0
        if self.rules:
            min_g = self.rules.min_green_time
            if hasattr(self.rules, 'max_green_time'): max_g = self.rules.max_green_time
        
        target_duration = np.clip(target_duration, min_g, max_g)
        
        # --- Thiết lập Phase Đèn ---
        phase_id = self.green_phases[self.current_green_phase_index]
        
        # [KPI 2] Ghi nhận hàng đợi ngay thời điểm chuyển xanh (Green Onset)
        if phase_id != self.last_phase:
            self._record_queue_at_green_onset(phase_id)
            self.last_phase = phase_id

        traci.trafficlight.setPhase(self.tls_id, phase_id)
        
        # Biến tích lũy để tính Reward
        accumulated_queue = 0
        accumulated_wait_time = 0 # Dùng Wait Time thay vì Loss Time cho Reward
        steps_run = 0
        
        # --- Vòng lặp mô phỏng (Pha Xanh) ---
        steps_green = int(target_duration)
        for _ in range(steps_green):
            traci.simulationStep()
            steps_run += 1
            self._update_vehicle_tracker() # Cập nhật KPI từng giây
            
            # Lấy thông tin quan sát tức thời
            obs = self._get_observation()
            
            # Cộng dồn chỉ số để tính trung bình sau này
            q_pcu = np.sum(obs[:4])     # Tổng hàng đợi 4 hướng
            w_pcu = np.sum(obs[4:8])    # Tổng thời gian chờ/loss 4 hướng (Lưu ý: obs trả về TimeLoss hoặc WaitTime tùy cài đặt ở dưới)
            
            accumulated_queue += q_pcu
            accumulated_wait_time += w_pcu
            
            # Kiểm tra xem simulation đã hết xe chưa
            if traci.simulation.getMinExpectedNumber() <= 0:
                terminated = True
                break
        
        # --- Vòng lặp mô phỏng (Pha Vàng - Cố định 3s) ---
        if not terminated:
            yellow_phase = phase_id + 1
            all_programs = traci.trafficlight.getAllProgramLogics(self.tls_id)
            if len(all_programs) > 0 and yellow_phase < len(all_programs[0].phases):
                traci.trafficlight.setPhase(self.tls_id, yellow_phase)
            
            for _ in range(3):
                traci.simulationStep()
                steps_run += 1
                self._update_vehicle_tracker() 
                
                obs = self._get_observation()
                accumulated_queue += np.sum(obs[:4])
                accumulated_wait_time += np.sum(obs[4:8])
        
        # --- Tính toán Reward (Quan trọng) ---
        # Reward = -(Trung bình Queue + Trọng số * Trung bình Wait Time)
        # Chia cho steps_run để chuẩn hóa, tránh việc Agent chọn phase ngắn để né phạt.
        if steps_run > 0:
            avg_queue = accumulated_queue / steps_run
            avg_wait = accumulated_wait_time / steps_run
            
            # Hệ số 0.05 giúp cân bằng vì Wait Time tích lũy thường lớn hơn Queue nhiều
            total_reward = - (1.0 * avg_queue + 0.05 * avg_wait)
        else:
            total_reward = 0
        
        # Chuyển sang phase xanh tiếp theo cho bước sau
        self.current_green_phase_index = (self.current_green_phase_index + 1) % len(self.green_phases)

        # Kiểm tra điều kiện dừng (Max Steps)
        if self.step_counter >= self.max_steps:
            truncated = True
            
        observation = self._get_observation()
        
        # Lấy thống kê KPI hiện tại để in ra log
        kpi_stats = self.get_kpi_stats()
        
        info = {
            "step_duration": target_duration,
            "avg_queue_pcu": accumulated_queue / max(1, steps_run),
            "KPI_AvgWait": kpi_stats["avg_wait_time"],
            "KPI_AvgQueue": kpi_stats["avg_queue_len"],
            "KPI_AvgTravel": kpi_stats["avg_travel_time"]
        }
        
        return observation, total_reward, terminated, truncated, info

    def _update_vehicle_tracker(self):
        """
        Hàm chạy ngầm mỗi giây để theo dõi từng xe.
        Mục đích: Tính chính xác Travel Time và Wait Time khi xe rời mạng lưới.
        """
        current_time = traci.simulation.getTime()
        current_ids = set(traci.vehicle.getIDList())
        
        # 1. Thêm xe mới vào DB
        for veh_id in current_ids:
            if veh_id not in self.vehicle_db:
                try:
                    vtype = traci.vehicle.getTypeID(veh_id)
                    pcu = self._get_start_vehicle_type_pcu(vtype)
                    self.vehicle_db[veh_id] = {
                        "entry_time": current_time,
                        "pcu": pcu,
                        "last_wait": 0.0
                    }
                except: pass
                
            # Cập nhật wait time tích lũy hiện tại (để lấy giá trị cuối cùng khi xe rời đi)
            try:
                wait_time = traci.vehicle.getAccumulatedWaitingTime(veh_id)
                self.vehicle_db[veh_id]["last_wait"] = wait_time
            except: pass

        # 2. Phát hiện xe đã rời đi (Arrived)
        # Xe có trong DB nhưng không còn trong current_ids -> Đã hoàn thành chuyến đi
        arrived_ids = set(self.vehicle_db.keys()) - current_ids
        
        for veh_id in arrived_ids:
            data = self.vehicle_db[veh_id]
            pcu = data["pcu"]
            
            # [KPI 1] Wait Time cuối cùng
            final_wait = data["last_wait"]
            self.kpi_history["wait_times"].append((final_wait, pcu))
            
            # [KPI 3] Travel Time = Thời gian hiện tại - Thời gian nhập
            travel_time = current_time - data["entry_time"]
            self.kpi_history["travel_times"].append((travel_time, pcu))
            
            # Xóa khỏi DB để tiết kiệm bộ nhớ
            del self.vehicle_db[veh_id]

    def _record_queue_at_green_onset(self, phase_id):
        """
        [KPI 2] Ghi nhận độ dài hàng chờ ngay khoảnh khắc đèn chuyển xanh.
        """
        current_queue_pcu = 0.0
        
        # Quét tất cả các làn đường đang quản lý
        all_lanes = [lane for group in self.direction_lanes for lane in group]
        
        for lane in all_lanes:
            try:
                vehicles = traci.lane.getLastStepVehicleIDs(lane)
                for veh in vehicles:
                    vtype = traci.vehicle.getTypeID(veh)
                    pcu = self._get_start_vehicle_type_pcu(vtype)
                    # Xe được tính là trong hàng chờ nếu vận tốc < 0.1 m/s
                    if traci.vehicle.getSpeed(veh) < 0.1:
                        current_queue_pcu += pcu
            except: pass
            
        self.kpi_history["queue_lengths"].append(current_queue_pcu)

    def get_kpi_stats(self):
        """Tính toán thống kê KPI trung bình từ lịch sử."""
        
        # 1. Avg Wait Time (Weighted by PCU)
        wait_data = self.kpi_history["wait_times"]
        if wait_data:
            total_weighted_wait = sum(w * p for w, p in wait_data)
            total_pcu = sum(p for _, p in wait_data)
            avg_wait = total_weighted_wait / total_pcu if total_pcu > 0 else 0
        else:
            avg_wait = 0
            
        # 2. Avg Queue Length (Unweighted average of snapshots)
        queue_data = self.kpi_history["queue_lengths"]
        avg_queue = np.mean(queue_data) if queue_data else 0
        
        # 3. Avg Travel Time (Weighted by PCU)
        travel_data = self.kpi_history["travel_times"]
        if travel_data:
            total_weighted_tt = sum(t * p for t, p in travel_data)
            total_pcu_tt = sum(p for _, p in travel_data)
            avg_tt = total_weighted_tt / total_pcu_tt if total_pcu_tt > 0 else 0
        else:
            avg_tt = 0
            
        return {
            "avg_wait_time": avg_wait,
            "avg_queue_len": avg_queue,
            "avg_travel_time": avg_tt,
            "vehicles_completed": len(wait_data)
        }

    def _get_observation(self):
        """Thu thập trạng thái môi trường (State)."""
        queues = []
        waits = []
        
        # Duyệt qua các nhóm làn đường (4 hướng)
        for lanes in self.direction_lanes:
            q_val = 0.0
            w_val = 0.0
            for lane in lanes:
                try:
                    vehicles = traci.lane.getLastStepVehicleIDs(lane)
                    for veh in vehicles:
                        vtype = traci.vehicle.getTypeID(veh)
                        pcu = self._get_start_vehicle_type_pcu(vtype)
                        
                        # Tính Queue
                        if traci.vehicle.getSpeed(veh) < 0.1: 
                            q_val += pcu
                        
                        # Tính Accumulated Waiting Time (Tốt hơn TimeLoss cho Reward)
                        # Dùng Accumulated Waiting Time giúp đồng bộ với KPI 1
                        w_time = traci.vehicle.getAccumulatedWaitingTime(veh)
                        w_val += w_time * pcu
                except: pass
            queues.append(q_val)
            waits.append(w_val)
            
        # Đảm bảo vector luôn có kích thước cố định (4 hướng)
        queues = queues[:4]
        waits = waits[:4]
        while len(queues) < 4: queues.append(0.0)
        while len(waits) < 4: waits.append(0.0)

        obs = np.array(queues + waits + [self.current_green_phase_index], dtype=np.float32)
        return obs
        
    def close(self):
        try: traci.close()
        except: pass