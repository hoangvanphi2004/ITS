# Dự Án Điều Khiển Đèn Giao Thông Thông Minh (RL-Based ITS)

Dự án này sử dụng Học Tăng Cường (Reinforcement Learning - RL) để tối ưu hóa điều khiển đèn giao thông trong môi trường mô phỏng SUMO. Hệ thống có khả năng tự động sinh ra các kịch bản giao thông thích ứng (adaptive scenarios) với nhiều loại phương tiện khác nhau.

## 1. Cấu Trúc Dự Án

Thư mục làm việc chính: `RandomGeneratedVehicles/`

*   **`sumo_env.py`**: Môi trường kết nối giữa SUMO và RL Model (theo chuẩn Gymnasium). Chịu trách nhiệm gửi lệnh điều khiển đèn và nhận dữ liệu cảm biến.
*   **`generate_traffic.py`**: Script tự động tạo file `routes.rou.xml`. Nó sinh ra ngẫu nhiên các loại xe (xe con, xe buýt, xe tải, xe máy) với tham số thực tế.
*   **`train.py`**: Script huấn luyện cơ bản (chạy 1 môi trường). Dùng để test code hoặc debug.
*   **`train_parallel.py`**: Script huấn luyện song song (Parallel Training) dùng thuật toán PPO. Tận dụng đa nhân CPU để huấn luyện nhanh gấp nhiều lần.
*   **`preview_scenario.py`**: Script để xem trước 1 kịch bản mô phỏng trên giao diện SUMO-GUI mà không cần chạy training.
*   **`rl_agent.py`**: Khung sườn (Skeleton) cho agent RL (hiện tại `train_parallel.py` dùng thư viện `stable-baselines3` nên file này chỉ để tham khảo cho việc custom sau này).

## 2. Cài Đặt

Yêu cầu: Python 3.8+, SUMO 1.18+ đã được cài đặt và biến môi trường `SUMO_HOME` đã được thiết lập.

Cài đặt các thư viện Python cần thiết:

```powershell
pip install -r requirements.txt
```

Nếu muốn dùng GPU cho PPO (PyTorch):
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## 3. Hướng Dẫn Sử Dụng

### A. Xem Trước Mô Phỏng (Preview)
Để kiểm tra xem kịch bản giao thông được sinh ra như thế nào, chạy:

```powershell
python preview_scenario.py
```
*   Cửa sổ SUMO sẽ hiện lên. Bấm nút **Play** (tam giác xanh) để chạy.
*   Script này đã được chỉnh chậm lại (delay 50ms) để mắt thường dễ quan sát xe cộ.

### B. Huấn Luyện Song Song (Khuyên Dùng)
Để tận dụng CPU nhiều nhân (ví dụ 20 core), chạy lệnh sau:

```powershell
python train_parallel.py --steps 100000 --cpu 16
```
*   `--steps`: Tổng số bước huấn luyện (ví dụ 10 triệu thì gõ 10000000).
*   `--cpu`: Số lượng môi trường chạy song song (nên để nhỏ hơn số luồng CPU của bạn một chút, ví dụ máy 28 thread thì để 20-24).
*   `--mode`: Chọn chế độ điều khiển:
    *   `phase_selection`: Model chọn pha đèn tiếp theo (Mặc định).
    *   `time_adjust`: Model tăng/giảm thời gian xanh.

Kết quả huấn luyện (Model) sẽ được lưu thành file `ppo_traffic_final.zip`.

### C. Huấn Luyện Cơ Bản (Single Process)
Chỉ dùng để debug lỗi:

```powershell
python train.py --episodes 5 --gui
```
*   Thêm cờ `--gui` nếu muốn xem giao diện (sẽ rất chậm).


## 5. Quy Trình End-to-End (Pipeline)

Quy trình chuẩn để phát triển và triển khai Model RL của bạn:

### Giai đoạn 1: Huấn Luyện (Training)
Chạy trên môi trường song song để tiết kiệm thời gian.
```powershell
# Chạy 20 CPU, 10 triệu bước (mất khoảng 2 tiếng)
cd RandomGeneratedVehicles
python train_parallel.py --steps 10000000 --cpu 20
```
*   **Output**: File `ppo_traffic_final.zip` (đây là "bộ não" của AI).

### Giai đoạn 2: Đánh Giá & Trực Quan (Evaluation)
Chạy thử "bộ não" này trên môi trường có giao diện để kiểm tra xem nó có thông minh thật không.
```powershell
# Chạy xem thử 1000 bước
python evaluate_model.py --model ppo_traffic_final
```
*   **Quan sát**:
    *   AI có ưu tiên làn đường đông xe không?
    *   AI có ưu tiên Xe Buýt (như config PCU) không?
    *   Đèn có bị nhấp nháy quá nhanh không?

### Giai đoạn 3: Tinh Chỉnh (Tuning)
Nếu kết quả chưa tốt (ví dụ: vẫn còn tắc đường), bạn cần quay lại sửa:
1.  **Sửa Reward**: Vào `sumo_env.py` chỉnh trọng số.
    *   Muốn ưu tiên giải tỏa tắc nghẽn -> Tăng hệ số `Queue` lên 1.5.
    *   Muốn giảm thời gian chờ -> Tăng hệ số `Wait` lên 0.2.
2.  **Sửa Input**: Thêm thông tin cho AI (ví dụ: vận tốc xe).
3.  **Lặp lại Giai đoạn 1**.


*   **Thay đổi lưu lượng xe**: Mở file `generate_traffic.py`, sửa tham số `num_vehicles` hoặc hàm `generate_route_file`.
*   **Thay đổi phần thưởng (Reward)**: Mở file `sumo_env.py`, tìm hàm `step()`. Hiện tại phần thưởng là âm của tổng thời gian chờ và độ dài hàng đợi (`reward = - (queue + wait)`).

---
**Lưu ý**: Khi chạy `train_parallel.py`, không nên bật GUI vì sẽ mở hàng chục cửa sổ SUMO cùng lúc gây treo máy.
