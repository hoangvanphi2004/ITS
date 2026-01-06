import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import argparse

# --- CẤU HÌNH MẶC ĐỊNH ---
DEFAULT_CSV_PATH = "progress.csv"
SMOOTHING = 0.95 # Độ mượt (0-1)

def smooth(scalars, weight):
    """Hàm làm mượt dữ liệu để dễ nhìn xu hướng"""
    if len(scalars) == 0:
        return np.array([])
    last = scalars.iloc[0]
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return np.array(smoothed)

def plot_results(csv_path):
    if not os.path.exists(csv_path):
        print(f"Lỗi: Không tìm thấy file '{csv_path}'.")
        print("Hãy chắc chắn bạn đã train xong hoặc trỏ đúng đường dẫn file progress.csv.")
        return

    # Đọc dữ liệu
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Lỗi khi đọc file CSV: {e}")
        return
    
    # Cài đặt giao diện
    sns.set_theme(style="darkgrid")
    fig, axs = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Kết Quả Huấn Luyện & Chỉ Số KPI', fontsize=20, weight='bold')
    
    # Trục X là Steps hoặc Episodes
    if 'time/total_timesteps' in df.columns:
        x = df['time/total_timesteps']
        xlabel = "Total Timesteps"
    elif 'index' in df.columns:
        x = df.index
        xlabel = "Updates"
    else:
        x = df.index
        xlabel = "Updates"

    # --- 1. BIỂU ĐỒ REWARD (Quan trọng nhất) ---
    ax = axs[0, 0]
    if 'rollout/ep_rew_mean' in df.columns:
        raw = df['rollout/ep_rew_mean']
        smoothed = smooth(raw, SMOOTHING)
        ax.plot(x, raw, alpha=0.3, color='orange', label='Raw')
        ax.plot(x, smoothed, color='red', linewidth=2, label='Trend')
        ax.set_title('Mean Episode Reward (Càng cao càng tốt)', fontsize=12, weight='bold')
        ax.set_ylabel('Reward')
        ax.set_xlabel(xlabel)
        ax.legend()
    else:
         ax.text(0.5, 0.5, 'Chưa có dữ liệu Reward', ha='center')

    # --- 2. BIỂU ĐỒ KPI: WAIT TIME ---
    ax = axs[0, 1]
    if 'kpi/avg_wait_time' in df.columns:
        raw = df['kpi/avg_wait_time']
        smoothed = smooth(raw, SMOOTHING)
        ax.plot(x, raw, alpha=0.3, color='lightblue')
        ax.plot(x, smoothed, color='blue', linewidth=2)
        ax.set_title('Average Wait Time (Càng thấp càng tốt)', fontsize=12, weight='bold')
        ax.set_ylabel('Seconds')
        ax.set_xlabel(xlabel)
    else:
        ax.text(0.5, 0.5, 'Chưa có dữ liệu KPI Wait Time\n(Cần dùng Custom Callback)', ha='center')

    # --- 3. BIỂU ĐỒ KPI: QUEUE LENGTH ---
    ax = axs[1, 0]
    if 'kpi/avg_queue_len' in df.columns:
        raw = df['kpi/avg_queue_len']
        smoothed = smooth(raw, SMOOTHING)
        ax.plot(x, raw, alpha=0.3, color='lightgreen')
        ax.plot(x, smoothed, color='green', linewidth=2)
        ax.set_title('Average Queue Length (PCU)', fontsize=12, weight='bold')
        ax.set_ylabel('Vehicles (PCU)')
        ax.set_xlabel(xlabel)
    else:
        ax.text(0.5, 0.5, 'Chưa có dữ liệu KPI Queue', ha='center')

    # --- 4. BIỂU ĐỒ LOSS ---
    ax = axs[1, 1]
    if 'train/value_loss' in df.columns:
        ax.plot(x, df['train/value_loss'], color='purple', alpha=0.7)
        ax.set_title('Value Loss (Critic Error)', fontsize=12, weight='bold')
        ax.set_ylabel('Loss')
        ax.set_yscale('log')
        ax.set_xlabel(xlabel)
    else:
        ax.text(0.5, 0.5, 'Chưa có dữ liệu Loss', ha='center')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vẽ biểu đồ huấn luyện từ file progress.csv")
    parser.add_argument("csv_file", nargs="?", default=DEFAULT_CSV_PATH, help="Đường dẫn đến file progress.csv")
    args = parser.parse_args()
    
    plot_results(args.csv_file)
