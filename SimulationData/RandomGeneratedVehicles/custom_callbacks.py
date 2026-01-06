from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

class KPILoggerCallback(BaseCallback):
    """
    Callback tùy chỉnh để ghi lại các chỉ số KPI từ môi trường SUMO vào TensorBoard/Log.
    """
    def __init__(self, verbose=0):
        super(KPILoggerCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # Lấy thông tin 'infos' từ môi trường
        # infos là một danh sách (do vectorized env), ta lấy phần tử đầu tiên
        # Lưu ý: Khi dùng SubprocVecEnv, infos có thể là list các dict
        infos = self.locals.get("infos", [{}])
        if isinstance(infos, list) and len(infos) > 0:
             info = infos[0] # Lấy môi trường đầu tiên
        else:
             info = {}

        # Ghi log các chỉ số KPI nếu chúng tồn tại trong info
        if "KPI_AvgWait" in info:
            self.logger.record("kpi/avg_wait_time", info["KPI_AvgWait"])
        if "KPI_AvgQueue" in info:
            self.logger.record("kpi/avg_queue_len", info["KPI_AvgQueue"])
        if "KPI_AvgTravel" in info:
            self.logger.record("kpi/avg_travel_time", info["KPI_AvgTravel"])
        
        # Ghi lại Queue thực tế (PCU)
        if "avg_queue_pcu" in info:
             self.logger.record("env/avg_queue_pcu", info["avg_queue_pcu"])

        return True
