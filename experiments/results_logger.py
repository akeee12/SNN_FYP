# experiments/results_logger.py
import os

class ResultsLogger:
    def __init__(self, results_dir: str):
        os.makedirs(results_dir, exist_ok=True)
        self.path = os.path.join(results_dir, "training_log.csv")
        with open(self.path, "w") as f:
            f.write("model,epoch,loss,train_acc,"
                    "test_acc,peak_mem_mb,epoch_time_s\n")

    def log_epoch(self, model_name, epoch, loss,
                  train_acc, test_acc, peak_mem, epoch_time):
        with open(self.path, "a") as f:
            f.write(f"{model_name},{epoch},{loss:.6f},"
                    f"{train_acc:.6f},{test_acc:.6f},"
                    f"{peak_mem:.2f},{epoch_time:.2f}\n")

    def save(self):
        pass
