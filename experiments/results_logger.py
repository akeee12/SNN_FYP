# experiments/results_logger.py
import os
import csv
from datetime import datetime


class ResultsLogger:
    """
    Logs per-epoch metrics for all models to a unified CSV.
    Columns: model, epoch, train_loss, train_acc, test_acc, peak_mem_mb, epoch_time_s
    """

    def __init__(self, results_dir: str):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = os.path.join(results_dir, f"benchmark_{timestamp}.csv")
        self._init_csv()

    def _init_csv(self):
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "model", "epoch", "train_loss",
                "train_acc", "test_acc",
                "peak_mem_mb", "epoch_time_s"
            ])

    def log_epoch(self, model_name, epoch, train_loss,
                  train_acc, test_acc, peak_mem_mb, epoch_time_s):
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                model_name, epoch,
                f"{train_loss:.6f}",
                f"{train_acc:.4f}",
                f"{test_acc:.4f}",
                f"{peak_mem_mb:.2f}",
                f"{epoch_time_s:.2f}"
            ])

    def save(self):
        print(f"  Results CSV saved: {self.csv_path}")