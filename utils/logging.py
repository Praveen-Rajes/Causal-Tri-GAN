"""
CausalTriGAN-StyleGAN2 - Logging Utilities
TensorBoard logging and console progress tracking.
"""
import os
import json
import time
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter


class Logger:
    """Combined TensorBoard + JSON logger."""

    def __init__(self, log_dir, run_name="stylegan2"):
        self.log_dir = log_dir
        self.tb_dir = os.path.join(log_dir, "tensorboard", run_name)
        self.writer = SummaryWriter(self.tb_dir)
        self.metrics = defaultdict(list)
        self.json_path = os.path.join(log_dir, f"{run_name}_metrics.json")
        self.start_time = time.time()

    def log_scalar(self, tag, value, step):
        """Log a scalar to TensorBoard and internal store."""
        self.writer.add_scalar(tag, value, step)
        self.metrics[tag].append((step, value))

    def log_scalars(self, tag_dict, step):
        """Log multiple scalars."""
        for tag, value in tag_dict.items():
            self.log_scalar(tag, value, step)

    def log_images(self, tag, images, step, nrow=8):
        """Log image grid to TensorBoard."""
        from torchvision.utils import make_grid
        grid = make_grid(images, nrow=nrow, normalize=True, value_range=(-1, 1))
        self.writer.add_image(tag, grid, step)

    def save_json(self):
        """Save all metrics to JSON."""
        serializable = {}
        for key, values in self.metrics.items():
            serializable[key] = [(int(s), float(v)) for s, v in values]
        with open(self.json_path, 'w') as f:
            json.dump(serializable, f, indent=2)

    def elapsed(self):
        """Return elapsed time string."""
        dt = time.time() - self.start_time
        hours = int(dt // 3600)
        minutes = int((dt % 3600) // 60)
        return f"{hours}h {minutes}m"

    def close(self):
        self.save_json()
        self.writer.close()
