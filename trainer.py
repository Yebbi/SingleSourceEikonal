import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from physics import ff, eikonal_residual


class EikonalTrainer:

    def __init__(self, model, data, optimizer, config):
        self.model = model
        self.data = data
        self.optimizer = optimizer
        self.config = config
        self.device = torch.device(config.device)

        self.global_step = 0
        self.training_time = 0.0
        self.best_loss = float("inf")

        self._prepare_grid_features()
        self._prepare_directories()

    # ==================================================
    # Setup
    # ==================================================

    def _prepare_grid_features(self):
        xy = self.data.train_xy

        self.FF = ff(xy[:, 0], xy[:, 1])

        self.inp_xy = torch.cat([
            xy,
            self.FF.unsqueeze(-1),
            1 / self.FF.unsqueeze(-1),
        ], dim=1)

    def _prepare_directories(self):
        self.exp_dir = self.config.exp_dir
        self.fig_dir = os.path.join(self.exp_dir, "Figures")
        self.ckpt_dir = os.path.join(self.exp_dir, "Models")

        os.makedirs(self.fig_dir, exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)

        self.log_file = os.path.join(self.exp_dir, "log.txt")

    # ==================================================
    # Training
    # ==================================================

    def train_one_epoch(self):
        self.model.train()

        losses = []

        for (sources,) in self.data.train_loader:
            sources = sources.to(self.device)
            batch_size = sources.shape[0]

            self.optimizer.zero_grad()

            diff = self.data.train_xy.unsqueeze(0).expand(batch_size, -1, -1) - sources[:, None, :]
            diff = torch.norm(diff, dim=-1)

            output = self.model(sources, self.inp_xy, diff)

            residual = eikonal_residual(
                output,
                self.FF.view(1, self.config.nx, self.config.ny),
                dx=self.config.dx,
                dy=self.config.dy,
            )

            zero = torch.zeros_like(residual)

            loss = F.mse_loss(residual, zero) + F.smooth_l1_loss(residual, zero)

            loss.backward()
            self.optimizer.step()

            losses.append(loss.item())

        return float(np.mean(losses))

    # ==================================================
    # Evaluation
    # ==================================================

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()

        errors = []

        for coord, coord_mask, mask, solution, distances in self.data.test_loader:
            batch_errors = []

            for i in range(len(solution)):
                valid = coord_mask[i]
                sources = coord[i][valid].to(self.device)

                diff = self.data.train_xy.unsqueeze(0).expand(sources.shape[0], -1, -1) - sources[:, None, :]
                diff = torch.norm(diff, dim=-1)

                output = self.model(sources, self.inp_xy, diff)
                output, _ = torch.min(output, dim=0)

                target = solution[i].to(self.device)
                error = F.mse_loss(output, target)

                batch_errors.append(error.item())

            errors.append(np.mean(batch_errors))

        return float(np.mean(errors))

    # ==================================================
    # Checkpoint
    # ==================================================

    def save_checkpoint(self, epoch):
        path = os.path.join(self.ckpt_dir, f"ep{epoch}.pth")

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_loss": self.best_loss,
            "training_time": self.training_time,
        }

        torch.save(checkpoint, path)

    # ==================================================
    # Logging
    # ==================================================

    def log(self, epoch, loss, test_error, elapsed, memory_mb):
        message = (
            f"[{epoch}/{self.config.epochs}] "
            f"loss: {loss:.8e}, "
            f"test error: {test_error:.8e}, "
            f"elapsed: {elapsed:.2f} s, "
            f"max memory: {memory_mb:.2f} MB\n"
        )

        print(message)

        with open(self.log_file, "a") as f:
            f.write(message)

    # ==================================================
    # Full training
    # ==================================================

    def fit(self):
        for epoch in tqdm(range(self.config.epochs)):
            start = time.time()

            loss = self.train_one_epoch()

            elapsed = time.time() - start
            self.training_time += elapsed

            if loss < self.best_loss:
                self.best_loss = loss

            memory_mb = 0.0

            if self.device.type == "cuda":
                memory_mb = torch.cuda.max_memory_allocated(self.device) / 1024**2

            if epoch % self.config.log_interval == 0:
                test_error = self.evaluate()
                self.log(epoch, loss, test_error, self.training_time, memory_mb)

            if epoch % self.config.checkpoint_interval == 0:
                self.save_checkpoint(epoch)