from dataclasses import dataclass
import torch


@dataclass
class Config:
    # -------------------------
    # Data
    # -------------------------
    data_path: str = (
        "/home/yesom/Codes/LENO/"
        "Eikonal_data_random_F=2X2+1_Nx=100_N=10000.pkl"
    )

    n_train: int = 4000
    n_test: int = 1000

    nx: int = 100
    ny: int = 100

    xylim: float = 1.0

    # -------------------------
    # Training
    # -------------------------
    train_batch_size: int = 2000
    test_batch_size: int = 200

    epochs: int = 1_000_000
    lr: float = 1e-5

    # -------------------------
    # Model
    # -------------------------
    hidden_dim: int = 256
    n_layers: int = 3

    # -------------------------
    # Logging / checkpoint
    # -------------------------
    log_interval: int = 100
    checkpoint_interval: int = 500

    exp_dir: str = (
        "exps/Single_Source_Distance_Operator/"
        "curve_F2X2_rand_source_pts"
    )

    # -------------------------
    # Device
    # -------------------------
    device: str = "cuda:1" if torch.cuda.is_available() else "cpu"

    @property
    def dx(self):
        return 2 * self.xylim / (self.nx - 1)

    @property
    def dy(self):
        return 2 * self.xylim / (self.ny - 1)