# SingleSourceEikonal


A PyTorch implementation of a Single-Source DeepONet for solving the Eikonal equation using physics-informed training.

The project is organized into separate modules for configuration, data handling, model definition, physics/PDE formulation, and training.

---

## Project Structure

```text
eikonal_deeponet/
│
├── train.py
├── config.py
├── dataset.py
├── trainer.py
├── physics.py
├── models.py
├── requirements.txt
├── README.md
│
└── exps/
    └── Single_Source_Distance_Operator/
        └── curve_F2X2_rand_source_pts/
            ├── Figures/
            ├── Models/
            └── log.txt
```

### File Description

| File               | Description                                                       |
| ------------------ | ----------------------------------------------------------------- |
| `train.py`         | Main entry point for training                                     |
| `config.py`        | Training, model, data, and experiment configuration               |
| `dataset.py`       | Dataset loading, preprocessing, grid construction, and DataLoader |
| `models.py`        | Branch network, trunk network, and DeepONet architecture          |
| `physics.py`       | Eikonal equation residual and slowness function.                  |
| `trainer.py`       | Training, evaluation, logging, and checkpoint management          |
| `requirements.txt` | Python dependencies                                               |

---

## Problem Formulation

The model is trained to approximate the solution of the Eikonal equation
```text
|∇u(x)| = F(x)
```
where `u(x)` is the solution field and `F(x)` is the spatially varying slowness function.

For the default setting, the slowness is fixed by
```text
F(x, y) = 2x² + 1
```
The DeepONet takes the source location and spatial information as inputs and predicts the corresponding solution field.
The PDE-informed loss is constructed from the Eikonal residual via Godunov scheme:

```text
R(x) = F(x)|∇u(x)| - 1
```
and the training objective minimizes the residual using a combination of MSE and Smooth L1 losses.

---

## Model

The network consists of three main components:

```text
Source coordinates
       │
       ▼
 Branch Network
       │
       │
       ├──────────────┐
       │              │
       ▼              ▼
   DeepONet      Trunk Network
                     ▲
                     │
              Spatial coordinates
              + F(x,y)
              + 1/F(x,y)
                     │
                     ▼
                  Output
                     │
                     ▼
              Eikonal residual
```

The branch network processes source coordinates, while the trunk network processes spatial coordinates and additional physical information.

---

## Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd eikonal_deeponet
```

### 2. Create a virtual environment

Using `conda`:

```bash
conda create -n eikonal python=3.10
conda activate eikonal
```

Or using `venv`:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```powershell
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Dataset

The training code expects a pickle file containing the following objects:

```python
(
    mask,
    solution,
    distances,
    coord,
    padded_coord,
    mask_coord,
    Fs,
)
```

The dataset path is specified in `config.py`:

```python
data_path = "/path/to/Eikonal_data.pkl"
```

For example:

```python
data_path = ("./Eikonal_data_random_F=2X2+1_Nx=100_N=10000.pkl")
```

### Dataset shapes

For the current dataset:

```text
Number of samples : 10,000
Grid resolution   : 100 × 100
Training samples  : 4,000
Test samples      : 1,000
```

The train/test split can be modified in `config.py`.

---

## Configuration

All major experimental parameters are controlled through `Config`.

Example:

```python
@dataclass
class Config:

    # Data
    data_path = "/path/to/data.pkl"
    n_train = 4000
    n_test = 1000

    nx = 100
    ny = 100
    xylim = 1.0

    # Training
    train_batch_size = 2000
    test_batch_size = 200

    epochs = 1_000_000
    lr = 1e-5

    # Model
    hidden_dim = 256
    n_layers = 3
```

This allows experiments to be modified without changing the training code.

---

## Training

After configuring the dataset path and experiment settings:

```bash
python train.py
```

The training pipeline performs the following steps:

```text
Config
  │
  ▼
DataModule
  │
  ▼
DeepONet
  │
  ▼
Optimizer
  │
  ▼
EikonalTrainer
  │
  ├── Training
  ├── Evaluation
  ├── Logging
  └── Checkpointing
```

---

## Checkpoints

Model checkpoints are automatically saved according to:

```python
checkpoint_interval = 500
```

Checkpoints are stored in:

```text
exps/
└── Single_Source_Distance_Operator/
    └── curve_F2X2_rand_source_pts/
        └── Models/
            ├── ep0.pth
            ├── ep500.pth
            ├── ep1000.pth
            └── ...
```

Each checkpoint contains:

```python
{
    "epoch": epoch,
    "model_state_dict": ...,
    "optimizer_state_dict": ...,
    "best_loss": ...,
    "training_time": ...
}
```

A checkpoint can be loaded using:

```python
checkpoint = torch.load("Models/ep500.pth")

model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
```

---

## Logging

Training logs are written to:

```text
exps/.../log.txt
```

Example:

```text
[0/1000000] loss: 1.234567e-02, test error: 2.345678e-02, elapsed: 12.34 s, max memory: 1024.00 MB
[100/1000000] loss: 8.765432e-03, test error: 1.876543e-02, elapsed: 1234.56 s, max memory: 1100.00 MB
```

The log contains:

* Epoch
* Training loss
* Test error
* Total training time
* Maximum GPU memory allocation

---

## Adding a Slowness

The physics formulation is isolated in:

```text
physics.py
```

For example, the current velocity field is:

```python
def ff(x, y):
    return 2 * x.pow(2) + 1
```

A different coefficient function can be implemented by modifying or extending this function.

The PDE residual is also isolated from the trainer:

```python
def eikonal_residual(u, F, dx, dy):
    ...
```

This separation makes it easier to experiment with different Eikonal problems and physics-informed objectives.

---

## TODO

* [ ] Add support for learning multiple velocity fields `F` simultaneously
* [ ] Add inverse problems
* [ ] Add automated reproducibility settings

---

## License

This project is released under the MIT License unless otherwise specified.

```
```
