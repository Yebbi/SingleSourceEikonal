import pickle as pkl
import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader, TensorDataset


class EikonalDataModule:

    def __init__(self, config):
        self.config = config

        self.device = torch.device(config.device)

        self._load_data()
        self._build_grid()
        self._build_train_data()
        self._build_test_data()

    # --------------------------------------------------
    # Load raw data
    # --------------------------------------------------
    def _load_data(self):

        (
            self.mask,
            self.solution,
            self.distances,
            self.coord,
            self.padded_coord,
            self.mask_coord,
            self.Fs,
        ) = pd.read_pickle(self.config.data_path)

    # --------------------------------------------------
    # Grid
    # --------------------------------------------------
    def _build_grid(self):

        n = self.config.nx

        x = np.linspace(
            -self.config.xylim,
            self.config.xylim,
            n,
        )

        y = np.linspace(
            -self.config.xylim,
            self.config.xylim,
            n,
        )

        X, Y = np.meshgrid(x, y)

        self.grid = torch.tensor(
            np.stack([X, Y], axis=2),
            dtype=torch.float32,
        )

        self.train_xy = (
            self.grid
            .view(-1, 2)
            .to(self.device)
            .requires_grad_(True)
        )

    # --------------------------------------------------
    # Train data
    # --------------------------------------------------
    def _build_train_data(self):

        n_train = self.config.n_train

        self.train_coord = self.padded_coord[:n_train]
        self.train_coord_mask = self.mask_coord[:n_train]

        self.train_mask = self.mask[:n_train]
        self.train_solution = self.solution[:n_train]
        self.train_distances = self.distances[:n_train]

        sources = []

        for i in range(n_train):

            valid = self.train_coord_mask[i]
            pts = self.train_coord[i][valid]

            for point in pts:
                sources.append(point)

        self.train_sources = torch.stack(sources)

        dataset = TensorDataset(self.train_sources)

        self.train_loader = DataLoader(
            dataset,
            batch_size=self.config.train_batch_size,
            shuffle=True,
        )

    # --------------------------------------------------
    # Test data
    # --------------------------------------------------
    def _build_test_data(self):

        start = self.config.n_train
        end = start + self.config.n_test

        self.test_coord = self.padded_coord[start:end]
        self.test_coord_mask = self.mask_coord[start:end]

        self.test_mask = self.mask[start:end]
        self.test_solution = self.solution[start:end]
        self.test_distances = self.distances[start:end]

        dataset = TensorDataset(
            self.test_coord,
            self.test_coord_mask,
            self.test_mask,
            self.test_solution,
            self.test_distances,
        )

        self.test_loader = DataLoader(
            dataset,
            batch_size=self.config.test_batch_size,
            shuffle=False,
        )