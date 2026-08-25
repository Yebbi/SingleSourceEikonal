import torch
import torch.nn as nn

from config import Config
from dataset import EikonalDataModule
from models import branch_net, trunk_net, deepOnet
from trainer import EikonalTrainer


def build_model(config):
    branch_model = branch_net(config.hidden_dim)

    trunk_model = trunk_net(
        hidden_dims=[4, 32, 32, 32, 32, config.hidden_dim],
        act=nn.ReLU(),
    )

    model = deepOnet(
        branch=branch_model,
        trunk_u=trunk_model,
        Nx=config.nx,
        Ny=config.ny,
    )

    return model.to(config.device)


def main():
    config = Config()

    # Data
    data = EikonalDataModule(config)

    # Model
    model = build_model(config)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    # Trainer
    trainer = EikonalTrainer(
        model=model,
        data=data,
        optimizer=optimizer,
        config=config,
    )

    # Train
    trainer.fit()


if __name__ == "__main__":
    main()
