import argparse
import os

import comet_ml
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CometLogger, TensorBoardLogger
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, random_split
from torchmetrics import Accuracy
from torchvision import transforms
from torchvision.datasets import MNIST

PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
AVAIL_GPUS = min(1, torch.cuda.device_count())
print("Available GPUS:", AVAIL_GPUS)
BATCH_SIZE = 256 if AVAIL_GPUS else 64

MAX_EPOCHS = 20
PROJECT_NAME = os.getenv("COMET_PROJECT_NAME", "ptl-dist")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_id")
    return parser.parse_args()


class Model(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(28 * 28, 10)

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_nb):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        self.logger.log_metrics({"train_loss": loss})
        return loss

    def validation_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        self.logger.log_metrics({"val_loss": loss})
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)


args = get_args()
# Init our model
model = Model()

node_rank = os.getenv("NODE_RANK")
os.environ["COMET_DISTRIBUTED_NODE_IDENTIFIER"] = f"node-{node_rank}"
print("Comet Node Identifier: ", f"node-{node_rank}")

if args.experiment_id:
    experiment = comet_ml.ExistingExperiment(
        previous_experiment=args.experiment_id,
        log_env_details=True,
        log_env_cpu=True,
        log_env_gpu=True,
    )
    # Use the default lightning logger on the worker nodes
    logger = TensorBoardLogger(save_dir=os.getcwd(), version=1, name="lightning_logs")
else:
    logger = CometLogger(log_env_cpu=True, log_env_gpu=True, project_name=PROJECT_NAME)

logger.log_hyperparams({"batch_size": BATCH_SIZE})

# Init DataLoader from MNIST Dataset
train_ds = MNIST(
    PATH_DATASETS, train=True, download=True, transform=transforms.ToTensor()
)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE)
eval_ds = MNIST(
    PATH_DATASETS, train=False, download=True, transform=transforms.ToTensor()
)
eval_loader = DataLoader(train_ds, batch_size=BATCH_SIZE)

# Initialize a trainer
trainer = Trainer(
    accelerator="gpu",
    max_epochs=MAX_EPOCHS,
    logger=logger,
    strategy="ddp",
    num_nodes=2,
    devices=1,
)
# Train the model ⚡
trainer.fit(model, train_loader, eval_loader)
