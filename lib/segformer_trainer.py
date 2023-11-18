import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, random_split

import segformer
from fancy_unet import Unet
import torch.utils.data
from dataset import ColonDataset
from path_util import out_dir
from plot_util import save_next

import torchmetrics
import lightning.pytorch as pl


class Model(pl.LightningModule):
    def __init__(self, args):
        super().__init__()

        self.internal = segformer.get_model()

    def forward(self, x):
        return self.internal(x)  # TODO add channels to input, upscale output

