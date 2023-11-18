import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, random_split
from fancy_unet import Unet
import torch.utils.data
from dataset import ColonDataset
from path_util import out_dir
from plot_util import save_next

import torchmetrics
import lighting.pytorch as pl


class BaselineModel(torch.nn.Module, pl.LightningModule):  # DEBUG
    def __init__(self):
        super().__init__()
        self.unet = Unet(
            in_channels=1,
            inter_channels=48,
            height=5,
            width=1,
            class_num=3
        )
        self.loss_fn = torch.nn.BCELoss() # THINK
        self.dice_score_fn = torchmetrics.Dice()

        self.dice_frequency = 100

        self.dataset = ColonDataset()
        self.train_data, self.val_data, self.test_data = random_split(
            self.dataset,
            [0.8, 0.1, 0.1],
            generator=torch.Generator().manual_seed(42)
        )

        self.batch_size = 4
        self.lr = 1e-4

    def forward(self, x):
        return self.unet(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, eps=1e-7, weight_decay=1e-4)
        return optimizer

    def training_step(self, batch, batch_idx):
        img, lb_mask, sb_mask, st_mask = batch
        gt = torch.cat((lb_mask, sb_mask, st_mask), 1)
        pred = self(img)
        loss = self.loss_fn(pred, gt)
        self.log('train_loss', loss, on_step=True, on_epoch=True)

        if batch_idx % self.dice_frequency == 0:
            dice_score = self.dice_score_fn(pred, gt)
            self.log('train_dice_score', dice_score, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        img, lb_mask, sb_mask, st_mask = batch
        gt = torch.cat((lb_mask, sb_mask, st_mask), 1)
        pred = self(img)
        loss = self.loss_fn(pred, gt)
        self.log('val_loss', loss, on_step=True, on_epoch=True)
        dice_score = self.dice_score_fn(pred, gt)
        self.log('val_dice_score', dice_score, on_step=True, on_epoch=True)

    def test_step(self, batch, batch_idx):
        img, lb_mask, sb_mask, st_mask = batch
        gt = torch.cat((lb_mask, sb_mask, st_mask), 1)
        pred = self(img)
        loss = self.loss_fn(pred, gt)
        self.log('test_loss', loss, on_step=True, on_epoch=True)
        dice_score = self.dice_score_fn(pred, gt)
        self.log('test_dice_score', dice_score, on_step=True, on_epoch=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        img, lb_mask, sb_mask, st_mask = batch
        gt = torch.cat((lb_mask, sb_mask, st_mask), 1)
        pred = self(img)
        loss = self.loss_fn(pred, gt)
        self.log('predict_loss', loss, on_step=True, on_epoch=True) # TODO
        dice_score = self.dice_score_fn(pred, gt)
        self.log('predict_dice_score', dice_score, on_step=True, on_epoch=True) # TODO

        fig, ax = plt.subplots()  # type: plt.Figure, plt.Axes
        img = np.stack([img.detach().cpu().numpy()[0, 0]] * 3, axis=-1)
        img[..., 0] += lb_mask.detach().cpu().numpy()[0, 0]
        img[..., 1] += sb_mask.detach().cpu().numpy()[0, 0]
        img[..., 2] += st_mask.detach().cpu().numpy()[0, 0]
        ax.imshow(img)
        save_next(fig, f'pred_{batch_idx}', with_fig_num=False)


def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        sample_val_data, _ = random_split(
            self.val_data,
            [0.1, 0.9],
            generator=torch.Generator().manual_seed(self.current_epoch)
        )
        return DataLoader(self.val_data, batch_size=self.batch_size, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, shuffle=True)

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
        optimizer.zero_grad(set_to_none=True)


    def configure_callbacks(self):
        return [
            pl.callbacks.ModelCheckpoint(
                monitor='val_dice_score',
                dirpath=out_dir,
                filename='{epoch}-{step}-{val_dice_score:.2f}',
                save_top_k=3,
                mode='min',
            ),
            pl.callbacks.EarlyStopping(
                monitor='val_dice_score',
                patience=10,
                mode='max',
            ),
            # pl.callbacks.LearningRateFinder(), # TODO
            # pl.callbacks.StochasticWeightAveraging(), # TODO
            # pl.callbacks.BatchSizeFinder(mode='bin_search') # TODO
        ]

def main():
    model = BaselineModel()
    trainer = pl.Trainer(
        precision='16-mixed',
        log_every_n_steps=10,
        max_epochs=50,
        deterministic=False,
        accumulate_grad_batches=32,
        reload_dataloaders_every_epoch=True,
        logger=pl.loggers.TensorBoardLogger(out_dir),

    )
    trainer.fit(model)
    trainer.test(model)