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
    def __init__(self):
        super().__init__()

        self.internal = Unet(
            in_channels=1,
            inter_channels=48,
            height=5,
            width=2,
            class_num=3
        )

        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        self.dice_score_fn = torchmetrics.Dice(zero_division=1)

        self.dice_frequency = 32  # MUST be min accumulate_grad_batches and SHOULD be equal

        self.dataset = ColonDataset()
        self.train_data, self.val_data, self.test_data = random_split(
            self.dataset,
            [0.8, 0.1, 0.1],
            generator=torch.Generator().manual_seed(42)
        )

        self.batch_size = 8
        self.lr = 1e-4

    def forward(self, x):
        return self.internal(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, eps=1e-7, weight_decay=1e-4)
        return optimizer

    def training_step(self, batch, batch_idx):
        img, lb_mask, sb_mask, st_mask = batch
        gt = torch.cat((lb_mask, sb_mask, st_mask), 1)
        pred_raw = self(img)
        loss = self.loss_fn(pred_raw, gt.float())
        self.log(f'train/loss', loss, on_step=True, on_epoch=True)

        if batch_idx % self.dice_frequency == 0:
            pred = torch.sigmoid(pred_raw)
            dice_score = self.dice_score_fn(pred, gt)
            self.log(f'train/dice_score', dice_score, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        img, lb_mask, sb_mask, st_mask = batch
        gt = torch.cat((lb_mask, sb_mask, st_mask), 1)
        pred_raw = self(img)
        loss = self.loss_fn(pred_raw, gt.float())
        self.log(f'train/loss', loss, on_epoch=True)

        if batch_idx % self.dice_frequency == 0:
            pred = torch.sigmoid(pred_raw)
            dice_score = self.dice_score_fn(pred, gt)
            self.log(f'val/dice_score', dice_score, on_epoch=True)

    def test_step(self, batch, batch_idx):
        img, lb_mask, sb_mask, st_mask = batch
        gt = torch.cat((lb_mask, sb_mask, st_mask), 1)
        pred_raw = self(img)
        loss = self.loss_fn(pred_raw, gt.float())
        self.log(f'test/loss', loss, on_epoch=True)

        pred = torch.sigmoid(pred_raw)
        dice_score = self.dice_score_fn(pred, gt)
        self.log(f'test/dice_score', dice_score, on_epoch=True)

        lb_time = 5 if self.dice_score_fn(pred[:, 0], gt[:, 0]) > 0.8 else -2
        sb_time = 3 if self.dice_score_fn(pred[:, 1], gt[:, 1]) > 0.7 else -3
        st_time = 2 if self.dice_score_fn(pred[:, 2], gt[:, 2]) > 0.85 else -4
        self.log(f'test/time_saved_large_bowel', lb_time, on_step=True, on_epoch=True)
        self.log(f'test/time_saved_small_bowel', sb_time, on_step=True, on_epoch=True)
        self.log(f'test/time_saved_stomach', st_time, on_step=True, on_epoch=True)

        if batch_idx <= 10:
            fig, (ax1, ax2) = plt.subplots(2, figsize=(20, 10))  # type: plt.Figure, plt.Axes
            img1 = np.stack([img.detach().cpu().numpy()[0, 0]] * 3, axis=-1)
            img1[..., 0] += pred[:, 0].detach().cpu().numpy()[0, 0]
            img1[..., 1] += pred[:, 1].detach().cpu().numpy()[0, 0]
            img1[..., 2] += pred[:, 2].detach().cpu().numpy()[0, 0]
            ax1.imshow(img1)
            img2 = np.stack([img.detach().cpu().numpy()[0, 0]] * 3, axis=-1)
            img2[..., 0] += lb_mask.detach().cpu().numpy()[0, 0]
            img2[..., 1] += sb_mask.detach().cpu().numpy()[0, 0]
            img2[..., 2] += st_mask.detach().cpu().numpy()[0, 0]
            fig.savefig(out_dir / f'unet_{batch_idx}.png')

        return loss

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=1,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            num_workers=1,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=1,
            num_workers=1,
        )

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
        optimizer.zero_grad(set_to_none=True)

    def configure_callbacks(self):
        return [
            pl.callbacks.ModelCheckpoint(
                monitor='val/dice_score_epoch',
                dirpath=out_dir,
                filename='{epoch}-{step}-{val/dice_score_epoch:.2f}',
                save_top_k=3,
                mode='min',
            ),
        ]


def main(args):
    if args.checkpoint is not None:
        model = Model.load_from_checkpoint(args.checkpoint)
    else:
        model = Model()
    trainer = pl.Trainer(
        log_every_n_steps=1,  # optimizer steps!
        max_epochs=5,
        deterministic=False,
        accumulate_grad_batches=16,
        reload_dataloaders_every_n_epochs=1,
        logger=pl.loggers.TensorBoardLogger(out_dir),
    )
    trainer.fit(model)
    trainer.test(model)
