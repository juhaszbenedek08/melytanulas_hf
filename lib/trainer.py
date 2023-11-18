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
    def __init__(self, model):
        super().__init__()

        if model == 'unet':
            internal = Unet(
                in_channels=1,
                inter_channels=48,
                height=5,
                width=1,
                class_num=3
            )
        elif model == 'segfomer':
            internal = segformer.get_model()

        if True:
            try:
                self.internal = torch.compile(internal, mode='default')
            except:
                self.internal = internal
        else:
            self.internal = internal

        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        self.dice_score_fn = torchmetrics.Dice(zero_division=1)
        self.dice_score_test = torchmetrics.Dice(num_classes=3, zero_division=1)

        self.dice_frequency = 32  # MUST be min accumulate_grad_batches and SHOULD be equal

        self.dataset = ColonDataset()
        self.train_data, self.val_data, self.test_data = random_split(
            self.dataset,
            [0.8, 0.1, 0.1],
            generator=torch.Generator().manual_seed(42)
        )

        self.batch_size = 4
        self.lr = 1e-4

    def forward(self, x):
        return self.internal(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, eps=1e-7, weight_decay=1e-4)
        return optimizer

    def step(self, batch, batch_idx, step):
        img, lb_mask, sb_mask, st_mask = batch
        gt = torch.cat((lb_mask, sb_mask, st_mask), 1)
        pred_raw = self(img)
        loss = self.loss_fn(pred_raw, gt.float())
        self.log(f'{step}/loss', loss, on_step=True, on_epoch=True)

        if step != 'train' or batch_idx % self.dice_frequency == 0:
            pred = torch.sigmoid(pred_raw)
            dice_score = self.dice_score_fn(pred, gt)
            self.log(f'{step}/dice_score', dice_score, on_step=True, on_epoch=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, 'val')

    def test_step(self, batch, batch_idx):
        step = 'test'
        img, lb_mask, sb_mask, st_mask = batch
        gt = torch.cat((lb_mask, sb_mask, st_mask), 1)
        pred_raw = self(img)
        loss = self.loss_fn(pred_raw, gt.float())
        self.log(f'{step}/loss', loss, on_step=True, on_epoch=True)

        pred = torch.sigmoid(pred_raw)
        dice_score = self.dice_score_fn(pred, gt)
        self.log(f'{step}/dice_score', dice_score, on_step=True, on_epoch=True)

        dice_score_classes = self.dice_score_test(pred, gt)
        lb_time = 5 if dice_score_classes[0] > 0.8 else -2
        sb_time = 3 if dice_score_classes[1] > 0.7 else -3
        st_time = 2 if dice_score_classes[2] > 0.85 else -4
        self.log(f'{step}/time_saved_large_bowel', lb_time, on_step=True, on_epoch=True)
        self.log(f'{step}/time_saved_small_bowel', sb_time, on_step=True, on_epoch=True)
        self.log(f'{step}/time_saved_stomach', st_time, on_step=True, on_epoch=True)

        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        img, lb_mask, sb_mask, st_mask = batch
        gt = torch.cat((lb_mask, sb_mask, st_mask), 1)
        pred_raw = self(img)

        fig, ax = plt.subplots()  # type: plt.Figure, plt.Axes
        img = np.stack([img.detach().cpu().numpy()[0, 0]] * 3, axis=-1)
        img[..., 0] += lb_mask.detach().cpu().numpy()[0, 0]
        img[..., 1] += sb_mask.detach().cpu().numpy()[0, 0]
        img[..., 2] += st_mask.detach().cpu().numpy()[0, 0]
        ax.imshow(img)
        save_next(fig, f'pred_{batch_idx}', with_fig_num=False)

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
    model = Model('unet')
    trainer = pl.Trainer(
        log_every_n_steps=1,  # optimizer steps!
        max_epochs=50,
        deterministic=False,
        accumulate_grad_batches=32,
        reload_dataloaders_every_n_epochs=1,
        logger=pl.loggers.TensorBoardLogger(out_dir),
    )
    trainer.fit(model)
    trainer.test(model)
