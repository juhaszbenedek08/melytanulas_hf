from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt

import segformer
from dataset import ColonDataModule
import torch.utils.data
from fancy_unet import Unet as FancyUnet
from path_util import out_dir

import gradio as gr

import torchmetrics
import lightning.pytorch as pl


class BaseModel(pl.LightningModule):
    """ Common settings for the two models """

    def __init__(self, internal):
        super().__init__()

        self.internal = internal

        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        self.dice_score_fn = torchmetrics.Dice(zero_division=1)

        self.dice_frequency = 32  # MUST be min accumulate_grad_batches and SHOULD be equal

    def forward(self, x):
        return self.internal(x)

    def training_step(self, batch, batch_idx):
        """ Train, occasionally calculate metrics, occasionally show figures"""
        img, lb_mask, sb_mask, st_mask = batch
        gt = torch.cat((lb_mask, sb_mask, st_mask), 1)
        pred_raw = self(img)
        loss = self.loss_fn(pred_raw, gt.float())
        self.log(f'train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)

        if batch_idx % self.dice_frequency == 0:
            pred = torch.sigmoid(pred_raw)
            dice_score = self.dice_score_fn(pred, gt)
            self.log(f'train/dice_score', dice_score, on_step=True, on_epoch=True, prog_bar=True)

        if batch_idx % (self.dice_frequency * 10) == 0:
            self.show_fig('train', img, lb_mask, sb_mask, st_mask, pred, batch_idx)

        return loss

    def validation_step(self, batch, batch_idx):
        """ Calculate metrics, show figures for 10 fixed images"""
        img, lb_mask, sb_mask, st_mask = batch
        gt = torch.cat((lb_mask, sb_mask, st_mask), 1)
        pred_raw = self(img)
        loss = self.loss_fn(pred_raw, gt.float())
        self.log(f'train/loss', loss, on_epoch=True)

        pred = torch.sigmoid(pred_raw)
        dice_score = self.dice_score_fn(pred, gt)
        self.log(f'val/dice_score', dice_score, on_epoch=True)

        if batch_idx <= 10:
            self.show_fig('val', img, lb_mask, sb_mask, st_mask, pred, batch_idx)

    def test_step(self, batch, batch_idx):
        """ Calculate metrics including custom metric, show figures for 10 fixed images"""
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
            self.show_fig('test', img, lb_mask, sb_mask, st_mask, pred, batch_idx)

        return loss

    def show_fig(self, phase, img, lb_mask, sb_mask, st_mask, pred, batch_idx):
        """ Create and save figure with 2 images: original and prediction """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        img1 = np.stack([img.detach().cpu().numpy()[0, 0]] * 3, axis=-1)
        pred = pred.detach().cpu().numpy()
        img1[..., 0] += pred[0, 0]
        img1[..., 1] += pred[0, 1]
        img1[..., 2] += pred[0, 2]
        ax1.imshow(img1)
        img2 = np.stack([img.detach().cpu().numpy()[0, 0]] * 3, axis=-1)
        img2[..., 0] += lb_mask.detach().cpu().numpy()[0, 0]
        img2[..., 1] += sb_mask.detach().cpu().numpy()[0, 0]
        img2[..., 2] += st_mask.detach().cpu().numpy()[0, 0]
        ax2.imshow(img2)
        fig.savefig(Path(self.logger.log_dir) / f'{self.name}_{phase}_epoch{self.current_epoch}_{batch_idx}.png')

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
        optimizer.zero_grad(set_to_none=True)

    def configure_callbacks(self):
        return [
            pl.callbacks.ModelCheckpoint(
                monitor='val/dice_score',
                dirpath=self.logger.log_dir,
                save_top_k=3,
                mode='max',
            ),
        ]


class FancyUNetModel(BaseModel):
    """ Model and parameters for using fancy_unet as backbone """

    def __init__(self):
        super().__init__(FancyUnet(
            in_channels=1,
            inter_channels=48,
            height=5,
            width=2,
            class_num=3
        ))

        self.batch_size = 6

        self.name = 'fancy_unet'

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4, eps=1e-7, weight_decay=1e-4)
        return optimizer

    def forward(self, x):
        return self.internal(x)


class SegformerModel(BaseModel):
    """ Model and parameters for using segformer as backbone """

    def __init__(self):
        super().__init__(segformer.get_model())

        self.batch_size = 12

        self.name = 'segformer'

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4, eps=1e-7, weight_decay=1e-4)
        return optimizer

    def forward(self, x):
        x = torch.cat([x] * 3, dim=1)
        x = self.internal(x)[0]
        x = torch.functional.F.interpolate(x, size=(384, 384), mode='bilinear')
        return x


class GradioDemo:
    def __init__(self, model: BaseModel, dm: ColonDataModule):
        self.model = model
        self.dm = dm
        self.iface = gr.Interface(
            fn=self.get_test_image,
            inputs=gr.Dropdown(
                sorted(self.dm.test_data.annots['id'].to_list()),
                label="Choose an id to see test image result!",
            ),
            outputs=[
                gr.Image(),
                gr.Number(label="Loss"),
                gr.Number(label="Dice Score"),
                gr.Number(label="LB Time"),
                gr.Number(label="SB Time"),
                gr.Number(label="ST Time"),
            ]
        )
        self.model.eval()

    def get_test_image(self, id_):
        index = self.dm.test_data.annots.index[self.dm.test_data.annots['id'] == id_].to_list()[0]
        batch = self.dm.test_data[index]
        return self.test_step(batch)

    def run(self):
        self.iface.launch(inline=True)

    def test_step(self, batch):
        img, lb_mask, sb_mask, st_mask = batch
        img = img[None, ...]
        lb_mask = lb_mask[None, ...]
        sb_mask = sb_mask[None, ...]
        st_mask = st_mask[None, ...]
        gt = torch.cat((lb_mask, sb_mask, st_mask), 1)
        pred_raw = self.model(img)
        loss = self.model.loss_fn(pred_raw, gt.float())
        pred = torch.sigmoid(pred_raw)
        dice_score = self.model.dice_score_fn(pred, gt)
        lb_time = 5 if self.model.dice_score_fn(pred[:, 0], gt[:, 0]) > 0.8 else -2
        sb_time = 3 if self.model.dice_score_fn(pred[:, 1], gt[:, 1]) > 0.7 else -3
        st_time = 2 if self.model.dice_score_fn(pred[:, 2], gt[:, 2]) > 0.85 else -4
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        img1 = np.stack([img.detach().cpu().numpy()[0, 0]] * 3, axis=-1)
        pred = pred.detach().cpu().numpy()
        img1[..., 0] += pred[0, 0]
        img1[..., 1] += pred[0, 1]
        img1[..., 2] += pred[0, 2]
        ax1.imshow(img1)
        img2 = np.stack([img.detach().cpu().numpy()[0, 0]] * 3, axis=-1)
        img2[..., 0] += lb_mask.detach().cpu().numpy()[0, 0]
        img2[..., 1] += sb_mask.detach().cpu().numpy()[0, 0]
        img2[..., 2] += st_mask.detach().cpu().numpy()[0, 0]
        ax2.imshow(img2)

        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        return data, loss, dice_score, lb_time, sb_time, st_time


def main(args):
    """ Create or load model, create data module and trainer, run training and testing """
    if args.model == 'fancy_unet':
        Model = FancyUNetModel
    elif args.model == 'segformer':
        Model = SegformerModel
    else:
        raise ValueError(f'Unknown model: {args.model}')

    if args.checkpoint is not None:
        model = Model.load_from_checkpoint(args.checkpoint)
    else:
        model = Model()

    if not args.colab:
        model.batch_size = 1  # Because of sanity even during inference

    dm = ColonDataModule(model.batch_size, 3 if args.colab else 1)

    trainer = pl.Trainer(
        log_every_n_steps=1,  # optimizer steps!
        max_epochs=10,
        deterministic=False,
        accumulate_grad_batches=16,
        reload_dataloaders_every_n_epochs=1,
        logger=pl.loggers.TensorBoardLogger(out_dir),
    )
    if args.gradio:
        demo = GradioDemo(model, dm)
        demo.run()
    else:
        if not args.only_test:
            trainer.fit(model, dm)
        trainer.test(model, dm)
