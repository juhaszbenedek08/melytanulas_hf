import gradio as gr
import torch
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from lib.dataset import ColonDataModule
from lib.trainer import BaseModel

class GradioDemo:
    def __init__(self, model : BaseModel, dataModule: ColonDataModule):
        self.model = model
        self.dataModule = dataModule
        self.iface = gr.Interface(
            fn=self.get_test_image,
            inputs=gr.Dropdown(
                sorted(self.dataModule.test_data.annots['id'].to_list()),
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


    def get_test_image(self, id):
        index = self.dataModule.test_data.annots.index[self.dataModule.test_data.annots['id']==id].to_list()[0]
        batch = self.dataModule.test_data[index]
        return self.test_step(self, batch)

    def run(self):
        self.iface.launch()

    def test_step(self, batch):
        img, lb_mask, sb_mask, st_mask = batch
        img = img[None, ...]
        lb_mask = lb_mask[None, ...]
        sb_mask = sb_mask[None, ...]
        st_mask = st_mask[None, ...]
        gt = torch.cat((lb_mask, sb_mask, st_mask), 1)
        pred_raw = self(img)
        loss = self.loss_fn(pred_raw, gt.float())
        pred = torch.sigmoid(pred_raw)
        dice_score = self.dice_score_fn(pred, gt)
        lb_time = 5 if self.dice_score_fn(pred[:, 0], gt[:, 0]) > 0.8 else -2
        sb_time = 3 if self.dice_score_fn(pred[:, 1], gt[:, 1]) > 0.7 else -3
        st_time = 2 if self.dice_score_fn(pred[:, 2], gt[:, 2]) > 0.85 else -4
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
    
    