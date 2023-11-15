import subprocess
from pathlib import Path

import torch
import torchvision
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
from PIL import Image
import gdown
from dataset import ColonDataset

data_dir = Path('/data')
csv_path = data_dir / 'train.csv'
raw_dir = data_dir / 'train'
out_dir = Path('/out')

fig_num = 0


def save_next(fig, name):
    global fig_num
    out_dir.mkdir(exist_ok=True)
    fig.savefig(out_dir / f'{name}_{fig_num}.png')
    fig_num += 1


def rl_decode(shape, sequence):
    """
    Run-length decoding of an array (starting with zeros).
    """
    arr = np.zeros(shape, dtype=int).reshape(-1)
    starts = sequence[0::2]
    lengths = sequence[1::2]
    for start, length in zip(starts, lengths):
        arr[start: start + length] = 1
    arr = arr.reshape(shape)
    return arr


def get_mask(shape, segmentation_str):
    if segmentation_str == '':
        return np.zeros(shape, dtype=int)
    else:
        lengths = [int(length) for length in segmentation_str.split(' ')]
        return rl_decode(shape, lengths)


def removeprefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text  # or whatever


def download():
    # Download

    if not data_dir.exists():
        data_dir.mkdir()
    if len(list(data_dir.iterdir())) > 0:
        return
    url = "https://drive.google.com/uc?id=1nq7DCNJsm27z8nKdvFRxphUnokU41ZY6"
    zip_path = data_dir / 'raw.zip'
    gdown.download(url, str(zip_path), quiet=False)
    subprocess.run(f'unzip "{str(zip_path)}" -d "{str(raw_dir)}"', shell=True, check=True)
    zip_path.unlink()


def main():
    download()

    ds = ColonDataset()

    train_data, val_data, test_data = random_split(
        ds,
        [0.8, 0.1, 0.1],
        generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=True)

    for img, lb_mask, sb_mask, st_mask in test_loader:
        if lb_mask.sum() > 0 and sb_mask.sum() > 0 and st_mask.sum() > 0:
            fig, ax = plt.subplots()  # type: plt.Figure, plt.Axes
            img = torch.stack([img[0, 0]] * 3, dim=-1)
            img[..., 0] += lb_mask[0, 0]
            img[..., 1] += sb_mask[0, 0]
            img[..., 2] += st_mask[0, 0]
            ax.imshow(img)
            save_next(fig, 'test')
            return

if __name__ == '__main__':
    main()
