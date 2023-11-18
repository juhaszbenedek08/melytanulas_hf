import subprocess

import gdown
import numpy as np
import torchvision
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, random_split, DataLoader
import torch
import pandas as pd
from PIL import Image

from path_util import data_dir, raw_dir, csv_path
from plot_util import save_next


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
    if not data_dir.exists():
        data_dir.mkdir()
    if len(list(data_dir.iterdir())) > 0:
        return
    url = "https://drive.google.com/uc?id=1nq7DCNJsm27z8nKdvFRxphUnokU41ZY6"
    zip_path = data_dir / 'raw.zip'
    gdown.download(url, str(zip_path), quiet=False)
    subprocess.run(f'unzip "{str(zip_path)}" -d "{str(raw_dir)}"', shell=True, check=True)
    zip_path.unlink()


class ColonDataset(Dataset):

    @staticmethod
    def get_case_id(day_dir, scan_path):
        slice_num = str(scan_path.name.split('_')[1])

        return f"{day_dir.name}_slice_{slice_num}"

    def __init__(self):

        self.annots = pd.read_csv(
            csv_path,
            dtype={'id': str, 'class': str, 'segmentation': str},
        ).fillna('').pivot(
            index='id',
            columns='class',
            values='segmentation'
        ).reset_index()
        self.annots = self.annots[
            (self.annots['large_bowel'] != '')
            | (self.annots['small_bowel'] != '')
            | (self.annots['stomach'] != '')
            ]

        all_scans = {
            self.get_case_id(day_dir, scan_path): str(scan_path)
            for case_dir in raw_dir.iterdir()
            for day_dir in case_dir.iterdir()
            for scan_path in (day_dir / 'scans').iterdir()
        }
        self.annots['path'] = self.annots['id'].map(all_scans)

        print(self.annots)

        self.size = (384, 384)
        self.img_resizer = torchvision.transforms.Resize(size=self.size)
        self.mask_resizer = torchvision.transforms.Resize(
            size=self.size,
            interpolation=torchvision.transforms.InterpolationMode.NEAREST
        )

    def normalize(self, img):
        intensities = torch.flatten(img)
        intensities = torch.sort(intensities).values
        idx = int(intensities.size(0) / 30)
        newmin = intensities[idx]
        newmax = intensities[-idx]
        img = (img - newmin) / (newmax - newmin)
        img[img < 0] = 0
        img[img > 1] = 1
        return img

    def __len__(self):
        return len(self.annots)

    def mask_from(self, img, row, column):
        return self.mask_resizer(
            torch.tensor(
                get_mask(
                    img.shape,
                    row[column]
                ),
                dtype=torch.int)[None]
        )

    def __getitem__(self, item):
        row = self.annots.iloc[item, :]
        img = self.normalize(torch.tensor(np.array(Image.open(row['path']))))

        lb_mask = self.mask_from(img, row, 'large_bowel')
        sb_mask = self.mask_from(img, row, 'small_bowel')
        st_mask = self.mask_from(img, row, 'stomach')
        img = self.img_resizer(img[None])

        return img, lb_mask, sb_mask, st_mask


def main():
    ds = ColonDataset()

    print('Dataset length:', len(ds))

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
