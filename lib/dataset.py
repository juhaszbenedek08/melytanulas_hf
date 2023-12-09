import subprocess

import gdown
import numpy as np
import torchvision
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
from PIL import Image

from path_util import data_dir, raw_dir, csv_path
from plot_util import save_next
import lightning.pytorch as pl


def removeprefix(text, prefix):
    """Removes prefix from text"""
    if text.startswith(prefix):
        return text[len(prefix):]
    return text  # or whatever


def download():
    """ Download the dataset from Google Drive """
    assert data_dir.exists()
    if len(list(data_dir.iterdir())) > 0:
        print('Data already downloaded')
        return
    url = "https://drive.google.com/uc?id=1nq7DCNJsm27z8nKdvFRxphUnokU41ZY6"
    zip_path = data_dir / 'raw.zip'
    gdown.download(url, str(zip_path), quiet=False)
    subprocess.run(f'unzip "{str(zip_path)}" -d "{str(raw_dir)}"', shell=True, check=True)
    zip_path.unlink()


class ColonDataset(Dataset):
    """ Segmentation dataset """

    def __init__(self, annots):
        self.annots = annots

        self.size = (384, 384)
        self.img_resizer = torchvision.transforms.Resize(size=self.size)
        self.mask_resizer = torchvision.transforms.Resize(
            size=self.size,
            interpolation=torchvision.transforms.InterpolationMode.NEAREST
        )

    def __len__(self):
        return len(self.annots)

    @staticmethod
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

    def get_mask(self, shape, segmentation_str):
        """ Convert segmentation string to mask """
        if segmentation_str == '':
            return np.zeros(shape, dtype=int)
        else:
            lengths = [int(length) for length in segmentation_str.split(' ')]
            return self.rl_decode(shape, lengths)

    def mask_from(self, img, row, column):
        """ Get specific mask from row """
        return self.mask_resizer(
            torch.tensor(
                self.get_mask(
                    img.shape,
                    row[column]
                ),
                dtype=torch.int)[None]
        )

    @staticmethod
    def normalize(img):
        """ Normalize image.
        This includes removing the top and bottom 3% of intensities for burn-in and burn-out correction.
        """
        intensities = torch.flatten(img)
        intensities = torch.sort(intensities).values
        idx = int(intensities.size(0) / 30)
        newmin = intensities[idx]
        newmax = intensities[-idx]
        img = (img - newmin) / (newmax - newmin)
        img[img < 0] = 0
        img[img > 1] = 1
        return img

    def __getitem__(self, item):
        row = self.annots.iloc[item, :]
        img = self.normalize(torch.tensor(np.array(Image.open(row['path']))))

        lb_mask = self.mask_from(img, row, 'large_bowel')
        sb_mask = self.mask_from(img, row, 'small_bowel')
        st_mask = self.mask_from(img, row, 'stomach')
        img = self.img_resizer(img[None])

        return img, lb_mask, sb_mask, st_mask


class ColonDataModule(pl.LightningDataModule):
    @staticmethod
    def get_case_id(day_dir, scan_path):
        """ Get full case id from path (to match with annotations)"""
        slice_num = str(scan_path.name.split('_')[1])

        return f"{day_dir.name}_slice_{slice_num}"

    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.annots = pd.read_csv(
            csv_path,
            dtype={'id': str, 'class': str, 'segmentation': str},
        ).fillna('').pivot(
            index='id',
            columns='class',
            values='segmentation'
        ).reset_index()
        self.annots['case'] = self.annots['id'].apply(lambda x: removeprefix(x.split('_')[0], 'case'))
        self.annots['day'] = self.annots['id'].apply(lambda x: removeprefix(x.split('_')[1], 'day'))

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

        cases = self.annots['case'].unique()
        np.random.seed(42)
        np.random.shuffle(cases)
        index_1 = int(len(cases) * 0.8)
        index_2 = int(len(cases) * 0.9)
        train_cases = cases[:index_1]
        val_cases = cases[index_1:index_2]
        test_cases = cases[index_2:]

        self.train_data = ColonDataset(self.annots[self.annots['case'].isin(train_cases)])
        self.val_data = ColonDataset(self.annots[self.annots['case'].isin(val_cases)])
        self.test_data = ColonDataset(self.annots[self.annots['case'].isin(test_cases)])
        print('Train length:', len(self.train_data))
        print('Val length:', len(self.val_data))
        print('Test length:', len(self.test_data))

    def __len__(self):
        return len(self.annots)

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=3,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            num_workers=3,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=1,
            num_workers=3,
        )


def main():
    """ Test the dataset"""
    dm = ColonDataModule(1)
    dm.setup('fit')

    print('Dataset length:', len(dm))

    for img, lb_mask, sb_mask, st_mask in dm.test_dataloader():
        if lb_mask.sum() > 0 and sb_mask.sum() > 0 and st_mask.sum() > 0:
            fig, ax = plt.subplots()  # type: plt.Figure, plt.Axes
            img = torch.stack([img[0, 0]] * 3, dim=-1)
            img[..., 0] += lb_mask[0, 0]
            img[..., 1] += sb_mask[0, 0]
            img[..., 2] += st_mask[0, 0]
            ax.imshow(img)
            save_next(fig, 'test')
            return
