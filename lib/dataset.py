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

    def __init__(self):
        # download()

        self.all_scans = [
            (
                str(removeprefix(case_dir.name, 'case')),
                str(removeprefix(day_dir.name, f'{case_dir.name}_day')),
                str(scan_path.name.split('_')[1]),
                scan_path
            )
            for case_dir in raw_dir.iterdir()
            for day_dir in case_dir.iterdir()
            for scan_path in (day_dir / 'scans').iterdir()
        ]
        self.annots = pd.read_csv(
            csv_path,
            dtype={'id': str, 'class': str, 'segmentation': str}
        ).fillna('')

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
        return len(self.all_scans)

    def __getitem__(self, item):
        scan = self.all_scans[item]
        fname = str(scan[3])
        img = self.normalize(torch.tensor(np.array(Image.open(fname))))

        # fig, ax = plt.subplots()
        # ax.imshow(img)
        # fig.show()

        annotname = f"case{scan[0]}_day{scan[1]}_slice_{scan[2]}"
        annots = self.annots.loc[self.annots['id'] == annotname]
        lb_annot = annots.loc[annots['class'] == 'large_bowel']['segmentation'].item()
        sb_annot = annots.loc[annots['class'] == 'small_bowel']['segmentation'].item()
        st_annot = annots.loc[annots['class'] == 'stomach']['segmentation'].item()
        lb_mask = torch.tensor(get_mask(img.shape, lb_annot), dtype=torch.float32)
        sb_mask = torch.tensor(get_mask(img.shape, sb_annot), dtype=torch.float32)
        st_mask = torch.tensor(get_mask(img.shape, st_annot), dtype=torch.float32)

        resizer = torchvision.transforms.Resize(size=(384, 384))
        img = resizer(img[None])
        lb_mask = resizer(lb_mask[None])
        sb_mask = resizer(sb_mask[None])
        st_mask = resizer(st_mask[None])

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
