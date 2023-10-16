import subprocess
from pathlib import Path
from torch.utils.data import Dataset
from torchvision.io import read_image
import pandas as pd


class ColonDataset(Dataset):

    def __init__(self):
        self.raw_dir = Path('')
        self.all_scans = [
            [
                [
                    (
                        int(case_dir.name.removeprefix('case_')),
                        int(day_dir.name.removeprefix(f'{case_dir.name}_day')),
                        int(scan_path.name.split('_')[1]),
                        scan_path
                    )
                    for scan_path in (day_dir / 'scans').iterdir()
                ]
                for day_dir in case_dir.iterdir()
            ]
            for case_dir in Path(self.raw_dir).iterdir()
        ]
        self.annots = pd.read_csv()

    def __len__(self):
        return len(self.all_scans)

    def __getitem__(self, item):
        scan = self.all_scans[item]
        fname = scan[3]
        img = read_image(fname)
        annotname = f"case{scan[0]}_day{scan[1]}_slice_{scan[2]}"
        annots = self.annots.where(self.annots['id'] == annotname)
        lb_annot = annots['segmentation'].where(annots['class'] == 'large_bowel')[0]
        sb_annot = annots['segmentation'].where(annots['class'] == 'small_bowel')[0]
        st_annot = annots['segmentation'].where(annots['class'] == 'stomach')[0]
#         todo make masks from annots, normalize img

        return img, lb_annot, sb_annot, st_annot


def main():
    # Download
    url = 'https://drive.google.com/file/d/1nq7DCNJsm27z8nKdvFRxphUnokU41ZY6/view?usp=sharing'
    zip_path = Path('./data.zip')
    raw_dir = Path('')
    subprocess.run(f'wget "{url}" -o "{str(zip_path)}"', shell=True)
    subprocess.run(f'unzip "{str(zip_path)}" -d "{str(raw_dir)}"', shell=True)

    # Read files # TODO TADAM16
    all_scans = [
        [
            [
                (
                    int(case_dir.name.removeprefix('case_')),
                    int(day_dir.name.removeprefix(f'{case_dir.name}_day')),
                    int(scan_path.name.split('_')[1]),
                    scan_path
                )
                for scan_path in (day_dir / 'scans').iterdir()
            ]
            for day_dir in case_dir.iterdir()
        ]
        for case_dir in Path(raw_dir).iterdir()
    ]

    # Read CSV # TODO TADAM16

    # Read PNG func # TODO TADAM16

    # Decode RLE func # TODO BENEDEK

    # Create splits

    # Create dataset & dataloader
