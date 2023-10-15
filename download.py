import subprocess
from pathlib import Path


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

    # Decode RLE func

    import numpy as np
    def rl_decode(shape, lengths):
        """
        Run-length decoding of an array (starting with zeros).
        """
        arr = np.zeros(shape, dtype=int).reshape(-1)
        value = 0
        start = 0
        for length in lengths:
            arr[start:start + length] = value
            start = start + length
            if value == 0:
                value = 1
            else:
                value = 0
        arr = arr.reshape(shape)
        return arr

    def get_mask(shape, segmentation_str):
        if segmentation_str is None:
            return np.zeros(shape, dtype=int)
        else:
            lengths = [int(length) for length in segmentation_str.split(' ')]
            return rl_decode(shape, lengths)

    # Create splits

    # Create dataset & dataloader
