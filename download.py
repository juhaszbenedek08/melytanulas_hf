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

    # Decode RLE func # TODO BENEDEK

    # Create splits

    # Create dataset & dataloader

