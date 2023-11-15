from torch.utils.data import Dataset
import torch
import pandas as pd


class ColonDataset(Dataset):

    def __init__(self):
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
        lb_mask = torch.tensor(get_mask(img.shape, lb_annot))
        sb_mask = torch.tensor(get_mask(img.shape, sb_annot))
        st_mask = torch.tensor(get_mask(img.shape, st_annot))

        resizer = torchvision.transforms.Resize(size=(384, 384))
        img = resizer(img[None])
        lb_mask = resizer(lb_mask[None])
        sb_mask = resizer(sb_mask[None])
        st_mask = resizer(st_mask[None])

        return img, lb_mask, sb_mask, st_mask