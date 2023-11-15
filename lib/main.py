import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, random_split

import dataset
from path_util import out_dir

fig_num = 0


def save_next(fig, name):
    global fig_num
    out_dir.mkdir(exist_ok=True)
    fig.savefig(out_dir / f'{name}_{fig_num}.png')
    fig_num += 1


import torchvision

def main():

    import train_pure_conv_baseline

    print(torch.cuda.is_available())

    with torch.device('cuda'):
        import torchvision
        model = torchvision.models.vit_b_16(
            weights=torchvision.models.ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1,
            progress=True
        )

        model.heads = torch.nn.Identity()

        print(model)
        input_ = torch.zeros((4, 3, 384,384))
        output = model(input_) # type: torch
        print(output.shape)

    # TODO 1 -> 3 channels
    # TODO mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225]
    # TODO keresni egy segmentation vision transformert (esetleg azt is h hogy mukszik)
    # TODO denseunet & lighnting (& determined akár) -> kevesebb memória


    dataset.download()

    ds = dataset.ColonDataset()

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
