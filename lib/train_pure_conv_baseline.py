import torch
from torch.utils.data import DataLoader, random_split
from unet import Unet
import torch.utils.data
from dataset import ColonDataset


def dice_score(pred, gt):
    tp = torch.sum(pred * gt)
    eps = 1e-6
    denom = torch.sum(pred + gt) + eps
    return (2 * tp + eps) / denom


def eval(model, loader, lossfun):
    with torch.no_grad():
        losses = []
        dscs = []

        for img, lb_mask, sb_mask, st_mask in loader:
            gt = torch.cat((lb_mask, sb_mask, st_mask), 1).to('cuda')
            img = img.to('cuda')
            pred = model(img)
            losses.append(lossfun(pred, gt))
            dscs.append(dice_score(pred, gt))

        loss = torch.cat(losses).mean()
        dsc = torch.cat(dscs).mean()

    return loss, dsc


def main():
    ds = ColonDataset()

    train_data, val_data, test_data = random_split(
        ds,
        [0.8, 0.1, 0.1],
        generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=True)

    lossfun = torch.nn.BCELoss()

    model = Unet(
        nbClasses=3,
        outSize=(384, 384),
    ).to('cuda')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, eps=1e-7, weight_decay=1e-4)

    for img, lb_mask, sb_mask, st_mask in train_loader:
        print('Picsa')

        gt = torch.cat((lb_mask, sb_mask, st_mask), 1).to('cuda')
        img = img.to('cuda')
        pred = model(img)
        loss = lossfun(pred, gt)

        print(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        shouldval = True  # todo
        if (shouldval):
            loss, dsc = eval(model, val_loader, lossfun)  # todo plotting and stuff

    loss, dsc = eval(model, test_loader, lossfun)  # todo plotting and stuff
