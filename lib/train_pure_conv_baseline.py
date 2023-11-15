import torch
from torch.utils.data import DataLoader
from fancy_unet import Unet
import torch.utils.data
from matplotlib import pyplot as plt
import numpy as np
import random
import pickle
import copy
from PIL import Image
import torchmetrics.functional
from dataset import ColonDataset
from torchmetrics.functional import dice_score

with torch.device("cuda"):

    def eval(model, loader):

        with torch.no_grad():
            losses = []
            dscs = []

            for img, lb_mask, sb_mask, st_mask in loader:
                gt = torch.cat((lb_mask, sb_mask, st_mask))

                pred = model(img)
                losses.append(lossfun(pred, gt))
                dscs.append(dice_score(pred, gt))

            loss = torch.cat(losses).mean()
            dsc = torch.cat(dscs).mean()

        return loss, dsc

    ds = ColonDataset()
    trainds = ds

    ds = ColonDataset()

    train_data, val_data, test_data = random_split(
        ds,
        [0.8, 0.1, 0.1],
        generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=True)

    lossfun = torch.nn.BCELoss()

    model = Unet()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, eps=1e-7, weight_decay=1e-4)

    for img, lb_mask, sb_mask, st_mask in train_loader:

        gt = torch.cat((lb_mask, sb_mask, st_mask))

        pred = model(img)
        loss = lossfun(pred)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        shouldval = True #todo
        if(shouldval):
            loss, dsc = eval(model, val_loader) #todo plotting and stuff


    loss, dsc = eval(model, test_loader) #todo plotting and stuff
