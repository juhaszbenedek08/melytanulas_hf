import torch

import unet_trainer
from dataset import ColonDataset
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', type=str, default=None)
    args = ap.parse_args()

    print(torch.cuda.is_available())

    # dataset.main()

    unet_trainer.main(args)

    # segmentation_vit.main()

    # return


if __name__ == '__main__':
    main()
