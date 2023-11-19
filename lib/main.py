import torch

import trainer
from dataset import ColonDataset
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', type=str, default=None)
    ap.add_argument('--model', type=str, default='unet')
    args = ap.parse_args()

    print(torch.cuda.is_available())

    # dataset.main()

    trainer.main(args)

    # segmentation_vit.main()


if __name__ == '__main__':
    main()
