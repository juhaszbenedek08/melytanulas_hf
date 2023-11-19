import torch

import trainer
from dataset import ColonDataset
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', type=str, default=None)
    ap.add_argument('--model', type=str, default='unet')
    ap.add_argument('--only_test', action='store_true', default=False)
    args = ap.parse_args()

    print(torch.cuda.is_available())

    # dataset.main()

    trainer.main(args)


if __name__ == '__main__':
    main()
