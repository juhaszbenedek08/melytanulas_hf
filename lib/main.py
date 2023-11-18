import torch

import train_pure_conv_baseline
from dataset import ColonDataset

def main():
    print(torch.cuda.is_available())

    # dataset.main()

    train_pure_conv_baseline.main()

    # segmentation_vit.main()

    # return


if __name__ == '__main__':
    main()
