import torch

import trainer
from dataset import ColonDataset

def main():
    print(torch.cuda.is_available())

    # dataset.main()

    trainer.main()

    # segmentation_vit.main()

    # return


if __name__ == '__main__':
    main()
