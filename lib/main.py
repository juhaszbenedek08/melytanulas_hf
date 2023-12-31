import torch

import dataset
import trainer
import argparse


def main():
    """ Entry point for the program.
    Parses command line arguments and runs the appropriate function(s), including:
    - Downloading the dataset
    - Training the model
    - Testing the model
    """
    ap = argparse.ArgumentParser()
    ap.add_argument('--download', action='store_true', default=False)
    ap.add_argument('--checkpoint', type=str, default=None)
    ap.add_argument('--model', type=str, default='fancy_unet')
    ap.add_argument('--only_test', action='store_true', default=False)
    ap.add_argument('--gradio', action='store_true', default=False)
    ap.add_argument('--colab', action='store_true', default=False)
    args = ap.parse_args()
    print(args)
    if torch.cuda.is_available():
        print('Using CUDA')

    if args.download:
        dataset.download()

    # dataset.main()

    trainer.main(args)


if __name__ == '__main__':
    main()
