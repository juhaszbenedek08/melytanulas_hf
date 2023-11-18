import torch


# import dataset
import train_pure_conv_baseline


# TODO 1 -> 3 channels
# TODO mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225] for these particular weights
# TODO convert_to_segmentation_model is so blatant,
#  look up a professional solution
def main():
    print(torch.cuda.is_available())

    # dataset.main()

    train_pure_conv_baseline.main()

    # segmentation_vit.main()

    # return


if __name__ == '__main__':
    main()
