import argparse
from ops import check_dir
import warnings
from chexgan import *

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description="CheXNet")

    # Set Data Directory
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./dataset/",
        help="dataset dir",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args is None:
        exit()

    check_dir("outputs")
    check_dir("parameters")

    # Declare Class
    model = CheXGAN(args)

    # Build Graph
    model.build_model()
    model.build_gan_model()

    # Run CheXGAN
    print(" [*] CheXGAN started!")
    model.generation()
    print(" [*] CheXGAN finished!")


if __name__ == "__main__":
    main()
