import os, argparse, math
from PIL import Image as image
import numpy as np


def main(ipt_img, opt_img, mask_img, background, times, align_height, align_width):
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A script to modify 24-bit RGB pictures."
        )
    parser.add_argument("-i", "--input", type=str,
                        help="input file",
                        required=True
                        )
    parser.add_argument("-o", "--output", type=str,
                        help="output file",
                        required=True
                        )
    parser.add_argument("-m", "--mask", type=str,
                        help="mask file",
                        required=True
                        )
    parser.add_argument("-b", "--background", type=str,
                        help="background file",
                        required=True
                        )
    parser.add_argument("-t", "--times", type=int,
                        help="iteration times",
                        required=True
                        )
    parser.add_argument("-h", "--height", type=int,
                        help="input align height on background",
                        required=True
                        )
    parser.add_argument("-w", "--width", type=int,
                        help="input align width on background",
                        required=True
                        )

    args = parser.parse_args()
    main(args.input, args.output, args.mask, args.background, args.times, args.height, args.width)
