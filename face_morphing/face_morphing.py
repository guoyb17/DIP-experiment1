import os, argparse, math
from PIL import Image as image
import numpy as np

from face_api import get_mark


def main(iptfrom, iptto, prefix, num):
    from_src = image.open(iptfrom).convert("RGB")
    from_width, from_height = from_src.size
    from_bitmap = np.array(from_src).astype(np.int)
    to_src = image.open(iptto).convert("RGB")
    to_width, to_height = to_src.size
    to_bitmap = np.array(to_src).astype(np.int)
    mark_from = get_mark(iptfrom)
    mark_from["global_left_up"] = {"y": 0, "x": 0}
    mark_from["global_mid_up"] = {"y": 0, "x": from_width // 2}
    mark_from["global_right_up"] = {"y": 0, "x": from_width - 1}
    mark_from["global_right_mid"] = {"y": from_height // 2, "x": from_width - 1}
    mark_from["global_right_down"] = {"y": from_height - 1, "x": from_width - 1}
    mark_from["global_mid_down"] = {"y": from_height - 1, "x": from_width // 2}
    mark_from["global_left_down"] = {"y": from_height - 1, "x": 0}
    mark_from["global_left_mid"] = {"y": from_height // 2, "x": 0}
    mark_to = get_mark(iptto)
    mark_to["global_left_up"] = {"y": 0, "x": 0}
    mark_to["global_mid_up"] = {"y": 0, "x": to_width // 2}
    mark_to["global_right_up"] = {"y": 0, "x": to_width - 1}
    mark_to["global_right_mid"] = {"y": to_height // 2, "x": to_width - 1}
    mark_to["global_right_down"] = {"y": to_height - 1, "x": to_width - 1}
    mark_to["global_mid_down"] = {"y": to_height - 1, "x": to_width // 2}
    mark_to["global_left_down"] = {"y": to_height - 1, "x": 0}
    mark_to["global_left_mid"] = {"y": to_height // 2, "x": 0}
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A script to do image fusion on given pictures."
        )
    parser.add_argument("-f", "--iptfrom", type=str,
                        help="input file, from: 0% merge",
                        required=True
                        )
    parser.add_argument("-t", "--iptto", type=str,
                        help="input file, to: 100% merge",
                        required=True
                        )
    parser.add_argument("-p", "--prefix", type=str,
                        help="output file name prefix",
                        required=True
                        )
    parser.add_argument("-n", "--num", type=int,
                        help="output file number(s); must >= 1 integer",
                        required=True
                        )

    args = parser.parse_args()
    main(args.iptfrom, args.iptto, args.prefix, args.num)
