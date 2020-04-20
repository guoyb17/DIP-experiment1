import os, argparse, math
from PIL import Image as image
import numpy as np


def gradient_cal(src, iter_x, iter_y, max_width, max_height):
    '''
    Calculate gradient: ∇(x,y) = 4I(x,y) − I(x − 1,y) − I(x,y − 1) − I(x + 1,y) − I(x,y + 1)
    For neighbor(s) out of range, use I(x,y) itself instead.
    src: input numpy array
    iter_x, iter_y: target point
    max_width, max_height: width and height of src
    '''
    left_part = src[0 if iter_x == 0 else iter_x - 1][iter_y]
    right_part = src[max_height - 1 if iter_x == max_height - 1 else iter_x + 1][iter_y]
    up_part = src[iter_x][0 if iter_y == 0 else iter_y - 1]
    down_part = src[iter_x][max_width - 1 if iter_y == max_width - 1 else iter_y + 1]
    return 4 * src[iter_x][iter_y] - left_part - right_part - up_part - down_part

def update(src, cover, max_width, max_height):
    '''
    Update src with cover info.
    src: input numpy array
    cover: Compressed Sparse Row (CSR) sparse matrix, LIST [[<row offsets>], [<column indices>], [values]]
    max_width, max_height: width and height of src
    '''
    for iter_r in range(max_height):
        for iter_c in range(cover[0][iter_r], cover[0][iter_r + 1]):
            src[iter_r][cover[1][iter_c]] = cover[2][iter_c]
    return src

def gauss_seidel(A, x0, b, n):
    '''
    Gauss-Seidel method to solve Ax = b.
    A: CSR sparse matrix
    x0: original x, it should've been like [[[1, 2, 3]], [[2, 3, 4]], ...], but [[1, 2, 3], [2, 3, 4], ...] seems better.
    b: use the SAME shape of x0
    n: iteration times
    '''
    len_x = len(x0) # A should be len_x * len_x size; NO check here!
    x = x0
    for _ in range(n):
        for iter_i in range(len_x):
            sum_a = np.zeros(3)
            the_iter_c = -1
            for iter_c in range(A[0][iter_i], A[0][iter_i + 1]):
                if iter_c != iter_i:
                    sum_a += A[2][iter_c]
                else:
                    the_iter_c = iter_c
            assert(the_iter_c != -1)
            x[iter_i] = (b[iter_i] - sum_a * x[iter_i]) / A[2][the_iter_c]
    return x

def get_mask(mask_map, mask_width, mask_height):
    '''
    Get CSR-sparse-matrix-like format of mask bitmap.
    '''
    ans = [[0], []] # Simplified CSR! No data part!
    for iter_r in range(mask_height):
        for iter_c in range(mask_width):
            if mask_map[iter_r][iter_c] != 0:
                ans[1].append(iter_c)
        ans[0].append(len(ans[1]))
    return ans

def in_mask(mask_map, r, c):
    '''
    Check whether (iter_r, iter_c) pixel within mask.
    '''
    if r + 1 >= len(mask_map[0]) or r < 0:
        return False
    for iter_c in range(mask_map[0][r], mask_map[0][r + 1]):
        if mask_map[1][iter_c] == c:
            return True
    return False

def at_boundary(mask_map, r, c):
    '''
    Check whether (iter_r, iter_c) pixel within mask.
    '''
    return in_mask(mask_map, r, c) and\
        not (in_mask(mask_map, r - 1, c) and in_mask(mask_map, r + 1, c)\
            and in_mask(mask_map, r, c - 1) and in_mask(mask_map, r, c + 1))

def main(ipt_img, opt_img, mask_img, background, times, align_height, align_width):
    ipt_src = image.open(ipt_img).convert("RGB")
    ipt_width, ipt_height = ipt_src.size
    ipt_bitmap = np.array(ipt_src)
    bg_src = image.open(background).convert("RGB")
    bg_width, bg_height = bg_src.size
    bg_bitmap = np.array(bg_src)
    mask_src = image.open(mask_img).convert("1")
    mask_width, mask_height = mask_src.size
    mask_bitmap = np.array(mask_src)
    mask_csr = get_mask(mask_bitmap, mask_width, mask_height)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A script to do image fusion on given pictures."
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
