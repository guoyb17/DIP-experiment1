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

def update(src, cover):
    '''
    Update src with cover info.
    src: input numpy array
    cover: Compressed Sparse Row (CSR) sparse matrix, LIST [[<row offsets>], [<column indices>], [values]]
    '''
    for iter_r in range(len(cover[0]) - 1):
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
            sum_ax = np.zeros(3)
            the_iter_c = -1
            for iter_c in range(A[0][iter_i], A[0][iter_i + 1]):
                c = A[1][iter_c]
                if c != iter_i:
                    sum_ax += A[2][iter_c] * x[c]
                else:
                    the_iter_c = c
            assert(the_iter_c != -1)
            x[iter_i] = (b[iter_i] - sum_ax) / A[2][the_iter_c]
    return x

def get_mask(mask_map, mask_width, mask_height):
    '''
    Get CSR-sparse-matrix-like format of mask bitmap.
    '''
    ans = [[0], []] # Simplified CSR! No data part!
    for iter_r in range(mask_height):
        for iter_c in range(mask_width):
            if mask_map[iter_r][iter_c] != False:
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
    # return in_mask(mask_map, r, c) and ...
    return not (in_mask(mask_map, r - 1, c) and in_mask(mask_map, r + 1, c)\
            and in_mask(mask_map, r, c - 1) and in_mask(mask_map, r, c + 1))

def main(ipt_img, opt_img, mask_img, background, times, align_height, align_width):
    ipt_src = image.open(ipt_img).convert("RGB")
    ipt_width, ipt_height = ipt_src.size
    ipt_bitmap = np.array(ipt_src).astype(np.int)
    bg_src = image.open(background).convert("RGB")
    bg_width, bg_height = bg_src.size
    bg_bitmap = np.array(bg_src).astype(np.int)
    mask_src = image.open(mask_img).convert("1")
    mask_width, mask_height = mask_src.size
    assert(ipt_width == mask_width)
    assert(ipt_height == mask_height)
    mask_bitmap = np.array(mask_src)
    mask_csr = get_mask(mask_bitmap, mask_width, mask_height)
    total_n = mask_csr[0][-1]
    cnt = 0
    A = [[0], [], []]
    A_buff = {} # iter_r * mask_width + mask_csr[1][iter_c]: (cnt, type), type = True for boundary, False for ordinary
    x0 = []
    b = []
    for iter_r in range(len(mask_csr[0]) - 1):
        for iter_c in range(mask_csr[0][iter_r], mask_csr[0][iter_r + 1]):
            # Target: (iter_r, mask_csr[1][iter_c])
            c = mask_csr[1][iter_c]
            A_buff[iter_r * mask_width + c] = (cnt, at_boundary(mask_csr, iter_r, c))
            cnt += 1
    assert(cnt == total_n)
    cnt = 0
    for iter_i in A_buff.keys():
        iter_r = iter_i // mask_width
        iter_c = iter_i % mask_width
        if A_buff[iter_i][1]:
            # boundary
            A[1].append(cnt)
            A[2].append(1)
            b.append(bg_bitmap[iter_r + align_height][iter_c + align_width])
        else:
            # ordinary
            A[1].append(cnt)
            A[2].append(4)
            A[1].append(A_buff[(iter_r - 1) * mask_width + iter_c][0])
            A[2].append(-1)
            A[1].append(A_buff[(iter_r + 1) * mask_width + iter_c][0])
            A[2].append(-1)
            A[1].append(A_buff[iter_r * mask_width + iter_c - 1][0])
            A[2].append(-1)
            A[1].append(A_buff[iter_r * mask_width + iter_c + 1][0])
            A[2].append(-1)
            b.append(gradient_cal(ipt_bitmap, iter_r, iter_c, ipt_width, ipt_height))
        A[0].append(len(A[1]))
        x0.append(ipt_bitmap[iter_r][iter_c])
        cnt += 1
    assert(cnt == total_n)
    x = gauss_seidel(A, x0, b, times)
    cover = [[], [], []]
    cnt = 0
    current_iter_r = -1
    for iter_i in A_buff.keys():
        iter_r = (iter_i // mask_width) + align_height
        if current_iter_r != iter_r:
            for _ in range(iter_r - current_iter_r):
                cover[0].append(len(cover[1]))
            current_iter_r = iter_r
        iter_c = iter_i % mask_width + align_width
        cover[1].append(iter_c)
        cover[2].append(x[cnt])
        cnt += 1
    assert(cnt == total_n)
    for _ in range(ipt_height - current_iter_r):
        cover[0].append(len(cover[1]))
    ans_bitmap = update(bg_bitmap, cover)
    dst = image.fromarray(np.uint8(ans_bitmap))
    dst.save(opt_img)

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
    parser.add_argument("-y", "--height", type=int,
                        help="input align height on background",
                        required=True
                        )
    parser.add_argument("-x", "--width", type=int,
                        help="input align width on background",
                        required=True
                        )

    args = parser.parse_args()
    main(args.input, args.output, args.mask, args.background, args.times, args.height, args.width)
