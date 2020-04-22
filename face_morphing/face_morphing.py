import os, argparse
from math import ceil
from PIL import Image as image
import numpy as np

from face_api import get_mark
from delaunay import delaunay


def get_y_range(triangle, x):
    '''
    Get legal y range for given x in given triangle.
    triangle: [(x1, y1), (x2, y2), (x3, y3)]
    return: (y_min, y_max)
    '''
    legal_1 = True
    y1 = None
    if triangle[1][0] == triangle[0][0]:
        legal_1 = False
    else:
        r1 = (x - triangle[0][0]) / (triangle[1][0] - triangle[0][0])
        if r1 < 0 or r1 > 1:
            legal_1 = False
        else:
            y1 = (1 - r1) * triangle[0][1] + r1 * triangle[1][1]
    legal_2 = True
    y2 = None
    if triangle[2][0] == triangle[0][0]:
        legal_2 = False
    else:
        r2 = (x - triangle[0][0]) / (triangle[2][0] - triangle[0][0])
        if r2 < 0 or r2 > 1:
            legal_2 = False
        else:
            y2 = (1 - r2) * triangle[0][1] + r2 * triangle[2][1]
    legal_3 = True
    y3 = None
    if triangle[1][0] == triangle[2][0]:
        legal_3 = False
    else:
        r3 = (x - triangle[1][0]) / (triangle[2][0] - triangle[1][0])
        if r3 < 0 or r3 > 1:
            legal_3 = False
        else:
            y3 = (1 - r3) * triangle[1][1] + r3 * triangle[2][1]
    ans_1 = None
    ans_2 = None
    if legal_1:
        if legal_2:
            ans_1 = min(y1, y2)
            ans_2 = max(y1, y2)
        else:
            # assert(legal_3)
            ans_1 = min(y1, y3)
            ans_2 = max(y1, y3)
    else:
        # assert(legal_2)
        # assert(legal_3)
        ans_1 = min(y2, y3)
        ans_2 = max(y2, y3)
    return (ceil(ans_1) - 1, int(ans_2) + 1)

def remove_duplicate(point_set):
    '''Remove duplicated points and store.'''
    points = list(point_set.items())
    pop_list = []
    for iter_i in range(len(points)):
        for iter_j in range(iter_i + 1, len(points)):
            if points[iter_i][1]["x"] == points[iter_j][1]["x"] and points[iter_i][1]["y"] == points[iter_j][1]["y"]:
                pop_list.append(iter_j)
    pop_list.sort(reverse=True)
    poped_items = []
    for pop_iter in pop_list:
        poped_items.append(points.pop(pop_iter))
    return (dict(poped_items), dict(points))

def in_triangle(mark_out, triangle_data, x, y):
    '''
    Get which triangle (x, y) is in.
    return: iterator index of triangle_data.
    '''
    for iter_t in range(len(triangle_data)):
        triangle = triangle_data[iter_t][0]
        x_from = min(mark_out[triangle[0]]["y"], mark_out[triangle[1]]["y"], mark_out[triangle[2]]["y"])
        x_to = max(mark_out[triangle[0]]["y"], mark_out[triangle[1]]["y"], mark_out[triangle[2]]["y"])
        if x < x_from or x > x_to:
            continue
        triangle_range = [(mark_out[triangle[0]]["y"], mark_out[triangle[0]]["x"]),
                          (mark_out[triangle[1]]["y"], mark_out[triangle[1]]["x"]),
                          (mark_out[triangle[2]]["y"], mark_out[triangle[2]]["x"])]
        (y_min, y_max) = get_y_range(triangle_range, x)
        if y >= y_min and y <= y_max:
            return iter_t
    bias = 1
    while True:
        for iter_t in range(len(triangle_data)):
            triangle = triangle_data[iter_t][0]
            x_from = min(mark_out[triangle[0]]["y"], mark_out[triangle[1]]["y"], mark_out[triangle[2]]["y"])
            x_to = max(mark_out[triangle[0]]["y"], mark_out[triangle[1]]["y"], mark_out[triangle[2]]["y"])
            if x < x_from or x > x_to:
                continue
            triangle_range = [(mark_out[triangle[0]]["y"], mark_out[triangle[0]]["x"]),
                            (mark_out[triangle[1]]["y"], mark_out[triangle[1]]["x"]),
                            (mark_out[triangle[2]]["y"], mark_out[triangle[2]]["x"])]
            (y_min, y_max) = get_y_range(triangle_range, x)
            if y >= y_min - bias and y <= y_max + bias:
                # print("[WARN] (" + str(x) + ", " + str(y) + ") not found directly. Return with bias = " + str(bias))
                return iter_t
        bias += 1
    # assert(False) # Typically should not reach!
    return 0

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
    # assert(len(mark_from) == len(mark_to))
    (pop_from, mark_from) = remove_duplicate(mark_from)
    for item in pop_from.keys():
        mark_to.pop(item)
    triangles = delaunay(mark_from, from_height, from_width)
    for iter_n in range(num):
        triangles_result = []
        alpha = (iter_n + 1) / (num + 1)
        opt_height = int(round((1 - alpha) * from_height + alpha * to_height))
        opt_width = int(round((1 - alpha) * from_width + alpha * to_width))
        opt_bitmap = np.zeros((opt_height, opt_width, 3))
        mark_out = {}
        for point in mark_from.keys():
            tmp_y = (1 - alpha) * mark_from[point]["y"] + alpha * mark_to[point]["y"]
            if tmp_y < 0:
                tmp_y = 0
            elif tmp_y >= opt_height - 1:
                tmp_y = opt_height - 1
            else:
                tmp_y = int(round(tmp_y))
            tmp_x = (1 - alpha) * mark_from[point]["x"] + alpha * mark_to[point]["x"]
            if tmp_x < 0:
                tmp_x = 0
            elif tmp_x >= opt_width - 1:
                tmp_x = opt_width - 1
            else:
                tmp_x = int(round(tmp_x))
            mark_out[point] = {
                "y": tmp_y,
                "x": tmp_x
            }
        mark_out["global_left_up"] = {"y": 0, "x": 0}
        mark_out["global_mid_up"] = {"y": 0, "x": opt_width // 2}
        mark_out["global_right_up"] = {"y": 0, "x": opt_width - 1}
        mark_out["global_right_mid"] = {"y": opt_height // 2, "x": opt_width - 1}
        mark_out["global_right_down"] = {"y": opt_height - 1, "x": opt_width - 1}
        mark_out["global_mid_down"] = {"y": opt_height - 1, "x": opt_width // 2}
        mark_out["global_left_down"] = {"y": opt_height - 1, "x": 0}
        mark_out["global_left_mid"] = {"y": opt_height // 2, "x": 0}
        for triangle in triangles:
            from_array = np.array([
                [mark_from[triangle[0]]["y"], mark_from[triangle[1]]["y"], mark_from[triangle[2]]["y"]],
                [mark_from[triangle[0]]["x"], mark_from[triangle[1]]["x"], mark_from[triangle[2]]["x"]],
                [1, 1, 1]
            ])
            to_array = np.array([
                [mark_out[triangle[0]]["y"], mark_out[triangle[1]]["y"], mark_out[triangle[2]]["y"]],
                [mark_out[triangle[0]]["x"], mark_out[triangle[1]]["x"], mark_out[triangle[2]]["x"]],
                [1, 1, 1]
            ])
            to_all = np.array([
                [mark_to[triangle[0]]["y"], mark_to[triangle[1]]["y"], mark_to[triangle[2]]["y"]],
                [mark_to[triangle[0]]["x"], mark_to[triangle[1]]["x"], mark_to[triangle[2]]["x"]],
                [1, 1, 1]
            ])
            try:
                affine_array = np.dot(from_array, np.linalg.inv(to_array))
                full_array = np.dot(to_all, np.linalg.inv(to_array))
                triangles_result.append([triangle, affine_array, full_array])
            except:
                print("[WARN] to_array is a singular matrix.")
        for iter_x in range(opt_height):
            for iter_y in range(opt_width):
                index = in_triangle(mark_out, triangles_result, iter_x, iter_y)
                affine_array = triangles_result[index][1]
                full_array = triangles_result[index][2]
                to_vec = np.array([[iter_x], [iter_y], [1]])
                from_vec = np.dot(affine_array, to_vec)
                if from_vec[0][0] < 0:
                    from_vec[0][0] = 0
                elif from_vec[0][0] >= from_height - 1:
                    from_vec[0][0] = from_height - 1
                else:
                    from_vec[0][0] = int(round(from_vec[0][0]))
                if from_vec[1][0] < 0:
                    from_vec[1][0] = 0
                elif from_vec[1][0] >= from_width - 1:
                    from_vec[1][0] = from_width - 1
                else:
                    from_vec[1][0] = int(round(from_vec[1][0]))
                full_vec = np.dot(full_array, to_vec)
                if full_vec[0][0] < 0:
                    full_vec[0][0] = 0
                elif full_vec[0][0] >= to_height - 1:
                    full_vec[0][0] = to_height - 1
                else:
                    full_vec[0][0] = int(round(full_vec[0][0]))
                if full_vec[1][0] < 0:
                    full_vec[1][0] = 0
                elif full_vec[1][0] >= to_width - 1:
                    full_vec[1][0] = to_width - 1
                else:
                    full_vec[1][0] = int(round(full_vec[1][0]))
                opt_bitmap[iter_x][iter_y] = (1 - alpha) * from_bitmap[int(from_vec[0][0])][int(from_vec[1][0])] + alpha * to_bitmap[int(full_vec[0][0])][int(full_vec[1][0])]
        dst = image.fromarray(np.uint8(opt_bitmap))
        dst.save(prefix + "_" + str(num) + "_" + str(iter_n) + ".png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A script to do image fusion on given pictures."
        )
    parser.add_argument("-f", "--iptfrom", type=str,
                        help="input file, from: 0%% merge",
                        required=True
                        )
    parser.add_argument("-t", "--iptto", type=str,
                        help="input file, to: 100%% merge",
                        required=True
                        )
    parser.add_argument("-p", "--prefix", type=str,
                        help="output file name prefix",
                        required=True
                        )
    parser.add_argument("-n", "--num", type=int,
                        help="output middle stage file number(s); must >= 1 integer",
                        required=True
                        )

    args = parser.parse_args()
    main(args.iptfrom, args.iptto, args.prefix, args.num)
