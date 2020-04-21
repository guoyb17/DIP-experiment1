from copy import deepcopy
from math import acos, sqrt, pi
import numpy as np
import time


def x_y_sort(point_set):
    '''
    Sort point_set with:
    1. x
    2. y
    '''
    mid = sorted(point_set.items(), key=lambda item: item[1]["y"])
    return sorted(mid, key=lambda item: item[1]["x"])

def get_degree(O, F, T):
    '''
    Get degree of angle FOT within [0, pi).
    Get degree direction of angle FOT, whose direction is decided by F(rom) and T(o).
    return[0]: angle FOT within [0, pi).
    return[1]: 1 for positive, 0 for common line, -1 for negative.
    '''
    vec_from = {
        "x": F["x"] - O["x"],
        "y": F["y"] - O["y"]
    }
    vec_to = {
        "x": T["x"] - O["x"],
        "y": T["y"] - O["y"]
    }
    if sqrt(vec_from["x"] ** 2 + vec_from["y"] ** 2) * sqrt(vec_to["x"] ** 2 + vec_to["y"] ** 2) != 0:
        cos_value = (vec_from["x"] * vec_to["x"] + vec_from["y"] * vec_to["y"]) / (sqrt(vec_from["x"] ** 2 + vec_from["y"] ** 2) * sqrt(vec_to["x"] ** 2 + vec_to["y"] ** 2))
        if cos_value > 1:
            cos_value = 1
        elif cos_value < -1:
            cos_value = -1
    else:
        cos_value = 1
    degree = acos(cos_value)
    ans = vec_from["x"] * vec_to["y"] - vec_from["y"] * vec_to["x"]
    if ans > 0:
        return (degree, 1)
    elif ans < 0:
        return (degree, -1)
    else:
        return (degree, 0)

def ccw(from_point, sub_set):
    '''
    Find counter-clockwise-next edge point in sub_set.
    Look for min of angle FOT, which is within [-pi, pi].
    '''
    ans = from_point
    for candidate in sub_set["edges"][from_point]:
        if ans == from_point:
            ans = candidate
        else:
            (tmp, sign) = get_degree(sub_set["point_set"][from_point], sub_set["point_set"][ans], sub_set["point_set"][candidate])
            if sign == -1:
                tmp = -tmp
            if tmp < 0:
                ans = candidate
    return ans

def cw(from_point, sub_set):
    '''
    Find counter-clockwise-next edge point in sub_set.
    Look for max of angle FOT, which is within [-pi, pi].
    '''
    ans = from_point
    for candidate in sub_set["edges"][from_point]:
        if ans == from_point:
            ans = candidate
        else:
            (tmp, sign) = get_degree(sub_set["point_set"][from_point], sub_set["point_set"][ans], sub_set["point_set"][candidate])
            if sign == -1:
                tmp = -tmp
            if tmp > 0:
                ans = candidate
    return ans

def get_convex_hull(sub_set, is_cw=True):
    '''
    Get convex hull of sub_set.
    '''
    start = sub_set["points"][0 if is_cw else -1][0]
    next_point = start
    ans = []
    ans.append(start)
    if is_cw:
        next_point = cw(next_point, sub_set)
    else:
        next_point = ccw(next_point, sub_set)
    while next_point not in ans:
        ans.append(next_point)
        if is_cw:
            next_point = cw(next_point, sub_set)
        else:
            next_point = ccw(next_point, sub_set)
    return ans

def test_convex_hull(sub_set, is_cw=False):
    '''
    Simple test of get_convex_hull, cw, and ccw.
    '''
    start = sub_set["points"][0 if is_cw else -1][0]
    next_point = start
    ans = []
    ans.append(start)
    if is_cw:
        next_point = cw(next_point, sub_set)
    else:
        next_point = ccw(next_point, sub_set)
    while next_point != start:
        ans.append(next_point)
        if is_cw:
            next_point = cw(next_point, sub_set)
        else:
            next_point = ccw(next_point, sub_set)
    to_test = get_convex_hull(sub_set, not is_cw)
    for item in to_test:
        if item not in ans:
            return False
    return True

def in_circle(A, B, C, D):
    '''
    Judge whether D is within Circle ABC.
    '''
    sign = np.array([
        [A["x"], A["y"], A["x"] * A["x"] + A["y"] * A["y"]],
        [B["x"], B["y"], B["x"] * B["x"] + B["y"] * B["y"]],
        [C["x"], C["y"], C["x"] * C["x"] + C["y"] * C["y"]]
    ])
    ret = np.array([
        [A["x"], A["y"], A["x"] * A["x"] + A["y"] * A["y"], 1],
        [B["x"], B["y"], B["x"] * B["x"] + B["y"] * B["y"], 1],
        [C["x"], C["y"], C["x"] * C["x"] + C["y"] * C["y"], 1],
        [D["x"], D["y"], D["x"] * D["x"] + D["y"] * D["y"], 1]
    ])
    return np.linalg.det(ret) * np.linalg.det(sign) < 0

def zig_zag(left_set, right_set):
    '''
    Find common tangent.
    return: {
        "top": (TL, TR),
        "bottom": (BL, BR)
    }
    '''
    # assert(test_convex_hull(left_set, False))
    # assert(test_convex_hull(right_set, True))
    ans = {}
    left_convex_hull = get_convex_hull(left_set, False)
    right_convex_hull = get_convex_hull(right_set, True)
    L = left_convex_hull[0]
    R = right_convex_hull[0]
    iter_l = 0
    iter_r = 0
    while True:
        if (iter_r < len(right_convex_hull) - 1) and (get_degree(left_set["point_set"][L], right_set["point_set"][R], right_set["point_set"][right_convex_hull[iter_r + 1]])[1] == 1):
            iter_r += 1
            R = right_convex_hull[iter_r]
        elif (iter_l < len(left_convex_hull) - 1) and (get_degree(left_set["point_set"][L], right_set["point_set"][R], left_set["point_set"][left_convex_hull[iter_l + 1]])[1] == 1):
            iter_l += 1
            L = left_convex_hull[iter_l]
        else:
            break
    ans["top"] = (L, R)
    L = left_convex_hull[0]
    R = right_convex_hull[0]
    iter_l = 0
    iter_r = 0
    while True:
        if get_degree(left_set["point_set"][L], right_set["point_set"][R], right_set["point_set"][right_convex_hull[iter_r - 1]])[1] == -1:
            iter_r -= 1
            R = right_convex_hull[iter_r]
        elif get_degree(left_set["point_set"][L], right_set["point_set"][R], left_set["point_set"][left_convex_hull[iter_l - 1]])[1] == -1:
            iter_l -= 1
            L = left_convex_hull[iter_l]
        else:
            break
    ans["bottom"] = (L, R)
    return ans

def merge_ans(left_ans, right_ans):
    '''
    Merge two subsets.
    params and return: dict {
        "edges": {
            "position1": ["position2", ...],
            "position2": ["position1", ...],
            ...
        },
        "points": [
            ("position", {
                "y": 123,
                "x": 234
            }),
            ...
        ],
        "point_set": {
            "position": {
                "y": 123,
                "x": 234
            },
            ...
        }
    }
    '''
    boundary = zig_zag(left_ans, right_ans)
    edges = deepcopy(left_ans["edges"])
    edges.update(deepcopy(right_ans["edges"]))
    point_set = deepcopy(left_ans["point_set"])
    point_set.update(deepcopy(right_ans["point_set"]))
    ans = {
        "edges": edges,
        "points": left_ans["points"] + right_ans["points"],
        "point_set": point_set
    }
    L = boundary["bottom"][0]
    R = boundary["bottom"][1]
    ans["edges"][L].append(R)
    ans["edges"][R].append(L)
    while L != boundary["top"][0] and R != boundary["top"][1]:
        left_candidates = {}
        for candidate in left_ans["edges"][L]:
            (tmp, sign) = get_degree(ans["point_set"][L], ans["point_set"][R], ans["point_set"][candidate])
            if sign == -1: # should not happen
                tmp = -tmp
            left_candidates[candidate] = tmp
        left_candidates = dict(sorted(left_candidates.items(), key=lambda item: item[1]))
        right_candidates = {}
        for candidate in right_ans["edges"][R]:
            (tmp, sign) = get_degree(ans["point_set"][R], ans["point_set"][candidate], ans["point_set"][L])
            if sign == -1: # should not happen
                tmp = -tmp
            right_candidates[candidate] = tmp
        right_candidates = dict(sorted(right_candidates.items(), key=lambda item: item[1]))
        iter_l = 0
        selected_l = ""
        left_list = list(left_candidates.keys())
        while True:
            tmp_left = left_list[iter_l]
            if left_candidates[tmp_left] < 0:
                break
            else:
                if (iter_l < len(left_list) - 1) and in_circle(ans["point_set"][L], ans["point_set"][R], ans["point_set"][tmp_left], ans["point_set"][left_list[iter_l + 1]]):
                    iter_l += 1
                    if tmp_left in ans["edges"][L]:
                        ans["edges"][L].remove(tmp_left)
                        ans["edges"][tmp_left].remove(L)
                else:
                    selected_l = tmp_left
                    break
        iter_r = 0
        selected_r = ""
        right_list = list(right_candidates.keys())
        while True:
            tmp_right = right_list[iter_r]
            if right_candidates[tmp_right] < 0:
                break
            else:
                if (iter_r < len(right_list) - 1) and in_circle(ans["point_set"][L], ans["point_set"][R], ans["point_set"][tmp_right], ans["point_set"][right_list[iter_r + 1]]):
                    iter_r += 1
                    if tmp_right in ans["edges"][R]:
                        ans["edges"][R].remove(tmp_right)
                        ans["edges"][tmp_right].remove(R)
                else:
                    selected_r = tmp_right
                    break
        if selected_l == "":
            if selected_r == "":
                # ERROR: Unreachable!
                break
            else:
                # case LR1
                R = selected_r
                ans["edges"][L].append(R)
                ans["edges"][R].append(L)
        else:
            if selected_r == "":
                # case L1R
                L = selected_l
                ans["edges"][L].append(R)
                ans["edges"][R].append(L)
            else:
                if in_circle(ans["point_set"][L], ans["point_set"][R], ans["point_set"][selected_l], ans["point_set"][selected_r]):
                    # R1 in Circle LRL1, select R1
                    R = selected_r
                    ans["edges"][L].append(R)
                    ans["edges"][R].append(L)
                else:
                    # R1 out of Circle LRL1, select L1
                    L = selected_l
                    ans["edges"][L].append(R)
                    ans["edges"][R].append(L)
    return ans

def triangulate(subset):
    '''
    Recursively do delaunay triangulation.
    return: {
        "edges": {
            "position1": ["position2", ...],
            "position2": ["position1", ...],
            ...
        },
        "points": [
            ("position", {
                "y": 123,
                "x": 234
            }),
            ...
        ],
        "point_set": {
            "position": {
                "y": 123,
                "x": 234
            },
            ...
        }
    }
    Note: translate array to dict:
        point_set = {}
        for item in subset:
            point_set[item[0]] = item[1]
    '''
    total_len = len(subset)
    point_set = dict(subset)
    if total_len == 1:
        # ERROR: Unreachable!
        ans = {
            "edges": {
                subset[0][0]: []
            },
            "points": subset,
            "point_set": point_set
        }
        return ans
    elif total_len == 2:
        ans = {
            "edges": {},
            "points": subset,
            "point_set": point_set
        }
        ans["edges"][subset[0][0]] = []
        ans["edges"][subset[1][0]] = []
        ans["edges"][subset[0][0]].append(subset[1][0])
        ans["edges"][subset[1][0]].append(subset[0][0])
        return ans
    elif total_len == 3:
        ans = {
            "edges": {},
            "points": subset,
            "point_set": point_set
        }
        ans["edges"][subset[0][0]] = []
        ans["edges"][subset[1][0]] = []
        ans["edges"][subset[2][0]] = []
        check_case = get_degree(ans["point_set"][subset[0][0]], ans["point_set"][subset[1][0]], ans["point_set"][subset[2][0]])
        if check_case[1] == 0:
            # Special case: three points on the same line
            if check_case[0] == pi:
                ans["edges"][subset[0][0]].append(subset[1][0])
                ans["edges"][subset[1][0]].append(subset[0][0])
                ans["edges"][subset[0][0]].append(subset[2][0])
                ans["edges"][subset[2][0]].append(subset[0][0])
            elif get_degree(ans["point_set"][subset[1][0]], ans["point_set"][subset[2][0]], ans["point_set"][subset[0][0]])[0] == pi:
                ans["edges"][subset[0][0]].append(subset[1][0])
                ans["edges"][subset[1][0]].append(subset[0][0])
                ans["edges"][subset[1][0]].append(subset[2][0])
                ans["edges"][subset[2][0]].append(subset[1][0])
            else:
                ans["edges"][subset[0][0]].append(subset[2][0])
                ans["edges"][subset[2][0]].append(subset[0][0])
                ans["edges"][subset[1][0]].append(subset[2][0])
                ans["edges"][subset[2][0]].append(subset[1][0])
        else:
            ans["edges"][subset[0][0]].append(subset[1][0])
            ans["edges"][subset[0][0]].append(subset[2][0])
            ans["edges"][subset[1][0]].append(subset[0][0])
            ans["edges"][subset[1][0]].append(subset[2][0])
            ans["edges"][subset[2][0]].append(subset[0][0])
            ans["edges"][subset[2][0]].append(subset[1][0])
        return ans
    else:
        left = subset[0 : (total_len // 2)]
        right = subset[(total_len // 2) : total_len]
        left_ans = triangulate(left)
        right_ans = triangulate(right)
        return merge_ans(left_ans, right_ans)

def delaunay(point_set):
    '''
    Do Delaunay Triangulation.
    point_set: dict {
        "position": {
            "y": 123,
            "x": 234
        },
        ...
    }
    data_set: array [
        ("position", {
            "y": 123,
            "x": 234
        }),
        ...
    ]
    results: dict {
        "edges": {
            "position1": ["position2", ...],
            "position2": ["position1", ...],
            ...
        },
        "points": [
            ("position", {
                "y": 123,
                "x": 234
            }),
            ...
        ],
        "point_set": {
            "position": {
                "y": 123,
                "x": 234
            },
            ...
        }
    }
    return: array [
        ["position1", "position2", "position3"],
        ...
    ]
    e = p - a + n, e = 1, p = number of points, a = number of edges, n = number of triangles.
    '''
    data_set = x_y_sort(point_set)
    results = triangulate(data_set)
    total_len = len(results["points"])
    # edge_cnt = 0
    # for edge_set in results["edges"].values():
    #     edge_cnt += len(edge_set)
    # edge_cnt = edge_cnt // 2
    ans = []
    for iter1 in range(total_len):
        for iter2 in range(iter1 + 1, total_len):
            if results["points"][iter1][0] in results["edges"][results["points"][iter2][0]]:
                # assert(results["points"][iter2][0] in results["edges"][results["points"][iter1][0]])
                for iter3 in range(iter2 + 1, total_len):
                    if results["points"][iter3][0] in results["edges"][results["points"][iter1][0]] and results["points"][iter3][0] in results["edges"][results["points"][iter2][0]]:
                        # assert(results["points"][iter1][0] in results["edges"][results["points"][iter3][0]] and results["points"][iter2][0] in results["edges"][results["points"][iter3][0]])
                        ans.append([results["points"][iter1][0], results["points"][iter2][0], results["points"][iter3][0]])
    # assert(1 == len(results["points"]) - edge_cnt + len(ans))
    return ans
