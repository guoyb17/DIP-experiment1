from math import acos, sqrt, pi
import numpy as np


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
    degree = acos((vec_from["x"] * vec_to["x"] + vec_from["y"] * vec_to["y"]) / (sqrt(vec_from["x"] ** 2 + vec_from["y"] ** 2) * sqrt(vec_to["x"] ** 2 + vec_to["y"] ** 2)))
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
    start = sub_set["points"][0][0 if is_cw else -1]
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
    return ans

def test_convex_hull(sub_set, is_cw=False):
    '''
    Simple test of get_convex_hull, cw, and ccw.
    '''
    start = sub_set["points"][0][0 if is_cw else -1]
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
    pass

def zig_zag(left_set, right_set):
    '''
    Find common tangent.
    return: {
        "top": (TL, TR),
        "bottom": (BL, BR)
    }
    '''
    assert(test_convex_hull(left_set, False))
    assert(test_convex_hull(right_set, True))
    ans = {}
    left_convex_hull = get_convex_hull(left_set, False)
    right_convex_hull = get_convex_hull(right_set, True)
    L = left_convex_hull[0]
    R = right_convex_hull[0]
    iter_l = 0
    iter_r = 0
    while True:
        if get_degree(left_set["point_set"][L], right_set["point_set"][R], right_set["point_set"][right_convex_hull[iter_r + 1]])[1] == 1:
            iter_r += 1
            R = right_convex_hull[iter_r]
        elif get_degree(left_set["point_set"][L], right_set["point_set"][R], left_set["point_set"][left_convex_hull[iter_l + 1]])[1] == 1:
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
    ans = {
        "edges": dict(left_ans["edges"].items() + right_ans["edges"].items()),
        "points": left_ans["points"] + right_ans["points"],
        "point_set": dict(left_ans["point_set"].items() + right_ans["point_set"].items())
    }
    L = boundary["bottom"][0]
    R = boundary["bottom"][1]
    ans["edges"][L].append(R)
    ans["edges"][R].append(L)
    while L != boundary["top"][0] and R != boundary["top"][1]:
        left_candidates = {}
        for candidate in left_ans["edges"][L]:
            (tmp, sign) = get_degree(left_ans["point_set"][L], right_ans["point_set"][R], left_ans["point_set"][candidate])
            if sign == -1: # should not happen
                tmp = -tmp
            left_candidates[candidate] = tmp

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
        ans["edges"][subset[0][0]].append(subset[1][0])
        ans["edges"][subset[0][0]].append(subset[2][0])
        ans["edges"][subset[1][0]].append(subset[0][0])
        ans["edges"][subset[1][0]].append(subset[2][0])
        ans["edges"][subset[2][0]].append(subset[0][0])
        ans["edges"][subset[2][0]].append(subset[1][0])
        return ans
    else:
        left = subset[0 : (total_len // 2)]
        right = subset[((total_len // 2) + 1) : (total_len - 1)]
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
    '''
    data_set = x_y_sort(point_set)
    pass
