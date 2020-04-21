import math
import numpy as np


def x_y_sort(point_set):
    '''
    Sort point_set with:
    1. x
    2. y
    '''
    mid = sorted(point_set.items(), key=lambda item: item[1]["y"])
    return sorted(mid, key=lambda item: item[1]["x"])

def convex_hull(point_set):
    '''
    Get convex hull of point set.
    '''
    pass

def ccw(from_point, sub_set):
    '''
    Find counter-clockwise-next edge point in sub_set.
    '''
    pass

def cw(sub_set):
    '''
    Find clockwise-next edge point in sub_set.
    '''
    pass

def zig_zag(left_set, right_set):
    '''
    Find common tangent.
    '''
    pass

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
        ]
    }
    '''
    pass

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
        ]
    }
    Note: translate array to dict:
        point_set = {}
        for item in subset:
            point_set[item[0]] = item[1]
    '''
    total_len = len(subset)
    if total_len == 1:
        # ERROR: Unreachable!
        ans = {
            "edges": {
                subset[0][0]: []
            },
            "points": subset
        }
        return ans
    elif total_len == 2:
        ans = {
            "edges": {},
            "points": subset
        }
        ans["edges"][subset[0][0]] = []
        ans["edges"][subset[1][0]] = []
        ans["edges"][subset[0][0]].append(subset[1][0])
        ans["edges"][subset[1][0]].append(subset[0][0])
        return ans
    elif total_len == 3:
        ans = {
            "edges": {},
            "points": subset
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
