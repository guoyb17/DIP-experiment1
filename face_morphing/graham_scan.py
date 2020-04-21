from math import acos, sqrt, pi


def get_bottom_point(points):
    """
    返回points中纵坐标最小的点的索引，如果有多个纵坐标最小的点则返回其中横坐标最小的那个
    :param points:
    :return:
    """
    min_index = 0
    n = len(points)
    for i in range(0, n):
        if points[i][1]["y"] < points[min_index][1]["y"] or (points[i][1]["y"] == points[min_index][1]["y"] and points[i][1]["x"] < points[min_index][1]["x"]):
            min_index = i
    return min_index
 
 
def sort_polar_angle_cos(points, center_point):
    """
    按照与中心点的极角进行排序，使用的是余弦的方法
    :param points: 需要排序的点
    :param center_point: 中心点
    :return:
    """
    n = len(points)
    cos_value = []
    rank = []
    norm_list = []
    for i in range(0, n):
        point_ = points[i][1]
        point = [point_["x"] - center_point["x"], point_["y"] - center_point["y"]]
        rank.append(i)
        norm_value = sqrt(point[0] * point[0] + point[1] * point[1])
        norm_list.append(norm_value)
        if norm_value == 0:
            cos_value.append(1)
        else:
            cos_value.append(point[0] / norm_value)
 
    for i in range(0, n-1):
        index = i + 1
        while index > 0:
            if cos_value[index] > cos_value[index - 1] or (cos_value[index] == cos_value[index - 1] and norm_list[index] > norm_list[index - 1]):
                temp = cos_value[index]
                temp_rank = rank[index]
                temp_norm = norm_list[index]
                cos_value[index] = cos_value[index - 1]
                rank[index] = rank[index - 1]
                norm_list[index] = norm_list[index - 1]
                cos_value[index - 1] = temp
                rank[index - 1] = temp_rank
                norm_list[index - 1] = temp_norm
                index = index - 1
            else:
                break
    sorted_points = []
    for i in rank:
        sorted_points.append(points[i])
 
    return sorted_points
 
 
def vector_angle(vector):
    """
    返回一个向量与向量 [1, 0]之间的夹角， 这个夹角是指从[1, 0]沿逆时针方向旋转多少度能到达这个向量
    :param vector:
    :return:
    """
    norm_ = sqrt(vector[0] * vector[0] + vector[1] * vector[1])
    if norm_ == 0:
        return 0
 
    angle = acos(vector[0] / norm_)
    if vector[1] >= 0:
        return angle
    else:
        return 2 * pi - angle
 
 
def coss_multi(v1, v2):
    """
    计算两个向量的叉乘
    :param v1:
    :param v2:
    :return:
    """
    return v1[0] * v2[1] - v1[1] * v2[0]
 
 
def graham_scan(points):
    '''
    points: array [("position1", {"x": 123, "y": 234}), ...]
    e.g. points = [("position1", {"x": 1.1, "y": 3.6}),
                   ("position2", {"x": 2.1, "y": 5.4}),
                   ("position3", {"x": 2.5, "y": 1.8}),
                   ("position4", {"x": 3.3, "y": 3.98}),
                   ("position5", {"x": 4.8, "y": 6.2}),
                   ("position6", {"x": 4.3, "y": 4.1}),
                   ("position7", {"x": 4.2, "y": 2.4}),
                   ("position8", {"x": 5.9, "y": 3.5}),
                   ("position9", {"x": 6.2, "y": 5.3}),
                   ("position10", {"x": 6.1, "y": 2.56}),
                   ("position11", {"x": 7.4, "y": 3.7}),
                   ("position12", {"x": 7.1, "y": 4.3}),
                   ("position13", {"x": 7, "y": 4.1})]
    return: array [("position1", {"x": 123, "y": 234}), ...]
    '''
    # print("Graham扫描法计算凸包")
    bottom_index = get_bottom_point(points)
    bottom_point = points.pop(bottom_index)
    sorted_points = sort_polar_angle_cos(points, bottom_point[1])
 
    m = len(sorted_points)
    if m < 2:
        print("点的数量过少，无法构成凸包")
        return
 
    stack = []
    stack.append(bottom_point)
    stack.append(sorted_points[0])
    stack.append(sorted_points[1])
 
    for i in range(2, m):
        length = len(stack)
        top = stack[length - 1]
        next_top = stack[length - 2]
        v1 = [sorted_points[i][1]["x"] - next_top[1]["x"], sorted_points[i][1]["y"] - next_top[1]["y"]]
        v2 = [top[1]["x"] - next_top[1]["x"], top[1]["y"] - next_top[1]["y"]]
 
        while coss_multi(v1, v2) >= 0:
            stack.pop()
            length = len(stack)
            top = stack[length - 1]
            next_top = stack[length - 2]
            v1 = [sorted_points[i][1]["x"] - next_top[1]["x"], sorted_points[i][1]["y"] - next_top[1]["y"]]
            v2 = [top[1]["x"] - next_top[1]["x"], top[1]["y"] - next_top[1]["y"]]
 
        stack.append(sorted_points[i])
 
    return stack
