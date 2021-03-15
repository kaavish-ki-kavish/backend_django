import numpy as np
import math
import random


def hbr_feature_extract(x, y, penups):
    '''
    x = list of x coordinates normalized such that the min of x is 0 and max is width
    y = list of y coordinates normalized such that the min of y is 0 and max is width
    penups = a set specifying indices where a penup happened
    returns a list of length 49 indicating HBR49 features
    '''

    # unit vectors in x and y direction
    u_x = np.array([1, 0])
    u_y = np.array([0, 1])

    # total number of points
    n = len(x)
    assert n == len(y)

    # first point
    s1 = x[0], y[0]
    # last point
    s_n = x[n - 1], y[n - 1]

    # total path length
    L = path_length(x, y, 0, n - 1, penups)

    # middle_point_index
    m = middle_point(x, y, L, penups)

    # bounding box points
    x_min, x_max = min(x), max(x)
    y_min, y_max = min(y), max(y)
    w = x_max - x_min
    h = y_max - y_min

    w = 1 if w == 0 else w
    h = 1 if h == 0 else h

    # center of b_box
    c_x = w / 2
    c_y = h / 2

    # meu center of gravity
    # NOT 100% SURE OF THIS
    meu_x = sum(x) / n
    meu_y = sum(y) / n

    # vector to store features
    features = np.zeros(49)
    l = max(w, h)

    # starting point
    features[0] = ((x[0] - c_x) / l) + 0.5
    features[1] = ((y[0] - c_y) / l) + 0.5

    # end point
    features[2] = ((x[n - 1] - c_x) / l) + 0.5
    features[3] = ((y[n - 1] - c_y) / l) + 0.5

    # first to last vector
    v = s_n[0] - s1[0], s_n[1] - s1[1]

    # norm
    features[4] = np.linalg.norm(v)

    # x-angle and condition to make sure the value doesnt blow
    features[5] = 0 if features[4] < l / 4 else np.dot(v, u_x) / np.linalg.norm(v)

    # y-angle and condition to make sure the values doesnt blow
    features[6] = 0 if features[4] < l / 4 else np.dot(v, u_y) / np.linalg.norm(v)

    # closure, would be zero for closed figures
    features[7] = features[4] / L

    # initial vector
    start = 0
    end = 2
    init_vector = x[end] - x[start], y[end] - y[start]

    # angle of initital vector
    # need to adjust spacing of points
    features[8] = np.dot(init_vector, u_x) / np.linalg.norm(init_vector)
    features[9] = np.dot(init_vector, u_y) / np.linalg.norm(init_vector)

    # inflexions

    features[10] = (1 / w) * (x[m] - ((x[0] - x[n - 1]) / 2))
    features[11] = (1 / h) * (y[m] - ((y[0] - y[n - 1]) / 2))

    down_strokes = random.randint(1, 6)  # to be computed
    features[12] = down_strokes

    # number of strokes
    features[13] = len(penups)

    # diagonal angle
    features[14] = math.atan(h / w)

    # trajectory Length
    features[15] = L

    # ratio between half parameter and trajectory length
    features[16] = (w + h) / L

    # average distance from the center of gravity: deviationx
    features[17] = (1 / n) * sum([distance.euclidean((x[i], y[i]), (meu_x, meu_y)) for i in range(n)])

    # average angle
    features[18] = (1 / n) * sum(
        [math.atan((y[i + 1] - y[i]) / (x[i + 1] - x[i])) for i in range(n - 1) if i + 1 not in penups])

    # angles for curvature and perpendicularity
    angles = np.array([stroke_angle(i, x, y) for i in range(1, n - 1)])

    # curvature
    features[19] = np.sum(angles)

    # perpendicularity
    features[20] = np.sum(np.sin(angles) ** 2)

    k = 2  # HYPERPARAM
    k_angles = np.array([stroke_angle(i, x, y, k) for i in range(k, n - k - 1)])

    features[21] = np.sum(np.sin(k_angles) ** 2)
    features[22] = np.max(k_angles)

    alpha = [np.arccos(np.clip(np.dot(np.array([x[i + 1] - x[i], y[i + 1] - y[i]]), u_x), -1, 1)) for i in
             range(0, n - 1) if i + i not in penups]
    angles_bins = np.array([(math.pi / 4) * i for i in range(8)])  # 8 angles, 45 degress
    h_bins = [0 for i in range(8)]  # bins for angles

    for angle in alpha:
        # angle ka bin

        bin = math.floor(angle / (math.pi / 4))

        # weight of each angle to each bin #NOT 100% sure of this
        w1 = 1 - (abs(angle - angles_bins[bin]) / (math.pi / 4))
        w2 = 1 - w1

        # adding weight to bin
        h_bins[bin] += w1
        h_bins[(bin + 1) % 8] += w2

    for i in range(4):
        features[23 + i] = (h_bins[i] + h_bins[i + 4]) / (n - len(penups))

    angles_bins = np.array([(math.pi / 4) * i for i in range(4)])  # 4 div between 0 to 180
    gamma = 0.25
    gammas = [(gamma * angles[i]) + ((1 - gamma) * k_angles[i]) for i in range(0, abs(n - k - 1 - k))]
    bins = [0 for i in range(4)]

    SMALL_VAL = 0.0000001
    for angle in gammas:
        # angle ka bin
        bin = math.floor((angle / (math.pi / 4)) - SMALL_VAL)

        # weight of each angle to each bin #NOT 100% sure of this
        w1 = 1 - (abs(angle - angles_bins[bin]) / (math.pi / 4))
        w2 = 1 - w1

        # adding weight to bin
        bins[bin] += w1
        bins[(bin + 1) % 4] += w2

    for i in range(4):
        features[27 + i] = bins[i] / (n - len(penups))

    bbox_bins = [[0 for i in range(3)] for j in range(3)]
    start_x = x_min
    start_y = y_min
    bbox_w = (x_max - x_min) / 3
    bbox_h = (y_max - y_min) / 3
    center_x = (start_x + bbox_w) / 2
    center_y = (start_y + bbox_h) / 2
    centers = [[((center_x + start_x) + (bbox_w * i), center_y + start_y + (bbox_h * j)) for i in range(3)] for j in
               range(3)]
    bins = [[0 for i in range(3)] for j in range(3)]

    dists = [0 for i in range(9)]
    for i in range(n):
        point = x[i], y[i]
        count = 0
        for row in range(3):
            for col in range(3):
                box_num = row, col
                center = centers[row][col]
                dists[count] = (1 / distance.euclidean(point, center), box_num)
                count += 1

        ##find closest 4 centers
        dists.sort(reverse=True)
        total_distance = sum([dists[i][0] for i in range(4)])
        weighted_dist = [dists[i][0] / total_distance for i in range(4)]

        for i in range(4):
            dist = dists[i][0]
            row = dists[i][1][0]
            col = dists[i][1][1]
            bins[row][col] += dist

    count = 0
    for row in range(3):
        for col in range(3):
            features[31 + count] = bins[row][col] / n
            count += 1

    moments = {(p, q): 0 for p in range(4) for q in range(4)}
    for p in range(4):
        for q in range(4):
            for i in range(n):
                x_i = x[i]
                y_i = y[i]
                moment = ((x_i - meu_x) ** p) * ((y_i - meu_y) ** q)
                moments[(p, q)] += moment

    for p in range(4):
        for q in range(4):
            m_pq = moments[(p, q)]
            gamma = 1 + ((p + q) / 2)
            moments[(p, q)] = (m_pq) / (moments[(0, 0)] ** gamma)

    features[40] = moments[(0, 2)] + moments[(2, 0)]
    features[41] = ((moments[(2, 0)] - moments[(0, 2)]) ** 2) + (4 * moments[(1, 1)]) ** 2
    features[42] = ((moments[(3, 0)] - 3 * moments[(1, 2)]) ** 2) + (3 * moments[2, 1] - moments[(0, 3)]) ** 2
    features[43] = ((moments[(3, 0)] + moments[(1, 2)]) ** 2) + (moments[(2, 1)] + moments[(0, 3)]) ** 2

    features[44] = ((moments[(3, 0)] - 3 * moments[(1, 2)]) ** 2) * (moments[(3, 0)] + moments[(1, 2)]) * (
            ((moments[(3, 0)] + moments[(1, 2)]) ** 2) - 3 * ((moments[(2, 1)] + moments[(0, 3)]) ** 2)) + (
                           ((3 * moments[(2, 1)]) - moments[(0, 3)]) * (moments[(2, 1)] + moments[(0, 3)]) * (
                       (3 * ((moments[(3, 0)] + moments[(1, 2)]) ** 2) - ((moments[(2, 1)] + moments[(0, 3)]) ** 2))))

    features[45] = ((moments[(2, 0)] - moments[(0, 2)]) * (
            ((moments[(3, 0)] + moments[(1, 2)]) ** 2) - ((moments[(2, 1)] + moments[(0, 3)]) ** 2))) + (
                           4 * (moments[(1, 1)]) * (moments[(3, 0)] + moments[(1, 2)]))

    features[46] = (((3 * moments[(2, 1)]) + moments[(0, 3)]) * (moments[(2, 1)] + moments[(0, 3)])) * (
        ((moments[(2, 1)] + moments[(0, 3)]) ** 2)) - (
                           ((moments[(3, 0)] + moments[(1, 2)]) * (moments[(2, 1)] + moments[(0, 3)])) * (
                           (3 * ((moments[(3, 0)] + moments[(1, 2)]) ** 2)) - (
                           moments[(2, 1)] + moments[(0, 3)]) ** 2))

    points = [(x[i], y[i]) for i in range(len(x))]
    from scipy.spatial import ConvexHull
    hull = ConvexHull(points)
    hull_vertices = hull.vertices
    a = 0
    b = 0
    for i in range(len(hull_vertices) - 1):
        a += x[i] * y[i + 1]
        b += -(x[i + 1] * y[i])
    summation = a, b

    A_h = np.linalg.norm(np.array(summation)) / 2
    features[47] = A_h / (w * h)
    features[48] = (L ** 2) / A_h
    return features


def stroke_angle(i, x, y, k=1):
    '''
    theta_i = refer to paper, eq . 14
    '''
    before = np.array(x[i - k], y[i - k])
    current = np.array(x[i], y[i])
    after = np.array(x[i + k], y[i + k])

    dist_before_current = current - before
    dist_current_after = after - current

    numerator = np.clip(np.dot(dist_before_current, dist_current_after), -1, 1)
    return np.arccos(numerator)  # equation 14


from scipy.spatial import distance


def get_downstrokes(x, y, penups):
    '''
    x = list of x coordinates
    y = list of y coordinates
    penups = a set specifying indices where a penup happened
    
    ki = 0
    penups = sorted(penups)
    strokes = []
    
    
    for i in penups:
        UP = dict()
        DOWN = dict()
        UP_COND = dict()
        poss_strokes = []
        kj = i - 1
        longest_path = 0
        for _start in range(ki, kj + 1):
            for _end in range(_start, kj + 1):
                if (_start, _end - 1) in DOWN:
                    #computing distance from end - 1 to end
                    DOWN[(_start, _end)] = DOWN[_start, _end - 1] + max(0, (y[_end] - y[_end - 1]))
                    
                    #if (_start, _end) exists in down, it also exists in UP.
                    UP[(_start, _end)] = UP[(_start, _end - 1)] + max(0, y[_end - 1] + y[_end])
                    UP_COND([_start, _end]) = UP[_start,_end] <= t2
                    
                    
                else:
                    assert (_start == _end)
                    DOWN[(_start, _end)] = 0
                    UP[(_start, _end)] = 0
                    DOWN_STROKE[(_start, _end)] = True
        
        
        _start = ki
        _end = kj
        
        while (_start != kj and _end != ki):
            if UP[(_start, _end)] < t:
            
        
        for _start in range(ki, kj + 1):
            for _end in range(kj, ki - 1, -1):
    '''


def is_downstroke(x, y, start, end, t1=0, t2=0):
    '''
    x = list of x coordinates
    y = list of y coordinates
    start = integer representing the start of path
    end = integer representing the end of path
    returns True or false specifying whether the stroke is a downstroke
    '''
    D = 0
    for k in range(start, end):
        D += max(0, y[k + 1] - y[k])

    if (D <= t1):
        return False


def path_length(x, y, start, end, penups):
    '''
    x = list of x coordinates 
    y = list of y coordiantes
    start = integer representing where to start the path length from
    end = integer representing where to end the path length to
    penups = a set specifying indices where a penup happened
    
    returns the path length of the path specified by x and y from start to end.
    '''
    dist = 0
    for index in range(start, end + 1):
        if index + 1 not in penups:
            start_point = [x[index], y[index]]
            end_point = [x[index + 1], y[index + 1]]
            dist += distance.euclidean(start_point, end_point)
    return dist


def middle_point(x, y, L, penups):
    '''
    x = list of x coordinates 
    y = list of y coordiantes
    penups = a set specifying indices where a penup happened
    
    returns the index of the middle point, which has path length approximately equal to L/2.
    '''
    half_length = L / 2
    for index in range(1, len(x)):
        current_path_length = path_length(x, y, 0, index, penups)
        if current_path_length > half_length:
            # return as soon as the path length becomes greater than half.
            return index - 1


def kaligo_Hist_of_orientations(x, y, penups):
    '''
    x = list of x coordinates
    y = list of y coordiantes
    penups = a set specifying indices where a penup happened
    :return: a numpy array of shape (48) each of each indicates the weighted histogram of orientations
    '''
    features = np.zeros(48)
    n = len(x)

    # final features
    cells = np.zeros((4, 3, 4))

    # mapping between range and index: eg: angles between 0 to math.pi/4 go to 0.
    bins = {math.pi / 4: 0, math.pi / 2: 1, (math.pi * 3) / 4: 2, (math.pi): 3}

    # bounding box points
    x_min, x_max = min(x), max(x)
    y_min, y_max = min(y), max(y)
    w = x_max - x_min
    h = y_max - y_min

    # dictionary to keep points for all cells
    # keys = x,y of the cell, values = tuples of the x,y of the points
    points_in_cell = {(i, j): [] for i in range(3) for j in range(4)}

    start_x = x_min
    start_y = y_min
    bbox_w = (x_max - x_min) / 4  # since 4 boxes in x direction
    bbox_h = (y_max - y_min) / 3  # since 3 boxes in y direction
    center_x = (start_x + bbox_w) / 2
    center_y = (start_y + bbox_h) / 2
    centers = {(i, j): ((center_x + start_x) + (bbox_w * i), center_y + start_y + (bbox_h * j)) for i in range(4) for j
               in range(3)}

    # just creating a sequencial list of points for each cell
    for i in range(n):
        x_dist = abs(x_min - x[i])
        y_dist = abs(y_min - y[i])
        x_bin = x_dist // bbox_w
        y_bin = y_dist // bbox_h
        points_in_cell[(x_bin, y_bin)].append((x, y))

    # going through each point in each cell and computing angle with next point
    for cell_x in range(4):
        for cell_y in range(3):
            center_gravity = np.mean([i[0] for i in points_in_cell[(cell_x, cell_y)]]), np.mean(
                [i[1] for i in points_in_cell[(cell_x, cell_y)]])
            angle_bins = np.zeros(4)
            for num in range(len(points_in_cell[(cell_x, cell_y)]) - 1):
                point = points_in_cell[(cell_x, cell_y)][num]
                n_point = points_in_cell[(cell_x, cell_y)][num + 1]
                # arccos angle
                angle = np.arccos(np.clip(np.dot(n_point, point) / np.linalg.norm(n_point) * np.linalg.norm, -1, 1))
                bin = max(angle // (math.pi / 4), 3)
                angle_bins[bin] += 1

            # now we have computed the bin for that cell

            # finding closest centers to the center of gravity
            closest_cells = sorted([(i, j) for i in range(4) for j in range(3)],
                                   key=lambda i, j: (distance.euclidean(center_gravity, centers[i, j])), reverse=True)
            closest_cells = closest_cells[:4]  # keeping closest four

            # sum of total distance (denominator of eq.1 in kaligo paper pg 15)
            total_sum = np.sum(
                [math.exp(-np.linalg.norm(np.array(center_gravity) - np.array(centers[(i, j)]))) for i, j in
                 closest_cells])

            for center in closest_centers:
                i, j = center
                # distance of center from center of gravity
                dist_from_gravity = math.exp(-np.linalg.norm(np.array(center_gravity) - np.array(centers[(i, j)])))

                # weight for that center
                w_ij = dist_from_gravity / total_sum

                # finallyyy adding the weighted orientation to our list of features
                cells[i, j] += w_ij * angle_bins

    return np.flatten(cells)


def get_chain_codes(x, y, dirs=8):
    # takes a list of x and y coordinates and returns a list denoting the chaincodes
    to_divide = (np.pi / 4) if dirs == 8 else (np.pi / 2)
    x = np.array(x)
    y = np.array(y)
    delta_x = np.diff(x)
    delta_y = np.diff(y)
    thetas = np.atan2(delta_y, delta_x)  # angles
    less_than_zero = thetas < 0  # finding negative angles
    thetas[less_than_zero] = thetas[less_than_zero] + (2 * np.pi)
    directions = np.clip(np.floor_divide(thetas, to_divide), 0, dirs - 1)
    return list(directions)


def scale_strokes(x, y, width, height):
    '''
    x = list of x-cordinates
    y = list of y-cordinates
    width = final width to scale to
    height = final height to scale to

    takes a list of x, y cordinates and normalizes then to the given width and height such that the minimum x
    becomes 0 and maximum x width
    '''
    min_x = min(x)
    max_x = max(x)
    min_y = min(y)
    max_y = max(y)

    # current width and height of bbox
    # adding +1 to avoid division by zero errors in exceptional cases.
    w = (max_x - min_x) + 1
    h = (max_y - min_y) + 1

    x = np.array(x)
    y = np.array(y)

    # so that minimum x and y becomes 0
    x = (x - min_x) / w
    y = (y - min_y) / h

    # scaling them to the new given range
    scaled_x = x * width
    scaled_y = y * height

    return scaled_x, scaled_y


# import pickle
# import pandas as pd
# import os
# import numpy as np
#
# __location__ = os.path.realpath(
#     os.path.join(os.getcwd(), os.path.dirname(__file__)))
#
#
# class RandomForestClassifier:
#     def __init__(self):
#         file = open(os.path.join(__location__, 'random_forest_model.sav'), 'rb')
#         self.model = pickle.load(file)
#
#     def preprocessing(self, input_data):
#         return input_data
#
#     def predict(self, input_data):
#         a = self.model.predict_proba(input_data)
#         return a
#
#     def postprocessing(self, input_data):
#         # label = "<=50K"
#         # if input_data[1] > 0.5:
#         #     label = ">50K"
#         # return {"probability": input_data, "label": label, "status": "OK"}
#         return input_data
#
#     def compute_prediction(self, input_data):
#         try:
#             input_data = self.preprocessing(input_data)
#             #prediction = self.predict(input_data)[0][0]  # only one sample
#             prediction = np.amax(self.predict(input_data))
#             # prediction = self.postprocessing(prediction)
#         except Exception as e:
#             return {"status": "Error", "message": str(e)}
#
#         return prediction
#
#
# import random
# import copy
#
#
# # points = np.arange(20)
# # points = points.tolist()
# # whole_x = copy.deepcopy(points)
# # whole_y = copy.deepcopy(points)
# # random.shuffle(whole_y)
# #
# # print("x:  ", whole_x)
# # print("y:  ", whole_y)
#
# whole_x = [106, 79, 61, 43, 25, 14, 3, 0, 1, 10, 23, 42, 65, 83, 99, 121, 139, 157, 174, 209, 221, 233, 251, 255, 255, 254, 248, 226, 201, 163, 109, 107, 104, 118, 139, 160, 162, 159, 139, 124, 117]
# whole_y = [89, 73, 71, 82, 102, 121, 150, 168, 195, 219, 231, 241, 247, 245, 238, 213, 225, 230, 233, 231, 225, 213, 181, 165, 143, 130, 118, 100, 89, 83, 87, 85, 62, 33, 10, 0, 5, 12, 34, 57, 86]
# scale_x, scale_y = scale_strokes(whole_x, whole_y, 128, 128)
# hbr = hbr_feature_extract(scale_x, scale_y, {len(whole_x)})
#
# r_m = RandomForestClassifier()
# print(r_m.compute_prediction(hbr))

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw


def cordToImage(x, y):
    x_0, y_0 = min(x), min(y)
    x_n, y_n = max(x), max(y)
    w = x_n - x_0 + 1
    h = y_n - y_0 + 1
    padding = 20
    x_origin = [x_cord - x_0 for x_cord in x]
    y_origin = [y_cord - y_0 for y_cord in y]
    x_y = list(zip(x_origin, y_origin))

    im = Image.new('RGB', (w + padding, h + padding), (0, 0, 0))
    draw = ImageDraw.Draw(im)
    draw.line(x_y)
    img_array = np.array(im)
    plt.axis('off')
    plt.imshow(img_array)
    plt.show()








x = [106, 79, 61, 43, 25, 14, 3, 0, 1, 10, 23, 42, 65, 83, 99, 121, 139, 157, 174, 209, 221, 233, 251, 255, 255, 254, 248, 226, 201, 163, 109, 107, 104, 118, 139, 160, 162, 159, 139, 124, 117]
y = [89, 73, 71, 82, 102, 121, 150, 168, 195, 219, 231, 241, 247, 245, 238, 213, 225, 230, 233, 231, 225, 213, 181, 165, 143, 130, 118, 100, 89, 83, 87, 85, 62, 33, 10, 0, 5, 12, 34, 57, 86]
cordToImage(x, y)