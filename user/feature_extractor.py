import numpy as np
import math
import random
import os

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))

def hbr_feature_extract(x, y, penups):
    '''
    x = list of x coordinates normalized such that the min of x is 0 and max is width
    y = list of y coordinates normalized such that the min of y is 0 and max is width
    penups = a set specifying indices where a penup happened
    returns a list of length 49 indicating HBR49 features
    '''
    
    #unit vectors in x and y direction
    u_x = np.array([1, 0])
    u_y = np.array([0, 1])
    
    #total number of points
    n = len(x)
    assert n == len(y) 
    
    #first point
    s1 = x[0], y[0]
    #last point
    s_n = x[n - 1], y[n - 1]
    
    #total path length
    L = path_length(x, y, 0, n - 1, penups)
    
    #middle_point_index
    m = middle_point(x, y, L, penups)
    
    
    
    #bounding box points
    x_min, x_max = min(x), max(x)
    y_min, y_max = min(y), max(y)
    w = x_max - x_min
    h = y_max - y_min
    
    w = 1 if w == 0 else w
    h = 1 if h == 0 else h
    
    
    #center of b_box
    c_x = w / 2
    c_y = h / 2
    
    #meu center of gravity
    # NOT 100% SURE OF THIS
    meu_x = sum(x) / n
    meu_y = sum(y) / n
    
    #vector to store features
    features = np.zeros(49)
    l = max(w, h)
    
    #starting point
    features[0] = ((x[0] - c_x)/l) + 0.5
    features[1] = ((y[0] - c_y)/l) + 0.5
    
    #end point
    features[2] = ((x[n - 1] - c_x)/l) + 0.5
    features[3] = ((y[n - 1] - c_y)/l) + 0.5
    
    #first to last vector
    v = s_n[0] - s1[0], s_n[1] - s1[1]
    
    #norm
    features[4] = np.linalg.norm(v)

    #x-angle and condition to make sure the value doesnt blow
    features[5] = 0 if features[4] < l/4 else np.dot(v, u_x)/np.linalg.norm(v)
    
    #y-angle and condition to make sure the values doesnt blow
    features[6] = 0 if features[4] < l/4 else np.dot(v, u_y)/np.linalg.norm(v)
    
    
    
    #closure, would be zero for closed figures
    features[7] = features[4] / L
    
    #initial vector
    start = 0
    end = 2
    init_vector = x[end] - x[start], y[end] - y[start]
    
    #angle of initital vector 
    #need to adjust spacing of points
    features[8] = np.dot(init_vector, u_x) / np.linalg.norm(init_vector)
    features[9] = np.dot(init_vector, u_y) / np.linalg.norm(init_vector)
    
    #inflexions
    

    features[10] = (1/w) * (x[m] - ((x[0] - x[n-1])/2))
    features[11] = (1/h) * (y[m] - ((y[0] - y[n-1])/2))
    
    down_strokes = random.randint(1, 6) #to be computed
    features[12] = down_strokes
    
    #number of strokes
    features[13] = len(penups)
    
    #diagonal angle
    features[14] = math.atan(h / w)
    
    #trajectory Length
    features[15] = L
    
    #ratio between half parameter and trajectory length
    features[16] = (w + h) / L
    
    #average distance from the center of gravity: deviationx
    features[17] = (1/n) * sum([distance.euclidean((x[i], y[i]), (meu_x, meu_y)) for i in range(n)])
    
    #average angle
    features[18] = (1 / n) * sum([math.atan((y[i + 1] - y[i]) / (x[i + 1] - x[i])) for i in range(n - 1) if i + 1 not in penups])
    
    #angles for curvature and perpendicularity
    angles = np.array([stroke_angle(i, x, y) for i in range(1, n - 1)])
    
    #curvature
    features[19] = np.sum(angles)
    
    #perpendicularity
    features[20] = np.sum(np.sin(angles) ** 2)
    
    k = 2 #HYPERPARAM
    k_angles = np.array([stroke_angle(i, x, y, k) for i in range(k, n - k - 1)])
    
    features[21] = np.sum(np.sin(k_angles) ** 2)
    features[22] = np.max(k_angles)
    
    alpha = [np.arccos(np.clip(np.dot(np.array([x[i + 1] - x[i], y[i + 1] - y[i]]), u_x), -1, 1)) for i in range(0, n - 1) if i + i not in penups]
    angles_bins = np.array([(math.pi / 4) * i for i in range(8)]) # 8 angles, 45 degress
    h_bins = [0 for i in range(8)] #bins for angles
    
    
    
    for angle in alpha:
        #angle ka bin
        
        bin = math.floor(angle/(math.pi / 4))
        
        #weight of each angle to each bin #NOT 100% sure of this
        w1 = 1 - (abs(angle - angles_bins[bin]) / (math.pi / 4))
        w2 = 1 - w1
        
        #adding weight to bin
        h_bins[bin] += w1
        h_bins[(bin + 1) % 8] += w2
    
    for i in range(4):
        features[23 + i] = (h_bins[i] + h_bins[i + 4]) / (n - len(penups))
        
        
    angles_bins = np.array([(math.pi / 4) * i for i in range(4)]) #4 div between 0 to 180
    gamma = 0.25
    gammas = [(gamma * angles[i]) + ((1 - gamma) * k_angles[i]) for i in range(0, abs(n - k - 1 - k))]
    bins = [0 for i in range(4)]
    
    SMALL_VAL = 0.0000001
    for angle in gammas:
        #angle ka bin
        bin = math.floor((angle/(math.pi / 4)) - SMALL_VAL)
        
        #weight of each angle to each bin #NOT 100% sure of this
        w1 = 1 - (abs(angle - angles_bins[bin]) / (math.pi / 4))
        w2 = 1 - w1
        
        #adding weight to bin
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
    centers = [[((center_x + start_x) + (bbox_w * i), center_y + start_y + (bbox_h * j)) for i in range(3)] for j in range(3)]
    bins = [[0 for i in range(3)] for j in range(3)]
    
    dists = [0 for i in range(9)]
    for i in range(n):
        point = x[i], y[i]
        count = 0
        for row in range(3):
            for col in range(3):
                box_num = row,col
                center = centers[row][col]
                dists[count] = (1/distance.euclidean(point, center), box_num)
                count += 1
        
        ##find closest 4 centers
        dists.sort(reverse = True)
        total_distance = sum([dists[i][0] for i in range(4)])
        weighted_dist = [dists[i][0]/total_distance for i in range(4)]
        
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
            gamma = 1 + ((p + q)/2)
            moments[(p, q)] = (m_pq) / (moments[(0,0)] ** gamma)
            
    features[40] = moments[(0, 2)] + moments[(2, 0)]
    features[41] = ((moments[(2, 0)] - moments[(0, 2)]) ** 2) + (4 * moments[(1, 1)]) ** 2
    features[42] = ((moments[(3, 0)] - 3 * moments[(1, 2)]) ** 2) + (3 * moments[2, 1] - moments[(0, 3)]) ** 2
    features[43] = ((moments[(3, 0)] + moments[(1, 2)]) ** 2) + (moments[(2, 1)] + moments[(0, 3)]) ** 2
    
    features[44] = ((moments[(3, 0)] - 3 * moments[(1,2)])**2) * (moments[(3,0)] + moments[(1,2)]) * (((moments[(3, 0)] + moments[(1, 2)]) ** 2) - 3*((moments[(2,1)] + moments[(0,3)])**2)) + (((3*moments[(2,1)]) - moments[(0, 3)])*(moments[(2,1)] + moments[(0, 3)]) * ((3*((moments[(3,0)] + moments[(1, 2)])**2) - ((moments[(2, 1)] + moments[(0, 3)])** 2))))
    
    features[45] = ((moments[(2, 0)] - moments[(0, 2)]) * (((moments[(3,0)] + moments[(1, 2)]) ** 2) - ((moments[(2,1)] + moments[(0, 3)]) ** 2))) + (4 * (moments[(1, 1)])*(moments[(3, 0)] + moments[(1, 2)]))
    
    
    features[46] = (((3 * moments[(2, 1)]) + moments[(0, 3)]) * (moments[(2, 1)] + moments[(0, 3)])) * (((moments[(2, 1)] + moments[(0, 3)]) ** 2)) - (((moments[(3, 0)] + moments[(1, 2)])*(moments[(2, 1)] + moments[(0, 3)])) * ((3*((moments[(3, 0)] + moments[(1, 2)])**2)) - (moments[(2, 1)] + moments[(0, 3)])**2))
    
    points = [(x[i], y[i]) for i in range(len(x))]
    from scipy.spatial import ConvexHull
    hull = ConvexHull(points)
    hull_vertices = hull.vertices
    a = 0
    b = 0
    for i in range(len(hull_vertices) - 1):
        a += x[i] * y[i + 1]
        b += -(x[i+1] * y[i])
    summation = a,b
        
    A_h = np.linalg.norm(np.array(summation)) / 2
    features[47] = A_h / (w * h)
    features[48] = (L ** 2) / A_h
    return features
    
        
        
        
    
    
    
        
        
    
    
    
    
    
    
    
     
    
    
    
    
    
    
    
def stroke_angle(i, x, y, k = 1):
    '''
    theta_i = refer to paper, eq . 14
    '''
    before = np.array([x[i - k], y[i - k]])
    current = np.array([x[i], y[i]])
    after = np.array([x[i + k], y[i + k]])


    dist_before_current = current - before

    dist_current_after = after - current

    if np.all(dist_before_current == 0) or np.all(dist_current_after == 0):
        return 0


    numerator = np.clip(np.dot(dist_before_current, dist_current_after) / (np.linalg.norm(dist_before_current, ord= 2) * np.linalg.norm(dist_current_after, ord = 2)), -1, 1)

    return np.arccos(numerator) #equation 14

                
                
                    
                
                
                    
            
                
            
            
        
        
def is_downstroke(x, y, start, end, t1 = 0, t2 = 0):
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
            #return as soon as the path length becomes greater than half. 
            return index - 1


from scipy.spatial import distance


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
    points_in_cell = {(i, j): [] for i in range(4) for j in range(3)}
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
        x_bin = min(int(x_dist // bbox_w), 3)
        y_bin = min(int(y_dist // bbox_h), 2)
        points_in_cell[(x_bin, y_bin)].append((x[i], y[i]))

    # going through each point in each cell and computing angle with next point
    gravity_centers = dict()
    for cell_x in range(4):
        for cell_y in range(3):
            if not points_in_cell[(cell_x, cell_y)]:
                continue

            center_gravity = np.mean([i[0] for i in points_in_cell[(cell_x, cell_y)]]), np.mean(
                [i[1] for i in points_in_cell[(cell_x, cell_y)]])
            gravity_centers[(cell_x, cell_y)] = center_gravity

            angle_bins = np.zeros(4)
            for num in range(len(points_in_cell[(cell_x, cell_y)]) - 1):
                point = points_in_cell[(cell_x, cell_y)][num]
                n_point = points_in_cell[(cell_x, cell_y)][num + 1]
                # arccos angle

                # print(n_point, point)
                angle = abs(np.arctan2((n_point[1] - point[1]), (n_point[0] - point[0])))
                # (np.clip(np.dot(n_point, point)/ np.linalg.norm(n_point) * np.linalg.norm(point), -1, 1))
                bin = int(min(angle // (math.pi / 4), 3))
                angle_bins[bin] += 1

            # now we have computed the bin for that cell
            # finding closest centers to the center of gravity
            closest_cells = sorted([(i, j) for i in range(4) for j in range(3)],
                                   key=lambda i: (np.linalg.norm(np.array(center_gravity) - centers[(i[0], i[1])], 1)))
            closest_cells = closest_cells[:4]  # keeping closest four
            # print(closest_cells, cell_x, cell_y)

            # sum of total distance (denominator of eq.1 in kaligo paper pg 15)
            total_sum = np.sum(
                [math.exp(-np.linalg.norm(np.array(center_gravity) - np.array(centers[(i, j)]), 1)) for i, j in
                 closest_cells])

            for center in closest_cells:
                i, j = center
                # distance of center from center of gravity
                dist_from_gravity = math.exp(-np.linalg.norm(np.array(center_gravity) - np.array(centers[(i, j)]), 1))

                # weight for that center
                w_ij = dist_from_gravity / total_sum

                # finallyyy adding the weighted orientation to our list of features
                cells[i, j] += w_ij * angle_bins

    return np.ravel(cells)



def get_chain_codes(x, y, dirs = 8):
    #takes a list of x and y coordinates and returns a list denoting the chaincodes
    to_divide = (np.pi / 4) if dirs == 8 else (np.pi / 2)
    x = np.array(x)
    y = np.array(y)
    delta_x = np.diff(x)
    delta_y = np.diff(y)
    thetas = np.atan2(delta_y, delta_x) #angles
    less_than_zero = thetas < 0 #finding negative angles
    thetas[less_than_zero] = thetas[less_than_zero] + (2 * np.pi)
    directions = np.clip(np.floor_divide(thetas, to_divide), 0, dirs - 1)
    return list(directions)


files_dir = ".\\files\\files\\"
from joblib import dump, load

import math


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



def get_mahalanobis_distance(whole_x, whole_y, penups, label):
    files_dir = os.path.realpath(os.path.join(os.path.join(__location__, 'files'), 'files'))
    scaled_x, scaled_y = scale_strokes(whole_x, whole_y, 128, 128)
    scaler = load(os.path.join(files_dir,f'cat_{label}_scaler.joblib'))
    hbr = hbr_feature_extract(scaled_x, scaled_y, penups)
    klg = kaligo_Hist_of_orientations(scaled_x, scaled_y, penups)
    all_models = np.concatenate([hbr, klg])
    feature_vector = [all_models]
    feature_vector = scaler.transform(feature_vector)[0]

    scores = np.load(os.path.join(files_dir, f'cat_{label}_score_dist.npy'))[:, 1]
    u = np.nan_to_num(feature_vector)
    v = np.load(os.path.join(files_dir, f'cat_{label}_mean_vector.npy'))
    cov = np.load(os.path.join(files_dir, f'cat_{label}_cov.npy'))
    dist = distance.mahalanobis(u=u, v=v, VI=cov)
    quants = [0.10 * i for i in range(1, 11)]

    for quant in quants:
        if dist < np.quantile(scores, quant):
            if dist < np.quantile(scores, quant - 0.05):
                return (1 - quant - 0.05)
            return (1 - quant)
    return 0 #the condition did not pass

s_x = [ 57.78285714,  44.61714286,  29.25714286,  16.82285714,   2.19428571, 0.,  6.58285714,  25.6, 32.18285714,  48.27428571, 65.09714286,  82.65142857,  97.28,       109.71428571, 116.29714286, 120.68571429, 125.80571429, 127.26857143, 125.07428571, 118.49142857, 109.71428571, 103.13142857,  89.23428571,  54.85714286,  39.49714286]
s_y = [0., 3.,   11., 22.5,  42.,   59.,   80.5, 109.5, 116.5, 126.,  127.5, 124., 118.5, 108.,   99.5,  91.,   72.,   55.5,  38.5,  25.5,  16.,   12.5,   8.5,   5., 1. ]
p = set([25])

if __name__ == "__main__":
    print(get_mahalanobis_distance(s_x, s_y, p, 0))
