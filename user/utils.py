from django.contrib.auth import authenticate, get_user_model, login
from rest_framework import serializers
from .models import ChildProfile
import numpy as np
import os
import cv2

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))

def get_and_authenticate_user(request, email, password):
    user = authenticate(username=email, password=password)
    if user is None:
        raise serializers.ValidationError("Invalid username/password. Please try again!")
    else:
        login(request, user) # Redirect to a success page.
    return user

def create_user_account(email, password, first_name="",
                        last_name="", **extra_fields):

    user = get_user_model().objects.create_user(
        email=email, password=password, first_name=first_name,
        last_name=last_name, **extra_fields)

    user.save()
    return user

def create_child_profile(parent_id, name, dob, gender, level, **extra_fields):
    child = ChildProfile(user_id = parent_id, name = name, dob = dob, gender = gender, level = level)
    child.save()
    return child

def delete_child_profile(profile_id ):
    ChildProfile.objects.filter(profile_id=profile_id).delete()

def edit_child_profile(profile_id,name,dob,gender,level, parent):
    child = ChildProfile.objects.get(pk=profile_id)
    if child.user_id.pk == parent:
        child.name = name
        child.dob = dob
        child.gender = gender
        child.level = level
        child.save()
    else:
        raise serializers.ValidationError("Parent does not have authorization to edit child")

def get_whole_stroke(drawing):
    whole_x = []
    whole_y = []
    penup = set()  # points at which there is a penup
    for stroke in drawing:
        for x, y in stroke:
            whole_x += [x]
            whole_y += [y]
        penup.add(len(whole_x))  # appending the index for penup

    return whole_x, whole_y, penup


def get_feature_vector(t_char):
    all_chars = ['alif', 'alif mad aa', 'ayn', 'baa', 'bari yaa', 'cheey', 'choti yaa', 'daal', 'dhaal', 'faa',
                          'gaaf', 'ghain', 'haa1', 'haa2', 'haa3', 'hamza', 'jeem', 'kaaf', 'khaa', 'laam', 'meem',
                          'noon', 'noonghunna', 'paa', 'qaaf', 'raa', 'rhraa', 'seen', 'seey', 'sheen', 'swaad', 'taa',
                          'ttaa', 'twa', 'waw', 'zaaa', 'zaal', 'zhaa', 'zwaa', 'zwaad']

    secondary_features = ['1_nuqta', '2_nuqta', '3_nuqta', 'ttaa', 'top', 'bottom', 'middle', 'kaaf', 'gaaf',
                              '3_nuqta_invert']
    primary_features = {'endVerticalDown': 0, 'sharpEdge': 1, 'daal': 2, 'downIntersection': 3, 'chashmiHay': 4,
                        'startLoopDown': 5, 'endChotiYay': 6, 'ray': 7, 'jeem': 8, 'seen': 9, 'longHorR2L': 10,
                        'goolHay': 11, 'semiCircleU2D': 12, 'endVerticalUp': 13, 'startAien': 14, 'startSwaad': 15,
                        'endSemiCircle': 16, 'longHorL2R': 17, 'semiCircleR2L': 18, 'alif': 19, 'startLoopUp': 20,
                        'startUpDownVertical': 21}

    cluster_features = {}
    cluster_features['alif'] = ['startUpDownVertical', 'endVerticalDown', 'alif']
    cluster_features['bey'] = ['startUpDownVertical', 'endVerticalUp', 'longHorR2L']
    cluster_features['jeem'] = ['longHorL2R', 'semiCircleU2D', 'jeem']
    cluster_features['daal'] = ['daal']
    cluster_features['ray'] = ['ray', 'startUpDownVertical', 'sharpEdge']
    cluster_features['seen'] = ['endSemiCircle', 'sharpEdge', 'seen']
    cluster_features['swad'] = ['endSemiCircle', 'startSwaad']
    cluster_features['twa'] = ['startUpDownVertical', 'downIntersection']
    cluster_features['ayn'] = ['startAien', 'semiCircleU2D']
    cluster_features['faa'] = ['longHorL2R', 'startLoopUp', 'endVerticalUp']
    cluster_features['qaaf'] = ['startLoopUp', 'endSemiCircle']
    cluster_features['kaaf'] = ['startUpDownVertical', 'longHorR2L', 'endVerticalUp']
    cluster_features['laam'] = ['startUpDownVertical', 'semiCircleR2L']
    cluster_features['noon'] = ['semiCircleR2L']
    cluster_features['meem'] = ['endVerticalDown', 'startLoopDown']
    cluster_features['waw'] = ['startLoopUp']
    cluster_features['gool-hay'] = ['goolHay']
    cluster_features['chashmi-ha'] = ['chashmiHay']
    cluster_features['choti-yay'] = ['endChotiYay', 'endSemiCircle']
    cluster_features['bari-yay'] = ['longHorL2R']

    clusters = {}
    clusters['alif'] = ['alif', 'alif mad aa']
    clusters['bey'] = ['ttaa', 'paa', 'seey', 'baa', 'taa']
    clusters['jeem'] = ['khaa', 'jeem', 'haa1', 'cheey']
    clusters['daal'] = ['daal', 'zaal', 'dhaal']
    clusters['ray'] = ['rhraa', 'raa', 'zhaa', 'zaaa']
    clusters['seen'] = ['seen', 'sheen']
    clusters['swad'] = ['zwaad', 'swaad']
    clusters['twa'] = ['twa', 'zwaa', 'Twaa']
    clusters['ayn'] = ['ayn', 'ghain']
    clusters['faa'] = ['faa']
    clusters['qaaf'] = ['qaaf']
    clusters['kaaf'] = ['gaaf', 'kaaf']
    clusters['laam'] = ['laam']
    clusters['meem'] = ['meem']
    clusters['noon'] = ['noon', 'noonghunna']
    clusters['waw'] = ['waw']
    clusters['gool-hay'] = ['haa3']
    clusters['chashmi-ha'] = ['haa2']
    clusters['choti-yay'] = ['choti-yaa']
    clusters['bari-yay'] = ['bari-yaa']

    s_features = {}

    s_features['baa'] = ['1_nuqta', 'bottom']
    s_features['paa'] = ['3_nuqta', 'bottom']
    s_features['taa'] = ['2_nuqta', 'top']
    s_features['ttaa'] = ['ttaa', 'top']
    s_features['seey'] = ['3_nuqta', 'top', '3_nuqta_invert']
    s_features['jeem'] = ['1_nuqta', 'middle']
    s_features['khaa'] = ['1_nuqta', 'top']
    s_features['cheey'] = ['3_nuqta', 'middle', '3_nuqta_invert']
    s_features['zaal'] = ['1_nuqta', 'top']
    s_features['dhaal'] = ['ttaa', 'top']
    s_features['rhraa'] = ['ttaa', 'top']
    s_features['zhaa'] = ['3_nuqta', 'top', '3_nuqta_invert']
    s_features['zaaa'] = ['1_nuqta', 'top']
    s_features['sheen'] = ['3_nuqta', 'top', '3_nuqta_invert']
    s_features['zwaad'] = ['1_nuqta', 'top']
    s_features['zwaa'] = ['1_nuqta', 'top']
    s_features['ghain'] = ['1_nuqta', 'top']
    s_features['faa'] = ['1_nuqta', 'top']
    s_features['qaaf'] = ['2_nuqta', 'top']
    s_features['kaaf'] = ['kaaf']
    s_features['gaaf'] = ['gaaf']
    s_features['noon'] = ['1_nuqta', 'middle']

    key_cluster_map = {}
    for char in all_chars:
        for cluster in clusters.keys():
            if char in clusters[cluster]:
                key_cluster_map[char] = cluster

    p_features = np.zeros(len(primary_features))
    s_features_vec = np.zeros(len(secondary_features))

    cluster = key_cluster_map[t_char]
    rel_features = cluster_features[cluster]
    for i in rel_features:
        p_features[primary_features[i]] = 1

    if t_char in s_features:
        s_feature_set = [secondary_features.index(i) for i in s_features[t_char]]
        s_features_vec[s_feature_set] = 1
    else:
        s_features[t_char] = []


    print(f'for char = {t_char} the relevant primary features were {rel_features}')
    print(f'for char = {t_char} the relevant secondary features were {s_features[t_char]}')
    return p_features, s_features_vec


import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import SGD, Adam
from PIL import Image, ImageDraw

def get_p_model():
    primary_model_weights = os.path.join(__location__, 'primary_model')
    primary_model = Sequential()
    primary_model.add(Conv2D(128, (2, 2), padding='same', activation='relu', input_shape=(28, 28, 1)))
    primary_model.add(MaxPool2D(pool_size=(2, 2)))
    primary_model.add(Dropout(0.5))
    primary_model.add(Conv2D(64, (2, 2), activation='relu'))
    primary_model.add(MaxPool2D(pool_size=(2, 2)))
    primary_model.add(Dropout(0.25))
    primary_model.add(Conv2D(32, (2, 2), activation='relu'))
    primary_model.add(MaxPool2D(pool_size=(2, 2)))
    primary_model.add(Flatten())
    primary_model.add(Dense(1024, activation='relu'))
    primary_model.add(Dropout(0.25))
    primary_model.add(Dense(22, activation='sigmoid'))
    sgd = SGD(lr=0.01, momentum=0.9)
    primary_model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=[tf.keras.metrics.binary_accuracy])
    primary_model.load_weights(primary_model_weights)
    return primary_model

def get_s_model():
    secondary_model_weights = os.path.join(__location__, 'secondary_model')
    secondary_model = Sequential()
    secondary_model.add(Conv2D(128, (2, 2), padding='same', activation='relu', input_shape=(28, 28, 1)))
    secondary_model.add(MaxPool2D(pool_size=(2, 2)))
    secondary_model.add(Dropout(0.5))
    secondary_model.add(Conv2D(64, (2, 2), activation='relu'))
    secondary_model.add(MaxPool2D(pool_size=(2, 2)))
    secondary_model.add(Dropout(0.25))
    secondary_model.add(Conv2D(32, (2, 2), activation='relu'))
    secondary_model.add(MaxPool2D(pool_size=(2, 2)))
    secondary_model.add(Flatten())
    secondary_model.add(Dense(1024, activation='relu'))
    secondary_model.add(Dropout(0.25))
    secondary_model.add(Dense(10, activation='sigmoid'))

    sgd = SGD(lr=0.01, momentum=0.9)
    secondary_model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=[tf.keras.metrics.binary_accuracy])
    secondary_model.load_weights(secondary_model_weights)
    return secondary_model

def crop_image(array):
    ret3, img = cv2.threshold(array, 10, 255, cv2.THRESH_BINARY)  # +cv.THRESH_OTSU)
    img = ~img

    # unique_elements, counts_elements = np.unique(img, return_counts=True)

    desired_size = 256
    # h, w = img.shape

    y, x = np.where(img == 0)
    y_min, y_max = y.min(), y.max()
    x_min, x_max = x.min(), x.max()

    img = img[y_min: y_max, x_min: x_max]

    old_size = img.shape[:2]  # old_size is in (height, width) format

    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])

    # new_size should be in (width, height) format
    img = cv2.resize(img, (new_size[1], new_size[0]))

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    color = [255, 0, 0]
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT,
                            value=color)

    return img

import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

def feature_scorer(test_image, correct_p, correct_s, w_p = 0.7, w_s = 0.3, verbose = 0):
    model_p = get_p_model()
    model_s = get_s_model()



    img = crop_image(test_image)



    img = 255 - img
    img1 = cv2.GaussianBlur(img, (3, 3), 0)
    img1 = cv2.resize(img1, (28, 28), cv2.INTER_AREA)
    ret3, img1 = cv2.threshold(img1, 0, 255, cv2.THRESH_BINARY)  # +cv.THRESH_OTSU)
    unique_elements, counts_elements = np.unique(img1, return_counts=True)
    
    ret3, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)  # +cv.THRESH_OTSU)
    unique_elements, counts_elements = np.unique(img, return_counts=True)
    # print(sorted(list(zip(unique_elements, counts_elements))))

    img = cv2.resize(img, (28, 28), cv2.INTER_AREA)
    ret3, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)


    '''
    data_transform_main = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))])

    data_transform_custom = transforms.Compose([transforms.ToPILImage(),
                                                transforms.Resize((28, 28)),
                                                data_transform_main])

    crop_img = data_transform_custom(img.astype(np.float32))

    
    test_image = np.array(crop_img.view(1, 1, 28, 28))
    '''
    test_image = np.reshape(img, (28, 28, 1))

    plt.imsave('final_image_to_feature.png', np.reshape(test_image, (28, 28)))

    p_features_prob = model_p.predict(np.array([test_image])).flatten()
    s_features_prob = model_s.predict(np.array([test_image])).flatten()



    correct_p_probs = []
    incorrect_p_probs = []

    for i in range(len(correct_p)):
        if correct_p[i] == 1:
          correct_p_probs.append(p_features_prob[i])
        else:
          incorrect_p_probs.append(p_features_prob[i])

    correct_p_score = np.mean(correct_p_probs)
    incorrect_p_score = np.mean(incorrect_p_probs)
    p_score = correct_p_score - incorrect_p_score

    correct_s_probs = []
    incorrect_s_probs = []

    for i in range(len(correct_s)):
        if correct_s[i] == 1:
          correct_s_probs.append(s_features_prob[i])
        else:
          incorrect_s_probs.append(s_features_prob[i])

    correct_s_score = np.mean(correct_s_probs) if correct_s_probs else 1
    incorrect_s_score = np.mean(incorrect_s_probs)
    s_score = correct_s_score - incorrect_s_score

    if verbose:
        print(f'p score = {p_score}')
        print(f'correct_p_score = {correct_p_score}')
        print(f'incorrect_p_score = {incorrect_p_score}')
        print(f's score = {s_score}')
        print(f'correct_s_score = {correct_s_score}')
        print(f'incorrect_s_score = {incorrect_s_score}')


    return (p_score * w_p) + (s_score * w_s)

import cv2

def get_perfect_model():
    perf_model = os.path.join(__location__, 'perfect_model')
    model = Sequential()
    model.add(Conv2D(128, (4, 4), padding='same', activation='relu', input_shape=(100, 100, 1)))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Conv2D(64, (2, 2), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(32, (2, 2), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(17, activation='sigmoid'))

    sgd = SGD(lr=0.01, momentum=0.9)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=[tf.keras.metrics.binary_accuracy])
    model.load_weights(perf_model)
    return model

def perfect_scorer(x, y, penup, char):
    labels = ['jeem', 'gaaf', 'zwad', 'kaaf', 'laam', 'hay', 'seen', 'sheen', 'swad', 'khay', 'chay', 'say', 'alif', 'baa', 'tay', 'noon', 'pay']
    x_0, y_0 = min(x), min(y)
    x_n, y_n = max(x), max(y)
    w = x_n - x_0 + 1
    h = y_n - y_0 + 1
    padding = 10
    x_origin = [x_cord + padding // 2 - x_0 for x_cord in x]
    y_origin = [y_cord + padding // 2 - y_0 for y_cord in y]
    im = Image.new('RGB', (w + padding, h + padding), (0, 0, 0))
    draw = ImageDraw.Draw(im)

    x_y = []
    for i in range(len(x_origin) - 1):
        if (i + 1) not in penup:
            x_y.append((x_origin[i], y_origin[i]))
        else:
            draw.line(x_y, fill = (255, 255, 255), width= 7)
            x_y = []


    draw.line(x_y, fill=(255, 255, 255), width= 7)


    im = np.array(im)
    # plt.imshow(img_array[:, :, 0])
    # plt.axis('off')
    # plt.show()
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    plt.imsave('perfect_image.png', im)

    ret, thresh1 = cv2.threshold(im, 127, 255, cv2.THRESH_BINARY_INV)
    thresh1 = ~thresh1

    kernel = np.ones((2, 2), np.uint8)
    dilation = cv2.dilate(thresh1, kernel, iterations=1)
    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print(contours)
    if len(contours) == 1:
        cnt = contours[0]
    else:
        cnt = np.concatenate(contours)

    img = cv2.cvtColor(dilation, cv2.COLOR_GRAY2BGR)
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 10)

    img = dilation[y:y + h, x:x + w]

    height, width = img.shape[:2]
    max_height = 70
    max_width = 70

    print(img.shape)

    # only shrink if img is bigger than required
    if max_height < height or max_width < width:
        # get scaling factor
        scaling_factor = max_height / float(height)
        if max_width / float(width) < scaling_factor:
            scaling_factor = max_width / float(width)
        # resize image
        img = cv2.resize(img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)



    blank_image = np.zeros((100, 100), np.uint8)
    height, width = img.shape[:2]
    print(img.shape)
    y_offset = 50 - (height // 2)
    x_offset = 50 - (width // 2)
    blank_image[y_offset:y_offset + img.shape[0], x_offset:x_offset + img.shape[1]] = img

    img = blank_image
    plt.imsave('final_image.png', blank_image)

    img = np.reshape(img, (1, 100, 100, 1))
    model = get_perfect_model()
    pred_labels = model.predict(img)[0]

    return pred_labels[labels.index(char)]
