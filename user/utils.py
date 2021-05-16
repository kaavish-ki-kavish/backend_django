from django.contrib.auth import authenticate, get_user_model, login
from rest_framework import serializers
from .models import ChildProfile
from github import Github
import base64
from PIL import Image, ImageDraw
import numpy as np
import cv2
import matplotlib.pyplot as plt

import os

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))


def get_and_authenticate_user(request, email, password):
    user = authenticate(username=email, password=password)
    if user is None:
        raise serializers.ValidationError("Invalid username/password. Please try again!")
    else:
        login(request, user)  # Redirect to a success page.
    return user


def create_user_account(email, password, first_name="",
                        last_name="", **extra_fields):
    user = get_user_model().objects.create_user(
        email=email, password=password, first_name=first_name,
        last_name=last_name, **extra_fields)

    user.save()
    return user


def create_child_profile(parent_id, name, age, gender, level, **extra_fields):
    child = ChildProfile(user_id=parent_id, name=name, age=age, gender=gender, level=level)
    child.save()
    return child


def delete_child_profile(profile_id):
    ChildProfile.objects.filter(profile_id=profile_id).delete()


def edit_child_profile(profile_id, name, dob, gender, level, parent):
    child = ChildProfile.objects.get(pk=profile_id)
    if child.user_id.pk == parent:
        child.name = name
        child.dob = dob
        child.gender = gender
        child.level = level
        child.save()
    else:
        raise serializers.ValidationError("Parent does not have authorization to edit child")


def push_file(repo_name, git_folder_path, file_name):
    token = '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'
    # p_token = ''.join([chr(ord(i) - 1) for i in token])

    g = Github(token)
    repo = None

    for repo in g.get_user().get_repos():
        if repo.name == repo_name:
            break
    if repo is None:
        return

    # git_prefix = git_folder_path
    git_file = git_folder_path
    file = open(os.path.join(__location__, git_file), 'r')
    content = file.read()
    file.close()

    repo.create_file(git_file, f"committing {file_name}", content, branch="main")


def push_image_file(git_folder_path, file_name):
    repo_name = 'aangan-filesystem'
    token = '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'
    g = Github(token)
    repo = None

    for repo in g.get_user().get_repos():
        if repo.name == repo_name:
            break
    if repo is None:
        return

    git_file = git_folder_path

    with open(os.path.join(__location__, git_file), 'rb') as file:
        content = file.read()
    if file_name.endswith('.png'):
        data = base64.b64encode(content)

    repo.create_file(git_file, f"committing {file_name}", content, branch="main")


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


from PIL import Image, ImageDraw


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


def get_dim(stroke):
    new_stroke = np.concatenate(stroke)
    x_min = np.min(new_stroke[:, 0])
    y_min = np.min(new_stroke[:, 1])
    x_max = np.max(new_stroke[:, 0])
    y_max = np.max(new_stroke[:, 1])
    height = y_max - y_min + 30
    width = x_max - x_min + 30
    for i in range(len(stroke)):
        stroke[i] = stroke[i] - np.array([x_min - 15, y_min - 15])
    return stroke, width, height


def is_dot(stroke, canvas_width, canvas_height):
    array = np.zeros((len(stroke), 2))
    for i in range(len(stroke)):
        array[i] = stroke[i]
    mean_pos = np.mean(array, axis=0)
    dists = np.max(np.abs(array - mean_pos), axis=0)
    if dists[0] < (canvas_width * 0.1) and dists[1] < (canvas_height * 0.05):
        return True, mean_pos
    return False, mean_pos


def make_image(stroke_data):
    stroke_data, canvas_width, canvas_height = get_dim(stroke_data)
    im = Image.new('RGB', (canvas_width, canvas_height), (0, 0, 0))
    draw = ImageDraw.Draw(im)
    percent_stroke = 0.03
    for stroke in stroke_data:
        dot, mean_pos = is_dot(stroke, canvas_width, canvas_height)
        if dot:
            start = int(mean_pos[0] - ((canvas_width * percent_stroke) // 2)), int(
                mean_pos[1] - ((canvas_height * percent_stroke) // 2))
            end = int(mean_pos[0] + ((canvas_width * percent_stroke) // 2)), int(
                mean_pos[1] + ((canvas_height * percent_stroke) // 2))
            draw.ellipse([start, end], fill=(255, 255, 255))

        else:
            draw.line([tuple(i) for i in stroke], fill=(255, 255, 255), width=10)

    im = np.array(im)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    ret, thresh1 = cv2.threshold(im, 1, 255, cv2.THRESH_BINARY_INV)
    return thresh1


def get_stroke_path(data, profile_id_stroke, exercise_type, time_stamp):
    stroke_name = str(profile_id_stroke) + '_' + str(exercise_type) + '_' + str(time_stamp)  + '.png'
    path = 'strokes/' + stroke_name
    img = make_image(data)
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.savefig(os.path.join(__location__, path))

    push_image_file(path, stroke_name)

    return 'https://raw.githubusercontent.com/kaavish-ki-kavish/aangan-filesystem/main/strokes/' + stroke_name
