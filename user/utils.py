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

import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

import cv2

