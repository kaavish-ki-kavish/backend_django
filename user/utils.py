from django.contrib.auth import authenticate, get_user_model, login
from rest_framework import serializers
from .models import ChildProfile


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
