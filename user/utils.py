from django.contrib.auth import authenticate, get_user_model, login
from rest_framework import serializers
from .models import ChildProfile
from github import Github
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


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


def create_child_profile(parent_id, name, dob, gender, level, **extra_fields):
    child = ChildProfile(user_id=parent_id, name=name, dob=dob, gender=gender, level=level)
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


def push_file(file_name, repo_name, git_folder):
    # token = '76148974bd3158362e:5e3e72fe28d385c632g4d'
    # p_token = ''.join([chr(ord(i) - 1) for i in token])
    # g = Github(p_token)
    # repo = None
    #
    # for repo in g.get_user().get_repos():
    #     if repo.name == repo_name:
    #         break
    # if repo is None:
    #     return
    #
    # # with open(file_name, 'r') as file:
    # #     content = file.read
    # from PIL import Image
    # content = Image.open(file_name)
    #
    # # Output Images
    # content.show()
    #
    #
    # # Upload to github
    # git_prefix = 'dashboard/'+git_folder + '/'
    # git_file = git_prefix + file_name
    # repo.create_file(git_file, f"committing {file_name}", content, branch="main")
    # print(git_file + ' CREATED')

    from git import Repo

    repo_dir = repo_name
    repo = Repo(repo_name)
    file_path = 'dashboard/' + git_folder + '/' + file_name
    file_list = [file_path]
    commit_message = 'committing' + file_name
    repo.index.add(file_list)
    repo.index.commit(commit_message)
    origin = repo.remote('origin')
    origin.push()
