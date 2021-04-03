from django.shortcuts import render

from django.contrib.auth import get_user_model, logout
from django.core.exceptions import ImproperlyConfigured
from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.permissions import AllowAny, IsAuthenticated
from django.http import HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404
from rest_framework.views import APIView
from django.db.models import Max, Min
import os, datetime, random, copy

from .models import ChildProfile, Characters, Session, History, ObjectWord, ColoringExercise, DrawingExercise

from rest_framework.response import Response
from . import serializers
from .utils import get_and_authenticate_user, create_user_account, create_child_profile, delete_child_profile, \
    edit_child_profile

from django.http import JsonResponse
from .classifier import RandomForestClassifier
from .feature_extractor import hbr_feature_extract, scale_strokes
from .urduCNN import UrduCnnScorer
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import cv2 as cv
import torch
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torch.utils import data
import torch.nn as nn

# from django.shortcuts import render
# from .apps import PredictorConfig
# from django.http import JsonResponse
# from rest_framework.views import APIView

User = get_user_model()

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))


class AuthViewSet(viewsets.GenericViewSet):
    permission_classes = [AllowAny, ]
    serializer_class = serializers.EmptySerializer
    serializer_classes = {
        'login': serializers.UserLoginSerializer,
        'register': serializers.UserRegisterSerializer,
        'password_change': serializers.PasswordChangeSerializer,
        'child_create': serializers.ChildRegisterSerializer,
        'get_all_child': serializers.ChildRegisterSerializer,
        'delete_child': serializers.DeleteChildSerializer,
        'get_most_recent_child': serializers.ChildRegisterSerializer,
        'edit_child': serializers.EditChildSerializer,
        'generate_character': serializers.Characters,
        'session': serializers.HistorySerializer
    }
    queryset = ''

    @action(methods=['POST', ], detail=False)
    def login(self, request):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        user = get_and_authenticate_user(request, **serializer.validated_data)
        data = serializers.AuthUserSerializer(user).data
        return Response(data=data, status=status.HTTP_200_OK)

    @action(methods=['POST', ], detail=False)
    def register(self, request):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        user = create_user_account(**serializer.validated_data)
        data = serializers.AuthUserSerializer(user).data
        return Response(data=data, status=status.HTTP_201_CREATED)

    @action(methods=['POST', ], detail=False)
    def logout(self, request):
        logout(request)
        request.session.flush()
        data = {'success': 'Sucessfully logged out'}
        return Response(data=data, status=status.HTTP_200_OK)

    @action(methods=['POST'], detail=False, permission_classes=[IsAuthenticated, ])
    def password_change(self, request):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        request.user.set_password(serializer.validated_data['new_password'])
        request.user.save()
        return Response(status=status.HTTP_204_NO_CONTENT)

    def get_serializer_class(self):
        if not isinstance(self.serializer_classes, dict):
            raise ImproperlyConfigured("serializer_classes should be a dict mapping.")

        if self.action in self.serializer_classes.keys():
            return self.serializer_classes[self.action]
        return super().get_serializer_class()

    @action(methods=['POST'], detail=False, permission_classes=[IsAuthenticated, ])
    def child_create(self, request):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        create_child_profile(request.user, **serializer.validated_data)
        return Response(
            data=serializers.ChildRegisterSerializer(ChildProfile.objects.filter(user_id=request.user), many=True).data,
            status=status.HTTP_204_NO_CONTENT)

    @action(methods=['GET'], detail=False, permission_classes=[IsAuthenticated, ])
    def get_all_child(self, request):
        return Response(
            data=serializers.ChildRegisterSerializer(ChildProfile.objects.filter(user_id=request.user), many=True).data,
            status=status.HTTP_204_NO_CONTENT)

    @action(methods=['POST'], detail=False, permission_classes=[IsAuthenticated, ])
    def delete_child(self, request):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        delete_child_profile(**serializer.validated_data)
        return Response(
            data=serializers.ChildRegisterSerializer(ChildProfile.objects.filter(user_id=request.user), many=True).data,
            status=status.HTTP_204_NO_CONTENT)

    @action(methods=['GET'], detail=False, permission_classes=[IsAuthenticated, ])
    def get_most_recent_child(self, request):
        return Response(data=serializers.ChildRegisterSerializer(
            ChildProfile.objects.filter(user_id=request.user).all().last()).data, status=status.HTTP_204_NO_CONTENT)

    @action(methods=['POST'], detail=False, permission_classes=[IsAuthenticated, ])
    def edit_child(self, request):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        edit_child_profile(**serializer.validated_data, parent=request.user.pk)
        return Response(status=status.HTTP_204_NO_CONTENT)

    @action(methods=['POST'], detail=False, permission_classes=[IsAuthenticated, ])
    def get_prediction_drawing(self, request):
        # pram = request.data.get('array_in',None)
        # r_m = RandomForestClassifier()
        # prediction = r_m.compute_prediction(pram)
        # response = {
        #     'message': 'Successful',
        #     'prediction': prediction,
        # }
        # return Response(response)

        whole_x = request.data.get('x', None)
        whole_y = request.data.get('y', None)
        scale_x, scale_y = scale_strokes(whole_x, whole_y, 128, 128)
        hbr = hbr_feature_extract(scale_x, scale_y, {len(whole_x)})
        hbr = [hbr]
        r_m = RandomForestClassifier()
        prediction = r_m.compute_prediction(hbr)
        response = {
            'message': 'Successful',
            'prediction': prediction,
        }
        return Response(response)

    @action(methods=['POST'], detail=False, permission_classes=[IsAuthenticated, ])
    def get_urdu_score(self, request):

        x = request.data.get('x', None)
        y = request.data.get('y', None)
        urdu_scorer = UrduCnnScorer(x, y)
        score = urdu_scorer.get_score()
        response = {
            'message': 'Successful',
            'prediction': score,
        }
        return Response(response)

    @action(methods=["GET"], detail=False)
    def get_random_exercise(self, request):
        number_of_records = DrawingExercise.objects.count()
        record_id = list(DrawingExercise.objects.all().aggregate(Min('drawing_id')).values())[0] + int(
            random.random() * number_of_records) + 1
        random_exercise = DrawingExercise.objects.get(pk=record_id)
        serializer = serializers.DrawingExerciseSerializer
        return Response(data=serializer(random_exercise).data, status=status.HTTP_204_NO_CONTENT)

    @action(methods=['POST'], detail=False, permission_classes=[IsAuthenticated, ])
    def save_stroke(self, request):
        x = request.data.get('x', None)
        y = request.data.get('y', None)
        stroke_name = request.data.get('s', None)
        profile_id_stroke = request.data.get('profile_id', None)
        path = '/strokes' + stroke_name + '.txt'
        f = open(os.path.join(__location__, path), 'w')
        f.write('x' + '\n')
        for x_cord in x:
            f.write(str(x_cord) + '\n')
        f.write('y' + '\n')
        for y_cord in y:
            f.write(str(y_cord) + '\n')
        f.close()

        # stroke_session_id = Session.objects.filter(profile_id=profile_id_stroke)
        stroke_session_id = Session.objects.values_list('session_id', flat=True).get(profile_id=profile_id_stroke)

        return Response(
            data=serializers.ColoringExerciseSerializer(ColoringExercise.objects.all(), many=True).data,
            status=status.HTTP_204_NO_CONTENT
        )

    #THIS FUNCTION WAS ONLY FOR ME TO TOO AND SEE  DATA SO PLEASE DON'T USE IT
    # @action(methods=['GET'], detail=False, permission_classes=[IsAuthenticated, ])
    # def insert_data(self, request):
    #     from django.utils import timezone
    #     words = ['alif', 'ttaa', 'paa', 'seey', 'baa', 'taa', 'daal', 'zaal', 'dhaal', 'seen', 'sheen', 'zwaad',
    #              'swaad']
    #     i = 3
    #     ch = Characters(
    #         character_id=32,
    #         level=1,
    #         ref_stroke_path='/strokes/' + words[i] + '.txt',
    #         ref_object_path='/strokes/' + words[i] + '.txt',
    #         label=words[i],
    #         sound_path='/strokes/' + words[i] + '.txt',
    #         sequence_id=i)
    #     ch.save()
    #     return Response(
    #         data=serializers.CharactersSerializer(
    #             Characters.objects.filter(character_id=32), many=True).data,
    #         status=status.HTTP_204_NO_CONTENT
    #     )
    #     for i in range(1,10):
    #         His = History(session_id=Session.objects.get(session_id=i),
    #                       stroke_score=random.random(),
    #                       stroke_path='/strokes/' + words[i] + '.txt',
    #                       time_taken=random.randint(1, 100),
    #                       datetime_attempt=timezone.now(),
    #                       similarity_score=random.random(),
    #                       character_id=Characters.objects.filter(character_id=i).latest('character_id'),
    #                       coloring_id=ColoringExercise.objects.get(coloring_id=1),
    #                       object_id=ObjectWord.objects.get(object_id=1),
    #                       is_completed=True,
    #                       drawing_id=DrawingExercise.objects.get(drawing_id=70 + i)
    #                       )
    #         His.save()
    #
    #     return Response(
    #         data=serializers.HistorySerializer(History.objects.all(),
    #                                               many=True).data,
    #         status=status.HTTP_204_NO_CONTENT
    #     )

    # from django.utils import timezone
    # import string
    # for i in range(10):
    #     print(i)
    #     s = Session(
    #         session_id=i,
    #         profile_id=ChildProfile.objects.get(profile_id=1),
    #         time_start=timezone.now(),
    #         time_end=timezone.now() + datetime.timedelta(seconds=random.randint(0, 86400)),
    #         token=''.join(random.choices(string.ascii_lowercase, k=5))
    #     )
    #     s.save()
    #

    # for i in range(2, 10):
    #     ch = Characters(
    #         character_id=i,
    #         level=1,
    #         ref_stroke_path='/strokes/' + words[i] + '.txt',
    #         ref_object_path='/strokes/' + words[i] + '.txt',
    #         label=words[i],
    #         sound_path='/strokes/' + words[i] + '.txt',
    #         sequence_id=i)
    #     ch.save()



    # o = ObjectWord(object_id=1, label='kursi', image_path='/strokes/' + 'stroke' + '.txt', is_object=True,
    #                ref_image_path='/strokes/' + 'stroke' + '.txt', category='animal',
    #                sound_path='/strokes/' + 'stroke' + '.txt')
    # o.save()
    #
    # co = ColoringExercise(
    #     coloring_id=1, ref_image_path='/strokes/' + 'stroke' + '.txt', level=1,
    #     sound_path='/strokes/' + 'stroke' + '.txt', label='ball')
    # co.save()
    #

    #
    #
    # return Response(
    #     data=serializers.HistorySerializer(History.objects.all(),
    #                                           many=True).data,
    #     status=status.HTTP_204_NO_CONTENT
    # )
    #
    # return Response(
    #     data=serializers.DrawingExerciseSerializer(DrawingExercise.objects.all(),
    #                                                many=True).data,
    #     status=status.HTTP_204_NO_CONTENT
    # )

    # Takes child profile id and gives queryset of generated drawing exercise
    @action(methods=['POST'], detail=False, permission_classes=[IsAuthenticated, ])
    def generate_drawing_exercise(self, request):
        profile_id_stroke = request.data.get('profile_id', None)
        profile_session_id = Session.objects.values_list('session_id', flat=True).get(profile_id=profile_id_stroke)
        profile_session_drawing_id = History.objects.values_list('drawing_id', flat=True).filter(
            session_id=profile_session_id).latest('attempt_id')
        # next exercise will be +1 of on which I am rn
        return Response(
            data=serializers.DrawingExerciseSerializer(
                DrawingExercise.objects.filter(drawing_id=(profile_session_drawing_id + 1)), many=True).data,
            status=status.HTTP_204_NO_CONTENT
        )

    @action(methods=['POST'], detail=False, permission_classes=[IsAuthenticated, ])
    def generate_character_exercise(self, request):

        # child profile_id to session session_id to History character_id to Character sequence_id

        profile_id_stroke = request.data.get('profile_id', None)
        profile_session_id = Session.objects.values_list('session_id', flat=True).filter(
            profile_id=profile_id_stroke).latest('session_id')
        profile_session_character_id = History.objects.values_list('character_id', flat=True).get(
            session_id=profile_session_id)

        character_sequence_id = Characters.objects.values_list('sequence_id', flat=True).filter(
            character_id=profile_session_character_id).latest('sequence_id')

        return Response(
            data=serializers.CharactersSerializer(
                Characters.objects.filter(sequence_id=character_sequence_id+1)[:1], many=True).data,
            status=status.HTTP_204_NO_CONTENT
        )

