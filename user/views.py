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
    edit_child_profile, get_whole_stroke, get_feature_vector, feature_scorer, perfect_scorer

from django.http import JsonResponse
from .classifier import RandomForestClassifier
from .feature_extractor import hbr_feature_extract, scale_strokes
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import cv2 as cv

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
        'session': serializers.HistorySerializer,
        'get_score': serializers.DataEntrySerializerTf,
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

    # @action(methods=['POST'], detail=False, permission_classes=[IsAuthenticated, ])
    # def get_prediction(self, request):
    #     r_m = RandomForestClassifier()
    #     return JsonResponse(r_m,safe=False)
    #     #return JsonResponse('prediction is'.format(r_m.compute_prediction(request.data)), safe=False)

    @action(methods=['POST'], detail=False, permission_classes=[IsAuthenticated, ])
    def get_prediction(self, request):
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

    # @action(methods=['GET'], detail=False, permission_classes=[IsAuthenticated, ])
    # def generate_excercise_letter_series(self, response):
    #     characters = ['alif', 'bay', 'pay']
    #     attempt = 1
    #     image_path = characters[0]+'.png'
    #     return Response(image_path,status=status.HTTP_204_NO_CONTENT)

    # @action(methods=['GET'], detail=False, permission_classes=[IsAuthenticated, ])
    # def generate_excercise_letter_series(self, response):
    #     number_of_records = Characters.objects.count()

    @action(methods=['POST'], detail=False, permission_classes=[IsAuthenticated, ])
    def save_stroke(self, request):
        x = request.data.get('x', None)
        y = request.data.get('y', None)
        stroke_name = request.data.get('s', None)

        # f = open('/users/stroke.txt', 'w')
        f = open(os.path.join(__location__, stroke_name + '.txt'), 'w')
        f.write('x' + '\n')
        for x_cord in x:
            f.write(str(x_cord) + '\n')
        f.write('y' + '\n')
        for y_cord in y:
            f.write(str(y_cord) + '\n')
        f.close()

        return Response(
            data=serializers.HistorySerializer(
                History.objects.all(), many=True).data,
            status=status.HTTP_204_NO_CONTENT)

    # @action(methods=['POST'], detail=False, permission_classes=[IsAuthenticated, ])
    # def insert_data(self, request):
    #     # a = ChildProfile.objects.get(profile_id=1)
    #     # s = Session(session_id=1, profile_id=a, time_start=datetime.datetime.now(), time_end=datetime.datetime.now(),
    #     #             token=1)
    #     # s.save()
    #
    #     #
    #     # h = History(attempt_id=1, session_id=Session.objects.get(session_id = 1), stroke_path='stroke.txt', time_taken=20, stroke_score=100,
    #     #             similarity_score=50, datetime_attempt=datetime.datetime.now(), character_id=5, drawing_id=3,
    #     #             coloring_id=6, object_id=6, is_completed=False)
    #     # h.save()
    #
    #     # a = ChildProfile.objects.filter(user_id=request.user).only('profile_id')
    #     # a = ChildProfile.objects.values.(profile_id)(user_id = request.user)
    #     #
    #     return Response(
    #         data=serializers.DrawingExerciseSerializer(DrawingExercise.objects.all(), many=True).data,
    #         status=status.HTTP_204_NO_CONTENT)



    @action(methods=['POST'], detail=False)
    def get_score(self, request):

        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data
        img = np.array(data['img'])
        scores = []

        if data['exercise'] == 0:  # drawing
            pass

        elif data['exercise'] == 1:  # urdu letters
            char = data['char']
            whole_x = data['whole_x']
            whole_y = data['whole_y']
            penup = data['pen_up']
            print('here0')
            p_features, s_features = get_feature_vector(char)
            print('here1')
            scores.append(feature_scorer(img, p_features, s_features, verbose=1))
            print('here2')
            scores.append(perfect_scorer(whole_x, whole_y, penup, char))

        response = {
            'scores': scores,
        }

        return Response(response)