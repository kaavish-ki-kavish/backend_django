from django.shortcuts import render

from django.contrib.auth import get_user_model, logout
from django.core.exceptions import ImproperlyConfigured
from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.permissions import AllowAny, IsAuthenticated
from django.http import HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404
from rest_framework.views import APIView
from django.db.models import Q
from django.utils import timezone
from django.db.models import Sum
from django.db.models import Count
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

    # THIS FUNCTION WAS ONLY FOR ME TO TOO AND SEE  DATA SO PLEASE DON'T USE IT
    # @action(methods=['GET'], detail=False, permission_classes=[IsAuthenticated, ])
    # def insert_data(self, request):
    #     from django.utils import timezone
    #     words = ['alif', 'ttaa', 'paa', 'seey', 'baa', 'taa', 'daal', 'zaal', 'dhaal', 'seen', 'sheen', 'zwaad','swaad']
    #
    #     for i in range(1,3):
    #         His = History(session_id=Session.objects.get(session_id=i),
    #                       stroke_score=random.random(),
    #                       stroke_path='/strokes/' + words[i] + '.txt',
    #                       time_taken=random.randint(1, 100),
    #                       datetime_attempt=timezone.now() - datetime.timedelta(days=random.randint(1, 7)),
    #                       similarity_score=random.random(),
    #                       character_id=None,
    #                       drawing_id=None,
    #                       coloring_id=None,
    #                       object_id=ObjectWord.objects.filter(object_id=i).latest('object_id'),
    #                       is_completed=True
    #
    #                       )
    #         His.save()
    #
    #         His = History(session_id=Session.objects.get(session_id=i),
    #                       stroke_score=random.random(),
    #                       stroke_path='/strokes/' + words[i] + '.txt',
    #                       time_taken=random.randint(1, 100),
    #                       datetime_attempt=timezone.now(),
    #                       similarity_score=random.random(),
    #                       character_id=None,
    #                       drawing_id=DrawingExercise.objects.get(drawing_id=70+i),
    #                       coloring_id=None,
    #                       object_id=None,
    #                       is_completed=False
    #
    #                       )
    #         His.save()
    #
    # return Response(
    #     data=serializers.HistorySerializer(History.objects.all(),
    #                                        many=True).data,
    #     status=status.HTTP_204_NO_CONTENT
    # )

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
        # return Response(
        #     data=serializers.SessionSerializer(
        #         Session.objects.filter(profile_id=(profile_id_stroke)), many=True).data,
        #     status=status.HTTP_204_NO_CONTENT
        # )

        profile_session_id = Session.objects.values_list('session_id', flat=True).filter(
            profile_id=profile_id_stroke).latest('session_id')
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
        profile_session_character_id = History.objects.values_list('character_id', flat=True).filter(
            session_id=profile_session_id).latest('character_id')

        character_sequence_id = Characters.objects.values_list('sequence_id', flat=True).filter(
            character_id=profile_session_character_id).latest('sequence_id')

        return Response(
            data=serializers.CharactersSerializer(
                Characters.objects.filter(sequence_id=character_sequence_id + 1)[:1], many=True).data,
            status=status.HTTP_204_NO_CONTENT
        )

    @action(methods=['POST'], detail=False, permission_classes=[IsAuthenticated, ])
    def time_on_exercise(self, request):

        profile_id_child = request.data.get('profile_id', None)
        days = request.data.get('days', None)
        child_all_sessions = Session.objects.filter(profile_id=profile_id_child).values_list('session_id', flat=True)

        graph_val_time_draw = []
        graph_val_score_draw = []

        graph_val_time_urdu = []
        graph_val_score_urdu = []
        for i in child_all_sessions:
            # for each session_id getting  time taken

            # DRAWING

            # for each session_id getting time taken
            date_time_ex_draw = History.objects.filter(session_id=i).filter(Q(drawing_id__isnull=False)).filter(
                datetime_attempt__range=(timezone.now() - datetime.timedelta(days=days), timezone.now())
            ).values('datetime_attempt').annotate(data_sum=Sum('time_taken'))

            # for each session_id getting sum stroke score
            date_score_ex_draw = History.objects.filter(session_id=i).filter(Q(drawing_id__isnull=False)).filter(
                datetime_attempt__range=(timezone.now() - datetime.timedelta(days=days), timezone.now())
            ).values('datetime_attempt').annotate(data_sum=Sum('stroke_score'))

            # URDU

            date_time_ex_urdu = History.objects.filter(session_id=i).filter(
                Q(character_id__isnull=False) | Q(object_id__isnull=False)).filter(
                datetime_attempt__range=(timezone.now() - datetime.timedelta(days=days), timezone.now())
            ).values('datetime_attempt').annotate(data_sum=Sum('time_taken'))

            # for each session_id getting sum stroke score
            date_score_ex_urdu = History.objects.filter(session_id=i).filter(
                Q(character_id__isnull=False) | Q(object_id__isnull=False)).filter(
                datetime_attempt__range=(timezone.now() - datetime.timedelta(days=days), timezone.now())
            ).values('datetime_attempt').annotate(data_sum=Sum('stroke_score'))

            draw_dates = list(date_time_ex_draw.values_list('datetime_attempt', flat=True))
            draw_time = list(date_time_ex_draw.values_list('data_sum', flat=True))
            draw_score = list(date_score_ex_draw.values_list('data_sum', flat=True))

            urdu_dates = list(date_time_ex_urdu.values_list('datetime_attempt', flat=True))
            urdu_time = list(date_time_ex_urdu.values_list('data_sum', flat=True))
            urdu_score = list(date_score_ex_urdu.values_list('data_sum', flat=True))

            for j in range(len(draw_dates)):
                # for draw
                graph_val_time_draw.append((draw_dates[j].strftime("%d-%B-%Y %H:%M:%S"), draw_time[j]))
                graph_val_score_draw.append((draw_dates[j].strftime("%d-%B-%Y %H:%M:%S"), draw_score[j]))

                # for urdu
                graph_val_time_urdu.append((urdu_dates[j].strftime("%d-%B-%Y %H:%M:%S"), urdu_time[j]))
                graph_val_score_urdu.append((urdu_dates[j].strftime("%d-%B-%Y %H:%M:%S"), urdu_score[j]))

        graph_val_time_draw.sort()
        graph_val_score_draw.sort()

        graph_val_time_urdu.sort()
        graph_val_score_urdu.sort()

        draw_dates = []
        draw_time = []
        draw_score = []
        for i in range(len(graph_val_time_draw)):
            draw_dates.append(graph_val_time_draw[i][0])
            draw_time.append(graph_val_time_draw[i][1])
            draw_score.append(graph_val_score_draw[i][1])

        urdu_dates = []
        urdu_time = []
        urdu_score = []
        for i in range(len(graph_val_time_urdu)):
            urdu_dates.append(graph_val_time_urdu[i][0])
            urdu_time.append(graph_val_time_urdu[i][1])
            urdu_score.append(graph_val_score_urdu[i][1])

        plt.plot(draw_dates, draw_time, label='Drawing')
        plt.plot(urdu_dates, urdu_time, label='Urdu')
        plt.legend()
        plt.title('Time Spent on Exercises')
        plt.xlabel('date')
        plt.ylabel('time spent')
        plt.savefig('temp.png')
        plt.clf()

        plt.plot(draw_dates, draw_score, label='Drawing')
        plt.plot(urdu_dates, urdu_score, label='Urdu')
        plt.title('Feedback Scores')
        plt.xlabel('date')
        plt.ylabel('score')
        plt.savefig('temp_score.png')

        return Response(
            data={'time_graph_path': 'temp.png',
                  'score_graph_path': 'temp_score.png'},
            status=status.HTTP_204_NO_CONTENT
        )

    @action(methods=['POST'], detail=False, permission_classes=[IsAuthenticated, ])
    def completion_exercise(self, request):
        drawing_completion = 0
        character_completion = 0
        profile_id_child = request.data.get('profile_id', None)
        child_all_sessions = Session.objects.filter(profile_id=profile_id_child).values_list('session_id', flat=True)
        for i in child_all_sessions:
            # drawing non null
            drawing_completion += History.objects.filter(session_id=i).filter(
                Q(drawing_id__isnull=False)).distinct().count()

            # character, object non null
            character_completion += History.objects.filter(session_id=i).filter(
                Q(character_id__isnull=False) | Q(object_id__isnull=False)).distinct().count()

        drawing_total = DrawingExercise.objects.all().count()
        urdu_total = Characters.objects.all().count() + ObjectWord.objects.all().count()
        return Response(
            data={'drawing': int((drawing_completion * 100) / drawing_total),
                  'character': int((character_completion * 100) / urdu_total)},
            status=status.HTTP_204_NO_CONTENT
        )

    @action(methods=['POST'], detail=False, permission_classes=[IsAuthenticated, ])
    def time_last_n_days(self, request):

        profile_id_child = request.data.get('profile_id', None)
        days = request.data.get('days', None)
        child_all_sessions = Session.objects.filter(profile_id=profile_id_child).values_list('session_id', flat=True)
        time_taken_sum = 0

        for i in child_all_sessions:
            # for each session_id getting last 7 days and sum of time taken

            date_time_query = History.objects.filter(session_id=i).filter(
                datetime_attempt__range=(timezone.now() - datetime.timedelta(days=days), timezone.now()))
            time_taken_sum += sum(date_time_query.values_list('time_taken', flat=True))

        return Response(
            data={'time_taken_sum': time_taken_sum},
            status=status.HTTP_204_NO_CONTENT
        )

    @action(methods=['POST'], detail=False, permission_classes=[IsAuthenticated, ])
    def average_time_exercise(self, request):
        profile_id_child = request.data.get('profile_id', None)
        child_all_sessions = Session.objects.filter(profile_id=profile_id_child).values_list('session_id', flat=True)
        total_time_taken = 0
        number_of_attempts = 0

        for i in child_all_sessions:
            filter_session = History.objects.filter(session_id=i)
            total_time_taken += sum(filter_session.values_list('time_taken', flat=True))
            number_of_attempts += filter_session.values('attempt_id').count()

        if number_of_attempts == 0:
            avg_time_taken = 0
        else:
            avg_time_taken = total_time_taken / number_of_attempts

        return Response(
            data={'avg_time_taken': int(avg_time_taken)},
            status=status.HTTP_204_NO_CONTENT
        )

    @action(methods=['POST'], detail=False, permission_classes=[IsAuthenticated, ])
    def average_score_exercise(self, request):
        profile_id_child = request.data.get('profile_id', None)
        days = request.data.get('days', None)
        child_all_sessions = Session.objects.filter(profile_id=profile_id_child).values_list('session_id', flat=True)
        score_sum = []

        for i in child_all_sessions:
            # for each session_id getting last 7 days and sum of time taken

            date_time_query = History.objects.filter(session_id=i).filter(
                datetime_attempt__range=(timezone.now() - datetime.timedelta(days=days), timezone.now()))
            score_sum.append(sum(date_time_query.values_list('stroke_score', flat=True)))

        print(score_sum)

        if len(score_sum) == 0:
            avg_score_sum = 0
        else:
            avg_score_sum = round((sum(score_sum) / len(score_sum)), 2) * 100

        return Response(
            data={'avg_score_sum': avg_score_sum},
            status=status.HTTP_204_NO_CONTENT
        )

    @action(methods=['POST'], detail=False, permission_classes=[IsAuthenticated, ])
    def exercises_last_n_days(self, request):
        profile_id_child = request.data.get('profile_id', None)
        days = request.data.get('days', None)
        child_all_sessions = Session.objects.filter(profile_id=profile_id_child).values_list('session_id', flat=True)
        completed_exercises = 0

        for i in child_all_sessions:
            filter_session = History.objects.filter(session_id=i).filter(is_completed=True).filter(
                datetime_attempt__range=(timezone.now() - datetime.timedelta(days=days), timezone.now()))
            completed_exercises += filter_session.values('character_id').distinct().count() + \
                                   filter_session.values('object_id').distinct().count() + \
                                   filter_session.values('drawing_id').distinct().count()

        return Response(
            data={'completed_exercises_sum': completed_exercises},
            status=status.HTTP_204_NO_CONTENT
        )

    @action(methods=['POST'], detail=False, permission_classes=[IsAuthenticated, ])
    def exercise_review_character(self, request):
        profile_id_child = request.data.get('profile_id', None)
        to_display_no = request.data.get('to_display_no', None)
        child_all_sessions = Session.objects.filter(profile_id=profile_id_child).values_list('session_id', flat=True)

        urdu_letter = History.objects.none()
        for i in child_all_sessions:
            urdu_letter = urdu_letter | History.objects.filter(session_id=i).filter(Q(character_id__isnull=False))

        lst_urdu_char = []
        for i in urdu_letter:
            lst_urdu_character_score = i.stroke_score
            lst_urdu_character_datetime = i.datetime_attempt
            lst_urdu_character_attempt = History.objects.filter(character_id=i.character_id).count()
            lst_urdu_character_timetaken = i.time_taken
            lst_urdu_character_image = \
                Characters.objects.filter(character_id=i.character_id.character_id).values_list('ref_stroke_path',
                                                                                                flat=True)[0]

            lst_urdu_char.append((lst_urdu_character_datetime, lst_urdu_character_image, lst_urdu_character_score,
                                  lst_urdu_character_timetaken, lst_urdu_character_attempt))

            lst_urdu_char = sorted(lst_urdu_char, reverse=True)

            if len(lst_urdu_char) > 0:
                if to_display_no < len(lst_urdu_char):
                    lst_urdu_char = lst_urdu_char[:to_display_no]

        return Response(
            data={'exercise_review': lst_urdu_char},
            status=status.HTTP_204_NO_CONTENT
        )

    @action(methods=['POST'], detail=False, permission_classes=[IsAuthenticated, ])
    def exercise_review_object(self, request):
        profile_id_child = request.data.get('profile_id', None)
        to_display_no = request.data.get('to_display_no', None)
        child_all_sessions = Session.objects.filter(profile_id=profile_id_child).values_list('session_id', flat=True)

        urdu_object = History.objects.none()
        for i in child_all_sessions:
            urdu_object = urdu_object | History.objects.filter(session_id=i).filter(Q(object_id__isnull=False))

        lst_urdu_object = []
        for i in urdu_object:
            lst_urdu_object_score = i.stroke_score
            lst_urdu_object_datetime = i.datetime_attempt
            lst_urdu_object_attempt = History.objects.filter(object_id=i.object_id).count()
            lst_urdu_object_timetaken = i.time_taken
            lst_urdu_object_image = \
                ObjectWord.objects.filter(object_id=i.object_id.object_id).values_list('image_path', flat=True)[0]

            lst_urdu_object.append((lst_urdu_object_datetime, lst_urdu_object_image, lst_urdu_object_score,
                                    lst_urdu_object_timetaken, lst_urdu_object_attempt))

            lst_urdu_char = sorted(lst_urdu_object, reverse=True)

            if len(lst_urdu_char) > 0:
                if to_display_no < len(lst_urdu_char):
                    lst_urdu_object = lst_urdu_char[:to_display_no]

        return Response(
            data={'exercise_review': lst_urdu_object},
            status=status.HTTP_204_NO_CONTENT
        )

    @action(methods=['POST'], detail=False, permission_classes=[IsAuthenticated, ])
    def exercise_review_drawing(self, request):
        profile_id_child = request.data.get('profile_id', None)
        to_display_no = request.data.get('to_display_no', None)
        child_all_sessions = Session.objects.filter(profile_id=profile_id_child).values_list('session_id', flat=True)

        drawing_ex = History.objects.none()
        for i in child_all_sessions:
            drawing_ex = drawing_ex | History.objects.filter(session_id=i).filter(Q(drawing_id__isnull=False))

        lst_drawing_ex = []
        for i in drawing_ex:
            lst_drawing_ex_score = i.stroke_score
            lst_drawing_ex_datetime = i.datetime_attempt
            lst_drawing_ex_attempt = History.objects.filter(drawing_id=i.drawing_id).count()
            lst_drawing_ex_timetaken = i.time_taken
            lst_drawing_ex_image = \
                DrawingExercise.objects.filter(drawing_id=i.drawing_id.drawing_id).values_list('ref_stroke_path',
                                                                                               flat=True)[0]

            lst_drawing_ex.append((lst_drawing_ex_datetime, lst_drawing_ex_image, lst_drawing_ex_score,
                                   lst_drawing_ex_timetaken, lst_drawing_ex_attempt))

            lst_drawing_ex = sorted(lst_drawing_ex, reverse=True)

            if len(lst_drawing_ex) > 0:
                if to_display_no < len(lst_drawing_ex):
                    lst_drawing_ex = lst_drawing_ex[:to_display_no]

        return Response(
            data={'exercise_review': lst_drawing_ex},
            status=status.HTTP_204_NO_CONTENT
        )
