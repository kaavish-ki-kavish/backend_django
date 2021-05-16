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
from django.db.models import Sum, Count, Max, Min, F
import os, datetime, random, copy

from .models import ChildProfile, Characters, History, ObjectWord, ColoringExercise, DrawingExercise, Clusters, \
    ClusterFeature, Features, AttemptFeatures, Dashoard, WordsUrdu

from rest_framework.response import Response
from . import serializers
from .utils import get_and_authenticate_user, create_user_account, create_child_profile, delete_child_profile, \
    edit_child_profile, push_file, push_image_file, get_whole_stroke, get_stroke_path

from .classifier import RandomForestClassifier
from .feature_extractor import hbr_feature_extract, scale_strokes
from .urduCNN import UrduCnnScorer
import numpy as np
import matplotlib.pyplot as plt
import sys
import requests
import time

# from django.shortcuts import render
# from .apps import PredictorConfig
# from django.http import JsonResponse
# from rest_framework.views import APIView

User = get_user_model()

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))


def feature_model_pseudo(file_path):
    return np.random.rand(1, 28)[0]


def attempt_score(char, data, exercise):
    stroke_score = random.randint(0, 100)
    similarity_score = random.randint(0, 100)
    return stroke_score, similarity_score


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
        'get_score': serializers.DataEntrySerializer,
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
        return Response(status=status.HTTP_200_OK)

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
            status=status.HTTP_200_OK)

    @action(methods=['GET'], detail=False, permission_classes=[IsAuthenticated, ])
    def get_all_child(self, request):
        return Response(
            data=serializers.ChildRegisterSerializer(ChildProfile.objects.filter(user_id=request.user), many=True).data,
            status=status.HTTP_200_OK)

    @action(methods=['POST'], detail=False, permission_classes=[IsAuthenticated, ])
    def delete_child(self, request):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        delete_child_profile(**serializer.validated_data)
        return Response(
            data=serializers.ChildRegisterSerializer(ChildProfile.objects.filter(user_id=request.user), many=True).data,
            status=status.HTTP_200_OK)

    @action(methods=['GET'], detail=False, permission_classes=[IsAuthenticated, ])
    def get_most_recent_child(self, request):
        return Response(data=serializers.ChildRegisterSerializer(
            ChildProfile.objects.filter(user_id=request.user).all().last()).data, status=status.HTTP_200_OK)

    @action(methods=['POST'], detail=False, permission_classes=[IsAuthenticated, ])
    def edit_child(self, request):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        edit_child_profile(**serializer.validated_data, parent=request.user.pk)
        return Response(status=status.HTTP_200_OK)

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
        return Response(data=serializer(random_exercise).data, status=status.HTTP_200_OK)

    @action(methods=['POST'], detail=False, permission_classes=[IsAuthenticated, ])
    def save_stroke(self, request):
        x = request.data.get('x', None)
        y = request.data.get('y', None)
        profile_id_stroke = request.data.get('profile_id', None)

        stroke_session_id = Session.objects.values_list('session_id', flat=True).filter(
            profile_id=profile_id_stroke).latest('session_id')
        stroke_attempt_id = History.objects.values_list('attempt_id', flat=True).filter(
            session_id=stroke_session_id).latest('attempt_id')

        stroke_name = str(profile_id_stroke) + '_' + str(stroke_attempt_id) + '.txt'
        path = 'strokes/' + stroke_name

        f = open(os.path.join(__location__, path), 'w')
        f.write('x' + '\n')
        for x_cord in x:
            f.write(str(x_cord) + '\n')
        f.write('y' + '\n')
        for y_cord in y:
            f.write(str(y_cord) + '\n')
        f.close()

        push_file('aangan-filesystem', path, stroke_name)

        return Response(
            data={
                'stroke_path': 'https://raw.githubusercontent.com/kaavish-ki-kavish/aangan-filesystem/main/strokes/' + stroke_name},
            status=status.HTTP_200_OK
        )

    @action(methods=['POST'], detail=False)
    def get_score(self, request):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data
        scores = []
        whole_x, whole_y, penup = get_whole_stroke(data['data'])
        char = data['char']

        url = 'http://aangantf.herokuapp.com/api/auth/get_score'

        msg = 'Successful'

        if data['exercise'] == 0:  # drawing
            tf_data = {
                'exercise': 0,
                'char': char,
                'img': [[0, 0]],
                'whole_x': whole_x,
                'whole_y': whole_y,
                'pen_up': list(penup),
                'data': data['data']
            }
            response = requests.post(url, json=tf_data)
            print(response)
            print(type(response))
            sys.stdout.flush()
            print(response.json()['scores'])
            sys.stdout.flush()
            a = response.json()['scores'][0]  # feature_scorer(img, p_features,s_features, verbose= 1)
            b = response.json()['scores'][1]  # perfect_scorer(whole_x, whole_y, penup, char)
            scores.append(a)
            scores.append(b)



        elif data['exercise'] == 1:  # urdu letters
            scorer = UrduCnnScorer(whole_x, whole_y, penup)
            label = scorer.NUM2LABEL.index(char)
            img = scorer.preprocessing()
            print(img.shape)
            # scores.append(scorer.test_img(img)[0, label])

            tf_data = {
                'exercise': 1,
                'char': char,
                'img': img.tolist(),
                'whole_x': whole_x,
                'whole_y': whole_y,
                'pen_up': list(penup),
                'data': data['data']
            }

            response = requests.post(url, json=tf_data)
            print(response)
            print(type(response))
            sys.stdout.flush()
            print(response.json()['scores'])
            sys.stdout.flush()
            # p_features, s_features = get_feature_vector(char)
            a = response.json()['scores'][0]  # feature_scorer(img, p_features,s_features, verbose= 1)
            b = response.json()['scores'][1]  # perfect_scorer(whole_x, whole_y, penup, char)
            scores.append(a)
            scores.append(b)
        elif data['exercise'] == 2:
            
            tf_data = {
                'exercise': 2,
                'char': char,
                'img': np.zeros((28, 28)).tolist(),
                'whole_x': whole_x,
                'whole_y': whole_y,
                'pen_up': list(penup),
                'data': data['data']
            }

            response = requests.post(url, json=tf_data)
            a = response.json()['scores'][0]
            scores.append(a)

            
        print(scores)
        response = {
            'message': msg,
            'prediction': np.mean(scores),
        }
        return Response(response)

    @action(methods=['POST'], detail=False, permission_classes=[IsAuthenticated, ])
    def generate_drawing_exercise(self, request):
        """
        Takes child profile id and gives queryset of generated drawing exercise
        """
        profile_id = request.data.get('profile_id', None)
        total_drawing_exercises = DrawingExercise.objects.all().count()

        profile_drawing_hist = History.objects.filter(profile_id=profile_id).filter(drawing_id__isnull=False)
        if profile_drawing_hist:
            count_is_completed = profile_drawing_hist.filter(is_completed=True).count()
            if count_is_completed < total_drawing_exercises:
                latest_drawing_id = profile_drawing_hist.latest('drawing_id').drawing_id.drawing_id
                next_exercise = latest_drawing_id + 1
            else:
                min_ordered_drawing = profile_drawing_hist.filter(drawing_id__isnull=False).annotate(
                    total_score=F('stroke_score') + F('similarity_score')
                ).order_by('total_score')
                next_exercise = min_ordered_drawing[0].drawing_id.drawing_id

        else:
            next_exercise = DrawingExercise.objects.all().first().drawing_id

        return Response(
            data=serializers.DrawingExerciseSerializer(
                DrawingExercise.objects.filter(drawing_id=next_exercise), many=True).data,
            status=status.HTTP_200_OK
        )

    @action(methods=['POST'], detail=False, permission_classes=[IsAuthenticated, ])
    def generate_urdu_words_exercise(self, request):
        """
        Takes child profile id and gives queryset of generated drawing exercise
        """
        profile_id = request.data.get('profile_id', None)
        total_word_exercises = WordsUrdu.objects.all().count()

        profile_words_hist = History.objects.filter(profile_id=profile_id).filter(word_id__isnull=False)
        if profile_words_hist:
            count_is_completed = profile_words_hist.filter(is_completed=True).count()
            if count_is_completed < total_word_exercises:
                latest_word_id = profile_words_hist.latest('word_id').word_id.word_id
                next_exercise = latest_word_id + 1
            else:
                min_ordered_words = profile_words_hist.annotate(
                    total_score=F('stroke_score') + F('similarity_score')
                ).order_by('total_score')
                next_exercise = min_ordered_words[0].word_id.word_id

        else:
            next_exercise = WordsUrdu.objects.all().first().word_id

        return Response(
            data=serializers.WordsUrduSerializer(
                WordsUrdu.objects.filter(word_id=next_exercise), many=True).data,
            status=status.HTTP_200_OK
        )

    @action(methods=['POST'], detail=False, permission_classes=[IsAuthenticated, ])
    def generate_character_exercise(self, request):
        """
        inputs: profile_id (child profile id), is_seq (1 / 0) : indicating if character will be generated in sequence or no.
        sequence if alif, bay, pay,..
        For non sequence
        ML model return a vector of length 28 containing feature  scores. For these scores feature ids are found by:
        character_id -> cluster_id -> feature_id.
        attempt_feature table is populated where score = feature_score_vector[feature_id - 1], feature_id = id we found.
        next exercise is generated by finding 3 feature_ids with least average score. A random feature_id among them is picked, the cluster to which the id belongs is
        found and a random character from the cluster is given as next exercise.
        """
        # child profile_id to session session_id to History character_id to Character sequence_id

        profile_id_stroke = request.data.get('profile_id', None)
        is_seq = request.data.get('is_seq', None)
        profile_session_id = Session.objects.values_list('session_id', flat=True).filter(
            profile_id=profile_id_stroke).latest('session_id')
        profile_session_character_id = History.objects.values_list('character_id', flat=True).filter(
            session_id=profile_session_id).latest('character_id')

        History_attempt_id = History.objects.filter(session_id=profile_session_id).latest('attempt_id')

        if is_seq == 1:
            character_sequence_id = Characters.objects.values_list('sequence_id', flat=True).filter(
                character_id=profile_session_character_id).latest('sequence_id')

            return Response(
                data=serializers.CharactersSerializer(
                    Characters.objects.filter(sequence_id=character_sequence_id + 1)[:1], many=True).data,
                status=status.HTTP_200_OK
            )
        else:

            feature_score_vector = feature_model_pseudo("pseudo_path")
            character_cluster_id = Characters.objects.values_list('cluster_id', flat=True).filter(
                character_id=profile_session_character_id).latest('cluster_id')

            cluster_feature_id = list(
                ClusterFeature.objects.filter(cluster_id=character_cluster_id).values_list('feature_id',
                                                                                           flat=True))
            ## populating attept_feature with feature ids and score

            for i in cluster_feature_id:
                AttemptFeatures.objects.create(feature_id=Features.objects.get(feature_id=i),
                                               score=feature_score_vector[i - 1],
                                               attempt_id=History_attempt_id)

            # finding average score of feature_id
            from django.db.models import Avg

            feature_averages_queryset = (AttemptFeatures.objects.values('feature_id').annotate(avg=Avg('score')))
            averages_lst = list(feature_averages_queryset.values_list('avg', flat=True))
            feature_id_lst = list(feature_averages_queryset.values_list('feature_id', flat=True))
            sorted_averages_list = sorted(averages_lst)
            if len(sorted_averages_list) > 3:
                sorted_averages_list = sorted_averages_list[:3]

            # selecting random feature from min 3 avg score

            features_min_score = Features.objects.get(
                feature_id=feature_id_lst[averages_lst.index(random.choice(sorted_averages_list))])
            cluster_feature_id = list(
                ClusterFeature.objects.filter(feature_id=features_min_score).values_list('cluster_id', flat=True))

            # randomly choosing cluster
            cluster_feature_id = random.choice(cluster_feature_id)
            cluster_cluster_id = Clusters.objects.get(cluster_id=cluster_feature_id)
            character_cluster = list(
                Characters.objects.filter(cluster_id=cluster_cluster_id).values_list('character_id', flat=True))

            return Response(
                data=serializers.CharactersSerializer(
                    Characters.objects.filter(character_id=random.choice(character_cluster)), many=True).data,
                status=status.HTTP_200_OK
            )

    @action(methods=["GET"], detail=False)
    def generate_urdu_object_exercise(self, request):
        """
        generating random urdu word exercise
        """

        records = list(ObjectWord.objects.values_list('object_id', flat=True))
        record_id = random.sample(records, 4)

        return Response(
            data=serializers.ObjectWordSerializer(
                ObjectWord.objects.filter(Q(pk=record_id[0]) | Q(pk=record_id[1]) | Q(pk=record_id[2]) | Q(pk=record_id[3])), many=True).data,
            status=status.HTTP_200_OK
        )

    @action(methods=['POST'], detail=False, permission_classes=[IsAuthenticated, ])
    def time_score_completion__exercise(self, request):
        """
        Takes two inputs:
        i)profile id of child : int id
        ii) number of days for which we have to see the graph :int days
        makes dashboard graphs for time, drawing, % completed. pushes them to github if they are not in database and returns their path. If they are in databse returns their path.
        """
        profile_id_child = request.data.get('profile_id', None)
        days = request.data.get('days', None)
        # child_all_sessions = Session.objects.filter(profile_id=profile_id_child).values_list('session_id', flat=True)
        latest_attempt_id = History.objects.values_list('attempt_id', flat=True).latest('attempt_id')
        file_name = str(profile_id_child) + '_' + str(latest_attempt_id) + '.png'

        dashboard_time_graph_path = 'https://raw.githubusercontent.com/kaavish-ki-kavish/aangan-filesystem/main/dashboard/time/' + file_name
        dashboard_score_graph_path = 'https://raw.githubusercontent.com/kaavish-ki-kavish/aangan-filesystem/main/dashboard/score/' + file_name
        dashboard_completion_graph_path = 'https://raw.githubusercontent.com/kaavish-ki-kavish/aangan-filesystem/main/dashboard/completion/' + file_name

        if Dashoard.objects.filter(time_path=dashboard_time_graph_path).exists():  # if any one exists, all exist
            return Response(
                data={
                    'time_graph_path': dashboard_time_graph_path,
                    'score_graph_path': dashboard_score_graph_path,
                    'score_completion_path': dashboard_completion_graph_path},
                status=status.HTTP_200_OK
            )
        else:

            graph_val_time_draw = []
            graph_val_score_draw = []

            graph_val_time_urdu = []
            graph_val_score_urdu = []

            drawing_completion = 0
            urdu_completion = 0

            drawing_completion = History.objects.filter(profile_id=profile_id_child).filter(
                Q(drawing_id__isnull=False)).distinct().count()

            # character, object non null
            urdu_completion = History.objects.filter(profile_id=profile_id_child).filter(
                Q(character_id__isnull=False) | Q(object_id__isnull=False)).distinct().count()

            # for each session_id getting  time taken

            # DRAWING

            # for each session_id getting time taken
            date_time_ex_draw = History.objects.filter(profile_id=profile_id_child).filter(
                Q(drawing_id__isnull=False)).filter(
                datetime_attempt__range=(timezone.now() - datetime.timedelta(days=days), timezone.now())
            ).values('datetime_attempt').annotate(data_sum=Sum('time_taken'))

            # for each session_id getting sum stroke score
            date_score_ex_draw = History.objects.filter(profile_id=profile_id_child).filter(
                Q(drawing_id__isnull=False)).filter(
                datetime_attempt__range=(timezone.now() - datetime.timedelta(days=days), timezone.now())
            ).values('datetime_attempt').annotate(data_sum=Sum('stroke_score'))

            # URDU

            date_time_ex_urdu = History.objects.filter(profile_id=profile_id_child).filter(
                Q(character_id__isnull=False) | Q(object_id__isnull=False)).filter(
                datetime_attempt__range=(timezone.now() - datetime.timedelta(days=days), timezone.now())
            ).values('datetime_attempt').annotate(data_sum=Sum('time_taken'))

            # for each session_id getting sum stroke score
            date_score_ex_urdu = History.objects.filter(profile_id=profile_id_child).filter(
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
                graph_val_time_draw.append((draw_dates[j].strftime("%d-%B-%Y  %H:%M:%S"), draw_time[j]))
                graph_val_score_draw.append((draw_dates[j].strftime("%d-%B-%Y  %H:%M:%S"), draw_score[j]))

                # for urdu
                graph_val_time_urdu.append((urdu_dates[j].strftime("%d-%B-%Y  %H:%M:%S"), urdu_time[j]))
                graph_val_score_urdu.append((urdu_dates[j].strftime("%d-%B-%Y  %H:%M:%S"), urdu_score[j]))

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
            figure_path = 'user/dashboard/time/'
            plt.savefig(figure_path + file_name)

            push_image_file('dashboard/time/' + file_name, file_name)

            plt.clf()

            plt.plot(draw_dates, draw_score, label='Drawing')
            plt.plot(urdu_dates, urdu_score, label='Urdu')
            plt.legend()
            plt.title('Feedback Scores')
            plt.xlabel('date')
            plt.ylabel('score')
            figure_path = 'user/dashboard/score/'
            plt.savefig(figure_path + file_name)

            push_image_file('dashboard/score/' + file_name, file_name)

            plt.clf()

            drawing_total = DrawingExercise.objects.all().count()
            urdu_total = Characters.objects.all().count() + ObjectWord.objects.all().count()

            completed_drawing = int((drawing_completion * 100) / drawing_total)
            completed_urdu = int((urdu_completion * 100) / urdu_total)

            exercise_name = ['Urdu', 'Drawing']
            exercise_completion = [completed_urdu, completed_drawing]
            plt.barh(exercise_name, exercise_completion)
            plt.title('Exercise Completion')
            plt.xlabel('percentage completed')
            figure_path = 'user/dashboard/completion/'
            plt.savefig(figure_path + file_name)

            push_image_file('dashboard/completion/' + file_name, file_name)

            Dashoard.objects.create(time_path=dashboard_time_graph_path, score_path=dashboard_score_graph_path,
                                    completion_path=dashboard_completion_graph_path,
                                    profile_id=ChildProfile.objects.get(profile_id=profile_id_child))

            return Response(
                data={
                    'time_graph_path': dashboard_time_graph_path,
                    'score_graph_path': dashboard_score_graph_path,
                    'score_completion_path': dashboard_completion_graph_path},
                status=status.HTTP_200_OK
            )

    @action(methods=['POST'], detail=False, permission_classes=[IsAuthenticated, ])
    def time_last_n_days(self, request):
        '''
        takes profile_id and days (number of days) s input and gives time in minutes spent. deafult days is 7
        '''
        profile_id_child = request.data.get('profile_id', None)
        days = request.data.get('days', 7)
        time_taken_sum = 0

        # for each session_id getting last 7 days and sum of time taken

        date_time_query = History.objects.filter(profile_id=profile_id_child).filter(
            datetime_attempt__range=(timezone.now() - datetime.timedelta(days=days), timezone.now()))
        time_taken_sum += sum(date_time_query.values_list('time_taken', flat=True))

        return Response(
            data={'time_taken_sum': time_taken_sum},
            status=status.HTTP_200_OK
        )

    @action(methods=['POST'], detail=False, permission_classes=[IsAuthenticated, ])
    def average_time_exercise(self, request):
        '''
        takes profile_id as input and gives time spent on average per exercise
        '''
        profile_id_child = request.data.get('profile_id', None)
        total_time_taken = 0
        number_of_attempts = 0

        filter_session = History.objects.filter(profile_id=profile_id_child)
        total_time_taken += sum(filter_session.values_list('time_taken', flat=True))
        number_of_attempts += filter_session.values('attempt_id').count()

        if number_of_attempts == 0:
            avg_time_taken = 0
        else:
            avg_time_taken = total_time_taken / number_of_attempts

        return Response(
            data={'avg_time_taken': int(avg_time_taken)},
            status=status.HTTP_200_OK
        )

    @action(methods=['POST'], detail=False, permission_classes=[IsAuthenticated, ])
    def average_score_exercise(self, request):
        '''
        takes profile_id and days (number of days we are averaging) as input and gives time spent on average per exercise.
        Default number of days is 7.
        '''
        profile_id_child = request.data.get('profile_id', None)
        days = request.data.get('days', 7)
        score_sum = 0

        # for each session_id getting last days and sum of time taken

        date_time_query = History.objects.filter(profile_id=profile_id_child).filter(
            datetime_attempt__range=(timezone.now() - datetime.timedelta(days=days), timezone.now()))
        score_sum = sum(date_time_query.values_list('stroke_score', flat=True))

        if score_sum == 0:
            avg_score_sum = 0
        else:
            avg_score_sum = round((score_sum / days), 2) * 100

        return Response(
            data={'avg_score_sum': avg_score_sum},
            status=status.HTTP_200_OK
        )

    @action(methods=['POST'], detail=False, permission_classes=[IsAuthenticated, ])
    def exercises_last_n_days(self, request):
        '''
        takes profile_id and number of days as input and gives exercises completed in last n days. Default number of days is 7.
        '''
        profile_id_child = request.data.get('profile_id', None)
        days = request.data.get('days', 7)

        filter_session = History.objects.filter(profile_id=profile_id_child).filter(is_completed=True).filter(
            datetime_attempt__range=(timezone.now() - datetime.timedelta(days=days), timezone.now()))
        completed_exercises = filter_session.values('character_id').distinct().count() + \
                              filter_session.values('object_id').distinct().count() + \
                              filter_session.values('drawing_id').distinct().count()

        return Response(
            data={'completed_exercises_sum': completed_exercises},
            status=status.HTTP_200_OK
        )

    @action(methods=['POST'], detail=False, permission_classes=[IsAuthenticated, ])
    def exercise_review_character(self, request):
        '''
        takes profile id and to_display_no as input. Default to_display_no is 6.
        to_display_no is how many exercises details are to be outputted.
        Output is exercise review for characters.
        '''
        profile_id_child = request.data.get('profile_id', None)
        to_display_no = request.data.get('to_display_no', 6)

        urdu_letter = History.objects.filter(profile_id=profile_id_child).filter(Q(character_id__isnull=False))

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
            status=status.HTTP_200_OK
        )

    @action(methods=['POST'], detail=False, permission_classes=[IsAuthenticated, ])
    def exercise_review_object(self, request):
        '''
        takes profile id and to_display_no as input. Default to_display_no is 6.
        to_display_no is how many exercises details are to be outputted.
        Output is exercise review for words.
        '''
        profile_id_child = request.data.get('profile_id', None)
        to_display_no = request.data.get('to_display_no', 6)

        urdu_object = History.objects.filter(profile_id=profile_id_child).filter(Q(object_id__isnull=False))

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
            status=status.HTTP_200_OK
        )

    @action(methods=['POST'], detail=False, permission_classes=[IsAuthenticated, ])
    def exercise_review_drawing(self, request):
        '''
        takes profile id and to_display_no as input. Default to_display_no is 6.
        to_display_no is how many exercises details are to be outputted.
        Output is exercise review for drawing.
        '''
        profile_id_child = request.data.get('profile_id', None)
        to_display_no = request.data.get('to_display_no', 6)
        drawing_ex = History.objects.filter(profile_id=profile_id_child).filter(Q(drawing_id__isnull=False))

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
            status=status.HTTP_200_OK
        )

    @action(methods=['POST'], detail=False, permission_classes=[IsAuthenticated, ])
    def submit_exercise(self, request):
        """
        will update history table.
        inputs: exercise, char, data, profile_id
        exercise are:-
        drawing = 0, character/letters = 1, words = 2
        """

        exercise_type = request.data.get('exercise', None)
        char = request.data.get('char', None)
        data = request.data.get('data', None)
        profile_id = request.data.get('profile_id', None)
        time_str = time.strftime("%Y%m%d-%H%M%S")

        stroke_path = get_stroke_path(data, profile_id, exercise_type, time_str)

        if exercise_type == 0:  # drawing
            stroke_score, similarity_score = attempt_score(char, data, exercise_type)
            score = (stroke_score + similarity_score) // 2
            if score > 50:
                is_completed = True
            is_completed = False
            History.objects.create(profile_id=ChildProfile.objects.get(profile_id=profile_id),
                                   stroke_score=stroke_score,
                                   stroke_path=stroke_path,
                                   time_taken=random.randint(1, 20),
                                   datetime_attempt=timezone.now(),
                                   similarity_score=similarity_score,
                                   character_id=None,
                                   coloring_id=None,
                                   object_id=None,
                                   is_completed=is_completed,
                                   drawing_id=DrawingExercise.objects.get(label=char),
                                   word_id=None
                                   )

        if exercise_type == 1:  # character
            stroke_score, similarity_score = attempt_score(char, data, exercise_type)
            score = (stroke_score + similarity_score) // 2
            if score > 60:
                is_completed = True
            is_completed = False
            History.objects.create(profile_id=ChildProfile.objects.get(profile_id=profile_id),
                                   stroke_score=stroke_score,
                                   stroke_path=stroke_path,
                                   time_taken=random.randint(1, 20),
                                   datetime_attempt=timezone.now(),
                                   similarity_score=similarity_score,
                                   character_id=Characters.objects.get(label=char),
                                   coloring_id=None,
                                   object_id=None,
                                   is_completed=is_completed,
                                   drawing_id=None,
                                   word_id=None
                                   )

        if exercise_type == 2:  # words
            stroke_score, similarity_score = attempt_score(char, data, exercise_type)
            score = (stroke_score + similarity_score) // 2
            if score > 50:
                is_completed = True
            is_completed = False
            History.objects.create(profile_id=ChildProfile.objects.get(profile_id=profile_id),
                                   stroke_score=stroke_score,
                                   stroke_path=stroke_path,
                                   time_taken=random.randint(1, 20),
                                   datetime_attempt=timezone.now(),
                                   similarity_score=similarity_score,
                                   character_id=None,
                                   coloring_id=None,
                                   object_id=None,
                                   is_completed=is_completed,
                                   drawing_id=None,
                                   word_id=WordsUrdu.objects.get(word_label=char)
                                   )
        return Response(
            data={'score': score},
            status=status.HTTP_200_OK
        )

    @action(methods=['GET'], detail=False, permission_classes=[IsAuthenticated, ])
    def check(self, request):
        return Response(
            data=serializers.HistorySerializer(
                History.objects.all(), many=True).data,
            status=status.HTTP_200_OK
        )
    
    @action(methods=['GET'], detail=False)
    def insert_data(self, request):
        words = ['alif', 'ttaa', 'paa', 'seey', 'baa', 'taa', 'daal', 'zaal', 'dhaal', 'seen', 'sheen', 'zwaad',
                 'swaad']
        for i in range(1, 5):
            His = History(profile_id=ChildProfile.objects.get(profile_id=i),
                          stroke_score=random.random(),
                          stroke_path='/strokes/' + words[i] + '.txt',
                          time_taken=random.randint(1, 100),
                          datetime_attempt=timezone.now(),
                          similarity_score=random.random(),
                          character_id=Characters.objects.get(character_id=i),
                          coloring_id=None,
                          object_id=None,
                          is_completed=True,
                          drawing_id=None,
                          word_id=None
                          )
            His.save()
            
        for i in range(1, 5):
            His = History(profile_id=ChildProfile.objects.get(profile_id=i),
                          stroke_score=random.random(),
                          stroke_path='/strokes/' + words[i] + '.txt',
                          time_taken=random.randint(1, 100),
                          datetime_attempt=timezone.now(),
                          similarity_score=random.random(),
                          character_id=None,
                          coloring_id=None,
                          object_id=None,
                          is_completed=True,
                          drawing_id=DrawingExercise.objects.get(drawing_id=i),
                          word_id=None
                          )
            His.save()
    
        for i in range(1, 5):
            His = History(profile_id=ChildProfile.objects.get(profile_id=i),
                          stroke_score=random.random(),
                          stroke_path='/strokes/' + words[i] + '.txt',
                          time_taken=random.randint(1, 100),
                          datetime_attempt=timezone.now(),
                          similarity_score=random.random(),
                          character_id=None,
                          coloring_id=None,
                          object_id=None,
                          is_completed=True,
                          drawing_id=None,
                          word_id=WordsUrdu.objects.get(word_id=i),
                          )
            His.save()
    
        return Response(
            data=serializers.HistorySerializer(
                History.objects.all(), many=True).data,
            status=status.HTTP_200_OK
        )
