from django.db import models
from django.contrib.auth.models import AbstractUser, BaseUserManager
from .managers import CustomUserManager
from datetime import datetime


class CustomUser(AbstractUser):
    username = None
    email = models.EmailField('email address', unique=True)
    first_name = models.CharField('First Name', max_length=255, blank=True,
                                  null=False)
    last_name = models.CharField('Last Name', max_length=255, blank=True,
                                 null=False)
    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = []

    objects = CustomUserManager()

    def __str__(self):
        return f"{self.id} - {self.email} - {self.first_name} {self.last_name}"


class ChildProfile(models.Model):
    profile_id = models.AutoField(primary_key=True)
    user_id = models.ForeignKey(CustomUser, on_delete=models.CASCADE)
    creation_date = models.DateField(auto_now_add=True, null=True, blank=True)
    name = models.CharField(max_length=255)
    age = models.IntegerField(null=True)
    gender_choice = [('F', 'Female'), ('M', 'Male'), ('O', 'Other')]
    gender = models.CharField(max_length=255, choices=gender_choice)
    level_choices = [(0, 'Montessori'), (1, 'Nursury'), (2, 'Prep'), (3, 'other')]
    level = models.IntegerField(choices=level_choices)

    def __str__(self):
        return f'{self.profile_id} - {self.name} - child of - {self.user_id}'


# class Session(models.Model):
#     session_id = models.AutoField(primary_key=True)
#     profile_id = models.ForeignKey(ChildProfile, on_delete=models.CASCADE)
#     time_start = models.DateTimeField(auto_now_add=True, null=True, blank=True)
#     time_end = models.DateTimeField(null=True, blank=True)
#     token = models.CharField(max_length=255)
#
#     def __str__(self):
#         return f'{self.session_id} - by {self.profile_id} - on {self.time_start}'


class Clusters(models.Model):
    cluster_id = models.AutoField(primary_key=True)
    cluster_name = models.CharField(max_length=255, null=False)


class Characters(models.Model):
    character_id = models.IntegerField(null=False, blank=False)
    level = models.SmallIntegerField(null=False, blank=False)
    ref_stroke_path = models.CharField(max_length=255, null=False)
    ref_object_path = models.CharField(max_length=255, null=False)
    label = models.CharField(max_length=255, null=False)
    sound_path = models.CharField(max_length=255, null=True)
    sequence_id = models.IntegerField(null=True)
    cluster_id = models.ForeignKey(Clusters, on_delete=models.CASCADE, null=True)

    # def __str__(self):
    #     return f'{self.character_id} - level {self.level} - stroke at {self.ref_stroke_path}'


class ObjectWord(models.Model):
    object_id = models.AutoField(primary_key=True)
    label = models.CharField(max_length=255, null=False)
    image_path = models.CharField(max_length=255, null=False)
    is_object = models.BooleanField()
    ref_image_path = models.CharField(max_length=255, null=False)
    category = models.CharField(max_length=255, null=False)
    sound_path = models.CharField(max_length=255, null=True)


class ColoringExercise(models.Model):
    coloring_id = models.AutoField(primary_key=True)
    ref_image_path = models.CharField(max_length=255, null=False)
    level = models.SmallIntegerField(null=False, blank=False)
    sound_path = models.CharField(max_length=255, null=True)
    label = models.CharField(max_length=255, null=False)


class DrawingExercise(models.Model):
    drawing_id = models.AutoField(primary_key=True)
    ref_img_path = models.CharField(max_length=255, null=False)
    ref_stroke_path = models.CharField(max_length=255, null=False)
    level = models.SmallIntegerField(null=False, blank=False)
    sound_path = models.CharField(max_length=255, null=True)
    label = models.CharField(max_length=255, null=False)


class WordsUrdu(models.Model):
    word_id = models.AutoField(primary_key=True)
    word_label = models.CharField(max_length=255, null=False)
    ref_stroke_path = models.CharField(max_length=255, null=False)


class History(models.Model):
    attempt_id = models.AutoField(primary_key=True)
    stroke_path = models.CharField(max_length=255, null=False)
    time_taken = models.IntegerField(null=False)
    stroke_score = models.FloatField(null=True, blank=True)
    similarity_score = models.FloatField(null=True, blank=True)
    datetime_attempt = models.DateTimeField(null=False)
    character_id = models.ForeignKey(Characters, on_delete=models.PROTECT, null=True, blank=True)
    drawing_id = models.ForeignKey(DrawingExercise, on_delete=models.PROTECT, null=True, blank=True)
    coloring_id = models.ForeignKey(ColoringExercise, on_delete=models.PROTECT, null=True, blank=True)
    object_id = models.ForeignKey(ObjectWord, on_delete=models.PROTECT, null=True, blank=True)
    is_completed = models.BooleanField()
    profile_id = models.ForeignKey(ChildProfile, on_delete=models.CASCADE)
    word_id = models.ForeignKey(WordsUrdu, on_delete=models.CASCADE, null=True, blank=True)


class Features(models.Model):
    feature_id = models.AutoField(primary_key=True)
    feature_name = models.CharField(max_length=255, null=False)


class ClusterFeature(models.Model):
    cluster_feature_id = models.AutoField(primary_key=True)
    feature_id = models.ForeignKey(Features, on_delete=models.CASCADE)
    cluster_id = models.ForeignKey(Clusters, on_delete=models.CASCADE)


class AttemptFeatures(models.Model):
    attempt_feature_id = models.AutoField(primary_key=True)
    feature_id = models.ForeignKey(Features, on_delete=models.CASCADE)
    attempt_id = models.ForeignKey(History, on_delete=models.CASCADE)
    score = models.FloatField(null=False)


class Dashoard(models.Model):
    dashboard_id = models.AutoField(primary_key=True)
    time_path = models.CharField(max_length=255, null=False)
    score_path = models.CharField(max_length=255, null=False)
    completion_path = models.CharField(max_length=255, null=False)
    profile_id = models.ForeignKey(ChildProfile, on_delete=models.CASCADE)
