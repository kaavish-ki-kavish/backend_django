from django.contrib.auth import get_user_model, password_validation
from rest_framework.authtoken.models import Token
from rest_framework import serializers
from django.contrib.auth.models import BaseUserManager
from .models import ChildProfile, Characters, Session, History, ObjectWord, ColoringExercise, DrawingExercise

User = get_user_model()


class UserLoginSerializer(serializers.Serializer):
    email = serializers.CharField(max_length=300, required=True)
    password = serializers.CharField(required=True, write_only=True, style={'input_type': 'password'})


class AuthUserSerializer(serializers.ModelSerializer):
    auth_token = serializers.SerializerMethodField()

    class Meta:
        model = User
        fields = ('id', 'email', 'first_name', 'last_name', 'is_active', 'is_staff', 'auth_token')
        read_only_fields = ('id', 'is_active', 'is_staff', 'auth_token')

    def get_auth_token(self, obj):
        try:
            token = Token.objects.get(user_id=obj.id)
        except Token.DoesNotExist:
            token = Token.objects.create(user=obj)
        return token.key


class EmptySerializer(serializers.Serializer):
    pass


class UserRegisterSerializer(serializers.ModelSerializer):
    """
    A user serializer for registering the user
    """
    email = serializers.CharField(max_length=300, required=True)
    password = serializers.CharField(required=True, write_only=True, style={'input_type': 'password'})

    class Meta:
        model = User
        fields = ('id', 'email', 'password', 'first_name', 'last_name')

    def validate_email(self, value):
        user = User.objects.filter(email=value)
        if user:
            raise serializers.ValidationError("Email is already taken")
        return BaseUserManager.normalize_email(value)

    def validate_password(self, value):
        password_validation.validate_password(value)
        return value


class ChildRegisterSerializer(serializers.ModelSerializer):
    """
    A child serializer for registering the child
    """

    class Meta:
        model = ChildProfile
        fields = ('name', 'dob', 'gender', 'level', 'profile_id')


class DeleteChildSerializer(serializers.Serializer):
    profile_id = serializers.IntegerField(required=True)


class PasswordChangeSerializer(serializers.Serializer):
    current_password = serializers.CharField(required=True, style={'input_type': 'password'})
    new_password = serializers.CharField(required=True, style={'input_type': 'password'})

    def validate_current_password(self, value):
        if not self.context['request'].user.check_password(value):
            raise serializers.ValidationError('Current password does not match')
        return value

    def validate_new_password(self, value):
        password_validation.validate_password(value)
        return value


class EditChildSerializer(serializers.Serializer):
    profile_id = serializers.IntegerField()
    name = serializers.CharField(max_length=255)
    level_choices = [(0, 'Montessori'), (1, 'Nursury'), (2, 'Prep'), (3, 'other')]
    level = serializers.ChoiceField(choices=level_choices)
    gender_choice = [('F', 'Female'), ('M', 'Male'), ('O', 'Other')]
    gender = serializers.ChoiceField(choices=gender_choice)
    dob = serializers.DateField('Date of Birth')


# class DrawingExerciseSerializer(serializers.ModelSerializer):
#     class Meta:
#         model = Characters
#         fields = ('character_id', 'level', 'ref_stroke_path', 'ref_object_path', 'label', 'sound_path', 'sequence_id')


class SessionSerializer(serializers.ModelSerializer):
    class Meta:
        model = Session
        fields = ('session_id', 'profile_id', 'time_start', 'time_end', 'token')


class HistorySerializer(serializers.ModelSerializer):
    class Meta:
        model = History
        fields = (
            'attempt_id', 'session_id', 'stroke_path', 'time_taken', 'stroke_score', 'similarity_score',
            'datetime_attempt',
            'character_id', 'drawing_id', 'coloring_id', 'object_id', 'is_completed')


class CharactersSerializer(serializers.ModelSerializer):
    class Meta:
        model = Characters
        fields = ('character_id', 'level''ref_stroke_path', 'ref_object_path', 'label', 'sound_path', 'sequence_id')


class ObjectWordSerializer(serializers.ModelSerializer):
    class Meta:
        model = ObjectWord
        fields = ('object_id', 'label', 'image_path', 'is_object', 'ref_image_path', 'category', 'sound_path')


class DrawingExerciseSerializer(serializers.ModelSerializer):
    class Meta:
        model = DrawingExercise
        fields = ('drawing_id', 'ref_img_path', 'ref_stroke_path', 'level', 'sound_path', 'label')


class ColoringExerciseSerializer(serializers.ModelSerializer):
    class Meta:
        model = ColoringExercise
        fields = ('coloring_id', 'ref_image_path', 'level', 'sound_path', 'label')


class DataEntrySerializer(serializers.Serializer):
    char = serializers.CharField(max_length= 255, required= True)
    data = serializers.ListField(
        child=serializers.ListField(
            child = serializers.ListField(
            child= serializers.IntegerField())))
    exercise = serializers.IntegerField()

class DataEntrySerializerTf(serializers.Serializer):
    char = serializers.CharField(max_length= 255, required= True)
    img = serializers.ListField(
        child=serializers.ListField(
            child= serializers.IntegerField())
    )
    whole_x = serializers.ListField(
                child= serializers.IntegerField()
            )
    whole_y = serializers.ListField(
                child= serializers.IntegerField()
            )
    pen_up = serializers.ListField(
                child= serializers.IntegerField()
            )
    data = serializers.ListField(
        child=serializers.ListField(
            child = serializers.ListField(
            child= serializers.IntegerField())))
    exercise = serializers.IntegerField()
