from django.contrib import admin
from .models import CustomUser, ChildProfile, History, Characters, ColoringExercise, DrawingExercise, ObjectWord


# Register your models here.
admin.site.register(CustomUser)
admin.site.register(ChildProfile)
admin.site.register(History)
admin.site.register(ColoringExercise)
admin.site.register(Characters)
#admin.site.register(Session)
admin.site.register(DrawingExercise)
admin.site.register(ObjectWord)