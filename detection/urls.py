from django.urls import path
from .views import upload_file, base

urlpatterns = [
    path('',base, name='base'),
    path('upload/', upload_file, name='upload_file'),
]
