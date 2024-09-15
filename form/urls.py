from django.urls import path
from . import views

urlpatterns = [
    path('', views.upload_id_card, name='form'),
]