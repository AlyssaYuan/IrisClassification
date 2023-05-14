from django.urls import path
from . import views

urlpatterns = [
    path('', views.Predictor, name = 'predictor'),
    path('result', views.formInfo, name = 'result')
]