from django.urls import path
from django import views
from .views import IndexView, PredictMovie

urlpatterns = [
    path('post', IndexView.as_view()),
    path('PredictMovie', PredictMovie.as_view())
]
