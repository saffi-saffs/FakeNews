
from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [path("",views.NewsDisplay,name="home"),
               path("claimcheck",views.ClaimCheck,name="claimcheck"),
               path("check123",views.check123, name="123")]


