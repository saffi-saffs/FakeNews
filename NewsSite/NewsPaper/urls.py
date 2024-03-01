from django.urls import path,include
from . import views
urlpatterns=[path("",views.News,name="news"),
path("database",views.News,name="show_articles"),
             
             
          
              
    ]
             