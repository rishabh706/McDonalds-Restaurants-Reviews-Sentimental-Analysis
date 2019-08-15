from django.urls import path
from django.conf.urls import url
from . import views
app_name='mymodel'
urlpatterns=[
url(r'^$',views.index,name='index'),
url(r'^predict/$',views.predict,name='predict'),
url(r'^upload/', views.predict_upload, name='predict_upload'),

]
