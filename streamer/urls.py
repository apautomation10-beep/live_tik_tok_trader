from django.urls import path
from . import views

urlpatterns = [
    path('', views.dashboard, name='dashboard'),
    path('go-live/', views.go_live, name='go_live'),
    path('recordings/', views.list_recordings, name='list_recordings'),
    path('delete-recordings/', views.delete_all_recordings, name='delete_recordings'),

    # TikTok integration endpoints
    path('tiktok/comment/', views.tiktok_comment, name='tiktok_comment'),
    path('replies/next/', views.next_reply, name='next_reply'),
]
