from django.urls import path

from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("api/ask/", views.ask, name="ask"),
    path("api/search/", views.search, name="search"),
    path("api/ingest/", views.ingest, name="ingest"),
    path("api/health/", views.health, name="health"),
    path("api/chats/", views.chats_list, name="chats_list"),
    path("api/chats/<str:session_id>/", views.chat_detail, name="chat_detail"),
    path("api/chats/<str:session_id>/delete/", views.chat_delete, name="chat_delete"),

    path("api/chats/<str:session_id>/rename/", views.rename_chat, name="rename_chat"),
]
