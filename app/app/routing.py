from django.urls import re_path
from app import consumers  # Adjust the import based on your app name

websocket_urlpatterns = [
    re_path(r'ws/some_path/$', consumers.YourConsumer.as_asgi()),
]