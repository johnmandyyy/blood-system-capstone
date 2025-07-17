"""
Definition of urls for app.
"""

from app import settings
from django.conf.urls.static import static
from django.urls import path, include
from django.contrib import admin
from app.views import TemplateView
from .api import *
from app.constants import app_constants
from rest_framework.schemas import get_schema_view
from django.views.generic import TemplateView as TV
from app.defined_api.cnn_api import Training, Prediction

import app.constants.url_constants as URLConstants
list_create_patterns = URLConstants.GenericAPI.list_create_patterns
list_get_patterns = URLConstants.GenericAPI.list_get_patterns
get_update_destroy_patterns = URLConstants.GenericAPI.retrieve_update_delete_patterns


# try:
    
#     import app.constants.url_constants as URLConstants
#     list_create_patterns = URLConstants.GenericAPI.list_create_patterns
#     list_get_patterns = URLConstants.GenericAPI.list_get_patterns
#     get_update_destroy_patterns = URLConstants.GenericAPI.retrieve_update_delete_patterns

# except Exception as e:
#     print(e, type(e), "No model found in the database.")
#     list_create_patterns = []
#     list_get_patterns = []
#     get_update_destroy_patterns = []


MainView = TemplateView()


api_patterns = [
    path(
        "api_schema/",
        get_schema_view(title="API Schema", description="Guide for the REST API"),
        name="api_schema",
    ),
    path(
        "docs/",
        TV.as_view(
            template_name="app/docs.html", extra_context={"schema_url": "api_schema"}
        ),
        name="swagger-ui",
    ),
    path("api/", include((list_get_patterns, app_constants.APP_NAME))),
    path("api/", include((list_create_patterns, app_constants.APP_NAME))),
    path("api/", include((get_update_destroy_patterns, app_constants.APP_NAME))),

    # Defined Endpoints
    path("api/login/", Login.as_view(), name="authenticate_user"),
    path("api/train-model/", Training.as_view(), name="train-model"),
    path("api/predict-disease/", Prediction.as_view(), name="prediction"),
    path("api/account-section/", AccountSection.as_view(), name="account-section"),
]

template_patterns = [
    path("home/", MainView.home, name="home"),
    path("datasets/", MainView.datasets, name="datasets"),
    path("patients/", MainView.patients, name="patients"),
    path("patients-record/<int:id>/", MainView.patients_records, name="patients-record"),
    path("administration/", MainView.administration, name="administration"),
    path("pathologist/", MainView.pathologist, name="pathologist"),
    path("admin/", admin.site.urls),
    path("logout/", MainView.user_logout, name="logout"),
    path("login/", MainView.login, name="login"),
]

urlpatterns = template_patterns + api_patterns +  static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
