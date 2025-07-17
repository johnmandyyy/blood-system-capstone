from django.contrib import admin
from app.models import *
from app.helpers.helpers import ModelHelpers
from django.contrib.sessions.models import Session

MODELS = ModelHelpers()

if len(MODELS.predefined_models) > 0:
    for each_models in MODELS.predefined_models:
        if each_models != 'User':
            admin.site.register(
                MODELS.get_model_instance(each_models)
            )

admin.site.register(Session)

