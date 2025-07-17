from django.apps import apps
from app.models import models
from app.constants import app_constants
from django.contrib.auth.models import User

class ModelHelpers:
    """
    A predefined for model helpers.
    predefined_models can be used to get all the models with the exemption of EXCEPT models in
    constants.
    """

    def __init__(self) -> None:
        """Can be used for models."""
        
        self.predefined_models = None
        self.all_models = []
        self.__initialize()

    # --------------------------------------------------------------------------------------------------------------- #
    def get_model_instance(self, model_name: str) -> models:
        """Get the model instance return the model itself."""
        if model_name == 'User':
            return User
        try:
            return apps.get_model("app", model_name)
        except Exception as e:
            return None

    # --------------------------------------------------------------------------------------------------------------- #
    def __initialize(self) -> None:
        """Use to initialize models."""
        # Get a list of all installed models
        installed_models = apps.get_models()
        self.predefined_models = []

        for model in installed_models:
            # Get the name of the model and append it to the list
            if model.__name__ not in app_constants.EXCEPT_MODELS:
                self.all_models.append(model)
                self.predefined_models.append(model.__name__)


