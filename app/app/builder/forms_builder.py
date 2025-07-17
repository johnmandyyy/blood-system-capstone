from django import forms
from django.forms import ModelForm
from django.apps import apps
from app.helpers.helpers import ModelHelpers

class FormBuilder:

    """
    Sample Code:
    obj = FormBuilder().get_targeted_form("<form name>")
    """

    def __init__(self) -> None:
        """Initialize based on models."""
        
        self.__available_forms = []
        self.create_form()

    # --------------------------------------------------------------------------------------------------------------- #
    def get_targeted_form(self, form_name: str) -> ModelForm:
        """To get the targeted form based on name."""

        if len(self.__available_forms) > 0:
            for each_forms in self.__available_forms:
                if each_forms["form_name"] == str(form_name):
                    return each_forms

            return None

    # --------------------------------------------------------------------------------------------------------------- #
    def get_all_forms(self) -> list:
        """Return all forms in string name inside list."""

        return self.__available_forms

    # --------------------------------------------------------------------------------------------------------------- #
    def get_available_forms(self):
        """Get the list of available forms."""

        form_names = []
        if len(self.__available_forms) > 0:
            for each_forms in self.__available_forms:
                form_names.append(each_forms["form_name"])

        return form_names

    # --------------------------------------------------------------------------------------------------------------- #
    def create_form(self):
        """To create forms for each available models in the database."""

        available_models = ModelHelpers().predefined_models
        if len(available_models) > 0:
            for each_models in available_models:
                self.__available_forms.append(
                    {
                        "form_name": str(each_models).lower(),
                        "form_object": self.build(each_models),
                    }
                )

    # --------------------------------------------------------------------------------------------------------------- #
    def build(self, model_name: str, use_vue = True) -> ModelForm:
        """A builder method for each model name."""

        class GenericForm(forms.ModelForm):
            """Authentication form which uses boostrap CSS."""

            class Meta:
                model = ModelHelpers().get_model_instance(model_name)
                fields = "__all__"

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                if use_vue == True:
                    for each_rows in self.fields:
                        self.fields[str(each_rows)].widget.attrs['v-model'] = str(each_rows)

        form = GenericForm()
        return form

