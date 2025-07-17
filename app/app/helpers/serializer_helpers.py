from rest_framework import serializers
from django.apps import apps
from app.models import models
from django.contrib.auth.models import User
from rest_framework import serializers

class UserCreateSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True)

    class Meta:
        model = User
        fields = ['username', 'email', 'password', 'first_name', 'last_name']

    def create(self, validated_data):
        user = User.objects.create_user(**validated_data)  # Uses built-in method
        return user

class SerializerHelpers:
    """A predefined for serializer helpers."""

    def __init__(self):
        pass
    
    # --------------------------------------------------------------------------------------------------------------- #
    def create_serializer(self, model_name: models, app_name: str) -> serializers:
        """
        Create an automatic serializer for your API that include(s) auto joining of table.
        """

        if model_name == 'User':
            django_model = User
        else:
            django_model = apps.get_model(app_label=str(app_name), model_name=model_name)

        class AutoSerializer(serializers.ModelSerializer):

            class Meta:
                model = django_model
                fields = "__all__"
                depth = 10  # 10 Depths of table auto joined.

        return AutoSerializer
    
    # --------------------------------------------------------------------------------------------------------------- #
    def create_serializer_no_depth(self, model_name: models, app_name: str) -> serializers:
        """
        Create an automatic serializer for your API that include(s) auto joining of table.
        """

        if model_name == 'User' or model_name == 'user':
            django_model = User
        else:
            # print(app_name, type(app_name), model_name)
            django_model = apps.get_model(app_label=str(app_name), model_name=model_name)

        class AutoSerializer(serializers.ModelSerializer):

            class Meta:
                model = django_model
                fields = "__all__"
                # depth = 10  # 10 Depths of table auto joined.

        return AutoSerializer