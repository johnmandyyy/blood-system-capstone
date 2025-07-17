from AesEverywhere import aes256
from django.conf import settings
import json
from datetime import datetime
from django.contrib.auth.models import User
from app.constants import app_constants

class Token:

    def __init__(self):
        pass

    # --------------------------------------------------------------------------------------------------------------- #
    def token_is_valid(self, token):

        """ To check if token is valid using custom token built from secret key. """
        try:

            token = aes256.decrypt(token, settings.SECRET_KEY)
            decrypted_dictionary = json.loads(token.decode("utf-8"))

            if app_constants.TOKEN_HAS_EXPIRY == True:
                return int(datetime.now().timestamp()) < decrypted_dictionary.get(
                    "expiry"
                )

            return True

        except Exception as e:
            return False

    # --------------------------------------------------------------------------------------------------------------- #
    def get_user(self, token):
        """ To check the user instance from token itself. """
        try:
            token = aes256.decrypt(token, settings.SECRET_KEY)
            decrypted_dictionary = json.loads(token.decode("utf-8"))
            username = decrypted_dictionary["user"]["username"]
            user_ = User.objects.get(username=username)
            return user_
        except Exception as e:
            return None

    # --------------------------------------------------------------------------------------------------------------- #
    def generate_token(self, request):
        """A custom class for generating token."""

        current_timestamp = int(datetime.now().timestamp())
        
        payload = {
            "user": {
                "user_id": request.user.pk,
                "username": str(request.user),
            },
            "created_at": current_timestamp,
            "expiry": current_timestamp + 30,
        }

        secret_key = settings.SECRET_KEY
        token = aes256.encrypt(json.dumps(payload), secret_key)
        token = token.decode('utf-8')
        return token
