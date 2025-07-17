from rest_framework.authtoken.models import Token
from django.contrib.auth.models import User
from rest_framework.authtoken.views import ObtainAuthToken
from rest_framework.authtoken.serializers import AuthTokenSerializer
from rest_framework.request import Request
from rest_framework.exceptions import AuthenticationFailed
from rest_framework.response import Response as DR

class TokenAuthenticator:
    def __init__(self):
        pass

    @staticmethod
    def authenticate_user_from_token(bearer_token: str) -> User:
        try:
            token_key = bearer_token.replace("Bearer ", "")
            token = Token.objects.get(key=token_key)
            user = token.user
            return user
        except Token.DoesNotExist:
            raise AuthenticationFailed("Invalid token")

class CustomTokenObtainer(ObtainAuthToken):
    def post(self, request: Request, *args, **kwargs):
        serializer = self.serializer_class(data=request.data, context={"request": request})
        serializer.is_valid(raise_exception=True)
        user = serializer.validated_data["user"]
        token, created = Token.objects.get_or_create(user=user)
        return DR({"token": token.key})

class BearerTokenAuthenticator:
    def __init__(self):
        pass

    @staticmethod
    def authenticate_with_bearer_token(request: Request):
        try:
            authorization_header = request.META.get("HTTP_AUTHORIZATION")
            if authorization_header and authorization_header.startswith("Bearer "):
                bearer_token = authorization_header.split(" ")[1]
                user = TokenAuthenticator.authenticate_user_from_token(bearer_token)
                if user:
                    print(f"Authenticated user: {user.username}")
                    return user
            raise AuthenticationFailed("Invalid Bearer token")
        except AuthenticationFailed as e:
            print(str(e))
            return None