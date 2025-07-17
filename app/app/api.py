from rest_framework.views import APIView
from rest_framework.response import Response
from django.contrib.auth import logout, authenticate, login
from app.helpers import authentication
from django.contrib.auth.models import User
from app.constants import response_constants as MESSAGES
from app.builder.api_builder import APIBuilder
from app.models import Pathologist
from django.contrib.auth.models import User
from rest_framework import generics
from rest_framework.permissions import AllowAny
from app.helpers.serializer_helpers import UserCreateSerializer

class AccountSection(generics.CreateAPIView):
    
    queryset = User.objects.all()
    serializer_class = UserCreateSerializer
    permission_classes = [AllowAny]  # Allows anyone to register

    def post(self, request, *args, **kwargs):
        response = super().post(request)
        if response.status_code == 201:
            User.objects.filter(username = response.data['username']).update(is_active = False)
            Pathologist.objects.create(user = User.objects.get(username = response.data['username']))
        
        return response
        
class Login(APIView):

    def post(self, request, *args, **kwargs):
        
        try:

            un = request.data['username']
            pw = request.data['password']
            active = User.objects.all().filter(username = un)

            if len(active) > 0:
                for user in active:
                    if user.is_active == False:
                        return MESSAGES.USER_IS_INACTIVE

            user = authenticate(request, username=un, password=pw)
            login(request, user)

            if user is not None:

                token = authentication.Token()
                token = token.generate_token(request)
                login(request, user)  # Library level not instance.

                return Response({
                    "details": "Login Successful",
                    "token": token
                }, 200)
            
        except Exception as e:
            print(e, type(e))
            return Response({
                "details": "Invalid Credentials"
            }, 401)