from app.helpers.authentication import Token

class APIHelpers:

    def __init__(self, request):
        self.request = request
    
    # --------------------------------------------------------------------------------------------------------------- #
    def is_permissible(self):
        """To check whether it is permissible or not"""
        if self.request.headers.get("Authorization") != None:
            return Token().token_is_valid(
                self.request.headers.get("Authorization", "").split(" ")[1]
            )
        else:
            return False

    # --------------------------------------------------------------------------------------------------------------- #
    def get_user_from_token(self):
        """To get the token"""
        if self.request.headers.get("Authorization") != None:
            return Token().get_user(
                self.request.headers.get("Authorization", "").split(" ")[1]
            )

    # --------------------------------------------------------------------------------------------------------------- #
    def is_login_required(self):
        """A flag to check whether user is authenticated."""
        return self.request.user.is_authenticated