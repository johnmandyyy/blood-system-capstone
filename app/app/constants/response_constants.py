from rest_framework.response import Response as __Response
from rest_framework import status as __STATUS

class __Messages:

    PERMISSION_DENIED_MESSAGE = "Permission Denied."
    INVALID_REQUEST = "Request was invalid."
    VALID = "Accepted."
    TRAINING_IS_DONE = "Training of AI Model is Done."
    USER_IS_INACTIVE = "User is inactive."

PERMMISSION_DENIED = __Response(
    {"details": __Messages.PERMISSION_DENIED_MESSAGE}, __STATUS.HTTP_401_UNAUTHORIZED
)

INVALID_REQUEST = __Response(
    {"details": __Messages.INVALID_REQUEST}, __STATUS.HTTP_400_BAD_REQUEST
)

TRAINING_DONE = __Response(
    {"details": __Messages.TRAINING_IS_DONE}, __STATUS.HTTP_201_CREATED
)

USER_IS_INACTIVE = __Response(
    {"details": __Messages.USER_IS_INACTIVE}, __STATUS.HTTP_403_FORBIDDEN
)

