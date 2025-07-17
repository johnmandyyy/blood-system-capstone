from app.logs.logging import Levels, LogTypes

#App Related
APP_NAME = "app"
SOFTWARE_NAME = "Lymphoblastic Cell(s) Detection"
SOFTWARE_DESCRIPTION = "Capstone"



#Database Related
EXCEPT_MODELS = [
    "LogEntry",
    "Permission",
    "Group",
    "ContentType",
    "Session",
]

# Permission Related
TOKEN_HAS_EXPIRY = False
TOKEN_VALIDITY = 60 * 60

# Logging Related
LOG_LEVEL = Levels()
LOG_TYPE = LogTypes()
HAS_LOGGING = False
IS_DATABASE_LOGGING = False

# Styling Related
THEMES = None

MODEL_NAME = 'xception_model.h5'
SEG_MODEL_NAME = 'blood_cell_segmentor_xception.h5'
EPOCHS = 5



