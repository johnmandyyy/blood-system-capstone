from app.builder.template_builder import Builder
from app.constants import app_constants
from datetime import datetime

HOME = (
    Builder()
    .addPage("app/home.html")
    .addTitle("home")
)

HOME.build()

DATASETS = Builder().addPage("app/datasets.html").addTitle("datasets")
DATASETS.build()

PATIENTS = Builder().addPage("app/patients.html").addTitle("patients")
PATIENTS.build()

PATIENTS_RECORD = Builder().addPage("app/patients_record.html").addTitle("patients_record")
PATIENTS_RECORD.build()

ADMINISTRATION = Builder().addPage("app/administration.html").addTitle("administration")
ADMINISTRATION.build()

PATHOLOGIST = Builder().addPage("app/pathologist.html").addTitle("pathologist")
PATHOLOGIST.build()

LOGIN = (
    Builder()
    .addPage("app/login.html")
    .addTitle("login")
    .addContext(
        {
            "runtime_instances": None,
            "title": "Login - Page",
            "obj_name": "login",
            "app_name": app_constants.SOFTWARE_NAME,
            "app_desc": app_constants.SOFTWARE_DESCRIPTION,
        }
    )
)
LOGIN.build()
