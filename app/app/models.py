import os
from django.db import models
from django.contrib.auth.models import User
from datetime import datetime
# Default Python Models for Auto API

class RouteExclusion(models.Model):
    """Model for URL Routes"""
    required_token = models.BooleanField(default = False)
    route = models.CharField(unique=True, max_length=255)
    is_enabled = models.BooleanField(default = False)

    def __str__(self):
        remarks = ""
        if self.is_enabled == True:
            remarks = "Enabled"
        else:
            remarks = "Disabled"

        return remarks + " : " + self.route

class AppLogs(models.Model):
    """Model for application logs, whether API Level or Function Level"""
    time_stamp = models.TextField(default=None, null=True, blank=True)
    log_type = models.TextField(default=None, null=True, blank=True)
    level = models.TextField(default=None, null=True, blank=True)
    source = models.TextField(default=None, null=True, blank=True)
    message = models.TextField(default=None, null=True, blank=True)
    user_id = models.TextField(default=None, null=True, blank=True)
    session_id = models.TextField(default=None, null=True, blank=True)
    ip_address = models.TextField(default=None, null=True, blank=True)
    request_method = models.TextField(default=None, null=True, blank=True)
    request_path = models.TextField(default=None, null=True, blank=True)
    response_status = models.TextField(default=None, null=True, blank=True)
    data = models.TextField(default=None, null=True, blank=True)
    error_type = models.TextField(default=None, null=True, blank=True)
    error_message = models.TextField(default=None, null=True, blank=True)
    execution_time = models.TextField(default=0.00, null=True, blank=True)

    def __str__(self):
        return f"{self.time_stamp}"
    
class StackTrace(models.Model):
    app_log = models.ForeignKey(AppLogs, on_delete=models.CASCADE)
    description = models.TextField()

class ModelInfo(models.Model):

    last_trained_state = models.DateField(default = None)
    accuracy = models.TextField(default = None)
    precision = models.TextField(default = None)
    recall = models.TextField(default = None)
    f1_score = models.TextField(default = None)
    json_info = models.TextField(default = None, null = True)

class Disease(models.Model):
    
    disease_name = models.TextField(default = 'No Disease')
    description = models.TextField(default = 'No Description')

    def __str__(self):
        return f"{self.disease_name}"
    
class Images(models.Model):

    disease = models.ForeignKey(Disease, on_delete=models.CASCADE)
    location = models.FileField(upload_to='datasets')
    fname = models.TextField(null = True)
    used_for = models.CharField(default = 'No Remark', max_length= 255)

    def __str__(self):
        return str(self.disease.disease_name) + ": " + self.location.name
    
    def save(self, *args, **kwargs):
        file_name = os.path.basename(self.location.name)
        new_path = f'{self.disease.disease_name}/{file_name}'
        self.location.name = new_path
        self.fname = self.location.name
        super().save(*args, **kwargs)

class Patient(models.Model):

    first_name = models.CharField(max_length=255, null = False)
    middle_name = models.CharField(max_length=255, null = True, default = None)
    last_name = models.CharField(max_length=255, null = False)
    birth = models.DateField(default = None)

    # General Questions
    fatigue = models.BooleanField(default=False, help_text="Feeling unusually tired or fatigued?")
    persistent_fever = models.BooleanField(default=False, help_text="Persistent or unexplained fevers?")
    weight_loss = models.BooleanField(default=False, help_text="Unintentional weight loss?")
    night_sweats = models.BooleanField(default=False, help_text="Night sweats or chills?")
    general_unwell = models.BooleanField(default=False, help_text="General feeling of being unwell?")
    frequent_infections = models.BooleanField(default=False, help_text="Frequent infections?")
    pale_skin = models.BooleanField(default=False, help_text="Pale skin or looking more pale than usual?")
    bone_joint_pain = models.BooleanField(default=False, help_text="Unexplained pain in bones or joints?")
    swelling = models.BooleanField(default=False, help_text="Swelling in neck, underarms, or abdomen?")
    shortness_of_breath = models.BooleanField(default=False, help_text="Shortness of breath or rapid heartbeat with mild activity?")
    date_updated = models.DateTimeField(default = None, null = True)

    def update(self, *args, **kwargs):
        print("Data updated.")
        self.date_updated = datetime.now()
        self.save(update_fields=['date_updated'])

    def __str__(self):
        full_name = f'{self.first_name} {self.middle_name or ""} {self.last_name}'
        return ' '.join(full_name.split())  # Removes extra spaces if middle_name is None
    
class Pathologist(models.Model):

    user = models.ForeignKey(User, on_delete=models.CASCADE, null = True)
    suffixes = models.TextField(default = None, null = True)
    e_signature = models.FileField(upload_to='pathologist_signature', null = True)

    def __str__(self):
        return f'{self.user}'

class Prediction(models.Model):

    patient = models.ForeignKey(Patient, on_delete=models.CASCADE, default = None, null = True)
    date_of_diagnosis = models.DateTimeField(default = datetime.now(), null = True)
    smear_image = models.FileField(upload_to='patient_smears', null = True)
    notes = models.TextField(default = None, null = True)
    is_done = models.BooleanField(default = False)
    generated_heatmap = models.TextField(default = None, null = True)
    predicted_diesease = models.ForeignKey(Disease, null = True, on_delete=models.DO_NOTHING)
    percentage_severity = models.TextField(null = True, default = "0")
    original_location = models.TextField(default = None, null = True)
    segmented_location = models.TextField(default = None, null = True)
    patient_symptoms = models.TextField(default = None, null = True)