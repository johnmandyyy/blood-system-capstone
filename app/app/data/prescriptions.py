from app.models import Prediction
from datetime import date

class Prescription(object):

    def __init__(self, id):
        self.id = id
        self.record = {
            'patient_name': '',
            'age': '',
            'date_today':  '',
            'remarks':  '',
            'email': '',
            'symptoms':  '',
            'notes':  '',
            'attending_physician': '',
            'signature': '',
            'license_no': ''
        }

    def _get_prescription_record(self):
        
        results = Prediction.objects.all().filter(id = self.id)
        if results:
            for each in results:
                self.record = {
                    'patient_name': str(each.patient.first_name) + " " + str(each.patient.middle_name) if each.patient.middle_name else "" + " " + str(each.patient.last_name),
                    'age': date.today().year - each.patient.birth.year,
                    'email': each.patient.email if each.patient.email else each.pathologist.user.email,
                    'date_today': str(date.today()),
                    'remarks': each.predicted_diesease.disease_name,
                    'symptoms': each.patient_symptoms,
                    'notes': each.notes,
                    'attending_physician': str(each.pathologist.user.first_name) + " " + str(each.pathologist.user.last_name) + " " + str(each.pathologist.suffixes) if each.pathologist.suffixes else "",
                    'e_signature': str(each.pathologist.e_signature),
                    'license_no': str(each.pathologist.license_no) if each.pathologist.license_no else ""
                }

                break

        return self.record
