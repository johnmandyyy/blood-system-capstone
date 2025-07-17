from app.models import Patient


class PatientRecord:

    def __init__(self):
        pass

    def get_patient(self, id):
        """ A method to retrieve patient object. """

        if id == None or id == '':
            return None
        
        try:
            patient_object = Patient.objects.all().filter(id = id)
            if len(patient_object) > 0:
                print("Patient record was retrieved.", patient_object)
                return patient_object
            return None
        except Exception as e:
            print(e)
            return None