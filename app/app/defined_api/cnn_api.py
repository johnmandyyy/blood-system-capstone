from app.defined_api.xception_algorithm import CNN
from rest_framework.views import APIView
from app.constants import response_constants as MESSAGE
from app.defined_api.xception_algorithm import CNN
from app.models import ModelInfo
from datetime import datetime
import json
from app.models import Prediction as PredictionTable
from rest_framework.response import Response
from app.models import Disease
import random
from app.helpers.symptoms_helper import SymptomsHelper

class Prediction(APIView):

    def __init__(self):
        self.medicines = [
            'Vincristine', 'Prednisone', 'L-Asparaginase', 'Daunorubicin', 'Methotrexate'
        ]

    def create_prescription(self):
        pass

    def post(self, request):
        symptoms = SymptomsHelper()
        file_loc = None
        patient = PredictionTable.objects.all().filter(id = request.data['id'])
        pk = None
        for each in patient:
            file_loc = each.smear_image
            pk = int(each.id)
            break
        
        Xception = CNN()
        predicted_disease = Xception.predict_image_v2(file_loc)
        heatmap_location = "/" + Xception.get_heatmap_location()
        segmented_location = Xception.get_segmented_image()
        original_location = Xception.get_original_image()

        symptoms.get_remarks(pk) # Generate a descriptive symptoms.
        description = symptoms.generate_description()
        severity = Xception.get_severity()
        patient.update(
            generated_heatmap = heatmap_location,
            predicted_diesease = Disease.objects.get(disease_name = predicted_disease),
            percentage_severity = severity,
            original_location = original_location,
            segmented_location = segmented_location,
            patient_symptoms = description
            )
        
        indications = ''
        if predicted_disease == 'Early':
            indications = random.choice(self.medicines)
            indications = "Take " + indications + " as prescribed."
 
        return Response({
            'message': predicted_disease + " is the predicted diesease.",
            "patient_symptoms": description,
            "original_smear_image": original_location,
            "heatmap_location": heatmap_location,
            "segmented_location": segmented_location,
            "smear_image": "/media/" + str(file_loc),
            "percentage_severity": severity,
            "prescription": indications
        }, 200)

class Training(APIView):

    def get(self, request):
        
        Xception = CNN()
        Xception.train_images()
        model_report = ModelInfo.objects.all()

        if len(model_report) > 0:

            ModelInfo.objects.all().update(last_trained_state = datetime.now(),
                                     accuracy = Xception.accuracy, 
                                     precision = Xception.precision, 
                                     recall = Xception.recall, 
                                     f1_score = Xception.f1_score, json_info = json.dumps(Xception.json_info))
        else:

            ModelInfo.objects.create(last_trained_state = datetime.now(),
                                     accuracy = Xception.accuracy, 
                                     precision = Xception.precision, 
                                     recall = Xception.recall, 
                                     f1_score = Xception.f1_score, json_info = json.dumps(Xception.json_info))

        return MESSAGE.TRAINING_DONE