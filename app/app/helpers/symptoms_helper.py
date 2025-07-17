from app.models import Patient, Prediction

class SymptomsHelper:

    def __init__(self):
        self.final_remarks = []

    def get_remarks(self, prediction_id):
        """Builds a descriptive list of symptom-related remarks for a prediction."""
        predictions = Prediction.objects.filter(id=prediction_id)

        if not predictions.exists():
            return

        for prediction in predictions:
            patient = prediction.patient  # Assuming ForeignKey 'patient' in Prediction

            if not patient:
                continue

            # Mapping of symptom field to descriptive text
            symptom_descriptions = {
                "fatigue": "The patient has a feeling of unusual tiredness or fatigue.",
                "persistent_fever": "The patient experiences persistent or unexplained fevers.",
                "weight_loss": "The patient has experienced unintentional weight loss.",
                "night_sweats": "The patient has reported night sweats or chills.",
                "general_unwell": "The patient feels generally unwell.",
                "frequent_infections": "The patient suffers from frequent infections.",
                "pale_skin": "The patient has pale skin or looks paler than usual.",
                "bone_joint_pain": "The patient reports unexplained pain in bones or joints.",
                "swelling": "The patient has swelling in the neck, underarms, or abdomen.",
                "shortness_of_breath": "The patient experiences shortness of breath or a rapid heartbeat with mild activity."
            }

            # Iterate over each symptom field
            for field, description in symptom_descriptions.items():
                if getattr(patient, field, False):
                    self.final_remarks.append(description)

    def generate_description(self):
        """Returns a single string description generated from symptoms."""
        return " ".join(self.final_remarks) if self.final_remarks else "No significant symptoms reported."
