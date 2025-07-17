

# from app.defined_api.xception_algorithm import CNN
# Xception = CNN()

class Tasks:

    def __init__(self):
        """A class for tasks."""
        self.model_instance = None
        pass
    
    # --------------------------------------------------------------------------------------------------------------- #
    def sample_job(self):

        # if self.model_instance is None:

        #     try:
        #         from app.models import ModelInfo
        #         self.model_instance = ModelInfo
        #     except Exception as e:
        #         print(e)
        
        # else:

        #     from app.models import Prediction
        #     predict = Prediction.objects.all()

        #     for each in predict:
        #         file_loc = each.smear_image
        #         break

            
        #     predicted_disease = Xception.predict_image(file_loc)
        #     print(predicted_disease)



        print("Task(s) is running.")