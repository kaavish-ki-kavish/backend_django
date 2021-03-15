import pickle
import pandas as pd
import os
import numpy as np

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))


class RandomForestClassifier:
    def __init__(self):
        file = open(os.path.join(__location__, 'random_forest_model.sav'), 'rb')
        self.model = pickle.load(file)

    def preprocessing(self, input_data):
        return input_data

    def predict(self, input_data):
        a = self.model.predict_proba(input_data)
        return a

    def postprocessing(self, input_data):
        # label = "<=50K"
        # if input_data[1] > 0.5:
        #     label = ">50K"
        # return {"probability": input_data, "label": label, "status": "OK"}
        return input_data

    def compute_prediction(self, input_data):
        try:
            input_data = self.preprocessing(input_data)
            #prediction = self.predict(input_data)[0][0]  # only one sample
            prediction = np.amax(self.predict(input_data))
            # prediction = self.postprocessing(prediction)
        except Exception as e:
            return {"status": "Error", "message": str(e)}

        return prediction


