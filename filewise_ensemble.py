from Ensemble import ensemble_model, voting, decide
from sklearn.metrics import classification_report
import numpy as np
from Pre_processing import file_names



def file_wise_ensemble(only_motor=False, list_area= None ):
    for file in file_names:
        print("***********",file,"***********")
        predictions,prediction_scores,y_test,y_train = ensemble_model(file)
        prediction_votes = voting(predictions, prediction_scores,only_motor, list_area)
        final_prediction = decide(prediction_votes)
        print("\n Final classification result after voting \n")
        print(classification_report(final_prediction, np.array(y_test)))

