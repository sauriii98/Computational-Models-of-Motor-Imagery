import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree

import Pre_processing


def ensemble_model(file_name):
    dataset = pd.read_csv(file_name+'.csv')
    dataset = dataset.sample(frac=1).reset_index(drop=True)
    
    mat = Pre_processing.take_input('Datasets/CLA-SubjectJ-170504-3St-LRHand-Inter' + '.mat')
    sub_data_sets = []
    shape = dataset.shape[1]-1
    
    for i in range(21):
        
        i = i * int(shape/21)
        
        point = dataset.iloc[:, i:i + int(shape/21)]
        sub_data_sets.append(point)

    x_train_sets = []
    x_test_sets = []
    for sub_data_set in sub_data_sets:
        train, test = train_test_split(sub_data_set, test_size=0.2, shuffle=False, random_state=0)
        x_train_sets.append(train)
        x_test_sets.append(test)

    y = dataset.iloc[:, -1]
    y_train, y_test = train_test_split(y, test_size=0.2, shuffle=False, random_state=0)

    predictions = {}
    prediction_scores = {}
    for i in range(len(x_train_sets)):
        tree_sub_train = tree.DecisionTreeClassifier()
        tree_sub_train.fit(x_train_sets[i], y_train)
        predicted = tree_sub_train.predict(x_test_sets[i])
        predictions[mat['o'][0][0][7][i][0][0]] = predicted
        prediction_scores[mat['o'][0][0][7][i][0][0]] = tree_sub_train.score(x_test_sets[i], y_test)
    
    print("Prediction scores of each tree corresponding to each point")
    print(prediction_scores)
    print("maximum accuracy of data point:",max(prediction_scores, key=prediction_scores.get))

    return predictions, prediction_scores,y_test,y_train


def voting(predictions, prediction_scores, only_motor=False,list_area = None):
    prediction_votes = []
    for j in range(len(predictions['F3'])):
        votes = {}
        for key, values in predictions.items():
            
            if only_motor == False:

                try:
                    votes[predictions[key][j]] += 1
                except:
                    votes[predictions[key][j]] = 1
            else:
                if key in list_area:
                    try:
                        votes[predictions[key][j]] += 1
                    except:
                        votes[predictions[key][j]] = 1
                

        prediction_votes.append(votes)
    return prediction_votes

def decide(prediction_votes, only_motor=False):
    final_prediction = []
    
    for votes in prediction_votes:
        
        max_votes = max(votes,key=votes.get)
        final_prediction.append(max_votes)
        
    return final_prediction




    
