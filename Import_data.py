from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
import pandas as pd
import numpy as np

def import_data(file_name):
    dataset = pd.read_csv(file_name+'.csv')
    dataset = dataset.sample(frac=1).reset_index(drop=True)

    train, test = train_test_split(dataset, test_size=0.2, shuffle=False, random_state=0)
    test = test.reset_index(drop=True)

    X_train = train[train.columns[:-1]]
    y_train = train['label']

    X_test = test[test.columns[:-1]]
    y_test = test['label']
    np.array(X_test).shape

    return (X_train,y_train),(X_test,y_test)
