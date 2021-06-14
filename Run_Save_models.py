from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.metrics import classification_report,confusion_matrix
import pandas as pd
from Import_data import import_data
from c_models import classifier_model, svm_ml, dTree_ml, rforest_ml, adaBoost_ml, bagging_ml, gaussianNB_ml, \
    bernoulliNB_ml, MLP_ml, QDA_ml, KNN_ml


def run_save_models(file_names):

    filewise_train_reports = {"models": ['SVM','Decision_tree','rforest','AdaBoost','Bagging','gaussianNB_ml','bernoulliNB_ml','MLP_ml','QDA_ml','KNN_ml']}
    filewise_test_reports = {"models": ['SVM','Decision_tree','rforest','AdaBoost','Bagging','gaussianNB_ml','bernoulliNB_ml','MLP_ml','QDA_ml','KNN_ml']}
    i=0
    for file in file_names:
        (X_train,y_train),(x_test,y_test) = import_data(file)
        reports = []




        reports.append(svm_ml(X_train, y_train, x_test, y_test))

        reports.append(dTree_ml(X_train, y_train, x_test, y_test))

        reports.append(rforest_ml(X_train, y_train, x_test, y_test))

        reports.append(adaBoost_ml(X_train, y_train, x_test, y_test))

        reports.append(bagging_ml(X_train, y_train, x_test, y_test))

        reports.append(gaussianNB_ml(X_train, y_train, x_test, y_test))

        reports.append(bernoulliNB_ml(X_train, y_train, x_test, y_test))

        reports.append(MLP_ml(X_train, y_train, x_test, y_test))

        reports.append(QDA_ml(X_train, y_train, x_test, y_test))

        reports.append(KNN_ml(X_train, y_train, x_test, y_test))
        train_accuracy = []
        test_accuracy = []

        for report in reports:
            train_accuracy.append(report[0])
            test_accuracy.append(report[1])




        filewise_train_reports[file] = train_accuracy
        filewise_test_reports[file] = test_accuracy

    filewise_test_reports_df = pd.DataFrame.from_dict(filewise_test_reports)
    filewise_train_reports_df = pd.DataFrame.from_dict(filewise_train_reports)

    filewise_test_reports_df.to_csv('Reports/file_wise_test_reports.csv')
    filewise_train_reports_df.to_csv('Reports/filewise_train_reports.csv')



    return filewise_train_reports_df,filewise_test_reports_df

