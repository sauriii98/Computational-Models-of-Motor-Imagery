from sklearn import svm
from sklearn import tree
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier



def classifier_model(clf, X_train, y_train, X_test, y_test):
    clf.fit(X_train, y_train)
    train_acc = clf.score(X_train, y_train)
    test_acc = clf.score(X_test, y_test)
    
    predicted_test = clf.predict(X_test)
    confusion=confusion_matrix(y_test, predicted_test)
    report = classification_report(y_test, predicted_test)
    
    return train_acc, test_acc, predicted_test, confusion, report

def svm_ml(X_train, y_train, X_test, y_test):
    clf = svm.SVC(kernel='rbf', max_iter=31)
    
    return classifier_model(clf, X_train, y_train, X_test, y_test)

def dTree_ml(X_train, y_train, X_test, y_test):
    clf = tree.DecisionTreeClassifier(max_depth=5)
    
    return classifier_model(clf, X_train, y_train, X_test, y_test)

def rforest_ml(X_train, y_train, X_test, y_test):
    clf = RandomForestClassifier()
    
    return classifier_model(clf, X_train, y_train, X_test, y_test)

def adaBoost_ml(X_train, y_train, X_test, y_test):
    clf = AdaBoostClassifier(n_estimators=100, random_state=0)
    
    return classifier_model(clf, X_train, y_train, X_test, y_test)

def bagging_ml(X_train, y_train, X_test, y_test):
    clf = BaggingClassifier()
    
    return classifier_model(clf, X_train, y_train, X_test, y_test)

def gaussianNB_ml(X_train, y_train, X_test, y_test):
    clf = GaussianNB()
    
    return classifier_model(clf, X_train, y_train, X_test, y_test)

def bernoulliNB_ml(X_train, y_train, X_test, y_test):
    clf = BernoulliNB()
    
    return classifier_model(clf, X_train, y_train, X_test, y_test)

def MLP_ml(X_train, y_train, X_test, y_test):
    clf = MLPClassifier()

    return classifier_model(clf, X_train, y_train, X_test, y_test)

def QDA_ml(X_train, y_train, X_test, y_test):
    clf = QuadraticDiscriminantAnalysis()
    
    return classifier_model(clf, X_train, y_train, X_test, y_test)

def KNN_ml(X_train, y_train, X_test, y_test):
    clf = KNeighborsClassifier(3)
    
    return classifier_model(clf, X_train, y_train, X_test, y_test)    