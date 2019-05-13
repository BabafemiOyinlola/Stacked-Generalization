import math
import random
import numpy as np
import pandas as pd

from sklearn.svm import SVC
from sklearn.model_selection import cross_val_predict
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from experiments import cross_validate_clfs, cross_validate_clfs_noise
from stacked_generalization import StackedGeneralization

def split_data(filepath, lbl_pos=-1):
    data = pd.read_csv(filepath, header=None)
    data = np.array(data)
    X, y = None, None
    if lbl_pos == -1:
        y = data[:, -1]
        X = data[:, 0:data.shape[1]-1]
    elif lbl_pos == 0:
        y = data[:, 0]
        X = data[:, 1:data.shape[1]]
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    return (X_train, X_test, y_train, y_test) 

if __name__ == "__main__":
    knn, dt, svm, lr  = KNeighborsClassifier(), DecisionTreeClassifier(), SVC(), LogisticRegression()
    classifiers = [KNeighborsClassifier, LogisticRegression, DecisionTreeClassifier]

    data = pd.read_csv('data/waveform/waveform.txt', header=None)
    # data = pd.read_csv('data/glass/glass.csv', header=None)

    # data = pd.read_csv('data/ionosphere/ionosphere.csv', header=None)
    # data = pd.read_csv('data/breast_cancer/breast_cancer.csv', header=None)

    data = data.sample(frac=1).reset_index(drop=True) #shuffle

    classifiers_cv = [KNeighborsClassifier, LogisticRegression, DecisionTreeClassifier, StackedGeneralization, StackedGeneralization]   
    classifiers_cv_names = ["knn", "lr", "dt", "stacked", "stacked_prob"]

    cv_results = cross_validate_clfs(data, classifiers_cv, classifiers_cv_names, meta_clf=SVC)
    print("\nCROSS VALIDATION NOISELESS \n")
    for clf, acc in cv_results:
        print(clf + " : " + str(round(acc, 3)) + " \t\t\tError: " + str(round(1 - acc, 3)))
    
    print("\nCROSS VALIDATION NOISE \n")
    for i in range(20, 80, 10):
        print("Noise level: ", i)
        cv_results_noise = cross_validate_clfs_noise(data, classifiers_cv, classifiers_cv_names, meta_clf=SVC, level=i, encode=False)
        print("\n")
        for clf, acc in cv_results_noise:
            print(clf + " : " + str(round(acc, 3)) + " \t\t\tError: " + str(round(1 - acc, 3)))

    print("\nDone")