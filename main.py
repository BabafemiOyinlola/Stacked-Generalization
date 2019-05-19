import math
import random
import numpy as np
import pandas as pd

from sklearn.model_selection import cross_val_predict

from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score

from stacked_generalization import StackedGeneralization
from experiments import cross_validate_clfs, cross_validate_clfs_noise, cross_validate_kappa, \
                        q_statistic_and_correlation, classifer_diversity


if __name__ == "__main__":
    #base classifiers - level 0
    knn = KNeighborsClassifier(n_neighbors=1)
    dt = DecisionTreeClassifier(max_depth=1)
    mlp = MLPClassifier(hidden_layer_sizes=(10, 1))

    #meta classifier - level 1
    lr = LogisticRegression()
    
    stk = StackedGeneralization()
    stk_p = StackedGeneralization()

    classifiers_cv = [mlp, dt, knn, stk, stk_p]
    classifiers_cv_names = ["MLP", "DT", "KNN", "STK", "STK_PR"]


    # data = pd.read_csv('data/hepatitis/hepatitis.csv', header=None)
    
    #binary 
    # data = pd.read_csv('data/sonar/sonar.csv', header=None)
    data = pd.read_csv('data/ionosphere/ionosphere.csv', header=None)
    # data = pd.read_csv('data/breast_cancer/breast_cancer.csv', header=None)
    # data = pd.read_csv('data/iris/iris.csv', header=None)
    # data = pd.read_csv('data/diabeties/diabetes.csv', header=None)
    

    data = data.sample(frac=1).reset_index(drop=True) #shuffle


    # cv_results = cross_validate_clfs(data, classifiers_cv, classifiers_cv_names, meta_clf=LogisticRegression(), encode=True)
    # print("\nCROSS VALIDATION - NOISELESS DATA \n")
    # for clf, acc in cv_results:
    #     print(clf + " : " + str(round(acc, 3)) + " \t\t\tError: " + str(round(1 - acc, 3)))
    

    # print("\nCROSS VALIDATION - NOISY DATA \n")
    # for i in range(5, 30, 5):
    #     print("\n")
    #     print("Noise level: ", i)
    #     cv_results_noise = cross_validate_clfs_noise(data, classifiers_cv, classifiers_cv_names, meta_clf=LogisticRegression(), level=i, encode=False)
    #     for clf, acc in cv_results_noise:
    #         print(clf + " : " + str(round(acc, 3)) + " \t\t\tError: " + str(round(1 - acc, 3)))

    classifer_diversity(data, mlp, dt, knn, encode=True)

    #various combinations of base classifers 
    cmb1, cmb1_names = [mlp, dt, stk, stk_p], ["MLP", "DT", "STK", "STK_PR"]
    cmb2, cmb2_names = [dt, knn, stk, stk_p], ["DT", "KNN", "STK", "STK_PR"]
    cmb3, cmb3_names = [mlp, knn, stk, stk_p], ["MLP", "KNN", "STK", "STK_PR"]


    cv_results = cross_validate_clfs(data, cmb1, cmb1_names, meta_clf=LogisticRegression(), encode=True)
    print("\nCROSS VALIDATION - DIFFERENT CLASSIFIER COMBINATION 1: MLP & DT \n")
    for clf, acc in cv_results:
        print(clf + " : " + str(round(acc, 3)) + " \t\t\tError: " + str(round(1 - acc, 3)))

    cv_results = cross_validate_clfs(data, cmb2, cmb2_names, meta_clf=LogisticRegression(), encode=True)
    print("\nCROSS VALIDATION - DIFFERENT CLASSIFIER COMBINATION  2: DT & KNN \n")
    for clf, acc in cv_results:
        print(clf + " : " + str(round(acc, 3)) + " \t\t\tError: " + str(round(1 - acc, 3)))

    cv_results = cross_validate_clfs(data, cmb3, cmb3_names, meta_clf=LogisticRegression(), encode=True)
    print("\nCROSS VALIDATION - DIFFERENT CLASSIFIER COMBINATION 3: MLP & KNN \n")
    for clf, acc in cv_results:
        print(clf + " : " + str(round(acc, 3)) + " \t\t\tError: " + str(round(1 - acc, 3)))

    print("\nDone")