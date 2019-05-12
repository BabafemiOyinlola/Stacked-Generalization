import math
import random
import numpy as np
import pandas as pd
from sklearn.svm import SVC

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

from stacked_generalization import StackedGeneralization

scaler = StandardScaler()
label_encoder = LabelEncoder()

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

def add_noise(data, filepath=None, encode_label=False):
    mu, sigma = 0, 0.1 
    shape = [i for i in data.shape]
    noise = np.random.normal(mu, sigma, shape)
    data_copy = data.copy()
    if encode_label:
        data_copy.iloc[:,-1] = label_encoder.fit_transform(data_copy.iloc[:,-1])

    clean = np.array(data_copy)  
    noise = clean + noise

    noise = np.delete(noise, -1, axis=1)
    noise = np.column_stack((noise, np.array(data)[:, -1]))
    
    noise = pd.DataFrame(noise)
    if filepath is not None:
        noise.to_csv(filepath, header=None, index=False)
        
    return noise
        
def cross_validate_clfs(X, y, classifiers_cv, classifiers_cv_names, meta_clf, folds=5):
    clfs_accuracies = []

    skf = StratifiedKFold(n_splits=folds, shuffle=True)
    for i in range(len(classifiers_cv)):
        acc = []
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            score = 0
            pred = None
            clf = None
            if i == len(classifiers_cv) - 1:
                if hasattr(classifiers_cv[i](classifiers_cv[0:i-1], meta_clf), 'clf_name'):
                    clf = classifiers_cv[i](classifiers_cv[0:i-1], meta_clf)
                else:
                    clf = classifiers_cv[i]()
                clf.fit(X_train, y_train)
                pred = clf.predict(X_test, prob=True)  
            else:
                if hasattr(classifiers_cv[i](classifiers_cv[0:i], meta_clf), 'clf_name'):
                    clf = classifiers_cv[i](classifiers_cv[0:i], meta_clf)
                else:
                    clf = classifiers_cv[i]()
                clf.fit(X_train, y_train)
                pred = clf.predict(X_test)
            score = accuracy_score(y_test, pred)
            acc.append(score)
        clfs_accuracies.append(np.mean(acc))

    combine = zip(classifiers_cv_names, clfs_accuracies)
    return combine

def cross_validate_clfs_noise(X, y, X_noise, y_noise, classifiers_cv, classifiers_cv_names, meta_clf, folds=5):
    clfs_accuracies = []

    skf = StratifiedKFold(n_splits=folds, shuffle=True)
    for i in range(len(classifiers_cv)):
        acc = []
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X_noise[train_index], X[test_index]
            y_train, y_test = y_noise[train_index], y[test_index]
            score = 0
            pred = None
            clf = None
            if i == len(classifiers_cv) - 1:
                if hasattr(classifiers_cv[i](classifiers_cv[0:i-1], meta_clf), 'clf_name'):
                    clf = classifiers_cv[i](classifiers_cv[0:i-1], meta_clf)
                else:
                    clf = classifiers_cv[i]()
                clf.fit(X_train, y_train)
                pred = clf.predict(X_test, prob=True)  
            else:
                if hasattr(classifiers_cv[i](classifiers_cv[0:i], meta_clf), 'clf_name'):
                    clf = classifiers_cv[i](classifiers_cv[0:i], meta_clf)
                else:
                    clf = classifiers_cv[i]()
                clf.fit(X_train, y_train)
                pred = clf.predict(X_test)
            score = accuracy_score(y_test, pred)
            acc.append(score)
        clfs_accuracies.append(np.mean(acc))

    combine = zip(classifiers_cv_names, clfs_accuracies)
    return combine


if __name__ == "__main__":
    # X_train, X_test, y_train, y_test = split_data('data/waveform/waveform.txt', -1)

    knn, dt, svm, nb = KNeighborsClassifier(), DecisionTreeClassifier(), SVC(), GaussianNB()
    lr = LogisticRegression()
    classifiers = [KNeighborsClassifier, LogisticRegression, DecisionTreeClassifier]

    # data = pd.read_csv('data/waveform/waveform.txt', header=None)
    # data = pd.read_csv('data/glass/glass.csv', header=None)
    data = pd.read_csv('data/ionosphere/ionosphere.csv', header=None)
    # data = pd.read_csv('data/breast_cancer/breast_cancer.csv', header=None)
    noise_data = add_noise(data, encode_label=True)
    df_len = int(len(noise_data)/4)
    # noise_to_add = noise_data.iloc[0:df_len]
    # noise_left = noise_data.iloc[df_len:]

    data_copy = data.copy()
    noise_copy  = noise_data.copy()

    data_copy.iloc[0:df_len] = noise_data.iloc[0:df_len]
    data_copy.iloc[df_len:] = data.iloc[df_len:]
    data_copy = data_copy.sample(frac=1).reset_index(drop=True) #shuffle

    noise_copy.iloc[0:df_len] = data.iloc[0:df_len]
    noise_copy.iloc[df_len:] = noise_data.iloc[df_len:]
    noise_copy = noise_copy.sample(frac=1).reset_index(drop=True)

    data_copy = np.array(data_copy)
    y = data_copy[:, -1]
    y = label_encoder.fit_transform(y)

    X = data_copy[:, 0:data.shape[1]-1]
    X = scaler.fit_transform(X)

    noise_copy = np.array(noise_copy)
    y_noise = noise_copy[:, -1]
    y_noise = label_encoder.fit_transform(y)

    X_noise = noise_copy[:, 0:noise_copy.shape[1]-1]
    X_noise = scaler.fit_transform(X_noise)

    stk = StackedGeneralization(classifiers, svm)

    # classifiers_cv = [knn, lr, dt, stk, stk]
    classifiers_cv = [KNeighborsClassifier, LogisticRegression, DecisionTreeClassifier, StackedGeneralization, StackedGeneralization]
    classifiers_cv_names = ["knn", "lr", "dt", "stacked", "stacked_prob"]

    cv_results = cross_validate_clfs(X, y, classifiers_cv, classifiers_cv_names, meta_clf=SVC)
    cv_results_noise = cross_validate_clfs_noise(X, y, X_noise, y_noise, classifiers_cv, classifiers_cv_names, meta_clf=SVC)


    print("CROSS VALIDATION NOISELESS \n")

    for clf, acc in cv_results:
        print(clf + " : " + str(round(acc, 2)) + " \t\t\tError: " + str(round(1 - acc, 2)))

    print("\n\nCROSS VALIDATION NOISE \n")

    for clf, acc in cv_results_noise:
        print(clf + " : " + str(round(acc, 2)) + " \t\t\tError: " + str(round(1 - acc, 2)))

    print("\nDone")


# for i in range(10):
    # dt = load_breast_cancer()
    # X = dt.data
    # y = dt.target

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


    # stacked = StackedGeneralization(classifiers, lr)
    # stacked.fit(X_train, y_train)
    # stk_pred = stacked.predict(X_test, cv=5)

    # stacked_prob = StackedGeneralization(classifiers, lr)
    # stacked_prob.fit(X_train, y_train)
    # stk_pred_prob = stacked_prob.predict(X_test, cv=5, prob=True)

    # KNN = knn.fit(X_train, y_train)
    # knn_pred = KNN.predict(X_test)

    # SVM  = svm.fit(X_train, y_train)
    # svm_pred = SVM.predict(X_test)

    # # NB = nb.fit(X_train, y_train)
    # # nb_pred = NB.predict(X_test)

    # DT = dt.fit(X_train, y_train)
    # dt_pred = DT.predict(X_test)

    # stk_acc = round(accuracy_score(y_test, stk_pred), 2)
    # stk_acc_prob = round(accuracy_score(y_test, stk_pred_prob), 2)
    # knn_acc = round(accuracy_score(y_test, knn_pred), 2)
    # svm_acc = round(accuracy_score(y_test, svm_pred), 2)
    # # nb_acc = round(accuracy_score(y_test, nb_pred), 2)
    # dt_acc = round(accuracy_score(y_test, dt_pred), 2)

    # print("\n\n")
    # print("stk_acc: ", stk_acc)
    # print("stk_acc_prob: ", stk_acc_prob)
    # print("knn_acc: ", knn_acc)
    # print("svm_acc: ", svm_acc)
    # # print("nb_acc: ", nb_acc)
    # print("dt_acc: ", dt_acc)