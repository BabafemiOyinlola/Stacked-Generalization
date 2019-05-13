import math
import random
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, KFold

# 1) K-FOLD CROSS VALIDATION
def cross_validate_clfs(data, classifiers_cv, classifiers_cv_names, meta_clf, folds=10, encode=False):
    data = np.array(data)
    y = data[:, -1]
    if encode:
        y = LabelEncoder().fit_transform(y)

    X = data[:, 0:data.shape[1]-1]
    X = StandardScaler().fit_transform(X)

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
       
# 2) NOISE
def add_noise(data, filepath=None, encode_label=False):
    data = np.array(data)
    mu, sigma = 0, 0.1 
    shape = [data.shape[0], 1]
    data_copy = data
    
    for i in range(data.shape[1]):
        if i == data.shape[1] -1: # ignore labels
            data_copy[:, i] = data[:, i]
        else:
            col_max = np.max(data[:, i])
            col_min = np.min(data[:, i])
            
            sigma = (col_max - col_min)
            
            noise = np.random.normal(mu, sigma, shape).reshape(data.shape[0],)
            temp = data[:, i]
            data_copy[:, i] = temp + noise
    data_copy = pd.DataFrame(data_copy)
    return data_copy    

def cross_validate_clfs_noise(noiseless_data, classifiers_cv, classifiers_cv_names, meta_clf, folds=5, level=20, encode=False):
    noiseless_data = pd.DataFrame(noiseless_data)

    noisy_data = add_noise(noiseless_data)

    data_copy = noiseless_data.copy()

    noise_proportion = int((level/100) * len(noiseless_data))
    
    test_data = noiseless_data.iloc[0: noise_proportion]
    noise_to_add = noisy_data[0 : noise_proportion]
    data_copy.iloc[0: noise_proportion] = noise_to_add

    data_with_noise = data_copy
    data_with_noise = data_with_noise.sample(frac=1).reset_index(drop=True) #shuffle
    
    test_data = test_data.sample(frac=1).reset_index(drop=True)

    clfs_accuracies = []

    data_with_noise = np.array(data_with_noise)
    test_data = np.array(test_data)

    y = data_with_noise[:, -1]
    y_test = test_data[:, -1]

    if encode:
        y = LabelEncoder().fit_transform(y)
        y_test = LabelEncoder().fit_transform(y_test)

    X = data_with_noise[:, 0:data_with_noise.shape[1]-1]
    X = StandardScaler().fit_transform(X)

    X_test = test_data[:, 0:test_data.shape[1]-1]
    X_test = StandardScaler().fit_transform(X_test)

    skf = StratifiedKFold(n_splits=folds, shuffle=True)
    for i in range(len(classifiers_cv)):
        acc = []
        for train_index, test_index in skf.split(X, y):
            X_train = X[train_index]
            y_train = y[train_index]
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