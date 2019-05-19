import math
import random
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, cohen_kappa_score

# 1) K-FOLD CROSS VALIDATION
def cross_validate_clfs(data, classifiers_cv, classifiers_cv_names, meta_clf, folds=5, encode=False):
    std_scaler = StandardScaler()
    lbl_encoder = LabelEncoder()

    data = np.array(data)
    y = data[:, -1]

    X = data[:, 0:data.shape[1]-1]

    clfs_accuracies = []
    
    clfs = len(classifiers_cv) - 2
    
    skf = StratifiedKFold(n_splits=folds, shuffle=True)
    for i in range(len(classifiers_cv)):
        acc = []
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            X_train = std_scaler.fit_transform(X_train)
            X_test = std_scaler.transform(X_test)
            y_train, y_test = y[train_index], y[test_index]
            if encode:
                y_train = lbl_encoder.fit_transform(y_train)
                y_test = lbl_encoder.transform(y_test)
            score = 0
            pred = None
            clf = None
            if i == len(classifiers_cv) - 1:
                if hasattr(classifiers_cv[i], 'clf_name'):
                    classifiers_cv[i].classifiers = classifiers_cv[0:clfs]
                    classifiers_cv[i].meta_clf = meta_clf
                clf = classifiers_cv[i]
                clf.fit(X_train, y_train)
                pred = clf.predict(X_test, prob=True)
            else:
                if hasattr(classifiers_cv[i], 'clf_name'):
                    classifiers_cv[i].classifiers = classifiers_cv[0:clfs]
                    classifiers_cv[i].meta_clf = meta_clf
                clf = classifiers_cv[i]
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
            
            sigma = (col_max - col_min)/0.5
            
            noise = np.random.normal(mu, sigma, shape).reshape(data.shape[0],)
            temp = data[:, i]
            data_copy[:, i] = temp + noise
    data_copy = pd.DataFrame(data_copy)
    return data_copy    

def cross_validate_clfs_noise(noiseless_data, classifiers_cv, classifiers_cv_names, meta_clf, folds=5, level=20, encode=False):
    std_scaler = StandardScaler()
    lbl_encoder  = LabelEncoder()

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
    y = y.reshape(data_with_noise.shape[0],)

    y_test = test_data[:, -1]
    y_test = y_test.reshape(test_data.shape[0],)

    if encode:
        y = lbl_encoder.fit_transform(y)
        y_test = lbl_encoder.transform(y_test)

    X = data_with_noise[:, 0:data_with_noise.shape[1]-1]
    X = std_scaler.fit_transform(X)

    X_test = test_data[:, 0:test_data.shape[1]-1]
    X_test = std_scaler.transform(X_test)

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
                if hasattr(classifiers_cv[i], 'clf_name'):
                    classifiers_cv[i].classifiers = classifiers_cv[0:i-1]
                    classifiers_cv[i].meta_clf = meta_clf
                clf = classifiers_cv[i]
                clf.fit(X_train, y_train)
                pred = clf.predict(X_test, prob=True)  
            else:
                if hasattr(classifiers_cv[i], 'clf_name'):
                    classifiers_cv[i].classifiers = classifiers_cv[0:i-1]
                    classifiers_cv[i].meta_clf = meta_clf
                clf = classifiers_cv[i]
                clf.fit(X_train, y_train)
                pred = clf.predict(X_test)
            score = accuracy_score(y_test, pred)
            acc.append(score)
        clfs_accuracies.append(np.mean(acc))

    combine = zip(classifiers_cv_names, clfs_accuracies)
    return combine

def cross_validate_kappa(data, classifiers_cv, classifiers_cv_names, meta_clf, folds=10, encode=False):
    data = np.array(data)
    y = data[:, -1]
    if encode:
        y = LabelEncoder().fit_transform(y)
    y = y.reshape(data.shape[0],)


    X = data[:, 0:data.shape[1]-1]
    X = StandardScaler().fit_transform(X)

    kappa_scores = []

    skf = StratifiedKFold(n_splits=folds, shuffle=True)
    for i in range(len(classifiers_cv)):
        kappa = []
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            score = 0
            pred = None
            clf = None
            if i == len(classifiers_cv) - 1:
                if hasattr(classifiers_cv[i], 'clf_name'):
                    clf = classifiers_cv[i](classifiers_cv[0:i-1], meta_clf)
                else:
                    clf = classifiers_cv[i]
                clf.fit(X_train, y_train)
                pred = clf.predict(X_test, prob=True)  
            else:
                if hasattr(classifiers_cv[i], 'clf_name'):
                    clf = classifiers_cv[i](classifiers_cv[0:i], meta_clf)
                else:
                    clf = classifiers_cv[i]
                clf.fit(X_train, y_train)
                pred = clf.predict(X_test)
            score = cohen_kappa_score(y_test, pred)
            kappa.append(score)
        kappa_scores.append(np.mean(kappa))

    combine = zip(classifiers_cv_names, kappa_scores)
    return combine

# 3) DIVERSITY
def q_statistic_and_correlation(clf1_pred, clf2_pred, y_true):
    '''
    Q is a measure of diversity
    create a 2x2 contigency table from the predictions obtained from both classifiers
    if the predicted label obtained is same as the true label, label == 1 otherwise, 0
    '''
    clf1_label = []
    for i in range(len(clf1_pred)):
        if clf1_pred[i] == y_true[i]:
            temp = 1
        else:
            temp = 0
        clf1_label.append(temp) 

    clf2_label = []
    for i in range(len(clf2_pred)):
        if clf2_pred[i] == y_true[i]:
            temp = 1
        else:
            temp = 0        
        clf2_label.append(temp) 

    contigency_table = np.empty((2,2))
    correct_correct = 0
    incorrect_correct = 0
    correct_incorrect = 0
    incorrect_incorrect = 0

    for i in range(len(clf1_label)):
        if clf1_label[i] == 1 and clf2_label[i] == 1:
            correct_correct += 1
        elif clf1_label[i] == 0 and clf2_label[i] == 1:
            incorrect_correct += 1
        elif clf1_label[i] == 1 and clf2_label[i] == 0:
            correct_incorrect += 1
        elif clf1_label[i] == 0 and clf2_label[i] == 0:
            incorrect_incorrect += 1

    contigency_table[0, 0] = correct_correct
    contigency_table[1, 0] = incorrect_correct
    contigency_table[0, 1] = correct_incorrect
    contigency_table[1, 1] = incorrect_incorrect

    num = (contigency_table[0, 0] * contigency_table[1, 1]) - (contigency_table[1, 0] * contigency_table[0, 1])
    q_den = (contigency_table[0, 0] * contigency_table[1, 1]) + (contigency_table[1, 0] * contigency_table[0, 1]) #q denominator
    
    ab_cd = (contigency_table[0, 0] + contigency_table[1, 0]) * (contigency_table[0, 1] + contigency_table[1, 1])
    ac_bd = (contigency_table[0, 0] + contigency_table[0, 1]) * (contigency_table[1, 0] + contigency_table[1, 1])
    
    Q = round(num / q_den, 3)

    correlation = round(num / (math.sqrt(ab_cd * ac_bd)), 3)

    return Q, correlation

def classifer_diversity(data, mlp, dt, knn, encode=False):
    data = np.array(data)
    y = data[:, -1]
    X = data[:, 0:data.shape[1]-1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    label_encoder = LabelEncoder()
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    if encode:
        y_train = label_encoder.fit_transform(y_train)
        y_test = label_encoder.transform(y_test)

    mlp.fit(X_train, y_train)
    mlp_pred = mlp.predict(X_test)
    
    dt.fit(X_train, y_train)
    dt_pred = dt.predict(X_test)

    knn.fit(X_train, y_train)
    knn_pred = knn.predict(X_test)

    mlp_dt = q_statistic_and_correlation(mlp_pred, dt_pred, y_test)
    dt_knn = q_statistic_and_correlation(dt_pred, knn_pred, y_test)
    mlp_knn = q_statistic_and_correlation(mlp_pred, knn_pred, y_test)

    mlp_dt_Q, mlp_dt_corr = mlp_dt[0], mlp_dt[1]
    dt_knn_Q, dt_knn_corr = dt_knn[0], dt_knn[1]
    mlp_knn_Q, mlp_knn_corr = mlp_knn[0], mlp_knn[1]


    print("\n\nMLP & DT Q: " + str(mlp_dt_Q) + " \t\t\tMLP & DT corr: " + str(mlp_dt_corr))
    print("DT & KNN Q: " + str(dt_knn_Q) + " \t\t\tDT & KNN corr: " + str(dt_knn_corr))
    print("MLP & KNN Q: " + str(mlp_knn_Q) + " \t\t\tMLP & KNN corr: " + str(mlp_knn_corr))


