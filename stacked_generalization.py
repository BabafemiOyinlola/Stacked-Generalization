import math
import random
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer


class StackedGeneralization():
    def __init__(self, classifiers, meta_clf):
        self.classifiers = classifiers
        self.meta_clf = meta_clf
    
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def base_classifiers(self, X_test, cv=5): #MAKE METHOD PRIVATE
        #1) Obtain the prediction for each classifier using a cross validation
        #2) Fit all the training items on each classifier

        #1) cv for each classifier
        feat_train_set = self.X_train
        lbl_trian_set = self.y_train
        pred = [None] * len(feat_train_set)
        stack = np.column_stack((feat_train_set, lbl_trian_set))
        stack = np.column_stack((stack, pred))

        combined_set = pd.DataFrame(pd.DataFrame(stack))
        combined_set=combined_set.rename(columns = {len(combined_set.columns) - 1:'pred'})
        combined_set=combined_set.rename(columns = {len(combined_set.columns) - 2:'y_train'})
        combined_set_copy = combined_set

        all_predictions = []
        
        fold_size = int(len(feat_train_set)/cv)

        for classifier in self.classifiers:
            start_fold = 0
            end_fold = fold_size

            # split into folds then pick
            for i in range(cv):
                #split into folds
                print("Start fold: ", start_fold)
                print("End fold: ", end_fold)
                
                if i == (cv - 1):
                    test_fold = combined_set_copy[start_fold:]
                    train_fold = combined_set_copy.drop(combined_set_copy.index[start_fold: ])
                else:
                    test_fold = combined_set_copy[start_fold: end_fold]
                    train_fold = combined_set_copy.drop(combined_set_copy.index[start_fold: end_fold])
                
                X_train_fold = train_fold.drop(columns={'y_train', 'pred'})
                y_train_fold = train_fold['y_train']
                y_train_fold = list(y_train_fold)
    
                X_test_fold = test_fold.drop(columns={'y_train', 'pred'})
                
                clf = classifier.fit(X_train_fold, y_train_fold)
                pred = clf.predict(X_test_fold)
                
                if i == (cv - 1):
                    rowIndex = combined_set.index[start_fold: (start_fold +len(test_fold))]
                    combined_set.loc[rowIndex, 'pred'] = pred
                    print(rowIndex)
                else:
                    rowIndex = combined_set.index[start_fold: end_fold]
                    combined_set.loc[rowIndex, 'pred'] = pred
                    print(rowIndex)
            
                start_fold = end_fold
                end_fold = end_fold + fold_size
                pred = combined_set['pred']
            all_predictions.append(pred)

        stacked_cv_predictions = None
        for i in range(len(all_predictions)):
            if i == len(all_predictions) - 1: break
            if i == 0:
                stacked_cv_predictions = np.column_stack((all_predictions[i], all_predictions[i + 1]))
            else:
                stacked_cv_predictions = np.column_stack((stacked_cv_predictions, all_predictions[i+1]))

        #2) train classifiers on full training set
        full_predictions= []
        for classifier in self.classifiers:
            clf = clf.fit(self.X_train, self.y_train) #classifiers trained on full training set
            pred = clf.predict(X_test) #predict labels of the test set
            full_predictions.append(pred)

        stacked_test_predictions = None 
        for i in range(len(full_predictions)):
            if i == len(full_predictions) - 1: break
            if i == 0:
                stacked_test_predictions = np.column_stack((full_predictions[i], full_predictions[i + 1]))
            else:
                stacked_test_predictions = np.column_stack((stacked_test_predictions, full_predictions[i+1]))

        return (stacked_cv_predictions, stacked_test_predictions)

    #RE-WRITE
    def base_classifiers_prob(self, X_test, cv=5): #MAKE METHOD PRIVATE
        #1) Obtain the prediction for each classifier using a cross validation
        #2) Fit all the training items on each classifier

        #1) cv for each classifier
        feat_train_set = self.X_train
        lbl_trian_set = self.y_train
        pred = [None] * len(feat_train_set)
        stack = np.column_stack((feat_train_set, lbl_trian_set))
        stack = np.column_stack((stack, pred))

        combined_set = pd.DataFrame(pd.DataFrame(stack))
        combined_set=combined_set.rename(columns = {len(combined_set.columns) - 1:'pred'})
        combined_set=combined_set.rename(columns = {len(combined_set.columns) - 2:'y_train'})
        combined_set_copy = combined_set

        all_predictions = []
        
        fold_size = int(len(feat_train_set)/cv)

        for classifier in self.classifiers:
            start_fold = 0
            end_fold = fold_size

            # split into folds then pick
            for i in range(cv):
                #split into folds
                print("Start fold: ", start_fold)
                print("End fold: ", end_fold)
                
                if i == (cv - 1):
                    test_fold = combined_set_copy[start_fold:]
                    train_fold = combined_set_copy.drop(combined_set_copy.index[start_fold: ])
                else:
                    test_fold = combined_set_copy[start_fold: end_fold]
                    train_fold = combined_set_copy.drop(combined_set_copy.index[start_fold: end_fold])
                
                X_train_fold = train_fold.drop(columns={'y_train', 'pred'})
                y_train_fold = train_fold['y_train']
                y_train_fold = list(y_train_fold)
    
                X_test_fold = test_fold.drop(columns={'y_train', 'pred'})
                
                clf = classifier.fit(X_train_fold, y_train_fold)
                pred = clf.predict_proba(X_test_fold)
                
                if i == (cv - 1):
                    rowIndex = combined_set.index[start_fold: (start_fold +len(test_fold))]
                    combined_set.loc[rowIndex, 'pred'] = pred
                    print(rowIndex)
                else:
                    rowIndex = combined_set.index[start_fold: end_fold]
                    combined_set.loc[rowIndex, 'pred'] = pred
                    print(rowIndex)
            
                start_fold = end_fold
                end_fold = end_fold + fold_size
                pred = combined_set['pred']
            all_predictions.append(pred)

        stacked_cv_predictions = None
        for i in range(len(all_predictions)):
            if i == len(all_predictions) - 1: break
            if i == 0:
                stacked_cv_predictions = np.column_stack((all_predictions[i], all_predictions[i + 1]))
            else:
                stacked_cv_predictions = np.column_stack((stacked_cv_predictions, all_predictions[i+1]))

        #2) train classifiers on full training set
        full_predictions= []
        for classifier in self.classifiers:
            clf = clf.fit(self.X_train, self.y_train) #classifiers trained on full training set
            pred = clf.predict_proba(X_test) #predict labels of the test set
            full_predictions.append(pred)

        stacked_test_predictions = None 
        for i in range(len(full_predictions)):
            if i == len(full_predictions) - 1: break
            if i == 0:
                stacked_test_predictions = np.column_stack((full_predictions[i], full_predictions[i + 1]))
            else:
                stacked_test_predictions = np.column_stack((stacked_test_predictions, full_predictions[i+1]))

        return (stacked_cv_predictions, stacked_test_predictions)

    def predict(self, X_test, cv=5):
        #train the meta classifer with the results of the predictions from the base classifers using a cross validated method
        predictions = self.base_classifiers_prob(X_test, cv=cv)

        stacked_cv_predictions = predictions[0]
        stacked_test_predictions = predictions[1]

        clf = self.meta_clf
        clf.fit(stacked_cv_predictions, self.y_train)
        final_pred = clf.predict(stacked_test_predictions)

        return final_pred

    def accuracy(self, y_true, y_test):
        correct = []
        for i in range(len(y_test) - 1):
            if y_test[i] == y_true[i]:
                correct.append(1)
            else:
                correct.append(0)
        match = sum(correct)
        acc = (match / len(y_test)) * 100
        acc = round(acc, 2)
        return acc


dt = load_breast_cancer()
X = dt.data
y = dt.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

knn, dt = KNeighborsClassifier(), DecisionTreeClassifier()
lr = LogisticRegression()
classifiers = [knn, dt]

stacked = StackedGeneralization(classifiers, lr)
stacked.fit(X_train, y_train)
max_folds = 20

accuracy = []

for i in range(2, max_folds+1):
    pred = stacked.predict(X_test, i)
    acc = stacked.accuracy(pred, y_test)
    accuracy.append(acc)


print("Accurracy: ", accuracy)
print("\nDone")
