import math
import random
import numpy as np
import pandas as pd

class StackedGeneralization():
    '''
    Stacked Generalization ensembles:
    1) Using predictions of each base classifier to train the meta classifier
    2) Using predicted probabilities of labels by base classifiers to train the meta classifer
    '''
    def __init__(self, classifiers, meta_clf):
        self.classifiers = classifiers
        self.meta_clf = meta_clf
        self.clf_name = "STG" #this was defined to identify classifier when using cv
    
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def __base_classifiers(self, X_test, cv=5):
        '''
        #1) Obtain the prediction for each classifier using a cross validation
        #2) Fit all the training items on each classifier
        '''

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

            # split into folds
            for i in range(cv):
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
                
                clf = classifier().fit(X_train_fold, y_train_fold)
                pred = clf.predict(X_test_fold)
                
                if i == (cv - 1):
                    rowIndex = combined_set.index[start_fold: (start_fold +len(test_fold))]
                    combined_set.loc[rowIndex, 'pred'] = pred
                else:
                    rowIndex = combined_set.index[start_fold: end_fold]
                    combined_set.loc[rowIndex, 'pred'] = pred
            
                start_fold = end_fold
                end_fold = end_fold + fold_size
                pred = combined_set['pred']
            all_predictions.append(np.array(pred))

        stacked_cv_predictions = None
        for i in range(len(all_predictions)):
            if i == len(all_predictions) - 1: break
            if i == 0:
                stacked_cv_predictions = np.column_stack((all_predictions[i], all_predictions[i + 1]))
            else:
                stacked_cv_predictions = np.column_stack((stacked_cv_predictions, np.array(all_predictions[i+1])))

        #2) train classifiers on full training set
        full_predictions= []
        for classifier in self.classifiers:
            clf = classifier().fit(self.X_train, self.y_train) #classifiers trained on full training set
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

    def __base_classifiers_prob(self, X_test, cv=5):
        '''
        #1) Obtain the probabilities for predicting a particular class for each classifier using a cross validation
        #2) Fit all the training items on each classifier
        '''

        #1) cv for each classifier
        feat_train_set = self.X_train
        lbl_trian_set = self.y_train
        pred = [None] * len(feat_train_set)

        #combine features and labels to numpy array
        stack = np.column_stack((feat_train_set, lbl_trian_set))
        stack = np.column_stack((stack, pred))

        combined_set = pd.DataFrame(pd.DataFrame(stack))
        combined_set=combined_set.rename(columns = {len(combined_set.columns) - 1:'pred'})
        combined_set=combined_set.rename(columns = {len(combined_set.columns) - 2:'y_train'})

        combined_set_copy = combined_set

        all_predictions = []
        all_preds = np.array([])
        fold_size = int(len(feat_train_set)/cv) 

        for classifier in self.classifiers:
            start_fold = 0
            end_fold = fold_size

            # split into folds then pick
            for i in range(cv):               
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
                
                clf = classifier().fit(X_train_fold, y_train_fold)
                pred = [np.array(i).tolist() for i in clf.predict_proba(X_test_fold)]

                if i == (cv - 1):
                    rowIndex = combined_set.index[start_fold: (start_fold +len(test_fold))]
                    combined_set.loc[rowIndex, 'pred'] = pred
                else:
                    rowIndex = combined_set.index[start_fold: end_fold]
                    combined_set.loc[rowIndex, 'pred'] = pred
            
                start_fold = end_fold
                end_fold = end_fold + fold_size

                pred = combined_set['pred']
            all_predictions.append(pred)
            if all_preds.size != 0:
                all_preds = np.column_stack((all_preds, pred))
            else:
                all_preds = pred

        flattened = np.mean(np.array(all_predictions), axis=0)
        stacked_cv_predictions = flattened #output to used in training meta-classifier

        #2) train classifiers on full training set
        full_predictions= []
        for classifier in self.classifiers:
            clf = classifier().fit(self.X_train, self.y_train) #classifiers trained newly on full training set
            pred = clf.predict_proba(X_test) #predict labels of the test set
            full_predictions.append(pred)

        stacked_test_predictions = np.mean(full_predictions, axis=0) #new test set used to test classifier
        
        return (stacked_cv_predictions, stacked_test_predictions)
    
    def predict(self, X_test, cv_=5, prob=False):
        #train the meta classifer with the results of the predictions from the base classifers using a cross validated method
        if prob:
            predictions = self.__base_classifiers_prob(X_test, cv=cv_)
        else:
            predictions = self.__base_classifiers(X_test, cv=cv_)

        stacked_cv_predictions = predictions[0]
        stacked_test_predictions = predictions[1]

        clf = self.meta_clf()
        clf.fit(stacked_cv_predictions, np.array(self.y_train).reshape(self.y_train.shape[0],-1))
        final_pred = clf.predict(stacked_test_predictions)

        return final_pred