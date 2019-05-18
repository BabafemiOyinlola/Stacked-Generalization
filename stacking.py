import numpy as np
import pandas as pd

class StackedGeneralization():
    def __init__(self, classifiers, meta_clf):
        self.classifiers = classifiers
        self.meta_clf = meta_clf
    
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test, cv=10, prob=False):
        return

    def base_classifier(self, X_test, folds=10):
        feat_train_set = self.X_train
        lbl_train_set = self.y_train

        all_data = np.column_stack((feat_train_set, lbl_train_set))
        df = pd.DataFrame(all_data)
        df = df.rename(columns = {len(df.columns) - 1:'y_train'})
        df['pred'] = [None] * len(df)
        df = df.sample(frac=1).reset_index(drop=True)

        df_copy = df.copy()

        clf_predictions = []

        fold_size = int(len(feat_train_set)/folds)

        for classifier in self.classifiers:
            start_fold = 0
            end_fold = fold_size

            train_fold, test_fold = None, None
            prediction = None

            for i in range(folds):
                #split into folds
                if i == (folds - 1):
                    test_fold = df_copy[start_fold:]
                    train_fold = df_copy.drop(df_copy.index[start_fold: ])
                else:
                    test_fold = df_copy[start_fold: end_fold]
                    train_fold = df_copy.drop(df_copy.index[start_fold: end_fold])
                
                y_train_fold = train_fold['y_train']
                y_train_fold = list(y_train_fold)
                X_train_fold = train_fold.drop(columns={'y_train', 'pred'})    
                X_test_fold = test_fold.drop(columns={'y_train', 'pred'})

                clf = classifier.fit(X_train_fold, y_train_fold)
                y_pred = clf.predict(X_test_fold)

                if i == (folds - 1):
                    rowIndex = df_copy.index[start_fold: (start_fold +len(test_fold))]
                    df_copy.loc[rowIndex, 'pred'] = y_pred
                else:
                    rowIndex = df_copy.index[start_fold: end_fold]
                    df_copy.loc[rowIndex, 'pred'] = y_pred
                
                start_fold = end_fold
                end_fold = end_fold + fold_size
                prediction = df_copy['pred']
            clf_predictions.append(list(df_copy['pred']))
        

        stacked_cv_predictions = None
        for i in range(len(clf_predictions)):
            if i == len(clf_predictions) - 1: break
            if i == 0:
                stacked_cv_predictions = np.column_stack((clf_predictions[i], clf_predictions[i + 1]))
            else:
                stacked_cv_predictions = np.column_stack((stacked_cv_predictions, clf_predictions[i+1]))

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
