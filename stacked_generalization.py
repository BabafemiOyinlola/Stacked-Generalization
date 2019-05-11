import numpy as np
import pandas as pd

class StackedGeneralization():
    def __init__(self, classifiers, meta_clf):
        self.classifiers = classifiers
        self.meta_clf = meta_clf
    
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    # def base_classifiers(self, X_test, cv=10): #MAKE METHOD PRIVATE
        
    def base_classifiers(self, X_test, folds=10):
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

    
    #RE-WRITE
    def base_classifiers_prob(self, X_test, cv=10): #MAKE METHOD PRIVATE
        #1) Obtain the prediction for each classifier using a cross validation
        #2) Fit all the training items on each classifier

        #1) cv for each classifier
        feat_train_set = self.X_train
        lbl_trian_set = self.y_train

        pred = [None] * len(feat_train_set)
        stack = np.column_stack((feat_train_set, lbl_trian_set))
        stack = np.column_stack((stack, pred))
        # np.random.shuffle(stack)

        combined_set = pd.DataFrame(pd.DataFrame(stack))
        combined_set=combined_set.rename(columns = {len(combined_set.columns) - 1:'pred'})
        combined_set=combined_set.rename(columns = {len(combined_set.columns) - 2:'y_train'})
        # combined_set= combined_set.sample(frac=1).reset_index(drop=True)
       
        combined_set_copy = combined_set.copy()

        all_predictions = []
        all_preds = np.array([])
        fold_size = int(len(feat_train_set)/cv)

        for classifier in self.classifiers:
            start_fold = 0
            end_fold = fold_size
            prediction = None
            # split into folds then pick
            for i in range(cv):
                #split into folds           
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
                
                clf = classifier
                if hasattr(clf, 'probability'):
                    clf.probability = True
                
                clf.fit(X_train_fold, y_train_fold)
                pred = [np.array(i).tolist() for i in clf.predict_proba(X_test_fold)]

                if i == (cv - 1):
                    rowIndex = combined_set.index[start_fold: (start_fold +len(test_fold))]
                    combined_set.loc[rowIndex, 'pred'] = pred
                else:
                    rowIndex = combined_set.index[start_fold: end_fold]
                    combined_set.loc[rowIndex, 'pred'] = pred
            
                start_fold = end_fold
                end_fold = end_fold + fold_size

                prediction = combined_set['pred']
            all_predictions.append(prediction)


        # flattened = []
        flattened = np.mean(np.array(all_predictions), axis=0)
        # for i in range(len(all_preds)):
        #     item = all_preds[i]
        #     test = np.array([])
        #     item = np.array([np.array(i) for i in item])
            # for i in item:
            #     c = np.array(i)
            #     test.append(c)
            # flat = []
            # for j in range(len(item)):
            #     flat.extend(item[j])
            # flat = np.array(flat).reshape((1,-1))
            # if flattened.size != 0:
            #     flattened = np.vstack([flattened, flat])
            # else:
                # flattened = np.array(flat)
            # flat = np.mean(item, axis=0)
            # flattened.append(flat)

        # flattened = np.array(flattened)
        
        stacked_cv_predictions = flattened #train meta-classifier with stacked probabilities

        #2) train classifiers on full training set
        full_predictions= []
        for classifier in self.classifiers:
            clf = clf.fit(self.X_train, self.y_train) #classifiers trained on full training set
            pred = clf.predict_proba(X_test) #predict labels of the test set
            full_predictions.append(pred)

        # stacked_test_predictions = None 
        stacked_test_predictions = np.mean(full_predictions, axis=0)
        # for i in range(len(full_predictions)):
        #     if i == len(full_predictions) - 1: break
        #     if i == 0:
        #         stacked_test_predictions = np.column_stack((full_predictions[i], full_predictions[i + 1]))
        #     else:
        #         stacked_test_predictions = np.column_stack((stacked_test_predictions, full_predictions[i+1]))

        return (stacked_cv_predictions, stacked_test_predictions)

    #try on different subsets
    def predict(self, X_test, cv_=10, prob=False):
        #train the meta classifer with the results of the predictions from the base classifers using a cross validated method
        if prob:
            predictions = self.base_classifiers_prob(X_test, cv=cv_)
        else:
            predictions = self.base_classifiers(X_test, folds=cv_)

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


