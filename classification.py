import pandas as pd
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from collections import Counter
import time
import csv

class Classification():

    def __init__(self):
        # data preparation
        data = pd.read_csv('data/training.csv')
        df_test = pd.read_csv('data/testing.csv')
        m_data = data.shape[0]
        df_train, df_valid = data[0:int(0.8*m_data)], data[int(0.8*m_data)+1:m_data]
        df_valid = df_valid.reset_index(drop=True)

        df = pd.read_csv('data/race-result-horse.csv')
        m, n = np.shape(df)
        m_train, _ = np.shape(df_train)
        m_valid, _ = np.shape(df_valid)
        m_test, _ = np.shape(df_test)

        #produce dicts for horse,jockey,trainer
        horse=list()
        jockey=list()
        trainer=list()
        for i in range(len(df)):
            if df.horse_name[i] not in horse:
                horse.append(df.horse_name[i])
            if df.jockey[i] not in jockey:
                jockey.append(df.jockey[i])
            if df.trainer[i] not in jockey:
                trainer.append(df.trainer[i])
        train_horse=np.zeros((m_train,1))
        train_jockey=np.zeros((m_train,1))
        train_trainer=np.zeros((m_train,1))

        # prepare training data
        for i in range(m_train):
            train_horse[i,0] = horse.index(df_train.horse_name[i])
            train_jockey[i,0] = jockey.index(df_train.jockey[i])
            train_trainer[i,0] = trainer.index(df_train.trainer[i])
        actual_weight = df_train.actual_weight.values.reshape((m_train,1))
        declared_weight = df_train.declared_horse_weight.values.reshape((m_train,1))
        draw = df_train.draw.values.reshape((m_train,1))
        win_odds = df_train.win_odds.values.reshape((m_train,1))
        race_distance = df_train.race_distance.values.reshape((m_train,1))

        # we use horse, jockey, trainer, actual weight, declared weight, win odds, race distance as independent variables
        self.X_train = np.hstack((train_horse, train_jockey, train_trainer, actual_weight,
                             declared_weight, draw, win_odds, race_distance))
        self.y_train = df_train.finishing_position

        # prepare valid data
        valid_horse=np.zeros((m_valid,1))
        valid_jockey=np.zeros((m_valid,1))
        valid_trainer=np.zeros((m_valid,1))

        #print(df_valid.horse_name[])
        for i in range(len(df_valid)):
            valid_horse[i,0] = horse.index(df_valid.horse_name[i])
            valid_jockey[i,0] = jockey.index(df_valid.jockey[i])
            valid_trainer[i,0] = trainer.index(df_valid.trainer[i])
        actual_weight_valid = df_valid.actual_weight.values.reshape((m_valid,1))
        declared_weight_valid = df_valid.declared_horse_weight.values.reshape((m_valid,1))
        draw_valid = df_valid.draw.values.reshape((m_valid,1))
        win_odds_valid = df_valid.win_odds.values.reshape((m_valid,1))
        race_distance_valid = df_valid.race_distance.values.reshape((m_valid,1))
        self.X_valid = np.hstack((valid_horse,valid_jockey,valid_trainer,actual_weight_valid,
                            declared_weight_valid, draw_valid, win_odds_valid, race_distance_valid))
        self.y_valid = df_valid.finishing_position

        # prepare test data
        test_horse=np.zeros((m_test,1))
        test_jockey=np.zeros((m_test,1))
        test_trainer=np.zeros((m_test,1))

        #print(df_test.horse_name[])
        for i in range(len(df_test)):
            test_horse[i,0] = horse.index(df_test.horse_name[i])
            test_jockey[i,0] = jockey.index(df_test.jockey[i])
            test_trainer[i,0] = trainer.index(df_test.trainer[i])
        actual_weight_test = df_test.actual_weight.values.reshape((m_test,1))
        declared_weight_test = df_test.declared_horse_weight.values.reshape((m_test,1))
        draw_test = df_test.draw.values.reshape((m_test,1))
        win_odds_test = df_test.win_odds.values.reshape((m_test,1))
        race_distance_test = df_test.race_distance.values.reshape((m_test,1))
        self.X_test = np.hstack((test_horse, test_jockey, test_trainer,
            actual_weight_test, declared_weight_test, draw_test, 
            win_odds_test, race_distance_test))
        self.y_test = df_test.finishing_position

        self.headers=['RaceID','HorseID','HorseWin','HorseRankTop3','HorseRankTop50Percent']

        #get actual results from y_valid
        horse_win_actual=np.zeros((len(self.y_valid),1))
        horse_top3_actual=np.zeros((len(self.y_valid),1))
        horse_top50percent_actual=np.zeros((len(self.y_valid),1))
        count_of_race_participation=Counter(df_valid.race_id)
        for i in range(len(self.y_valid)):
            if self.y_valid[i]==1:
                 horse_win_actual[i]=1
            if self.y_valid[i]<=3:
                 horse_top3_actual[i]=1
            if self.y_valid[i]<=np.floor(count_of_race_participation[df_valid.race_id[i]]/2):
                 horse_top50percent_actual[i]=1

        self.horse_win_actual = horse_win_actual 
        self.horse_top3_actual = horse_top3_actual 
        self.horse_top50percent_actual = horse_top50percent_actual 


    def logistic(self):
        # we get the good parameter by following code commented
        # And after finding a good parameter we comment them to faster the
        # speed
        """
        parameters = {'penalty': ['l1', 'l2'], 'C': [0.06, 0.08, 0.1, 1], 'random_state': [0]}
        lr = LogisticRegression()
        clf = GridSearchCV(lr, parameters)
        clf.fit(self.X_train, self.y_train)
        print(clf.best_estimator_)
        """

        start_time = time.time()
        lr_model = LogisticRegression(C=1, class_weight=None, dual=False,
                fit_intercept=True, intercept_scaling=1, max_iter=100, 
                solver='liblinear', tol=0.0001, verbose=0, 
                warm_start=False)
        lr_model.fit(self.X_train,self.y_train)
        print('Training time of linear regression:', time.time()-start_time, 'secends')
        print('Score of linear regression:', lr_model.score(self.X_valid, self.y_valid))
        self.lr_y_pred = lr_model.predict(self.X_test)
        
        # get predict
        df_test = pd.read_csv('data/testing.csv')
        lr_result = self.lr_y_pred
        horse_win_lr=np.zeros((len(lr_result),1))
        horse_top3_lr=np.zeros((len(lr_result),1))
        horse_top50percent_lr=np.zeros((len(lr_result),1))
        count_of_race_participation=Counter(df_test.race_id)
        for i in range(len(lr_result)):
            if lr_result[i] == 1:
                 horse_win_lr[i]=1
            if lr_result[i] <= 3:
                 horse_top3_lr[i] = 1
            if lr_result[i]<= np.floor(count_of_race_participation[df_test.race_id[i]]/2):
                 horse_top50percent_lr[i] = 1
        with open('data/lr_predictions.csv','w') as f1:
             lr_csv=csv.writer(f1)
             lr_csv.writerow(self.headers)
             for i in range(len(lr_result)):
                 lr_csv.writerow([df_test.race_id[i],df_test.horse_id[i],horse_win_lr[i][0],horse_top3_lr[i][0],horse_top50percent_lr[i][0]])

        self.evaluation(horse_win_lr, horse_top3_lr, 
                horse_top50percent_lr, 'logistic')





    def naiveBayes(self):
        start_time = time.time()
        nb_model = sklearn.naive_bayes.GaussianNB()
        nb_model.fit(self.X_train, self.y_train)
        print('Training time of Naive Bayes:', time.time()-start_time, 'secends')
        nb_score = nb_model.score(self.X_test, self.y_test)
        print('Score of Naive Bayes:', nb_score)
        self.nb_y_pred = nb_model.predict(self.X_test)

        df_test = pd.read_csv('data/testing.csv')
        nb_result = self.nb_y_pred
        horse_win_nb=np.zeros((len(nb_result),1))
        horse_top3_nb=np.zeros((len(nb_result),1))
        horse_top50percent_nb=np.zeros((len(nb_result),1))
        count_of_race_participation=Counter(df_test.race_id)
        for i in range(len(nb_result)):
            if nb_result[i] == 1:
                horse_win_nb[i]=1
            if nb_result[i] <= 3:
                horse_top3_nb[i] = 1
            if nb_result[i]<= np.floor(count_of_race_participation[df_test.race_id[i]]/2):
                horse_top50percent_nb[i] = 1
        with open('data/nb_predictions.csv','w') as f2:
             nb_csv=csv.writer(f2)
             nb_csv.writerow(self.headers)
             for i in range(len(nb_result)):
                 nb_csv.writerow([df_test.race_id[i],df_test.horse_id[i],horse_win_nb[i][0],horse_top3_nb[i][0],horse_top50percent_nb[i][0]])

        self.evaluation(horse_win_nb, horse_top3_nb, 
                horse_top50percent_nb, 'Naive Bayes')

    # TODO
    def supportVector(self):
        self.svm_y_pred = self.y_test

        df_test = pd.read_csv('data/testing.csv')
        svm_result = self.svm_y_pred
        horse_win_svm=np.zeros((len(svm_result),1))
        horse_top3_svm=np.zeros((len(svm_result),1))
        horse_top50percent_svm=np.zeros((len(svm_result),1))
        count_of_race_participation=Counter(df_test.race_id)
        for i in range(len(svm_result)):
            if svm_result[i] == 1:
                horse_win_svm[i]=1
            if svm_result[i] <= 3:
                horse_top3_svm[i] = 1
            if svm_result[i]<= np.floor(count_of_race_participation[df_test.race_id[i]]/2):
                horse_top50percent_svm[i] = 1
        with open('data/svm_predictions.csv','w') as f3:
            svm_csv=csv.writer(f3)
            svm_csv.writerow(self.headers)
            for i in range(len(svm_result)):
                svm_csv.writerow([df_test.race_id[i],df_test.horse_id[i],horse_win_svm[i][0],horse_top3_svm[i][0],horse_top50percent_svm[i][0]])

        self.evaluation(horse_win_svm, horse_top3_svm, 
                horse_top50percent_svm, 'SVM')

    def randomForest(self):
        # we get the good parameter by following code commented
        # And after finding a good parameter we comment them to faster the
        # speed
        """
        parameters = {'n_estimators': [30, 50, 100], 
            'max_features': ['sqrt', 'log2', None], 
            'max_depth': [None,1, 2, 5], 'random_state': [0]}
        raf = RandomForestClassifier()
        clf = GridSearchCV(raf, parameters)
        clf.fit(X_train, y_train)
        print('finish')
        sorted(clf.cv_results_.keys())
        print(clf.best_params_)
        raf_model = clf.best_estimator_
        """
        start_time = time.time()
        rf_model = RandomForestClassifier(max_depth=2, max_features=None,
            n_estimators=50, random_state=0)
        rf_model.fit(self.X_train, self.y_train)
        print('Training time of random forest:', 
            time.time()-start_time, 'secends')
        print('Score of random forest:', rf_model.score(self.X_valid,
            self.y_valid))
        self.rf_y_pred = rf_model.predict(self.X_test)

        df_test = pd.read_csv('data/testing.csv')
        rf_result = self.rf_y_pred
        horse_win_rf=np.zeros((len(rf_result),1))
        horse_top3_rf=np.zeros((len(rf_result),1))
        horse_top50percent_rf=np.zeros((len(rf_result),1))
        count_of_race_participation=Counter(df_test.race_id)
        for i in range(len(rf_result)):
            if rf_result[i] == 1:
                horse_win_rf[i]=1
            if rf_result[i] <= 3:
                horse_top3_rf[i] = 1
            if rf_result[i]<= np.floor(count_of_race_participation[df_test.race_id[i]]/2):
                horse_top50percent_rf[i] = 1
        with open('data/rf_predictions.csv','w') as f4:
            rf_csv=csv.writer(f4)
            rf_csv.writerow(self.headers)
            for i in range(len(rf_result)):
                rf_csv.writerow([df_test.race_id[i],df_test.horse_id[i],horse_win_rf[i][0],horse_top3_rf[i][0],horse_top50percent_rf[i][0]])
    
        self.evaluation(horse_win_rf, horse_top3_rf, 
                horse_top50percent_rf, 'random forest')


    def evaluation(self, win, top3, top50percent, model):
        TP=0;FP=0;FN=0;TN=0
        for i in range(len(self.X_valid)):
            if (win[i]==1 and self.horse_win_actual[i]==1):
                TP=TP+1
            if (win[i]==1 and self.horse_win_actual[i]==0):
                FP=FP+1
            if (win[i]==0 and self.horse_win_actual[i]==1):
                FN=FN+1
            if (win[i]==0 and self.horse_win_actual[i]==0):
                TN=TN+1
        print('Recall of '+ model +' model horse_win prediction= ', TP/(TP+FN))
        print('Precision of '+model+' model horse_win prediction= ', TP/(TP+FP))
        TP=0;FP=0;FN=0;TN=0
        for i in range(len(self.X_valid)):
            if (top3[i]==1 and self.horse_top3_actual[i]==1):
                TP=TP+1
            if (top3[i]==1 and self.horse_top3_actual[i]==0):
                FP=FP+1
            if (top3[i]==0 and self.horse_top3_actual[i]==1):
                FN=FN+1
            if (top3[i]==0 and self.horse_top3_actual[i]==0):
                TN=TN+1
        print('Recall of '+model+' model horse_top3 prediction= ', TP/(TP+FN))
        print('Precision of '+model+' model horse_top3 prediction= ', TP/(TP+FP))
        TP=0;FP=0;FN=0;TN=0
        for i in range(len(self.X_valid)):
            if (top50percent[i]==1 and self.horse_top50percent_actual[i]==1):
                TP=TP+1
            if (top50percent[i]==1 and self.horse_top50percent_actual[i]==0):
                FP=FP+1
            if (top50percent[i]==0 and self.horse_top50percent_actual[i]==1):
                FN=FN+1
            if (top50percent[i]==0 and self.horse_top50percent_actual[i]==0):
                TN=TN+1
        print('Recall of '+model+' model horse_top50percent prediction= ', TP/(TP+FN))
        print('Precision of '+model+' model horse_top50percent prediction= ', TP/(TP+FP), '\n')


if __name__=='__main__':
    clf = Classification()
    clf.logistic()
    clf.naiveBayes()
#     clf.supportVector()
    clf.randomForest()
    # clf.getPrediction()
#     clf.evaluation()

        
