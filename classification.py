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
import time

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
        print('training time:', time.time()-start_time, 'secends')
        print('Score of linear regression:', lr_model.score(self.X_valid, self.y_valid))

    def naiveBayes(self):
        start_time = time.time()
        nb_model = sklearn.naive_bayes.GaussianNB()
        nb_model.fit(self.X_train, self.y_train)
        nb_result=nb_model.predict(self.X_valid)
        print('training time of Naive Bayes:', time.time()-start_time, 'secends')
        nb_score = nb_model.score(self.X_test, self.y_test)
        print('Score of Naive Bayes:', nb_score)


    def supportVector(self):

        return

    def randomForest(self):
        return

    def getPrediction(self):
        return

    def evaluation(self):
        return

if __name__=='__main__':
    clf = Classification()
    clf.logistic()
    clf.naiveBayes()
#     clf.supportVector()
#     clf.randomForest()
#     clf.getPrediction()
#     clf.evaluation()

        
