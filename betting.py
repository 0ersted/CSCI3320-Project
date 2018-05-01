#import

from classification import Classification
import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import StandardScaler


clf = Classification()#clf is the dataframe of prediction ranks
clf.logistic()
data = pd.read_csv('data/training.csv')
m_data = data.shape[0]
y = clf.lr_y_pred
df_train, df_valid = data[0:int(0.8*m_data)], data[int(0.8*m_data)+1:m_data]
df_valid = df_valid.reset_index(drop=True)
y=y[0:len(df_valid)]
raceid=list()
for i in range(len(df_valid)):
    if df_valid.race_id[i] not in raceid:
        raceid.append(df_valid.race_id[i])
print("Totoally ", len(raceid)," races")
bet=list()
for i in range(len(raceid)):
    index=list()
    predict_winner=list()
    for j in np.where(df_valid.race_id==raceid[i]):
        index.append(j)#index is the row number in X_test whose race id is race_id[i]
    index=index[0]
    #find the winner for the race whose name is race_id[i]
    min_rank=np.min(y[index])
    for k in index:
        if y[k]==min_rank:
             predict_winner.append(k)
    bet.append(predict_winner[0])

money=0
win_time=0
for i in bet:
    if df_valid.finishing_position[i]==1:
       money=money+df_valid.win_odds[i]-1
       win_time=win_time+1
    else:
       money=money-1

print("money win: ", money)
print("We win ",win_time," times")