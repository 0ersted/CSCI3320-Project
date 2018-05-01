import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import StandardScaler
scaler_X=StandardScaler()
#first produce X_train
df_train = pd.read_csv('data/training.csv')
df_test = pd.read_csv('data/testing.csv')
df = pd.read_csv('data/race-result-horse.csv')
X_train=df_train.drop(['finishing_position','horse_number','horse_name','horse_id','jockey','trainer','length_behind_winner','running_position_1','running_position_2','running_position_3','running_position_4','finish_time','running_position_5','running_position_6','race_id','recent_6_runs'],axis=1)
X_train=pd.DataFrame(X_train,dtype=np.float)
scaler_X.fit(X_train)
X_train_normalized=scaler_X.transform(X_train)
m_train, _ = np.shape(X_train)
X_test=df_test.drop(['finishing_position','horse_number','horse_name','horse_id','jockey','trainer','length_behind_winner','running_position_1','running_position_2','running_position_3','running_position_4','finish_time','running_position_5','running_position_6','race_id','recent_6_runs'],axis=1)
X_test=pd.DataFrame(X_test,dtype=np.float)
scaler_X.fit(X_test)
X_test_normalized=scaler_X.transform(X_test)
m_test, _ = np.shape(X_test)
y_train=df_train.finish_time
def t2s(t):#transfer the finish_time into float
    min,sec,msec=t.strip().split(".")
    return (float(min)*60+float(sec)+float(msec)*0.01)
y_train=[t2s(y_train[i]) for i in range(len(y_train))]
y_train=pd.DataFrame(y_train,dtype=np.float)
scaler_y=StandardScaler()
scaler_y.fit(y_train)
y_train_normalized=scaler_y.transform(y_train)
y_test=df_test.finish_time
y_test=[t2s(y_test[i]) for i in range(len(y_test))]
y_test=pd.DataFrame(y_test,dtype=np.float)
scaler_y.fit(y_test)
y_test_normalized=scaler_y.transform(y_test)
m, n = np.shape(df)
m_train, _ = np.shape(df_train)
m_test, _ = np.shape(df_test)
#4.1.1
from sklearn.svm import SVR
svr_model=SVR(C=27,epsilon=0.2,kernel='rbf')
svr_model.fit(X_train,y_train)
svr_result=svr_model.predict(X_test)
svr_score=svr_model.score(X_test,y_test)
print("svr_score before normalization: ", svr_score)
#4.1.2
from sklearn.ensemble import GradientBoostingRegressor
gbrt_model=GradientBoostingRegressor(loss='ls',learning_rate=0.05,n_estimators=600,max_depth=1)
gbrt_model.fit(X_train,y_train)
gbrt_result=gbrt_model.predict(X_test)
gbrt_score=gbrt_model.score(X_test,y_test)
print("gbrt_score before normalization: ", gbrt_score)
#4.2
#root_mean_squared_error svr model
y_subtracted=np.power(np.array(y_test)-np.array(svr_result),2)
sum=0
for i in range(len(y_test)):
    sum=sum+y_subtracted[i][0]
root_mean_squared_error_SVR = np.power(sum/len(y_test),1/2)
#root_mean_squared_error GBRT model
y_subtracted=np.power(np.array(y_test)-np.array(gbrt_result),2)
sum=0
for i in range(len(y_test)):
    sum=sum+y_subtracted[i][0]
root_mean_squared_error_GBRT= np.power(sum/len(y_test),1/2)
#get rank of gbrt model from finish_time
raceid=list()
for i in range(len(df_test)):
    if df_test.race_id[i] not in raceid:
        raceid.append(df_test.race_id[i])
#svr
svr_predicted_rank=np.zeros((len(X_test),1))
for i in range(len(raceid)):
    index=list()
    for j in np.where(df_test.race_id==raceid[i]):
        index.append(j)#index is the row number in X_test whose race id is race_id[i]
    index=index[0]
    order=np.argsort(np.array(svr_result)[index],axis=0)
    k=0
    for j in index:
        svr_predicted_rank[j]=order[k]+1
        k=k+1
from collections import Counter
TP_top1=0
TP_top3=0
sum_of_actual_rank=0
count_of_1=0
for i in range(len(svr_result)):
    if svr_predicted_rank[i]==1:
        count_of_1=count_of_1+1
        sum_of_actual_rank=sum_of_actual_rank+df_test.finishing_position[i]
    if (svr_predicted_rank[i]==1 and df_test.finishing_position[i]==1):
        TP_top1=TP_top1+1
    if (svr_predicted_rank[i]==1 and df_test.finishing_position[i]<=3):
        TP_top3=TP_top3+1
Top_1=TP_top1/count_of_1
Top_3=TP_top3/count_of_1
ave_rank=sum_of_actual_rank/count_of_1
print("SVR Model before normalization : RMSE = ",root_mean_squared_error_SVR,"; Top_1 = ",Top_1,"; Top_3 = ",Top_3,"; Average_Rank = ",ave_rank)
#gbrt
gbrt_predicted_rank=np.zeros((len(X_test),1))
for i in range(len(raceid)):
    index=list()
    for j in np.where(df_test.race_id==raceid[i]):
        index.append(j)#index is the row number in X_test whose race id is race_id[i]
    index=index[0]
    order=np.argsort(np.array(gbrt_result)[index],axis=0)
    k=0
    for j in index:
        gbrt_predicted_rank[j]=order[k]+1
        k=k+1
from collections import Counter
TP_top1=0
TP_top3=0
sum_of_actual_rank=0
count_of_1=0
for i in range(len(gbrt_result)):
    if gbrt_predicted_rank[i]==1:
        count_of_1=count_of_1+1
        sum_of_actual_rank=sum_of_actual_rank+df_test.finishing_position[i]
    if (gbrt_predicted_rank[i]==1 and df_test.finishing_position[i]==1):
        TP_top1=TP_top1+1
    if (gbrt_predicted_rank[i]==1 and df_test.finishing_position[i]<=3):
        TP_top3=TP_top3+1
Top_1=TP_top1/count_of_1
Top_3=TP_top3/count_of_1
ave_rank=sum_of_actual_rank/count_of_1
print("Gradient Boosting Regression Tree Model before normalization: RMSE = ",root_mean_squared_error_GBRT,"; Top_1 = ",Top_1,"; Top_3 = ",Top_3,"; Average_Rank = ",ave_rank)
#normalized cases
#normalized svr
svr_normalized_model=SVR(C=20,epsilon=0.1,kernel='rbf')
svr_normalized_model.fit(X_train_normalized,y_train_normalized)
svr_normalized_result=svr_normalized_model.predict(X_test_normalized)
svr_normalized_score=svr_normalized_model.score(X_test_normalized,y_test_normalized)
print("svr_normalized_score after normalization: ", svr_normalized_score)
#normalized GBRT
gbrt_normalized_model=GradientBoostingRegressor(loss='ls',learning_rate=0.012,n_estimators=1000,max_depth=1)#very sensitive to leaning_rate
gbrt_normalized_model.fit(X_train_normalized,y_train_normalized)
gbrt_normalized_result=gbrt_normalized_model.predict(X_test_normalized)
gbrt_normalized_score=gbrt_normalized_model.score(X_test_normalized,y_test_normalized)
print("gbrt_normalized_score: ",gbrt_normalized_score)
#root_mean_squared_error svr model
y_subtracted=np.power(np.array(y_test_normalized)-np.array(svr_normalized_result),2)
sum=0
for i in range(len(y_test)):
    sum=sum+y_subtracted[i][0]
normalized_root_mean_squared_error_SVR = np.power(sum/len(y_test),1/2)
#root_mean_squared_error GBRT model
y_subtracted=np.power(np.array(y_test_normalized)-np.array(gbrt_normalized_result),2)
sum=0
for i in range(len(y_test)):
    sum=sum+y_subtracted[i][0]
normalized_root_mean_squared_error_GBRT= np.power(sum/len(y_test),1/2)
#get rank of gbrt model from finish_time
#svr
svr_normalized_predicted_rank=np.zeros((len(X_test),1))
for i in range(len(raceid)):
    index=list()
    for j in np.where(df_test.race_id==raceid[i]):
        index.append(j)#index is the row number in X_test whose race id is race_id[i]
    index=index[0]
    order=np.argsort(np.array(svr_normalized_result)[index],axis=0)
    k=0
    for j in index:
        svr_normalized_predicted_rank[j]=order[k]+1
        k=k+1
from collections import Counter
TP_normalized_top1=0
TP_normalized_top3=0
sum_of_actual_rank=0
count_of_1=0
for i in range(len(svr_normalized_result)):
    if svr_normalized_predicted_rank[i]==1:
        count_of_1=count_of_1+1
        sum_of_actual_rank=sum_of_actual_rank+df_test.finishing_position[i]
    if (svr_normalized_predicted_rank[i]==1 and df_test.finishing_position[i]==1):
        TP_normalized_top1=TP_normalized_top1+1
    if (svr_normalized_predicted_rank[i]==1 and df_test.finishing_position[i]<=3):
        TP_normalized_top3=TP_normalized_top3+1
Top_normalized_1=TP_normalized_top1/count_of_1
Top_normalized_3=TP_normalized_top3/count_of_1
ave_normalized_rank=sum_of_actual_rank/count_of_1
print("SVR Model after normalization: RMSE = ",normalized_root_mean_squared_error_SVR,"; Top_1 = ",Top_normalized_1,"; Top_3 = ",Top_normalized_3,"; Average_Rank = ",ave_normalized_rank)
#gbrt
gbrt_normalized_predicted_rank=np.zeros((len(X_test),1))
for i in range(len(raceid)):
    index=list()
    for j in np.where(df_test.race_id==raceid[i]):
        index.append(j)#index is the row number in X_test whose race id is race_id[i]
    index=index[0]
    order=np.argsort(np.array(gbrt_normalized_result)[index],axis=0)
    k=0
    for j in index:
        gbrt_normalized_predicted_rank[j]=order[k]+1
        k=k+1
from collections import Counter
TP_normalized_top1=0
TP_normalized_top3=0
sum_of_actual_rank=0
count_of_1=0
for i in range(len(gbrt_normalized_result)):
    if gbrt_normalized_predicted_rank[i]==1:
        count_of_1=count_of_1+1
        sum_of_actual_rank=sum_of_actual_rank+df_test.finishing_position[i]
    if (gbrt_normalized_predicted_rank[i]==1 and df_test.finishing_position[i]==1):
        TP_normalized_top1=TP_normalized_top1+1
    if (gbrt_normalized_predicted_rank[i]==1 and df_test.finishing_position[i]<=3):
        TP_normalized_top3=TP_normalized_top3+1
Top_normalized_1=TP_normalized_top1/count_of_1
Top_normalized_3=TP_normalized_top3/count_of_1
ave_normalized_rank=sum_of_actual_rank/count_of_1
print("Gradient Boosting Regression Tree Model after normalization: RMSE = ",normalized_root_mean_squared_error_GBRT,"; Top_1 = ",Top_normalized_1,"; Top_3 = ",Top_normalized_3,"; Average_Rank = ",ave_normalized_rank)
#print(svr_normalized_result)
#print(gbrt_normalized_result)
