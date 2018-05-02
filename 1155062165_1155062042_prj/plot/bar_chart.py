import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv('../data/training.csv')

m_data = data.shape[0]
df_train, df_test = data[0:int(0.7*m_data)], data[int(0.7*m_data)+1:m_data]
df_test = df_test.reset_index(drop=True)

df = pd.read_csv('../data/race-result-horse.csv')
m, n = np.shape(df)
m_train, _ = np.shape(df_train)
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
actual_weight = df_train.actual_weight.reshape((m_train,1))
declared_weight = df_train.declared_horse_weight.reshape((m_train,1))
draw = df_train.draw.reshape((m_train,1))
win_odds = df_train.win_odds.reshape((m_train,1))
race_distance = df_train.race_distance.reshape((m_train,1))

# we use horse, jockey, trainer, actual weight, declared weight, win odds, race distance as independent variables
X_train = np.hstack((train_horse, train_jockey, train_trainer, actual_weight,
                     declared_weight, draw, win_odds, race_distance))
y_train = df_train.finishing_position

rf = RandomForestClassifier(n_estimators=50, random_state=0)
rf.fit(X_train, y_train)

features = ['train_horse', 'train_jockey', 'train_trainer', 
        'actual_weight', 'declared_weight', 'draw', 
        'win_odds', 'race_distance']
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

features_reorder = [features[i] for i in indices]

index = range(len(features))

plt.bar(index, importances[indices], align='center')
plt.xlabel('feature names')
plt.ylabel('importance value')
plt.title('Feature Importance')
plt.xticks(index, features_reorder)
plt.show()

