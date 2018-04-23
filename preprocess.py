import pandas as pd
import numpy as np

def preprocess():
    # data preprocessing
    # 2.2.1

    df = pd.read_csv('data/race-result-horse.csv')
    drop_list = []
    for i in range(df.finishing_position.size):
        try:
            int(df.finishing_position[i])
        except:
            drop_list.append(i)

    df = df.drop(index=drop_list).reset_index(drop=True)
    m, n = df.shape

    # 2.2.2
    horse_list = list(set(df['horse_name']))
    horse_dict = {}
    for name in horse_list:
        name_index = []
        for i, k in enumerate(df['horse_name']):
            if k == name:
                name_index.append(i)
        horse_dict[name] = df.iloc[name_index][['finishing_position']]

    recent_str_list = [None] * m
    recent_ave_list = [None] * m

    for name, data in horse_dict.items():
        recent_runs = []
        recent_str = ''

        for row in horse_dict[name].itertuples():       
            recent_str = recentRunToStr(recent_runs)
            recent_ave = round(recentRunToAve(recent_runs), 2)
            recent_runs.insert(0, int(row.finishing_position))
            # print(row.Index, recent_str, recent_ave)
            recent_str_list[row.Index]=recent_str
            recent_ave_list[row.Index]=recent_ave

    df['recent_6_runs'] = recent_str_list
    df['recent_ave_rank'] = recent_ave_list

    # 2.2.3
    jockey_list = list(set(df['jockey']))
    trainer_list = list(set(df['trainer']))
    horse = dict((name, i) for i, name in enumerate(horse_list))
    jockey = dict((name, i) for i, name in enumerate(jockey_list))
    trainer = dict((name, i) for i, name in enumerate(trainer_list))
    print("Numer of horses: ", len(horse_list))
    print("Numer of jockeys: ", len(jockey_list))
    print("Numer of trainers: ", len(trainer_list))

    # compute jockey_ave_rank
    jockey_dict = {}
    for name in jockey_list:
        name_index = []
        for i, k in enumerate(df['jockey']):
            if k == name:
                name_index.append(i)
        jockey_dict[name] = df.iloc[name_index][['finishing_position']]

    recent_jockey_ave_list = [None] * m

    for name, data in jockey_dict.items():
        recent_runs = []
        for row in jockey_dict[name].itertuples():
            recent_ave = round(7 if len(recent_runs) == 0 else sum(recent_runs)/len(recent_runs))
            recent_runs.insert(0, int(row.finishing_position))
            recent_jockey_ave_list[row.Index]=recent_ave

    df['jockey_ave_rank'] = recent_jockey_ave_list

    # compute trainer_ave_rank
    trainer_dict = {}
    for name in trainer_list:
        name_index = []
        for i, k in enumerate(df['trainer']):
            if k == name:
                name_index.append(i)
        trainer_dict[name] = df.iloc[name_index][['finishing_position']]

    recent_trainer_ave_list = [None] * m

    for name, data in trainer_dict.items():
        recent_runs = []
        for row in trainer_dict[name].itertuples():
            recent_ave = round(7 if len(recent_runs) == 0 else sum(recent_runs)/len(recent_runs))
            recent_runs.insert(0, int(row.finishing_position))
            recent_trainer_ave_list[row.Index]=recent_ave

    df['trainer_ave_rank'] = recent_trainer_ave_list

    # 2.2.4
    dff = pd.read_csv('data/race-result-race.csv')
    race_id_dist = dff[['race_id', 'race_distance']]
    df = df.join(race_id_dist.set_index('race_id'), on='race_id')

    #2.2.5 (split)
    trainsize=int(np.floor(0.8*len(df)))
    df_train=df.head(trainsize)
    df_test=df.tail(len(df)-trainsize)
    df_train.to_csv('data/training.csv')
    df_test.to_csv('data/testing.csv')


def recentRunToStr(runs):
    l = len(runs)
    recent_str = ''
    if l == 0:
        return recent_str
    else:
        for i in range(min(6,l)):
            if i > 0:
                recent_str += '/'
            recent_str += str(runs[i])
        return recent_str

def recentRunToAve(runs):
    l = len(runs)
    if l == 0:
        return 7
    else:
        return sum(runs[0:min(6,l)])/min(6,l)


if __name__ == "__main__":
    preprocess()

