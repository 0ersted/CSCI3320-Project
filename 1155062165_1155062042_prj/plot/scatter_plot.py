import pandas as pd
import matplotlib.pyplot as plt 

data = pd.read_csv('../data/training.csv')


horse_plot_info = {'name':[], 'win_num':[], 'win_rate':[]}
jockey_plot_info = {'name':[], 'win_num':[], 'win_rate':[]}

horse_list = list(set(data['horse_name']))
horse_dict = {}
for name in horse_list:
    name_index = []
    for i, k in enumerate(data['horse_name']):
        if k == name:
            name_index.append(i)
    record = data.iloc[name_index][['finishing_position']]
    win_num = sum([1 for row in record.itertuples() if int(row.finishing_position) == 1])
    win_rate = win_num/len(record)
    horse_plot_info['name'].append(name)
    horse_plot_info['win_num'].append(win_num)
    horse_plot_info['win_rate'].append(win_rate)


jockey_list = list(set(data['jockey']))
jockey_dict = {}
for name in jockey_list:
    name_index = []
    for i, k in enumerate(data['jockey']):
        if k == name:
            name_index.append(i)
    record = data.iloc[name_index][['finishing_position']]
    win_num = sum([1 for row in record.itertuples() if int(row.finishing_position) == 1])
    jockey_dict[name] = {'win_num': win_num, 'win_rate': win_num/len(record)}
    win_rate = win_num/len(record)
    jockey_plot_info['name'].append(name)
    jockey_plot_info['win_num'].append(win_num)
    jockey_plot_info['win_rate'].append(win_rate)


# hyper parameter
horse_threshold = 0.65
jockey_threshold = 0.18

plt.subplot(1,2,1)
plt.scatter(horse_plot_info['win_rate'], horse_plot_info['win_num'], marker='8', c='g', alpha=0.35)
plt.xlabel('winning rate')
plt.ylabel('winning number')
plt.title('Good Horses')
for i, name in enumerate(horse_plot_info['name']):
    if horse_plot_info['win_rate'][i] >= horse_threshold:
        plt.annotate(name, (horse_plot_info['win_rate'][i], horse_plot_info['win_num'][i]))

plt.subplot(1,2,2)
plt.scatter(jockey_plot_info['win_rate'], jockey_plot_info['win_num'], marker='s', c='r', alpha=0.35)
plt.xlabel('winning rate')
plt.ylabel('winning number')
plt.title('Good Jockeys')
for i, name in enumerate(jockey_plot_info['name']):
    if jockey_plot_info['win_rate'][i] >= jockey_threshold:
        plt.annotate(name,( jockey_plot_info['win_rate'][i], jockey_plot_info['win_num'][i]))

plt.show()
