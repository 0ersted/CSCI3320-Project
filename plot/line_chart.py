import pandas as pd
import matplotlib.pyplot as plt 

global data
data = pd.read_csv('../data/training.csv')

def line_chart(horse_id):
    horse_index = []
    for i, k in enumerate(data['horse_id']):
        if k == horse_id:
            horse_index.append(i)
    horse_finish = data.iloc[horse_index][['finishing_position', 'race_id']]

    plt.plot( horse_finish.race_id[-7:-1],horse_finish.finishing_position[-7:-1])
    plt.xlabel('race id')
    plt.ylabel('finishing position')
    plt.title('line chart of recent 6 run of horse '+horse_id)
    plt.show()


line_chart('P106')
line_chart('K019')
