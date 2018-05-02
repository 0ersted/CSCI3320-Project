import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

data = pd.read_csv('../data/training.csv')

draw_list = np.unique(data.draw)
draw_win_rate = []
for draw in draw_list:
    draw_index = []
    for i, k in enumerate(data['draw']):
        if k == draw:
            draw_index.append(i)
    record = data.iloc[draw_index][['finishing_position']]
    win_num = sum([1 for row in record.itertuples() if int(row.finishing_position) == 1])
    draw_win_rate.append(win_num/len(record))
    
    
plt.pie(draw_win_rate, labels=draw_list, autopct='%1.1f%%')
plt.title('Pie Chart of the Draw Bias Effect')
plt.show()


