import pandas as pd
import numpy as np

df1=pd.read_csv('race-result-horse.csv',header=None,names=['finishing_position','horse_number','horse_name','horse_id','jockey','trainer','actual_weight','declared_horse_weight','draw','length_behind_winner','running_position_1','running_position_2','running_position_3','running_position_4','finish_time','win_odds','running_position_5','running_position_6','race_id'],na_values=['?'])
#print(df1.shape)
df1.finishing_position=pd.to_numeric(df1.finishing_position,errors='coerce')
df1=df1.dropna(subset=['finishing_position'])
#print(df1.finishing_position)
#2.2.2
