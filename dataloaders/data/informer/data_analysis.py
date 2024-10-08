import numpy as np
import os
import matplotlib.pyplot as plt 
import pandas as pd
from pathlib import Path

df_M4 = pd.read_csv('dataloaders/data/informer/M4/Monthly-train.csv')
df_annotations = pd.read_csv('dataloaders/data/informer/M4/M4-info.csv')
print(df_annotations.keys())




'''
plt.plot( hrs, hufl_series,label='hufl')
plt.plot( hrs, hull_series,label = 'hull')
plt.plot( hrs, mufl_series,label = 'mufl')
plt.plot( hrs, mull_series,label = 'mull')
plt.plot( hrs, lufl_series,label = 'lufl')
plt.plot( hrs, lull_series,label = 'lull')
plt.plot( hrs, ot_series,label = 'ot')

plt.show()
'''