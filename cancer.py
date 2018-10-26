import pandas as pd
import numpy as np
column_names=['a','b','c','d','e','f','g','h','i','j','k']
data=pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data',names=column_names)
data=data.replace(to_replace='?',value=np.nan)
data=data.dropna(how='any')
print(data.shape)