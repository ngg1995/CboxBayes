
#%%
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pba

from CboxBayes import CboxBayes
#%%
num_samples = 1000
num_features = 10
p = 0.5
rng = np.random.default_rng(1)

results = pd.Series(rng.random(size = (num_samples,))>p)

like = rng.random(size = [num_features,2])

columns = [chr(ord('A') + i) for i in range(num_features)]

dataset = pd.DataFrame(
    index = results.index,
    columns = columns
)

for i in results.index:
    for c, l in zip(columns,like):
        if results.loc[i]:
            dataset.loc[i,c] = rng.random() > l[0]
        else:
            dataset.loc[i,c] = rng.random() > l[1]
            

X_train, X_test, Y_train, Y_test = train_test_split(dataset, results, test_size=0.10, random_state=42)
CB = CboxBayes()
CB.fit(X_train,Y_train)

Y_pred = CB.predict(X_test.iloc[0],p=0.5)

fig,ax = plt.subplots(1,1)
Y_pred.iloc[0].values[0].show(figax = (fig,ax))
#%%