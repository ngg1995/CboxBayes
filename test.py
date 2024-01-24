#%%
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pba

from CboxBayes import CboxBayes, likelihoods

from sklearn.linear_model import LogisticRegression

#%%
num_samples = 2000
num_features = 2
p = 0.5
rng = np.random.default_rng(1234)

results = pd.Series(rng.random(size = (num_samples,))>p)

columns = [chr(ord('A') + i) for i in range(num_features)]

dataset = pd.DataFrame(
    index = results.index,
    columns = columns
)

like = {c: likelihoods().from_sens_spec(rng.random(),rng.random()) for c in columns}

for i in results.index:
    for c in columns:
        if results.loc[i]: 
            dataset.loc[i,c] = rng.random() < like[c].sensitivity
        else:
            dataset.loc[i,c] = rng.random() > like[c].specificity
        

#%%
X_train, X_test, Y_train, Y_test = train_test_split(dataset, results, test_size=0.01, random_state=42)
CB = CboxBayes()
CB.fit(X_train,Y_train)

#%%
# Y_pred = CB.predict(X_test,p=0.5)

# fig,ax = plt.subplots(1,1)
# Y_pred[0].show(figax = (fig,ax))
# fig.show()
#%%
# %%
def singh(cboxes,alpha):
    
    x = np.linspace(0,1,1001)

    left = 0
    right = 0

    if hasattr(alpha,"__iter__"):
        l_alphas = [sum(cbox.left  > a)/cbox.steps for cbox,a in zip(cboxes,alpha)]
        r_alphas = [sum(cbox.right > a)/cbox.steps for cbox,a in zip(cboxes,alpha)]
    else:
        l_alphas = [sum(cbox.left  > alpha)/cbox.steps for cbox in cboxes]
        r_alphas = [sum(cbox.right > alpha)/cbox.steps for cbox in cboxes]

    left = [sum([i<=j for i in l_alphas])/len(cboxes) for j in x]
    right = [sum([i<=j for i in r_alphas])/len(cboxes) for j in x]

    fig, ax = plt.subplots()
    ax.plot([0]+list(x)+[1],[0]+left+[1])
    ax.plot([0]+list(x)+[1],[0]+right+[1])
    ax.plot([0,1],[0,1],'k--',lw = 2)
    
    return fig, ax
# %%
