#%%
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pba
from tqdm import tqdm
from CboxBayes import CboxBayes, likelihoods

from sklearn.linear_model import LogisticRegression
#%%
def singh(cboxes,alpha,fig = None,ax = None):
    
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
    
    if fig is None:
        fig, ax = plt.subplots()
    ax.plot([0]+list(x)+[1],[0]+left+[1])
    ax.plot([0]+list(x)+[1],[0]+right+[1])
    ax.plot([0,1],[0,1],'k--',lw = 2)
    
    return fig, ax

def compute_ppv(PLR,p):
    p = 1/(1+(1/p-1)*(1/PLR))
    return p


def compute_npv(NLR,p):
    return 1/(1+(1/p-1)*(1/NLR))

#%%
num_samples = 1000
num_features = 1
p = 0.5
rng = np.random.default_rng(1234)

columns = [chr(ord('A') + i) for i in range(num_features)]

like = {
    'A': likelihoods().from_sens_spec(0.8,0.85)
}

A = compute_ppv(like['A'].PLR,p)
nA = compute_npv(like['A'].NLR,p)

#%%
c_boxes = {
    'A-PLR': [],
    'A-NLR': [],
    'A-sens': [],
    'A-spec': [],
    'A': [],
    'nA': []
}
s = []
t = []
for _ in tqdm(range(1000)): 

    results = pd.Series(rng.random(size = (num_samples,))>p)

    dataset = pd.DataFrame(
        index = results.index,
        columns = columns
    )

    for i in results.index:
        for c in columns:
            if results.loc[i]: 
                dataset.loc[i,c] = rng.random() < like[c].sensitivity
            else:
                dataset.loc[i,c] = rng.random() > like[c].specificity
        

    X_train, X_test, Y_train, Y_test = train_test_split(dataset, results, test_size=0.01, random_state=42)
    CB = CboxBayes()
    CB.fit(X_train,Y_train)

    c_boxes['A-PLR'].append(CB.likelihood_ratios['A'].PLR)
    c_boxes['A-NLR'].append(CB.likelihood_ratios['A'].NLR)
    c_boxes['A-sens'].append(CB.likelihood_ratios['A'].sensitivity)
    c_boxes['A-spec'].append(CB.likelihood_ratios['A'].specificity)

    test = pd.DataFrame(
        {
            'A': [True,False]
        },
        index = ['A','nA']
    )

    predictions = CB.predict(test,p = p)

    c_boxes['A'].append(predictions[0])
    c_boxes['nA'].append(predictions[1])

# %%
fig_A, ax_A = plt.subplots(2,2)
singh(c_boxes['A-PLR'],like['A'].PLR,fig=fig_A,ax=ax_A[0,0])
ax_A[0,0].set_title('A - PLR')
singh(c_boxes['A-NLR'],like['A'].NLR,fig=fig_A,ax=ax_A[0,1])
ax_A[0,1].set_title('A - NLR')
singh(c_boxes['A-sens'],like['A'].sensitivity,fig=fig_A,ax=ax_A[1,0])
ax_A[1,0].set_title('A - Sensitivity')
singh(c_boxes['A-spec'],like['A'].specificity,fig=fig_A,ax=ax_A[1,1])
ax_A[1,1].set_title('A - specificity')
fig_A.tight_layout()
fig_A.savefig('A_1D.png')

#%%
fig2, ax2 = plt.subplots(1,2)
singh(c_boxes['A'],A,fig=fig2,ax=ax2[0])
ax2[0].set_title('$A$')
singh(c_boxes['nA'],nA,fig=fig2,ax=ax2[1])
ax2[1].set_title(r'$\neg A$')
fig2.tight_layout()
fig2.savefig('singhing_1D.png')

# %%
