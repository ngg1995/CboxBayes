from operator import ne
import numpy as np
from numpy.random import default_rng
import random as rnd 
import pandas as pd
# import pba
from string import ascii_lowercase as alphabet

seed = 2
nprnd = default_rng(seed)
rnd.seed(seed)

some = 12
many = 1000
names = list(alphabet)[:some]
thresh  = nprnd.random(some)
print(*list(zip(names,thresh)),sep = '\n')
data = pd.DataFrame(nprnd.random((many,some))<thresh,columns = names)

few = 2
mrl = 2
# rules = {
#     i: {
#         'body':rnd.sample(names,rnd.randint(1,mrl)),
#         'conf': rnd.random()/2+0.5
#         } for i in range(few)
# }
rules = {
    0: {
        'body': ['a','b'],
        'conf': 0.8
    },
    1: {
        'body': ['f','g','j'],
        'conf': 0.6
    },
    2: {
        'body': ['d'],
        'conf': 0.9
    }
}
def print_rules(rules):
    for r in rules.values():
        o = "t :- "
        for b in r['body']:
            o += f'{b}, '
        o = o[:-2] + '.'
        print(o)

print_rules(rules)

def give_class(data, rules):
    results = pd.Series(np.zeros(len(data.index)), index = data.index,dtype = bool)
    for i in data.index:
        for r in rules.values():
            if sum(data.loc[i,r['body']]) == len(data.loc[i,r['body']]):
                if rnd.random() < r['conf']:
                    results[i] = True
                    break
        
            
    return results

results = give_class(data,rules)
data['truth'] = results
data.to_csv('dataset.csv')