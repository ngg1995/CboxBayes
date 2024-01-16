from dataclasses import dataclass
import pba
import numpy as np
import pandas as pd
from typing import Union
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
__all__= ["CboxBayes"]

@dataclass
class likelihoods:
    '''class for keeping track of sensitivity, specificity, PLR and NLR'''
    sensitivity: pba.Pbox
    specificity: pba.Pbox
    PLR: pba.Pbox
    NLR: pba.Pbox
    
def bayes_rule(p,s,t):
    return 1/(1+(1/p-1)*(1/s)*(1-t))

class CboxBayes:
    def __init__(self, one_sided = False):
        self.questions = []
        self.likelihood_ratios = {}
    
    
    def fit(self, data, results):
        '''
        Fit the algorithms
        '''
        
        if not isinstance(data,pd.DataFrame):
          data = pd.DataFrame(data)
        if not isinstance(data,(pd.DataFrame,pd.Series)):
          results = pd.Series(results)     
          
        for i in data.columns:
            tn, fp, fn, tp  = confusion_matrix(data[i].to_numpy(dtype=int),results.to_numpy(dtype=int)).ravel()
            print(tp,fp,fn,tn)
            s = pba.KM(tp,fn)
            t = pba.KM(tn,fp)
            plr = s/(1-t)
            nlr = (1-s)/t
            
            if plr.straddles(1) or nlr.straddles(1):
                print(f"{i} is unlikely to be a useful features since the PLR/NLR straddles 1")
            self.likelihood_ratios[i] = likelihoods(s,t,plr,nlr)

    
    def predict(self, X, p):
        
        def compute_ppv(LR,PPV):
            C_PPV = 1/(1+(1/PPV-1)/LR)
            # C_PPV = 1/(1+(1/s)*(1/p - 1)*(1-t))
            return C_PPV


        def compute_npv(LR,NPV):
            C_PPV = 1 / (1 + (LR/((1/NPV)-1)))
            return C_PPV

        if isinstance(X,pd.Series):
            X = pd.DataFrame(X).transpose()
        
        if not hasattr(p, "__iter__"):
            p = [p]*len(X.index)
            
        Y = []
        
        for i,ppv in tqdm(zip(X.index,p)):
            for c in X.columns:
                if X.loc[i,c]:
                    ppv = compute_ppv(self.likelihood_ratios[c].PLR,ppv)
                else:
                    ppv = compute_npv(self.likelihood_ratios[c].NLR,ppv)
            Y.append(ppv)
            
        return Y
