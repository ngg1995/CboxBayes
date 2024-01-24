from dataclasses import dataclass
import pba
import numpy as np
import pandas as pd
from typing import Union
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

def make_confustion_matrix(Y_pred,Y_true):
    a,b,c,d = 0,0,0,0
    
    for i,j in zip(Y_pred,Y_true):
        if i and j:
            a += 1
        elif i and not j:
            b += 1
        elif not i and j:
            c += 1
        elif not i and not j:
            d += 1
    
    return a,b,c,d


@dataclass
class likelihoods:
    '''class for keeping track of sensitivity, specificity, PLR and NLR'''
    sensitivity: pba.Pbox = None
    specificity: pba.Pbox = None
    PLR: pba.Pbox = None
    NLR: pba.Pbox = None
    
    def from_sens_spec(self, sensitivity, specificity):
        self.sensitivity = sensitivity
        self.specificity = specificity
        self.PLR = sensitivity/(1-specificity)
        self.NLR = (1-sensitivity)/specificity
        return self
        
    def from_PLR_NLR(self, PLR, NLR):
        self.PLR = PLR
        self.NLR = NLR
        self.sensitivity  = PLR*(NLR-1)/(NLR-PLR)
        self.specificity  = (1-PLR)/(NLR-PLR)
        return self
    
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
            a,b,c,d = make_confustion_matrix(data[i],results)
            s = pba.KN(a,a+c)
            t = pba.KN(d,b+d)
            
            plr = s/(1-t)
            nlr = (1-s)/t
            
            # if plr.straddles(1) or nlr.straddles(1):
            #     print(f"{i} may not be a useful features since the PLR/NLR straddles 1")
            self.likelihood_ratios[i] = likelihoods(s,t,plr,nlr)

    
    def predict(self, X, p):
        
        def compute_ppv(LR,p):
            if not isinstance(p,pba.Pbox):
                p = pba.box(p,p)
            return 1/(1+(1/p-1).div(LR,'i'))


        def compute_npv(LR,NPV):
            if not isinstance(NPV, pba.Pbox):
                NPV = pba.box(NPV,NPV)
            C_PPV = 1 / (1 + LR.div((1/NPV)-1,'i'))
            return C_PPV
        
        def compute_ppn(NLR,p):
            if not isinstance(p,pba.Pbox):
                p = pba.box(p,p)   
            return 1/(1+(1/p-1).div(NLR,'i'))
        

        if isinstance(X,pd.Series):
            X = pd.DataFrame(X).transpose()
        
        if not hasattr(p, "__iter__"):
            p = [p]*len(X.index)
            
        Y = []
        
        for i,ppv in zip(X.index,p):
            for c in X.columns:
                if X.loc[i,c]:
                    ppv = compute_ppv(self.likelihood_ratios[c].PLR,ppv)
                else:
                    ppv = compute_ppn(self.likelihood_ratios[c].NLR,ppv)
            Y.append(ppv)
            
        return Y
