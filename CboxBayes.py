from dataclasses import dataclass
import pba
import numpy as np
import pandas as pd
from typing import Union
from sklearn.metrics import confusion_matrix

__all__= ["CboxBayes"]

@dataclass
class sens_spec:
    '''class for keeping track of sensitivity and specificity'''
    sensitivity: pba.Pbox
    specificity: pba.Pbox
    
def bayes_rule(p,s,t):
    return 1/(1+(1/p-1)*(1/s)*(1-t))

class CboxBayes:
    def __init__(self, one_sided = False):
        self.questions = []
        self.lr = {}
    
    
    def fit(self, data, results):
        '''
        Fit the algorithms
        '''
        
        def _calc_sens_spec(X: pd.Series, Y: pd.Series) -> sens_spec:
            tn, fp, fn, tp  = confusion_matrix(Y.to_numpy(dtype=int),X.to_numpy(dtype=int)).ravel()
            print(tn,fp,fn,tp)
            s = pba.KM(tp,fn)
            t = pba.KM(tn,fp)

            return sens_spec(s,t)
        
        if not isinstance(data,pd.DataFrame):
          data = pd.DataFrame(data)
        if not isinstance(data,(pd.DataFrame,pd.Series)):
          results = pd.Series(results)     
          
        for i in data.columns:

            self.lr[i] = _calc_sens_spec(data[i],results)

    
    def predict(self, X, p):
        if isinstance(X,pd.Series):
            X = pd.DataFrame(X).transpose()
        Y = pd.DataFrame(np.ones(len(X))*p,index=X.index,dtype='O')
        for i in Y.index:
            for c in X.columns:
                Y.loc[i] = bayes_rule(Y.loc[i],self.lr[c].sensitivity,self.lr[c].specificity)
                
                Y.loc[i].values[0].show(title = str(c))
        
        return Y
