# -*- coding: utf-8 -*-
"""
============================================================
Logistic Regression with fused-lasso by split bregman method
============================================================

'Split Bregman method for large scale fused Lasso' Gui-Bo et. al 2010

"""

print(__doc__)
from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt

class LogitR:
    # y is labels (1 of -1)
    # X is design matrix (n_samples*n_feautes)
    # w is weights 
    # f is values of discrimination function
    # p is probability

    # initialization
    def __init__(self, lmbd1=0.1, lmbd2=0.1, mu1=1e-2, mu2=1e-2,
                 tol=1e-2, alpha=0.01, maxitr=1e+5):
                
        self.lmbd1=lmbd1
        self.lmbd2=lmbd2
        self.mu1=mu1
        self.mu2=mu2
        
        self.tol=tol
        self.alpha=alpha
        self.maxitr=int(maxitr)
        
        self.history_l=[]
        self.history_d=[]
        
        self.scaler=None
    
    # evaluate probability    
    def eval_p(self):
        f=self.X.dot(self.w)
        p=1/(1+np.exp(-f*self.y))
        return p
        
    # evaluate loss function 
    def eval_l(self):
        p=self.eval_p()
        P=p.prod()
        l=-np.log(P)
        return l
    
    # evaluate derivative of loss function  
    def eval_d(self):
        p=self.eval_p()
        b=self.y*(1-p)
        d=-self.X.T.dot(b)
        return d
        
    # evaluate hessian of negative log-likelihood
    def eval_h(self):
        p=self.eval_p()
        B=np.diag(p*(1-p))
        X=self.X
        h=X.T.dot(B).dot(X)+2*self.lmbd1
        return h
        
    def fit(self,X,y):
        n_features=X.shape[1]
        
        L=self.make_L(n_features)
        
        # initialize
        self.scaler=preprocessing.StandardScaler().fit(X)
        
        self.X=self.scaler.transform(X)
        self.y=y
        self.w=np.random.randn(n_features)
        self.a=self.w
        self.b=L.dot(self.w)
        self.u=np.random.randn(n_features)
        self.v=np.random.randn(n_features-1)
            
        # parameters
        tol=self.tol
        alpha=self.alpha
        maxitr=self.maxitr
        
        for i in range(maxitr):
            # iteration for w primal
            d=self.eval_d()+self.u+self.v.dot(L)+self.mu1*(self.w-self.a)+self.mu2*L.T.dot(L.dot(self.w)-self.b)
            if n_features<100:
                h=self.eval_h()+self.mu1*np.identity(n_features)+self.mu2*L.T.dot(L)
                invhess=np.linalg.inv(h)
                delt_w=invhess.dot(d)
            else:
                delt_w=d
                
            self.w=self.w-alpha*delt_w
            
            # iteration for a,b primal
            self.a=self.soft_thresholding(self.w+self.u/self.mu1,self.lmbd1/self.mu1)
            self.b=self.soft_thresholding(L.dot(self.w)+self.v/self.mu2,self.lmbd2/self.mu2)
            
            # iteration for u,v dual
            dlta=1e-2
            self.u=self.u+dlta*(self.w-self.a)
            self.v=self.v+dlta*(L.dot(self.w)-self.b)

            # history
            l=(self.eval_l()
            +self.lmbd1*np.linalg.norm(self.a,ord=1)
            +self.lmbd2*np.linalg.norm(self.b,ord=1)
            +self.u.dot(self.w-self.a)
            +self.v.dot(L.dot(self.w)-self.b)
            +self.mu1/2*np.linalg.norm(self.w-self.a,ord=2)
            +self.mu2/2*np.linalg.norm(L.dot(self.w)-self.b,ord=2))
            
            self.history_l.append(l)
            self.history_d.append(np.linalg.norm(d,ord=2))
            
            # terminal condition            
            if np.linalg.norm(d,ord=2)<tol:
                break
                
        return self
        
    def pred(self,X_test):
        X_test=self.scaler.transform(X_test)
        f=X_test.dot(self.w)
        return np.sign(f)
        
    def soft_thresholding(self,x,lmbd):
        ind_p=np.where(x>=lmbd)
        ind_n=np.where(x<=-lmbd)
        
        out=np.zeros_like(x)
        out[ind_p]=x[ind_p]-lmbd
        out[ind_n]=x[ind_n]+lmbd
        return out
        
    def make_L(self,n_features):
        L=np.identity(n_features)
        for i in range(n_features-1):
            L[i][i+1]=-1
        L=np.delete(L,-1,axis=0)
        return L
        
        
if __name__=="__main__":

    np.random.seed(1)
    n_samples,n_features=200,100
    
    X=np.random.normal(size=[n_samples,n_features])
    w=np.zeros(n_features)
    w[20:29]=1
    y=np.sign(X.dot(w))

    model = LogitR()

    model.fit(X,y)
    plt.figure()
    plt.plot(model.history_l,label = "Loss Function")
    plt.plot(model.history_d,label = "Norm of Derivative")
    plt.legend(loc='best')

    plt.figure()
    plt.plot(model.w)
