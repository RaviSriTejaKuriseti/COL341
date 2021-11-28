# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 21:06:08 2021

@author: Kuriseti Ravi Sri Teja
"""

import numpy as np
import pandas as pd
from sklearn import *

def extractfromcsv(fname):
    U=pd.read_csv(fname,delimiter=",")  #Assumes first row to be header by default
    U1=U.to_numpy("float64")
    return U1

def process_train_data(A):
    N=A.shape
    rows=N[0]
    cols=N[1]
    Y=np.array([1]*rows,"float64")
    for i in range(0,rows):
        (Y[i],A[i][0])=(A[i][cols-1],Y[i])
    A=np.delete(A,cols-1,1)
    return (A.astype("float64"),Y)


def process_test_data(A):
    N=A.shape
    rows=N[0]
    cols=N[1]
    Y=np.array([1],"float64")
    for i in range(0,rows):
       A[i][0]=Y[0]
    return A.astype("float64")

def Train(X,Y):
    reg=linear_model.LassoLarsCV(cv=10,eps=1e-10,max_iter=100).fit(X,Y)
    I=[]
    L=reg.coef_
    for i in range(0,len(L)):
        if(L[i]!=0):
            I.append(i)
    ans1=(reg.coef_.tolist(),I,reg.alpha_,reg.score(X,Y))
    return ans1

def extremum(X,Y,p):
    if(p==0):
        temp2=np.transpose(X)
        temp3=np.dot(temp2,X) #xtx
        temp4=np.linalg.inv(temp3)
        temp1=np.dot(temp2,Y) #xty
        ans=np.dot(temp4,temp1)
        return ans
    else:
        temp2=np.transpose(X)
        temp3=np.dot(temp2,X) #xtx
        I=np.identity(temp3.shape[0],"float64")
        temp4=np.linalg.inv(temp3+p*I) #(xtx+lI)^-1
        temp1=np.dot(temp2,Y) #xty
        ans=np.dot(temp4,temp1)
        return ans

def func(L):
   
   L_1=L[:,[5,14,16,18,20,11,13]]
   poly=preprocessing.PolynomialFeatures(2,include_bias=True)
   L_11=poly.fit_transform(L_1)
   print(L_11.shape)
   L_3=L[:,[1,2,6,7,8,9,10,12,23,24,25,26,27,28,30]]
   enc=preprocessing.OneHotEncoder(sparse=False)
   enc.fit(L_3)
   L_31=enc.transform(L_3)
   print(L_31.shape)
   L_f=np.concatenate((L_11,L_31),axis=1)
   print(L_f.shape)
   return L_f

A1=extractfromcsv("krst_train_2.csv")
A2=extractfromcsv("krst_test_2.csv")
U1,V1=process_train_data(A1)
U2=process_test_data(A1)
U1=func(U1)
U2=func(U2)
a1=Train(U1,V1)
print(a1)
M=a1[1]

U1=U1[:,M]
U2=U2[:,M]


N=[0.001,0.01,0.1,1,3,10,30,100,300,1000]
x1=100000
for e in N:
    Y_exp=np.dot(U2,extremum(U1,V1,e))
    x1=min(x1,np.linalg.norm(Y_exp-V1)/np.linalg.norm(V1))
print(x1)

#0.39