# -*- coding: utf-8 -*-
"""
Created on Thu Sep  10 21:31:02 2021

@author: Kuriseti Ravi Sri Teja
"""

import numpy as np
import pandas as pd
from scipy.special import softmax
import sys
import math
from sklearn import *


def extractfromcsv(train_path,test_path):

    train = pd.read_csv(train_path, index_col = 0)    
    test = pd.read_csv(test_path, index_col = 0)
        
    Y_train = np.array(train['Length of Stay'])

    train = train.drop(columns = ['Length of Stay'])

    #Ensuring consistency of One-Hot Encoding

    data = pd.concat([train, test], ignore_index = True)
    cols = train.columns
    cols = cols[:-1]
    data = pd.get_dummies(data, columns=cols, drop_first=True)
    data = data.to_numpy()
    X_train = data[:train.shape[0], :]
    X_test = data[train.shape[0]:, :]
    n=X_train.shape[0]

    Y_1=np.zeros((n,8),dtype="int")
    for i in range(0,n):
        Y_1[i][Y_train[i]-1]=1
    #print(Y_1)

    X1=np.ones((n,))
    X_train=np.array(np.column_stack((X1,X_train)))

    X_1=np.ones((X_test.shape[0],))
    X_test=np.array(np.column_stack((X_1,X_test)))

    return (X_train,Y_1,X_test)


def writetoout(fname,arr):
    f1=open(fname,"w")
    S=arr.shape
    if(len(S)==1):
        n=S[0]
        for i in range(0,n-1):
            f1.write(str(arr[i])+"\n")
        f1.write(str(arr[-1]))
        f1.close()
    else:
        m=S[0]
        n=S[1]
        for i in range(0,m-1):
            for j in range(0,n):
                f1.write(str(arr[i][j])+"\n")


        for u in range(0,n-1):
            f1.write(str(arr[-1][u])+"\n")
        f1.write(str(arr[-1][-1]))
        f1.close()

    return 0


def max_L(L):
    maxi=-10**9+7
    for i in range(0,L.shape[0]):
        for j in range(0,L.shape[1]):
            maxi=max(maxi,abs(L[i][j]))
    return maxi





def get_Y_(X,W):
    M1=np.dot(X,W)
    return softmax(M1,axis=1)
        



def f(X,W,Y):
    Z1=get_Y_(X,W)
    Z2=np.log(Z1)
    A=np.multiply(Y,Z2)
    return-1.000*np.sum(A,dtype="float32")


def g(X,Xt,Y,W):
    Y_1=get_Y_(X,W)
    grad=np.dot(Xt,(Y_1-Y))
    return grad/X.shape[0]




def frob_norm(X):
    return np.linalg.norm(X)


def h(X):
    return (frob_norm(X))**2/(X.shape[0]*X.shape[1])

def constant_rate_gradient_descent(n,rate,X,Xt,Y,W0):
    W=W0
    init_h=h(g(X,Xt,Y,W))
    i=0
    check_point=min(6,n)
    while(i<n):
        z=g(X,Xt,Y,W)
        W=W-rate*z
        i+=1
        if(i==check_point and h(z)>init_h):
            return constant_rate_gradient_descent(n,-1*rate,X,Xt,Y)
        
    return W

    



def adaptive_learning_rate_gradient_descent(n,rate,X,Xt,Y,W0):
    i=0
    W=W0
    curr_rate=rate
    ini_rate=rate
    z=g(X,Xt,Y,W)
    init_h=h(z)
    check_point=min(6,n)
    while(i<n):
        curr_rate=ini_rate/math.sqrt(i+1)
        z=g(X,Xt,Y,W)
        W=W-curr_rate*z        
        i+=1
        if(i==check_point and h(z)>init_h):
            return adaptive_learning_gradient_descent(n,-1*ini_rate,X,Xt,Y)
       

    return W


def h1(z):
    return (np.linalg.norm(z,"fro"))**2


def a_b_backtracking_gradient_descent(n,rate,a,b,X,Xt,Y,W0):
    i=0
    W=W0
    curr_rate=rate
    init_h=h(g(X,Xt,Y,W))
    #g is gradient
    #f is original function
    z=g(X,Xt,Y,W)
    n1=X.shape[0]
    check_point=min(6,n)
    while(i<n):
        z=g(X,Xt,Y,W)                
        curr_rate=a_b_backtracking_search(rate,X,W,Y,a,b,z)
        W=W-curr_rate*z       
        i+=1
        if(i==check_point and h(z)>init_h):
            return constant_rate_gradient_descent(n,-1*ini_rate,X,Xt,Y)
    return W


def a_b_backtracking_search(rate,X,W,Y,a,b,z):
    curr_rate=rate
    n1=X.shape[0]
    while(f(X,W,Y)-f(X,W-curr_rate*z,Y)<a*n1*curr_rate*h1(z)):
        curr_rate=b*curr_rate
    return curr_rate



def mini_batch_gradient_descent(batch_size,n,P,X,Y,r):
    W=np.zeros((X.shape[1],Y.shape[1]),dtype="float32")
    N=len(X)//batch_size
    l=batch_size
    if(r==1):
        for i in range(0,n):
            for j in range(0,N):
                X1=X[j*l:(j+1)*l,:]
                X1t=np.transpose(X1)
                Y1=Y[j*l:(j+1)*l,:]
                W=W-P[0]*g(X1,X1t,Y1,W)
        return W 
    elif(r==2):
        r0=P[0]
        for i in range(0,n):
            r1=r0/math.sqrt(i+1)
            for j in range(0,N):
                X1=X[j*l:(j+1)*l,:]
                X1t=np.transpose(X1)
                Y1=Y[j*l:(j+1)*l,:]                
                W=W-r1*g(X1,X1t,Y1,W)
        return W

    elif(r==3):
        (r0,a,b)=(P[0],P[1],P[2])
        for i in range(0,n):
            print(i)
            for j in range(0,N):
                X1=X[j*l:(j+1)*l,:]
                X1t=np.transpose(X1)
                Y1=Y[j*l:(j+1)*l,:] 
                z=g(X1,X1t,Y1,W)
                W=W-a_b_backtracking_search(rate,X,W,Y,a,b,z)*z

        return W
    else:
        return W
    


def get_output_vector(X,W):
    Y_1=get_Y_(X,W)
    n=Y_1.shape[0]
    Y=[0]*(n)
    for i in range(0,n):
       Y[i]=1+np.argmax(Y_1[i])
    Y=np.array(Y)
    return Y


L=sys.argv
if(L[1]=='a'):
    ftrain=L[2]
    ftest=L[3]
    fparam=L[4]
    fout=L[5]
    fweight=L[6]
    f1=open(fparam)
    A=f1.readlines()
    f1.close()
    descent_type=int(A[0])
    iters=int(A[2])
    X,Y,X_test=extractfromcsv(ftrain,ftest)
    X1=np.transpose(X)
    wt_vector=[]
    W0=np.zeros((X.shape[1],Y.shape[1]),dtype="float64")
    if(descent_type==1):
        rate=float(A[1])
        wt_vector=constant_rate_gradient_descent(iters,rate,X,X1,Y,W0)
    elif(descent_type==2):
        rate=float(A[1])
        wt_vector=adaptive_learning_rate_gradient_descent(iters,rate,X,X1,Y,W0)
    elif(descent_type==3):
        rate,a,b=map(float,A[1].split(","))
        wt_vector=a_b_backtracking_gradient_descent(iters,rate,a,b,X,X1,Y,W0)
    else:
        print("Invalid Input")

   
    ans=get_output_vector(X_test,wt_vector)
    writetoout(fout,ans)
    writetoout(fweight,wt_vector)
    
elif(L[1]=='b'):
    ftrain=L[2]
    ftest=L[3]
    fparam=L[4]
    fout=L[5]
    fweight=L[6]
    f1=open(fparam)
    A=f1.readlines()
    f1.close()
    X,Y,X_test=extractfromcsv(ftrain,ftest)
    descent_type=int(A[0])
    iters=int(A[2])
    batch_size=int(A[3])
    wt_vector=[]
    if(descent_type==1):
        rate=float(A[1])
        P=[rate]
        wt_vector=mini_batch_gradient_descent(batch_size,iters,P,X,Y,1)
    elif(descent_type==2):
        rate=float(A[1])
        P=[rate]
        wt_vector=mini_batch_gradient_descent(batch_size,iters,P,X,Y,2)
    elif(descent_type==3):
        P=list(map(float,A[1].split(",")))
        wt_vector=mini_batch_gradient_descent(batch_size,iters,P,X,Y,3)
    else:
        print("Invalid Input")

    ans=get_output_vector(X_test,wt_vector)
    writetoout(fout,ans)
    writetoout(fweight,wt_vector)


    

elif(L[1]=='c'):
    ftrain=L[2]
    ftest=L[3]
    fout=L[4]
    fweight=L[5]
    

elif(L[1]=='d'):
    ftrain=L[2]
    ftest=L[3]
    fout=L[4]
    fweight=L[5]
    
else:
    print("Invalid Input Command")