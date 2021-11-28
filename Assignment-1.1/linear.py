# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 21:31:02 2021

@author: Kuriseti Ravi Sri Teja
"""

import numpy as np
import pandas as pd
import sys
from sklearn import *


def extractfromcsv(fname):
    U=pd.read_csv(fname,delimiter=",")  #Assumes first row to be header by default
    U1=U.to_numpy("float64")
    return U1

def writetoout(fname,arr):
    f1=open(fname,"w")
    for i in range(0,arr.shape[0]-1):
        f1.write(str(arr[i])+"\n")
    f1.write(str(arr[-1]))
    f1.close()
    

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
    

def cross_split(data):
    S=data.shape
    points=S[0]
    N=[]
    if(points%10==0):
        m=(points//10)
        for j in range(0,10):
            U=[]            
            for i in range(m*j,m*j+m):
                X=data[i]
                U.append(X)
            U1=np.array(U)
            N.append(U1)
    else:
        m=(points//10)
        rem=points%10
        for j in range(0,rem):
            U=[]            
            for i in range(m*j+j,m*j+m+j+1):
                X=data[i]
                U.append(X)
            U1=np.array(U)
            N.append(U1)
            
        for j in range(rem,10):
            U=[]            
            for i in range(m*j+rem,m*j+m+rem):
                X=data[i]
                U.append(X)
            U1=np.array(U)
            N.append(U1)  
    return N


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
    
    
def R2_error(A,A1):
   #A-actual A1-experimental
   if(A.shape!=A1.shape):
       return -1
   else:      
        s=0
        s1=0
        for i in range(0,A.shape[0]):
            s+=(A[i]-A1[i])**2
            s1+=(A1[i])**2
        error_value=(s/s1)
        return error_value
    
    
def L1_norm(A):
    s=0
    for i in range(0,len(A)):
        s+=(A[i])
    error_value=s/len(A)
    return error_value

def L2_norm(A):
    s=0
    for i in range(0,len(A)):
        s+=(A[i])**2
    error_value=(s/len(A))**0.5
    return error_value
    
    

def error_analysis(cross_data,M):
    #M-parameters of lambda
    #cross-data :array of tuples that were split
    least_error=10**9
    corr=-1
    for e in M:
        error_vec=[0]*len(cross_data)
        for i in range(0,len(cross_data)):           
            if(i!=0):
                X1=cross_data[0][0]
                Y1=cross_data[0][1]
                test_in=cross_data[i][0]
                test_exp_out=cross_data[i][1]
                for j in range(1,len(cross_data)):
                    if(j!=i):
                        X1=np.concatenate((X1,cross_data[j][0]),axis=0)
                        Y1=np.concatenate((Y1,cross_data[j][1]),axis=None)
                wt_vector=extremum(X1,Y1,e)
                test_ac_out=np.dot(test_in,wt_vector)
                error_vec[i]=R2_error(test_exp_out,test_ac_out)
            else:
                test_in=cross_data[0][0]
                test_exp_out=cross_data[0][1]
                X1=cross_data[1][0]
                Y1=cross_data[1][1]
                for j in range(2,len(cross_data)):
                   X1=np.concatenate((X1,cross_data[j][0]),axis=0)
                   Y1=np.concatenate((Y1,cross_data[j][1]),axis=None)
                wt_vector=extremum(X1,Y1,e)
                test_ac_out=np.dot(test_in,wt_vector)
                error_vec[i]=R2_error(test_exp_out,test_ac_out)
        
        corr_error=L2_norm(error_vec)
        if(corr_error<least_error):
            least_error=corr_error
            corr=e
            
    return corr



def error_analysis_single(cross_data):
    #M-parameters of lambda
    #cross-data :array of tuples that were split
    least_error=10**9+7
    corr_error=-1
    e=0
    corr_vec=np.array([0])
    for i in range(0,len(cross_data)):           
        if(i!=0):
            X1=cross_data[0][0]
            Y1=cross_data[0][1]
            test_in=cross_data[i][0]
            test_exp_out=cross_data[i][1]
            for j in range(1,len(cross_data)):
                if(j!=i):
                    X1=np.concatenate((X1,cross_data[j][0]),axis=0)
                    Y1=np.concatenate((Y1,cross_data[j][1]),axis=None)
            wt_vector=extremum(X1,Y1,e)
            test_ac_out=np.dot(test_in,wt_vector)
            corr_error=R2_error(test_exp_out,test_ac_out)
        else:
            test_in=cross_data[0][0]
            test_exp_out=cross_data[0][1]
            X1=cross_data[1][0]
            Y1=cross_data[1][1]
            for j in range(2,len(cross_data)):
               X1=np.concatenate((X1,cross_data[j][0]),axis=0)
               Y1=np.concatenate((Y1,cross_data[j][1]),axis=None)
            wt_vector=extremum(X1,Y1,e)
            test_ac_out=np.dot(test_in,wt_vector)
            corr_error=R2_error(test_exp_out,test_ac_out)
        
       
        if(corr_error<least_error):
            least_error=corr_error
            corr_vec=wt_vector

    print(least_error)
    return corr_vec


def func(L,L_out):   
   L_1=L[:,[5,14,16,18,20,11,13]]
   L_t1=L_out[:,[5,14,16,18,20,11,13]]
   poly=preprocessing.PolynomialFeatures(2,include_bias=True)
   L_11=poly.fit_transform(L_1)
   L_tt=poly.fit_transform(L_t1)

   L_3=L[:,[1,2,6,7,8,9,10,12,23,24,25,26,27,28,30]]
   L_3t=L_out[:,[1,2,6,7,8,9,10,12,23,24,25,26,27,28,30]]
   enc=preprocessing.OneHotEncoder(sparse=False)
   enc.fit(L_3)
   L_31=enc.transform(L_3)
   L_31t=enc.transform(L_3t)
   L_f=np.concatenate((L_11,L_31),axis=1)
   L_ft=np.concatenate((L_tt,L_31t),axis=1)
   return (L_f,L_ft)
   


L=sys.argv
if(L[1]=='a'):
    ftrain=L[2]
    ftest=L[3]
    fout=L[4]
    fweight=L[5]
    in_array=extractfromcsv(ftrain)
    testing_data=extractfromcsv(ftest)    
    (X,Y)=process_train_data(in_array)
    ans=extremum(X,Y,0)
    writetoout(fweight,ans)
    N1=process_test_data(testing_data)
    N2=np.dot(N1,ans)
    writetoout(fout,N2)
    
elif(L[1]=='b'):
    ftrain=L[2]
    ftest=L[3]
    fregularization=L[4]
    fout=L[5]
    fweight=L[6]
    fparam=L[7]
    G1=open(fregularization,"r")
    M=G1.read()
    G1.close()
    L=list(map(float,M.split(",")))
    M=np.array(L,"float64")
    in_array=extractfromcsv(ftrain)
    X,Y=process_train_data(in_array)
    testing_data=extractfromcsv(ftest)
    Xt=process_test_data(testing_data)
    datasets=cross_split(in_array)
    datasets=list(map(process_train_data,datasets))
    ans=error_analysis(datasets,M)
    FF=open(fparam,"w")
    FF.write(str(ans)+"\n")
    FF.close()
    weight_vector=extremum(X,Y,ans)
    writetoout(fweight,weight_vector)
    final_output=np.dot(Xt,weight_vector)
    writetoout(fout,final_output)

elif(L[1]=='c'):
    ftrain=L[2]
    ftest=L[3]
    fout=L[4]
    in_array=extractfromcsv(ftrain)
    X,Y=process_train_data(in_array)
    testing_data=extractfromcsv(ftest)
    Xt=process_test_data(testing_data)
    X,Xt=func(X,Xt)
    cols= [2, 3, 6, 13, 16, 19, 20, 28, 30, 31, 33, 35, 36, 38, 40, 41, 43, 45, 46, 47, 49, 50, 54, 56, 60, 62, 66, 69, 71, 72, 73, 74, 77, 78, 80, 81, 82, 83, 84, 86, 87, 90, 93, 95, 98, 101, 103, 104, 105, 106, 107, 110, 111, 112, 113, 114, 116, 121, 122, 124, 129, 131, 132, 151, 155, 
    156, 157, 160, 161, 162, 165, 166, 167, 169, 171, 172, 173, 175, 177, 179, 181, 189, 199, 200, 201, 202, 215]
    final_wt=extremum(X[:,cols],Y,0.001)
    final_output=np.dot(Xt[:,cols],final_wt)    
    writetoout(fout,final_output)
else:
    print("Invalid Input Command")