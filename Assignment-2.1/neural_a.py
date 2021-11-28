import numpy as np
import pandas as pd
import sys
import math
import time


class Activation_Function:

	def __init__(self,key,outlayer,final_error):

		self.key=key
		self.outlayer=outlayer
		self.final_error=final_error

		self.activation=None
		self.activation_gradient=None
		self.output_activation=None
		self.output_activation_gradient=None
		self.error_func=None
		self.error_func_gradient=None

		if(key==0):
			self.activation=self.log_sigmoid
			self.activation_gradient=self.log_sigmoid_gradient
		elif(key==1):
			self.activation=self.tan_h
			self.activation_gradient=self.tan_h_gradient
		elif(key==2):
			self.activation=self.relu
			self.activation_gradient=self.relu_gradient


		if(outlayer==0):
			self.output_activation=self.softmax
			# self.output_activation_gradient=self.softmax_gradient
		elif(outlayer==1):
			self.output_activation=self.tan_h
			self.output_activation_gradient=self.tan_h_gradient


		if(outlayer==0):
			self.error_func=self.CE
			self.error_func_gradient=self.CE_grad

			
		elif(outlayer==1):
			self.error_func=self.MSE
			self.error_func_gradient=self.MSE_grad

			


	def tan_h(self,x):
		return np.tanh(x)

	def tan_h_gradient(self,x):
		z=np.power(np.tanh(x),2)
		return 1-z

	def log_sigmoid(self,x):
		return 1/(1+np.exp(-1*x))

	def log_sigmoid_gradient(self,x):
		u1=self.log_sigmoid(x)
		return u1*(1-u1)

	def relu(self,x):
		return np.maximum(x,0)

	def relu_gradient(self,x):
		return np.maximum(np.sign(x),0)
		

	def softmax(self,Y):
		Y1=np.zeros(Y.shape)
		for i in range(0,Y.shape[0]):			
			Y1[i]=np.exp(Y[i]-np.max(Y[i]))
			Y1[i]=Y1[i]/np.sum(Y1[i])
		return Y1


	# def softmax(self,Y):
	# 	Y1=np.zeros(Y.shape)
	# 	Y1=np.exp(Y-np.max(Y))
	# 	Y1=Y1/np.sum(Y1)
	# 	return Y1



	def MSE(self,Y,Y1):
		X1=Y-Y1
		X1=np.power(X1,2)
		s1=np.sum(np.sum(X1,axis=0))
		s1*=(-0.5/X1.shape[0])
		return s1


	def CE(self,Y,Y1):
		X1=np.dot(Y,np.log(Y1))
		s1=np.sum(np.sum(X1,axis=0))
		s1*=(-1/X1.shape[0])
		return s1


	def MSE_grad(self,Y,Y1):
		X1=Y1-Y
		return X1


	def CE_grad(self,Y,Y1):
		X1=-1*np.dot(Y,np.log(Y1))
		return X1

		



	


class Neural_Network:

	def __init__(self,Layers_Array,seed_val,act_func_param,out_layer_act,cost_func_param):

		self.Layers=Layers_Array #Sizes of each layers (X_in will be 1xm array)
		self.seed_val=seed_val	
		self.F=Activation_Function(act_func_param,out_layer_act,cost_func_param)
		V=[]
		np.random.seed(self.seed_val) #Layers will have input-layer size as well
		L1=self.Layers
		for i in range(0,len(L1)-1):
			temp=(L1[i]+1)*(L1[i+1])
			V.append((L1[i]+1,L1[i+1]))
		W=[(np.random.normal(0,1,size=(V[i][0],V[i][1]))*np.sqrt(2/(V[i][0]+V[i][1]))).astype("float32") for i in range(len(V))]
		W1=[e[1:,:] for e in W]
		b1=[e[0,:] for e in W]

		self.Weights=[e.astype("float64") for e in W1]
		self.Biases=[e.astype("float64") for e in b1]
		
		
	

	def get_output(self,X1,W1,b1):
		A_out=np.dot(X1,W1)+b1
		Z_out=self.F.activation(A_out)
		return (A_out,Z_out)

	def get_final_output(self,X1,W1,b1):
		A_out=np.dot(X1,W1)+b1
		Z_out=self.F.output_activation(A_out)
		return (A_out,Z_out)


	def forward_propagation(self,X0):
		Z=[X0]  #Z0=input,Z1=first-layer output,...
		A=[[]]  #A0=None #h(A)=Z
		We=self.Weights
		Bi=self.Biases
		for u in range(0,len(We)-1):
			V1=self.get_output(Z[-1],We[u],Bi[u])
			Z.append(V1[1])
			A.append(V1[0])
		V1=self.get_final_output(Z[-1],We[-1],Bi[-1])
		Z.append(V1[1])
		A.append(V1[0])
		return (Z,A)


	def back_propagation(self,X0,Y0):
		(Z,A)=self.forward_propagation(X0)
		
		W=self.Weights
		G=[np.zeros(u.shape) for u in self.Weights]
		G1=[np.zeros(u.shape) for u in self.Biases]
		delta=[np.zeros((2,2),"float64")]+[np.zeros(u.shape) for u in self.Weights]


		if(self.F.outlayer==1):
			Rth=self.F.output_activation_gradient(A[-1]).astype("float64")
			delta[-1]=np.multiply(self.F.error_func_gradient(Y0,Z[-1]),Rth).astype("float64")
		else:
			delta[-1]=(-Y0+Z[-1]).astype("float64")



		for u in range(len(G)-1,0,-1):
			Vth=self.F.activation_gradient(A[u]).astype("float64")
			delta[u]=np.multiply(np.dot(delta[u+1],np.transpose(W[u])),Vth).astype("float64")

		#delta k=dL/da(k) k=1 to last-level

		for u in range(len(G)-1,-1,-1):
			G[u]=np.dot(np.transpose(Z[u]),delta[u+1]).astype("float64")  #G[i]=dL/dW[i]
			G1[u]=delta[u+1]  
		return (G1,G)



	def make_batches(self,X11,Y11,batch_size):
		l=batch_size
		N=(X11.shape[0])//l
		Q=[]
		for i in range(0,N):
			Q.append((X11[i*l:(i+1)*l,:],Y11[i*l:(i+1)*l,:]))
		return Q
		

	


	def iteration_gradient_descent(self,U,r0):

		# nabla_b = [np.zeros(b.shape) for b in self.Biases]
		# nabla_w = [np.zeros(w.shape) for w in self.Weights]
	
		dE_dB,dE_dW= self.back_propagation(U[0],U[1])
		self.Weights = [w-(r0/len(U[0]))*nw for w, nw in zip(self.Weights, dE_dW)]
		dE_dB_f=[np.sum(e,axis=0) for e in dE_dB]
		self.Biases = [b-(r0/len(U[0]))*nb for b, nb in zip(self.Biases, dE_dB_f)]
		





	

	def gradient_descent(self,grad_param,batch_size,X1,Y1,epochs,ini_rate):
		U=self.make_batches(X1,Y1,batch_size)
		for j in range(0,epochs):
			for i in range(0,len(U)):			
				if(grad_param==1):
					r=ini_rate/math.sqrt(j+1)
					self.iteration_gradient_descent(U[i],r)
				else:
					self.iteration_gradient_descent(U[i],ini_rate)
			




	def train_model(self,Xtrain,Ytrain,grad_param,batch_size,epochs,ini_rate):
		self.gradient_descent(grad_param,batch_size,Xtrain,Ytrain,epochs,ini_rate)
		return (self.Weights,self.Biases)
		

		


	def test_model(self,Xtest):
		Z0=self.forward_propagation(Xtest)[0]
		final_ans=Z0[-1]
		U=np.argmax(final_ans,axis=0)
		return U





#######################################################################

def read_from_input(path,train_in_name,test_in_name):
	train=pd.read_csv(path+"/"+train_in_name,index_col=None,header=None)
	test=pd.read_csv(path+"/"+test_in_name,index_col=None,header=None)
	Y_train=train.iloc[:,0]
	X_train=train.iloc[:,1:]
	X_test=test.iloc[:,1:]
	Y_train=np.array(Y_train)
	X_train=np.array(X_train)/255
	X_test=np.array(X_test)/255
	return (X_train.astype("float64"),Y_train,X_test.astype("float64"))


def write_to_out_weights(output_path,W,B):
	for i in range(0,len(W)):
		Af=np.vstack((B[i],W[i]))
		np.save(output_path+"/"+"w_"+str(i+1),Af)
	return 0


def write_to_out_predictions(output_path,V):
	np.save(output_path+"/"+"predictions",V)
	return 0


def process_output_data(Y,nrs):
	Y1=np.zeros((Y.shape[0],nrs))
	for i in range(0,Y.shape[0]):
		Y1[i][Y[i]]=1
	return Y1


	







def main():
	L=sys.argv
	fin=L[1]
	fout=L[2]
	fparam=L[3]
	f1=open(fparam)
	A=f1.readlines()
	f1.close()
	epochs=int(A[0])
	batch_size=int(A[1])
	S=A[2].strip()
	Layers=list(map(int,S[1:len(S)-1].split(",")))
	learning_rate_type=int(A[3])
	learning_rate_value=float(A[4])
	activation_function_type=int(A[5])
	loss_function_type=int(A[6])
	seed_value=int(A[7])
	train_in_name="toy_dataset_train.csv"
	test_in_name="toy_dataset_test.csv"
	(X,Y000,Xt)=read_from_input(L[1],train_in_name,test_in_name)
	N=Neural_Network([X.shape[1]]+Layers,seed_value,activation_function_type,0,loss_function_type)
	Y=process_output_data(Y000,Layers[-1])
	(W11,B11)=N.train_model(X,Y,learning_rate_type,batch_size,epochs,learning_rate_value)
	write_to_out_weights(fout,W11,B11)
	V=N.test_model(Xt)
	write_to_out_predictions(L[2],V)


'''
    Xtrain,Ytrain,grad_param,batch_size,epochs,ini_rate
    (self,Layers_Array,seed_val,X_inp,Y_out,act_func_param,out_layer_act,cost_func_param):
	8 lines specifying epochs, batch size, a list specifying the architecture([100,50,10]
	implies 2 hidden layers with 100 and 50 neurons and 10 neurons in the output layer), learning rate
	type(0 for fixed and 1 for adaptive), learning rate value, activation function(0 for log sigmoid, 1 for
	tanh, 2 for relu), loss function(0 for CE and 1 for MSE), seed value for the numpy.random.normal
	used(some whole number).
'''


main()