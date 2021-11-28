import cv2
import numpy as np
import pandas as pd
from skimage import io,transform
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import torch
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import glob
import os
from IPython.display import Image
import sys

class DevanagariDataset(Dataset):
    
    def __init__(self, data_csv, train = True , img_transform = None):
        """
        Dataset init function
        
        INPUT:
        data_csv: Path to csv file containing [data, labels]
        train: 
            True: if the csv file has [data, labels] (Train data and Public Test Data) 
            False: if the csv file has only [data] and labels are not present.
        img_transform: List of preprocessing operations need to performed on image. 
        """
        self.data_csv = data_csv
        self.img_transform = img_transform
        self.is_train = train
        
        data = pd.read_csv(data_csv, header=None)
        if self.is_train:
            images = data.iloc[:,:-1].to_numpy()
            labels = data.iloc[:,-1].astype(int)
        else:
            images = data.iloc[:,:].to_numpy()
            labels = None
        
        self.images = images
        self.labels = labels
        print("Total Images: {}, Data Shape = {}".format(len(self.images), images.shape))
        
    def __len__(self):
        """Returns total number of samples in the dataset"""
        return len(self.images)
    
    def __getitem__(self, idx):
        """
        Loads image of the given index and performs preprocessing.
        
        INPUT: 
        idx: index of the image to be loaded.
        
        OUTPUT:
        sample: dictionary with keys images (Tensor of shape [1,C,H,W]) and labels (Tensor of labels [1]).
        """
        image = self.images[idx]
        image = np.array(image).astype(np.uint8).reshape(32, 32, 1)
        
        if self.is_train:
            label = self.labels[idx]
        else:
            label = -1
        
        image = self.img_transform(image)
#         print(image.shape, label, type(image))
        sample = {"images": image, "labels": label}
        return sample



class Neural_Network(Module):

	def __init__(self):
		super(Neural_Network,self).__init__()

		L1=Conv2d(1,32,3,1)
		L2=BatchNorm2d(32)
		L3=ReLU(inplace=True)
		L4=MaxPool2d(2,stride=2)
		L5=Conv2d(32,64,3,1)
		L6=BatchNorm2d(64)
		L7=ReLU(inplace=True)
		L8=MaxPool2d(2,stride=2)
		L9=Conv2d(64,256,3,1)
		L10=BatchNorm2d(256)
		L11=ReLU(inplace=True)
		L12=MaxPool2d(2,stride=1)
		L13=Conv2d(256,512,3,1)
		L14=ReLU(inplace=True)

		L16=ReLU(inplace=True)
		L17=Dropout(p=0.2)



		self.Seq1=Sequential(L1,L2,L3,L4,L5,L6,L7,L8,L9,L10,L11,L12,L13,L14)
		self.Seq2=Sequential(L16,L17)
		self.Linear1=Linear(1*1*512,256)
		self.Linear2=Linear(256,46)


	def forward(self,x):
		x=self.Seq1(x)
		x=torch.flatten(x,1)
		x=self.Linear1(x)
		x=self.Seq2(x)
		x=torch.flatten(x,1)
		x=self.Linear2(x)

		return x




def load_model(test_data):
	BATCH_SIZE = 200 # Batch Size. Adjust accordingly
	NUM_WORKERS = 20 # Number of threads to be used for image loading. Adjust accordingly.

	img_transforms = transforms.Compose([transforms.ToPILImage(),transforms.ToTensor()])


	# Test DataLoader
	# test_data = "" # Path to test csv file
	test_dataset = DevanagariDataset(data_csv = test_data, train=False, img_transform=img_transforms)
	test_loader = DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle=False, num_workers = NUM_WORKERS)
	return test_loader



def write_to_file(L,fname):
	f=open(fname,"w")
	for i in range(0,len(L)):
		s=str(L[i])+"\n"
		f.write(s)
	f.close()



	



def main():
	# train_data="/content/drive/MyDrive/train_data_shuffled.csv"
	# test_data="/content/drive/MyDrive/public_test.csv"
	L=sys.argv	
	test_data=L[1]
	import_model=L[2]
	prediction_file=L[3]

	model=torch.load(import_model)
	
	if torch.cuda.is_available():
		model=model.cuda()

	epochs=8
	test_loader=load_model(test_data)
	Pred=[]
	

	for i in range(0,epochs):
		model.train()
	with torch.no_grad():
		for batch_idx, sample in enumerate(test_loader):
			test_images = sample['images']
			if(torch.cuda.is_available()):
				test_images=test_images.cuda()
				outputs=model(test_images)
				_,pred=torch.max(outputs,1)
				Pred+=pred.tolist()

	
		
	
	
	write_to_file(Pred,prediction_file)

			
						
main()








		