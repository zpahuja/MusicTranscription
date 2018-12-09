# includes all the model classes
# they all have the following structure
# class <modelname>:
# 	def __init__(hyperparams):
# 	def train(dl): # takes train datalist
# 	def predict(dl): # takes test datalist and gives out labels

import musicnetIO as mn
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from collections import defaultdict
import csv

notelist=[i for i in range(40,60)]
#notelist=[67,68,69]

# assumes X is d x n matrix where d is dimensions and n is numsamples
# return Xnorm where Xnorm[:,i]=X[:,i]/norm(X[:,i])
def normalize(X,ord=None):
	X1=np.linalg.norm(X,ord=ord,axis=0,keepdims=True)
	return X/np.tile(X1,(X.shape[0],1))


def plotspec(spec):
	librosa.display.specshow(librosa.amplitude_to_db(spec,ref=np.max),sr=44100,y_axis='log', x_axis='time')
	plt.title('Power spectrogram')
	plt.colorbar(format='%+2.0f dB')
	plt.tight_layout()
	plt.show()

def filterbank(X,fb,window_size=4096,hop_length=512):
	if fb=='stft':
		l=[]
		for i in range(X.shape[1]):
			xfb=np.abs(librosa.stft(X[:,i],n_fft=window_size,
					hop_length=hop_length))
			l.append(xfb.flatten())
		Xfb=np.stack(l,axis=1)
		return Xfb

class classifier:
	def __init__(self,n):
		self.clf = [linear_model.SGDClassifier() for _ in range(n)]
		self.n=n

	def partial_fit(self,X,y):
		for i in range(self.n):
			self.clf[i].partial_fit(X,y[:,i],classes=np.array([0,1]))

	def predict(self,X):
		yhat=[self.clf[i].decision_function(X) for i in range(self.n)]
		yhat=np.stack(yhat,axis=1)
		return yhat

# filter bank model
# it just applies a given filterbank and trains a linear classifier on top of it
class fbmodel:
	def __init__(self,fb='stft'):
		self.fb='stft'
		self.clf=classifier(n=len(notelist))

	def train(self,traindl, epochs=1, batchsize=20):
		start=0
		self.mean,self.numsamples=0,0
		for ep in range(epochs):
			for it in range(len(traindl)//batchsize):
				td, start=mn.sampleData(traindl,firstf=start,numfiles=batchsize)
				X=np.concatenate(td['X'],axis=1)
				y=np.concatenate(td['y'],axis=1)[notelist] #keep only note labels
				Xnorm=normalize(X)
				H=filterbank(Xnorm,self.fb)
				logH=np.log(H)
				self.mean = self.mean*self.numsamples+np.mean(logH,axis=1,keepdims=True)*logH.shape[1]
				self.numsamples += logH.shape[1]
				self.mean/=self.numsamples
				self.clf.partial_fit(np.transpose(logH-np.tile(self.mean,(1,logH.shape[1]))), np.transpose(y))
				print ("completed iteration {}/{}, epoch {}/{}, numsamples {}".format(it,len(traindl)//batchsize,ep,epochs,logH.shape[1]))

	def predict(self,testdl):
		td, _ = mn.sampleData(testdl)
		X=np.concatenate(td['X'],axis=1)
		y=np.concatenate(td['y'],axis=1)[notelist] #keep only note labels
		Xnorm=normalize(X)
		H=filterbank(Xnorm,self.fb)
		logH=np.log(H)
		yhat=self.clf.predict(np.transpose(logH-np.tile(self.mean,(1,logH.shape[1]))))
		return np.transpose(yhat), y

def computescore(yhat,y,func='f1'):

	return average_precision_score(y.flatten(), yhat.flatten())

def newdiv(a,b):
	return a/b if b else 0

def results1():
	df = pd.read_csv("experiment1.csv",names=['w','s','note','span','beg'])
	W=[1024,2048,4096,8192,16384]
	S=[128,256,512,1024]
	for w in W:
		for s in [128]:
			for note in range(128):
				df1=df.loc[(df['w']==w) & (df['s']==s) & (df['note']==note)]
				if len(df1)>0:
					print (df1['span'].mean())
					print (df1['beg'].mean())
					if df1['span'].mean()<4:
						print("low span for note {}. (w,s)=({},{})".format(note,w,s))
					if df1['beg'].mean()<3:
						print("low beg for note {}. (w,s)=({},{})".format(note,w,s))
					print ("---")
			print("=================================")

def experiment1():
	W=[1024,2048,4096,8192,16384]
	S=[128,256,512,1024]
	datalist = mn.readData()
	newdatalist = mn.filterData(datalist,keepinstr=[1],keepnotes=0,excludenotes=0,excludeinstr=-1)
	traindl,valdl,testdl = mn.splitData(newdatalist)

	with open("experiment1.csv",'w') as f:
		f.write("") #clear the file

	for w in W:
		for s in S:
			span=[[] for _ in range(128)]
			numbegins=[[] for _ in range(128)]
			for it in range(len(traindl)):

				f,start,end,ftree=traindl[it]
				startcts=defaultdict(int)
				spancts=defaultdict(int)
				for i in range(start,end-w,s):
					s1=ftree[i]
					s3=ftree[i+w//2]
					for intvl in s3.difference(s1):
						startcts[intvl]+=1
					for intvl in s1:
						if intvl[0]==i:
							startcts[intvl]+=1
					for intvl in s3:
						spancts[intvl]+=1
				for intvl in ftree:
					numbegins[intvl[2][1]].append(startcts[intvl])
					span[intvl[2][1]].append(spancts[intvl])
				
			with open("experiment1.csv",'a') as f:
				writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
				for i in range(128):
					for j in range(len(span[i])):
						writer.writerow([w,s,i,span[i][j],numbegins[i][j]])
		print ("completed {} out of {}".format(w,W))

#the main function is a demo of fbmodel training and testing
def main():

	datalist = mn.readData()

	newdatalist = mn.filterData(datalist,keepinstr=[1],keepnotes=0,excludenotes=0,excludeinstr=-1)

	traindl,valdl,testdl = mn.splitData(newdatalist)

	stftmod = fbmodel(fb='stft')

	stftmod.train(traindl)

	yhat, y = stftmod.predict(valdl)

	print (y.shape)
	print (yhat.shape)
	print (y)
	print ("-----")
	print (yhat)

	print ("average_precision_score is: {}".format(computescore(yhat,y,func='f1')))

	yhat,y=np.transpose(yhat),np.transpose(y)
	precision, recall, _ = precision_recall_curve(y.flatten(), yhat.flatten())
	print (precision)
	print (recall)
	fig = plt.figure()
	plt.plot(recall,precision,color=(41/255.,104/255.,168/255.),linewidth=3)
	fig.axes[0].set_xlabel('recall')
	fig.axes[0].set_ylabel('precision')
	plt.show()

if __name__ == '__main__':
	#experiment1()
	#results1()
	main()