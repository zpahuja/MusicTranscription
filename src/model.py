# includes all the model classes
# they all have the following structure
# class <modelname>:
# 	def __init__(hyperparams):
# 	def train(dl): # takes train datalist
# 	def predict(dl): # takes test datalist and gives out labels

import musicnetIO as mn
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

notelist=[67,68,69]

# assumes X is d x n matrix where d is dimensions and n is numsamples
# return Xnorm where Xnorm[:,i]=X[:,i]/norm(X[:,i])
def normalize(X,ord=None):
	X1=np.linalg.norm(X,ord=ord,axis=0,keepdims=True)
	return X/np.tile(X1,(X.shape[0],1))


def plotspec(spec):
	librosa.display.specshow(librosa.amplitude_to_db(spec,ref=np.max),y_axis='log', x_axis='time')
	plt.title('Power spectrogram')
	plt.colorbar(format='%+2.0f dB')
	plt.tight_layout()
	plt.show()

def filterbank(X,fb):
	if fb=='stft':
		l=[]
		for i in range(X.shape[1]):
			l.append(np.abs(librosa.stft(X[:,i])).flatten())
		Xfb=np.stack(l,axis=1)
		return Xfb

class classifier:
	def __init__(self,n):
		self.clf = [linear_model.SGDClassifier(max_iter=10, tol=1e-3) for _ in range(n)]
		self.n=n

	def partial_fit(self,X,y):
		for i in range(self.n):
			print (i)
			self.clf[i].partial_fit(X,y[:,i],classes=np.array([0,1]))

	def predict(self,X):
		yhat=[self.clf[i].predict(X) for i in range(self.n)]
		yhat=np.stack(yhat,axis=1)
		return yhat

# filter bank model
# it just applies a given filterbank and trains a linear classifier on top of it
class fbmodel:
	def __init__(self,fb='stft'):
		self.fb='stft'
		self.clf=classifier(n=len(notelist))

	def train(self,traindl, epochs=1, batchsize=2):
		start=0
		for _ in range(epochs):
			td, start=mn.sampleData(traindl,start=start,numfiles=batchsize)
			X=np.concatenate(td['X'],axis=1)
			y=np.concatenate(td['y'],axis=1)[notelist] #keep only note labels
			Xnorm=normalize(X)
			H=filterbank(Xnorm,self.fb)
			logH=np.log(H)
			self.clf.partial_fit(np.transpose(X), np.transpose(y))

	def predict(self,testdl):
		td, _ = mn.sampleData(testdl)
		X=np.concatenate(td['X'],axis=1)
		y=np.concatenate(td['y'],axis=1)[notelist] #keep only note labels
		Xnorm=normalize(X)
		H=filterbank(Xnorm,self.fb)
		logH=np.log(H)
		yhat=self.clf.predict(np.transpose(X))
		return np.transpose(yhat), y

def computescore(yhat,y,func='f1'):

	return average_precision_score(y.flatten(), yhat.flatten())

#the main function is a demo of fbmodel training and testing
def main():

	datalist = mn.readData()

	newdatalist = mn.filterData(datalist,keepinstr=[1],keepnotes=0,excludenotes=0,excludeinstr=-1)

	traindl,valdl,testdl = mn.splitData(newdatalist)

	td, _ = mn.sampleData(valdl)
	y=np.concatenate(td['y'],axis=1)[notelist] #keep only note labels
	print (np.ndarray.tolist(np.sum(y,axis=1)))

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
	fig = plt.figure()
	plt.plot(recall,precision,color=(41/255.,104/255.,168/255.),linewidth=3)
	fig.axes[0].set_xlabel('recall')
	fig.axes[0].set_ylabel('precision')
	plt.show()

if __name__ == '__main__':
	main()