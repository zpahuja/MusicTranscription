# this script implements functions of musicnet input pipeline.
# IO pipeline is as follows:

# 1. We start with a list of files and their labels. 
# Files are represented by fileids, 
# and labels are represented by list of start, end, instrument and note.
# So we have list({fileid,x,intervaltree[list(start,end,{instrument,note})]}) at the start.
# where 'x' varies over all instruments and notes. 
# if x is positive its notes and if negative then instrument
# x takes values {-128,-127,...,-1,1,2,...,128}

# 2. Next step is to obtain only parts that contain a given set of instrument and note.
# So at the end of this step we have: 
# list({fileid,intervaltree[list(start,end,{instrument,note})]})

# 3. Next step is to split the data into training and validation sets.
# Here we will have 2 list of file ids one for val and one for train.

# 4. (optional) subsampling where we take a small part of the data for prototyping.
# there are 3 subsampling options.
# -> on a rolling window basis (for processing the whole data in its entirety)
# -> randomly sample frames
# -> randomly sample chunks

# 5. Final step in the input pipeline is to obtain raw frames from the interval notation.
# For both train and val we get X,y described below:
# -> Both are lists where, len(X)=len(y)=number of coherent music segments
# -> X[i] and y[i] numpy arrays where, 
# --> X[i][0:frame_size,j] is the wavesequence of jth frame in ith segment.
# --> y[i][0:128,j] is binary vector signifying note and instrument labels



import numpy as np
from intervaltree import Interval,IntervalTree
from os import listdir
from os.path import isfile, join
from scipy.io import wavfile
import csv
import random
import pickle

# constants
fs = 44100
frame_size = 16384 #taken from cnn_amt.pdf paper
stride = frame_size//4 
datadir="../data/musicnet/" #default datadirectory
demolist=['1727','1730','2303','2677','2678'] #used as toy list for testing the code

#returns list of fileids
def getFileids():
	with open(join(datadir,"metadata.csv"),'r') as f:
		reader=csv.reader(f,delimiter=',',)
		fileids=[row[0] for row in reader][1:] #skipping header
	return fileids

#first step of pipeline
def readData(datapath=0):
	if datapath!=0:
		global datadir
		datadir=datapath

	if isfile(join(datadir,'binaries/datalist.pkl')):
		with open(join(datadir,'binaries/datalist.pkl'),'rb') as f:
			datalist=pickle.load(f)
	else:
		fileids=getFileids()

		fileids=demolist #just for testing

		datalist=[]
		for f in fileids:
			print ("reading {}".format(f))
			with open(join(datadir,"labels/"+f+".csv")) as labf:
				reader=csv.reader(labf,delimiter=',',)
				labels=[row[:4] for row in reader][1:] #ignoring columns other than first 4 and header
			cts={}
			for i in range(1,129):
				cts[i]=IntervalTree()
				cts[-i]=IntervalTree()
			for start,end,instrument,note in labels:
				cts[-int(instrument)].addi(int(start),int(end),(int(instrument),int(note)))
				cts[int(note)].addi(int(start),int(end),(int(instrument),int(note)))
			datalist.append((f,cts))

		with open(join(datadir,'binaries/datalist.pkl'),'wb') as f:
			pickle.dump(datalist,f)
	return datalist

#second step of the pipeline
#keepnotes and keepinstrs are list of notes and instruments to keep
#similarly exclude notes and instrs are ones to avoid even if it means ignoring keep ones
def filterData(datalist, keepnotes=0, excludenotes=0, keepinstr=0, excludeinstr=0):
	paramstr=str(keepnotes)+str(excludenotes)+str(keepinstr)+str(excludeinstr)
	if isfile(join(datadir,'binaries/'+paramstr+".pkl")):
		with open(join(datadir,'binaries/'+paramstr+".pkl"),'rb') as f:
			newdatalist=pickle.load(f)
	else:
		if keepnotes==0:
			keepnotes=[x for x in range(1,129)]
		if keepinstr==0:
			keepinstr=[x for x in range(1,129)]

		if excludenotes==0:
			excludenotes=[]
		if excludenotes==-1:
			excludenotes=[x for x in range(1,129) if x not in keepnotes]

		if excludeinstr==0:
			excludeinstr=[]
		if excludeinstr==-1:
			excludeinstr=[x for x in range(1,129) if x not in keepinstr]

		newdatalist=[]
		for f,cts in datalist:
			print ("processing file {}".format(f))
			#starting with empty tree
			ftree=IntervalTree() 

			#taking OR with all keepinstr and notes
			for instr in keepinstr:
				ftree|=cts[-instr] 
			for note in keepnotes:
				ftree|=cts[note]

			#taking set difference with exclude instr and notes
			for instr in excludeinstr:
				ftree-=cts[-instr] 
			for note in excludenotes:
				ftree-=cts[note]
			newdatalist.append((f,ftree))

		with open(join(datadir,'binaries/'+paramstr+".pkl"),'wb') as f:
			pickle.dump(newdatalist,f)
	return newdatalist

#third step in the pipeline
def splitData(datalist):
	#computing duration for each file as it will decide which files go to which split
	datadict,fduration={},{}
	for f,ftree in datalist:
		L=[]
		for intvl in ftree:
			L.append((intvl[0],1))
			L.append((intvl[1],-1))
		L.sort(key=lambda x: x[0]-0.01*x[1])
		Lunion=[]
		duration=0
		start, end, ct=-1,-1,0
		for pt in L:
			if ct==0:
				start=pt[0]
			ct+=pt[1]
			if ct==0:
				end=pt[0]
				Lunion.append((start,end))
				duration+=(end-start)
				start,end=-1,-1
		datadict[f]=(ftree,Lunion)
		fduration[f]=duration

	#getting the list of test and train files
	train,test,val=[],[],[]
	trd,ted,vald=0,0,0
	with open(join(datadir,"metadata.csv"),'r') as f:
		reader=csv.reader(f,delimiter=',')
		fileids=[row[0:2] for row in reader][1:] #skipping header

		fileids=zip(demolist,['train','train','test','train','train'])

		for f,sp in fileids:
			if sp=='train':
				train.append(f)
				trd+=fduration[f]
			else:
				test.append(f)
				ted+=fduration[f]

	print ("duration of all test files: {} seconds".format(ted/fs))
	print ("test files: {} ".format(test))

	random.shuffle(train)
	while vald<0.1*trd:
		val.append(train[-1])
		trd-=fduration[train[-1]]
		vald+=fduration[train[-1]]
		del train[-1]

	print ("duration of all train files: {} seconds".format(trd/fs))
	print ("train files: {} ".format(train))
	print ("duration of all val files: {} seconds".format(vald/fs))
	print ("val files: {} ".format(val))

	newdatalist=[[],[],[]]
	L=[train,val,test]
	for i in range(3):
		for f in L[i]:
			ftree,Lunion = datadict[f]
			for start,end in Lunion:
				newdatalist[i].append((f,start,end,ftree))
	return newdatalist


def getLabel(tree,i):
	lab=np.zeros(128*2)
	for intvl in tree[i]:
		lab[intvl[2][0]-1+128]=1
		lab[intvl[2][1]-1]=1
	return lab

#final step in the pipeline. Sampling data can be done on a rolling bases
#start=index of starting file in the datalist
#numfiles= number files to retriev
def sampleData(datalist,firstf=0,numfiles=-1):
	flist=[i for i in range(len(datalist))]+[i for i in range(len(datalist))]
	
	currf,numsegs=firstf,0
	X,y=[],[]
	wavdata={}
	while numsegs!=numfiles:
		if ((numfiles==-1) and currf==len(datalist)):
			break
		f,start,end,ftree=datalist[flist[currf]]
		#print (f)
		if f not in wavdata:
			fs1, wavdata[f] = wavfile.read(join(datadir,"data/"+f+".wav"))
			if fs1!=fs:
				print ("sample rate doesn't match")
		dt=wavdata[f]
		xl,yl=[],[]
		for t in range(start+frame_size//2,end-frame_size//2,stride):
			xl.append(dt[t-frame_size//2:t+frame_size//2])
			yl.append(getLabel(ftree,t))
		if len(xl)>0:
			numsegs+=1
			xl=np.array(xl)
			yl=np.array(yl)
			X.append(np.transpose(xl))
			y.append(np.transpose(yl))
		currf+=1
			
	return {'X':X, 'y':y}, flist[currf]


#the main function is a demo of musicnetIO interface
def main():
	datalist=readData()
	# print (datalist[0][0])
	# print ("-----------------------")
	# print (datalist[0][1][-43])
	# print ("-----------------------")
	# print (datalist[0][1][53])
	# print ("-----------------------")

	newdatalist=filterData(datalist,keepinstr=[1],keepnotes=0,excludenotes=0,excludeinstr=-1)
	# print (newdatalist[0][1])
	# print ("-----------------------")

	traindl,valdl,testdl=splitData(newdatalist)

	trainData,_=sampleData(traindl,start=1,numfiles=1)


if __name__ == '__main__':
	main()