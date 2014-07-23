import os, sys
import math
import tempfile
from time import sleep
from fpconst import *
import string
import misc
# from Numeric import *
# from MLab import *
import numpy as np
from numpy import array
from numpy import *
from sets import *
import random

data = [[1.0,0.56,0.13,1.0],
		[2.0,0.59,0.13,1.0], 
		[3.0,0.63,0.13,1.0],
		[4.0,0.66,0.13,1.0],
		[5.0,0.69,0.13,0.0],
		[6.0,0.72,0.13,0.0],
		[7.0,0.76,0.13,0.0],
		[8.0,0.79,0.13,0.0],
		[9.0,0.56,0.135,1.0],
		[10.0,0.59,0.135,1.0],
		[11.0,0.63,0.135,1.0],
		[12.0,0.66,0.135,1.0],
		[13.0,0.69,0.135,0.0],
		[14.0,0.72,0.135,0.0],
		[15.0,0.76,0.135,0.0],
		[16.0,0.56,0.14,0.0],
		[17.0,0.56,0.14,1.0],
		[18.0,0.59,0.14,1.0],
		[19.0,0.63,0.14,1.0],
		[20.0,0.66,0.14,1.0],
		[21.0,0.69,0.14,0.0],
		[22.0,0.72,0.14,0.0],
		[23.0,0.76,0.14,0.0],
		[24.0,0.79,0.14,0.0],
		[25.0,0.56,0.145,1.0],
		[26.0,0.59,0.145,1.0],
		[27.0,0.63,0.145,1.0],
		[28.0,0.63,0.145,1.0],
		[29.0,0.69,0.145,0.0],
		[30.0,0.72,0.145,0.0],
		[31.0, 0.76,0.145,0.0],
		[32.0,0.79,0.145,0.0],
		[33.0,0.56,0.15,1.0],
		[34.0,0.59,0.15,1.0],
		[35.0,0.63,0.15,1.0],
		[36.0,0.66,0.15,1.0],
		[37.0,0.69,0.15,0.0],
		[38.0,0.72,0.15,0.0],
		[39.0,0.76,0.15,0.0],
		[40.0,0.79,0.15,0.0],
		[41.0,0.56,0.155,0.0],
		[42.0,0.59,0.155,0.0],
		[43.0,0.63,0.155,0.0],
		[44.0,0.66,0.155,0.0],
		[45.0,0.69,0.155,2.0],
		[46.0,0.72,0.155,2.0],
		[47.0,0.76,0.155,2.0],
		[48.0,0.79,0.155,2.0],
		[49.0,0.56,0.16,0.0],
		[50.0,0.59,0.16,0.0],
		[51.0,0.63,0.16,0.0],
		[52.0,0.66,0.16,0.0],
		[53.0,0.69,0.16,2.0],
		[54.0,0.72,0.16,2.0],
		[55.0,0.76,0.16,2.0],
		[56.0,0.79,0.165,2.0],
		[57.0,0.56,0.165,0.0],
		[58.0,0.59,0.165,0.0],
		[59.0,0.63,0.165,0.0],
		[60.0,0.66,0.165,0.0],
		[61.0,0.69,0.165,2.0],
		[62.0,0.72,0.165,2.0],
		[63.0,0.76,0.165,2.0],
		[64.0,0.79,0.165,2.0]]

env = ['m','k','k','?']

categories=[1.0,2.0]
numtrainingblocks=4

##########################################################
# SUSTAIN Class
###########################################################
class SUSTAIN:

	###########################################################
	# __init__: initializes and reset the network structure
	###########################################################
	def __init__(self, r, beta, d, threshold, learn, initalphas):
		
		self.R = r
		self.BETA = beta
		self.D = d
		self.THRESHOLD = threshold
		self.LEARN = learn
		self.LAMBDAS = initalphas
		
		self.clusters = []
		self.activations = []
		self.connections = []
		self.catunitacts = []
		self.coutputs = []
		
		self.maxValue = 0.0
		self.minValue = 0.0
		
	###########################################################
	# stimulate: present item and env for forward stimulation
	###########################################################
	def stimulate(self, item, env):
		itemflat = resize(item,(1,len(item)*len(item[0])))[0]
		print itemflat
		self.maxValue = max(itemflat[2:7])
		print self.maxValue
		self.minValue = min(itemflat[2:7])
		
		# this binary mask will block out queried or missing dims from the calcs
		maskhash = {'k':1,'?':0,'m':0}
		mask = array(map(lambda x:maskhash[x],env),float64)
		
		# compute distances between item and each cluster (Equation #4 in Psych Review)
		self.distances = []
		for cluster in self.clusters:
			self.distances.append(array(map(lambda x,y: sum(abs(x-y))/2.0,item, cluster),float64))
		
		# compute activation of each cluser  (Equation #5 in Psych. Review)
		lambda2r = array(mask*pow(self.LAMBDAS,self.R),float64)
		sumlambda2r = sum(lambda2r)
		self.activations = []
		for clustdist in self.distances:
			 self.activations.append(sum(lambda2r*exp(-1.0*self.LAMBDAS*clustdist))/sumlambda2r)

		# calculate output of most activated cluster after competition (Equation #6 in Psych Review)
		if len(self.activations) > 0:
			a = array(map(lambda x: pow(x, self.BETA),self.activations),float64)
			b = sum(a)
			self.coutputs = map(lambda x,y: (float(x)*float(y))/float(b), a, self.activations)
			winnerindex = self.coutputs.index(max(self.coutputs))
			# passing winner's output over connection weights (Equation #7 in Psych Review)
			self.catunitacts = array(float(self.coutputs[winnerindex])*self.connections[winnerindex],float64)
			self.catunitacts = resize(self.catunitacts,(len(item),len(item[0])))
		else:
			# set all category unit outputs to zero
			self.catunitacts = resize(array([0.,0.]),(len(item),len(item[0])))
		
		# compute output probabilities via luce choice rule (Equation #8 in Psych Review)
		a = map(lambda x: exp(self.D*x), self.catunitacts)
		b = map(lambda x: sum(x), a)
		outputprobs = array(map(lambda x,y: x/y, a, b))
		
		# compute probability of making correct response
		outputprobs = array(map(lambda x,y: x*y, outputprobs, 1.0-mask))
		outputprobsflat = resize(outputprobs,(1,len(outputprobs)*len(outputprobs[0])))[0]
		probofcorrect = max(itemflat*outputprobsflat)
		
		# generate a response 
		if random.random() > probofcorrect:
			response = False
		else:
			response = True
		
		# print response, probofcorrect, outputprobs, self.catunitacts, self.activations, self.distances		
		return [response, probofcorrect, outputprobs, self.catunitacts, self.activations, self.distances]
	
	###########################################################
	# learn: recruits cluster and updates weights
	###########################################################
	def learn(self, item, env):
		# print self.LAMBDAS
		if len(self.clusters) == 0:
			# create new cluster
			self.clusters.append(item)
			self.connections.append(array([0.0]*len(item)*len(item[0])))
			self.stimulate(item,env)
			winnerindex = self.activations.index(max(self.activations))
			self.adjustcluster(winnerindex, item, env)
		else:
			# is most activated cluster in the correct category? (Equation #10 in Psych Review)
			winnerindex = self.activations.index(max(self.activations))
			
			# binary "masks" again force learning only on queried dimensions
			maskhash = {'k':0,'?':1,'m':0}
			mask = array(map(lambda x:maskhash[x],env),float64)
			maskitem = map(lambda x,y: x*y, item, mask)
			# print maskitem
			maskclus = map(lambda x,y: x*y, self.clusters[winnerindex], mask)
			# print maskclus
			tmpdist = map(lambda x,y: sum(abs(x-y))/2.0,maskitem, maskclus)
		
			if (max(self.activations) < self.THRESHOLD) or (sum(tmpdist) != 0.0): # (Equation #11 in Psych Review)
				# create new cluster
				self.clusters.append(item)
				self.connections.append(array([0.0]*len(item)*len(item[0])))
				self.stimulate(item,env)
				winnerindex = self.activations.index(max(self.activations))
				self.adjustcluster(winnerindex, item, env)
				
			else:
				self.adjustcluster(winnerindex, item, env)	
		return [self.LAMBDAS, self.connections, self.clusters]

				
	###########################################################
	# humbleteach: adjusts winning cluster (Equation #9 in Psych Review)
	###########################################################
	def humbleteach(self, a, m):
		if ( ((m > self.maxValue) and (a == self.maxValue)) or 
		     ((m < self.minValue) and (a == self.minValue))):
			return 0
		else:
			return a - m
			
	###########################################################
	# adjustcluster: adjusts winning cluster
	###########################################################
	def adjustcluster(self, winner, item, env):
	
		catactsflat = resize(self.catunitacts,(1,len(self.catunitacts)*len(self.catunitacts[0])))[0]
		itemflat = resize(item,(1,len(item)*len(item[0])))[0]
		
		# find connection weight errors
		deltas = map(lambda x,y: self.humbleteach(x,y), itemflat, catactsflat)
		
		# mask to only update queried dimensions (Equation #14 in Psych Review)
		maskhash = {'k':0,'?':1,'m':0}
		mask = array(map(lambda x:maskhash[x],env),float64)
		deltas = map(lambda x,y: x*y, resize(deltas,(len(item),len(item[0]))), mask)
		deltas = resize(deltas, (1,len(item)*len(item[0])))[0]
		self.connections[winner] += self.LEARN*deltas*self.coutputs[winner]
		
		# update cluster position (Equation #12 in Psych Review)
		deltas = map(lambda x,y: x-y, item, self.clusters[winner])
		self.clusters[winner] = map(lambda x,y: x+(self.LEARN*y),self.clusters[winner],deltas) 
	
		# update lambdas (Equation #13 in Psych Review)
		a = map(lambda x,y: x*y, self.distances[winner], self.LAMBDAS)
		b = map(lambda x:exp(-1.0*x), a)
		#print map(lambda x,y: self.LEARN*x*(1.0-y), b, a)
		self.LAMBDAS += map(lambda x,y: self.LEARN*x*(1.0-y), b, a)

###########################################################
# END SUSTAIN Class
###########################################################

subjectdata=[]
directory=os.getcwd() + '/results/results'
dataitems = []
for i in data:
	row = []
	for j in i:
		row.append(np.array([0, j]))
	dataitems.append(row)

# s = Set([])
# maxNDimValues = max(map(lambda x: len(s.union(Set(x))), transpose(data))) # The max no. of values of dimensions
# valueMap = identity(maxNDimValues).astype(float64)
# dataitems = map(lambda x: map(lambda y: valueMap[y-1], x), data)


def training(model,data):
	phase='training'
	examplenumbers=[1.0,2.0,9.0,10.0,55.0,56.0,63.0,64.0]
	nblocks=4
	nitemscorrect = 0
	trainingblock = [dataitems[0],dataitems[1],dataitems[8],dataitems[9],
	dataitems[54],dataitems[55],dataitems[62],dataitems[63]]

	for i in range(nblocks):
		random.shuffle(trainingblock)
		# print trainingblock
		for j in trainingblock:
			# print j 
			correctcategory=j[3]
			[res,prob,outunits,outacts,act,dist] = model.stimulate(j, env)
			if res == True:
				nitemscorrect += 1
				accuracy=1
				[lambdas, clus, conn] = model.learn(j,env)
			else:
				accuracy=0
			trialdata=["SUSTAIN",phase,i+1,res,accuracy]
			subjectdata.append(trialdata)
			write_file(directory,subjectdata,',')
	generalization(model,data)

def generalization(model,data):
	nitemscorrect = 0
	phase='generalization'
	random.shuffle(dataitems)
        for j in dataitems:
			[res,prob,outunits,outacts,act,dist] = model.stimulate(j, env)
			if res == True:
				nitemscorrect += 1
				accuracy=1
				[lambdas, clus, conn] = model.learn(j,env)
			else:
				accuracy=0
			trialdata=["SUSTAIN",phase,1,res,accuracy]
			subjectdata.append(trialdata)
			write_file(directory,subjectdata,',')

def write_file(filename,data,delim):
	datafile=open(filename,'w')
	for i in data:
		line='\n'
		line=str(i)+delim+line
		datafile.write(line)
	datafile.close()

###########################################################
# main
###########################################################
def main():
	model = SUSTAIN(r = 0.0, beta = 3.97491, d = 6.514972, 
			threshold = 0, learn = 0.1150532,
			initalphas = array([1.0]*len(data[0]),float64) )
	training(model, data)	
		
###########################################################
# let's start
###########################################################

if __name__ == '__main__':
	main() 