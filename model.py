import os, sys
import math
import tempfile
from time import sleep
from fpconst import *
import string
import numpy as np
from numpy import array
from numpy import *
from sets import *
import random

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
        self.maxValue = max(itemflat[2:7])
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
                
        return [response, probofcorrect, outputprobs, self.catunitacts, self.activations, self.distances]
    
    ###########################################################
    # learn: recruits cluster and updates weights
    ###########################################################
    def learn(self, item, env):
        # print self.LAMBDAS
        accuracy = -1
        if len(self.clusters) == 0:
            # create new cluster
            self.clusters.append(item)
            self.connections.append(array([0.0]*len(item)*len(item[0])))
            self.stimulate(item,env)
            winnerindex = self.activations.index(max(self.activations))
            self.adjustcluster(winnerindex, item, env)
            maskclus = item
        else:
            # is most activated cluster in the correct category? (Equation #10 in Psych Review)
            winnerindex = self.activations.index(max(self.activations))
            
            # binary "masks" again force learning only on queried dimensions
            maskhash = {'k':0,'?':1,'m':0}
            mask = array(map(lambda x:maskhash[x],env),float64)
            maskitem = map(lambda x,y: x*y, item, mask)
            maskclus = map(lambda x,y: x*y, self.clusters[winnerindex], mask)
            tmpdist = map(lambda x,y: sum(abs(x-y))/2.0,maskitem, maskclus)

            if (sum(tmpdist) != 0.0): # (Equation #11 in Psych Review)
                # create new cluster
                self.clusters.append(item)
                self.connections.append(array([0.0]*len(item)*len(item[0])))
                self.stimulate(item,env)
                winnerindex = self.activations.index(max(self.activations))
                self.adjustcluster(winnerindex, item, env)
                accuracy = 0
            else:
                accuracy = 1
                self.adjustcluster(winnerindex, item, env)  
        return [self.LAMBDAS, self.connections, self.clusters, maskclus[3][1], accuracy, len(self.clusters)]

    ##########################################################
    #SALMA: Unsupervised learning for generalization mode:
    ##########################################################
    def learnunsupervised(self, item, env):

        # is most activated cluster in the correct category? (Equation #10 in Psych Review)
        winnerindex = self.activations.index(max(self.activations))

        # binary "masks" again force learning only on queried dimensions
        maskhash = {'k':0,'?':1,'m':0}
        mask = array(map(lambda x:maskhash[x],env),float64)
        maskclus = map(lambda x,y: x*y, self.clusters[winnerindex], mask)
        
        if (max(self.activations) < self.THRESHOLD): # (Equation #11 in Psych Review)
            # create new cluster
            cluster = item
            if (self.clusters[winnerindex][3][1] == 1.0):
                cluster[3][1] = 2.0
            else:
                cluster[3][1] = 1.0
            self.clusters.append(cluster)
            self.connections.append(array([0.0]*len(cluster)*len(cluster[0])))
            self.stimulate(item,env)
            winnerindex = self.activations.index(max(self.activations))

            self.adjustcluster(winnerindex, cluster, env)

            return [self.LAMBDAS, self.connections, self.clusters, cluster[3][1], len(self.clusters)]
        else:
            self.adjustclusterunsupervised(winnerindex, item, env)
            return [self.LAMBDAS, self.connections, self.clusters, int(round(maskclus[3][1])), len(self.clusters)]       

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
        
        #mask to only update queried dimensions (Equation #14 in Psych Review)
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

        self.LAMBDAS += map(lambda x,y: self.LEARN*x*(1.0-y), b, a)

    #############################################################
    #SALMA: adjusts winning cluster in unsupervised learning mode
    #############################################################
    def adjustclusterunsupervised(self, winner, item, env):
    
        catactsflat = resize(self.catunitacts,(1,len(self.catunitacts)*len(self.catunitacts[0])))[0]
        itemflat = resize(item,(1,len(item)*len(item[0])))[0]
        
        # find connection weight errors
        deltas = map(lambda x,y: self.humbleteach(x,y), itemflat, catactsflat)
        
        # update cluster position (Equation #12 in Psych Review)
        deltas = map(lambda x,y: x-y, item, self.clusters[winner])
        self.clusters[winner] = map(lambda x,y: x+(self.LEARN*y),self.clusters[winner],deltas) 
    
        # update lambdas (Equation #13 in Psych Review)
        a = map(lambda x,y: x*y, self.distances[winner], self.LAMBDAS)
        b = map(lambda x:exp(-1.0*x), a)

        self.LAMBDAS += map(lambda x,y: self.LEARN*x*(1.0-y), b, a)

###########################################################
# END SUSTAIN Class
###########################################################

