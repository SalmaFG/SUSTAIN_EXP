import os
import sys
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
from model import *

# Stimuli data of the experiment obtained from Conaway & Kurtz
data = [[1.0, 0.56, 1.3, 1.0], 
        [2.0, 0.59, 1.3, 1.0],  
        [3.0, 0.63, 1.3, -1.0], 
        [4.0, 0.66, 1.3, -1.0], 
        [5.0, 0.69, 1.3, -1.0], 
        [6.0, 0.72, 1.3, -1.0], 
        [7.0, 0.76, 1.3, -1.0], 
        [8.0, 0.79, 1.3, -1.0], 
        [9.0, 0.56, 1.35, 1.0], 
        [10.0, 0.59, 1.35, 1.0], 
        [11.0, 0.63, 1.35, -1.0], 
        [12.0, 0.66, 1.35, -1.0], 
        [13.0, 0.69, 1.35, -1.0], 
        [14.0, 0.72, 1.35, -1.0], 
        [15.0, 0.76, 1.35, -1.0], 
        [16.0, 0.79, 1.35, -1.0], 
        [17.0, 0.56, 1.4, -1.0], 
        [18.0, 0.59, 1.4, -1.0], 
        [19.0, 0.63, 1.4, -1.0], 
        [20.0, 0.66, 1.4, -1.0], 
        [21.0, 0.69, 1.4, -1.0], 
        [22.0, 0.72, 1.4, -1.0], 
        [23.0, 0.76, 1.4, -1.0], 
        [24.0, 0.79, 1.4, -1.0], 
        [25.0, 0.56, 1.45, -1.0], 
        [26.0, 0.59, 1.45, -1.0], 
        [27.0, 0.63, 1.45, -1.0], 
        [28.0, 0.66, 1.45, -1.0], 
        [29.0, 0.69, 1.45, -1.0], 
        [30.0, 0.72, 1.45, -1.0], 
        [31.0,  0.76, 1.45, -1.0], 
        [32.0, 0.79, 1.45, -1.0], 
        [33.0, 0.56, 1.5, -1.0], 
        [34.0, 0.59, 1.5, -1.0], 
        [35.0, 0.63, 1.5, -1.0], 
        [36.0, 0.66, 1.5, -1.0], 
        [37.0, 0.69, 1.5, -1.0], 
        [38.0, 0.72, 1.5, -1.0], 
        [39.0, 0.76, 1.5, -1.0], 
        [40.0, 0.79, 1.5, -1.0], 
        [41.0, 0.56, 1.55, -1.0], 
        [42.0, 0.59, 1.55, -1.0], 
        [43.0, 0.63, 1.55, -1.0], 
        [44.0, 0.66, 1.55, -1.0], 
        [45.0, 0.69, 1.55, -1.0], 
        [46.0, 0.72, 1.55, -1.0], 
        [47.0, 0.76, 1.55, -1.0], 
        [48.0, 0.79, 1.55, -1.0], 
        [49.0, 0.56, 1.6, -1.0], 
        [50.0, 0.59, 1.6, -1.0], 
        [51.0, 0.63, 1.6, -1.0], 
        [52.0, 0.66, 1.6, -1.0], 
        [53.0, 0.69, 1.6, -1.0], 
        [54.0, 0.72, 1.6, -1.0], 
        [55.0, 0.76, 1.6, 2.0], 
        [56.0, 0.79, 1.6, 2.0], 
        [57.0, 0.56, 1.65, -1.0], 
        [58.0, 0.59, 1.65, -1.0], 
        [59.0, 0.63, 1.65, -1.0], 
        [60.0, 0.66, 1.65, -1.0], 
        [61.0, 0.69, 1.65, -1.0], 
        [62.0, 0.72, 1.65, -1.0], 
        [63.0, 0.76, 1.65, 2.0], 
        [64.0, 0.79, 1.65, 2.0]]

env = ['m', 'k', 'k', '?']

directory1 = os.getcwd() + '/results/training.csv'
directory2 = os.getcwd() + '/results/generalization.csv'

# Preparing the data to fit the model
dataitems = []
for i in data:
    row = []
    for j in i:
        row.append(np.array([0,  j]))
    dataitems.append(row)

# The 8 examples that will be used for the training phase
trainingblock = [dataitems[0], dataitems[1], dataitems[8], dataitems[9], 
    dataitems[54], dataitems[55], dataitems[62], dataitems[63]]

##########################################################
  # Training phase:
##########################################################
def training(model, data):
    phase = "training"
    nblocks = 4
    subjectdata = []
    
    for i in range(nblocks):
        random.shuffle(trainingblock)
        for j in trainingblock:
            trialn = int(floor(j[0][1]))
            [res, prob, outunits, outacts, act, dist] = model.stimulate(j, env)
            [lambdas, clus, conn, response, accuracy, nclus] = model.learn(j, env)
            trialdata=["SUSTAIN", phase, i+1, trialn, response, accuracy, nclus]
            subjectdata.append(trialdata)
        # Writing the results of the phase in training.csv
        write_file(phase, directory1, subjectdata, ',')

##########################################################
  # Generalization phase:
##########################################################
def generalization(data):
    phase = "generalization"
    subjectdata2 = []

    for j in dataitems:
        trialn=int(floor(j[0][1]))
        [res, prob, outunits, outacts, act, dist] = model.stimulate(j, env)
        [lambdas, clus, conn, response, nclus] = model.learn_unsupervised(j, env)
        trialdata=["SUSTAIN", phase, 1, trialn, response, nclus]
        subjectdata2.append(trialdata)
    # Writing the results of the phase in generalization.csv
    write_file(phase, directory2, subjectdata2, ',')

###########################################################
  # Just a method that does both the training and 
  # generalization phase a 100 times to collect results
  # only for the experiment.
###########################################################
def testing(data):
    subjectdata = []
    subjectdata2 = []
    for i in range(100):
        model = SUSTAIN(r = 38.0,  beta = 5.386305,  d = 5.0,  
            threshold = 0.89,  learn = 0.09361126, 
            initalphas = array([1.0]*len(data[0]), float64) )
        for k in range(4):
            random.shuffle(trainingblock)
            for j in trainingblock:
                trialn=int(floor(j[0][1]))
                [res, prob, outunits, outacts, act, dist] = model.stimulate(j, env)
                [lambdas, clus, conn, response, accuracy, nclus] = model.learn(j, env)
                trialdata = ["SUSTAIN", "training", k+1, trialn, response, accuracy, nclus]
                subjectdata.append(trialdata)
        write_file("training", directory1, subjectdata, ',')
        random.shuffle(dataitems)
        for k in dataitems:
            trialn = int(floor(k[0][1]))
            [res, prob, outunits, outacts, act, dist] = model.stimulate(k, env)
            [lambdas,  clus,  conn,  response,  n] = model.learn_unsupervised(k, env)
            trialdata = ["SUSTAIN", "generalization", i+1, trialn, response, n]
            subjectdata2.append(trialdata)
    write_file("generalization", directory2, subjectdata2, ',')

##########################################################
  # Writes results to csv file:
##########################################################
def  write_file(phase, filename, data, delim):
    datafile = open(filename, 'w')
    if (phase == "training"):
        datafile.write("Model, Phase, Block, Item, Response, Accuracy, Clusters number" + '\n')
    else:
        datafile.write("Model, Phase, Block, Item, Response, Clusters number" + '\n')
    for i in data:
        si = ', '.join(str(ie) for ie in i)
        line = str(si) + delim + '\n'
        datafile.write(line)
    datafile.close()

###########################################################
# main
###########################################################
def main():
    testing(data)
        
###########################################################
# start
###########################################################
if __name__ == '__main__':
    main() 