"""
Script to create a simple linearly separable dataset
"""
#%%
from random import random
from math import *
def generateNonLinearData(filename, N):
    f = open(filename, 'w')
    count = 0
    while count < N:
        r = random()
        o = 2*3.14*random()
        if(r < 0.25):
            f.write(repr(0.5*r*cos(o)+0.5) +"," +repr(0.5*r*sin(o)+0.5) + ",1\n")
            count = count + 1
        if(r > 0.55):
            f.write(repr(0.5*r*cos(o)+0.5) +"," +repr(0.5*r*sin(o)+0.5) + ",0\n")
            count = count + 1
    f.close()
    
