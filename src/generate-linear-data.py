"""
Script to create a simple linearly separable dataset
"""
#%%
from random import random

def generateLinearData(filename, N):
    f = open(filename, 'w')
    count = 0
    while count < N:
        x = random()
        y = random()
        if(y > x + 0.1):
            f.write(repr(x) +"," +repr(y) + ",1\n")
            count = count + 1
        if(y < x - 0.1):
            f.write(repr(x) +"," +repr(y) + ",0\n")
            count = count + 1
    f.close()
    
